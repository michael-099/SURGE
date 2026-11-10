"""Train SURGE-D.

    python -u scripts/train.py --config configs/g1_vjepa21.yaml [--set key=value ...]

Stage 1: backbone frozen, decoder + prior trained.  Stage 2: LoRA on the backbone's last 12 blocks, decoder
initialized from a Stage-1 checkpoint (train.init_from).  The run stops cleanly (saving last.pt) if
SAFETY_STOP.txt exists in the repo root or STOP exists in the run folder; it resumes from last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from surge.backbones import add_lora, build_backbone  # noqa: E402
from surge.config import load_config  # noqa: E402
from surge.data import (ClipDataset, SaliconDataset, dhf1k_videos, parse_stavis, read_split_file,  # noqa: E402
                        stavis_split_videos, ucf_videos)
from surge.infer import evaluate  # noqa: E402
from surge.lora import lora_parameters  # noqa: E402
from surge.losses import saliency_loss  # noqa: E402
from surge.model import SurgeD, central  # noqa: E402
from surge.model import last as last_frame  # noqa: E402  (main() uses `last` for the checkpoint path)


def stop_requested(run_dir: Path) -> str | None:
    for p in (ROOT / "SAFETY_STOP.txt", run_dir / "STOP"):
        if p.exists():
            return str(p)
    return None


def videos_for(cfg, part: str):
    d = cfg["data"]
    if d["dataset"] == "dhf1k":
        ids = read_split_file(ROOT / d[f"{part}_split"])
        return dhf1k_videos(d["root"], ids)
    if d["dataset"] == "ucf":
        vids = ucf_videos(d["root"], "train" if part in ("train", "holdout") else "test")
        if part in ("train", "holdout"):
            hold = set(open(ROOT / d["holdout_split"]).read().split())
            vids = [v for v in vids if (v.name in hold) == (part == "holdout")]
        return vids
    if parse_stavis(d["dataset"]):
        return stavis_split_videos(d["dataset"], "val" if part == "val" else part, ROOT / "configs/splits")
    if d["dataset"] == "hollywood_vig":
        from surge.data import hollywood_vig_videos
        return hollywood_vig_videos(part, ROOT / "configs/splits")
    raise ValueError(d["dataset"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    run_dir = ROOT / cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), newline="\n")
    logf = open(run_dir / "log.jsonl", "a")

    def log(msg, **kv):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
        print(line, flush=True)
        if kv:
            logf.write(json.dumps({"time": time.time(), **kv}) + "\n")
            logf.flush()

    tr, md, d = cfg["train"], cfg["model"], cfg["data"]
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    bb = build_backbone(md["backbone"], grad_ckpt=tr.get("grad_ckpt", False))
    for p in bb.parameters():
        p.requires_grad_(False)
    stage2 = tr["stage"] == 2
    if stage2:
        names = add_lora(bb, r=tr["lora_r"], alpha=tr["lora_alpha"])
        log(f"LoRA on {len(names)} linears")
    use_audio = md.get("audio", False)
    model = SurgeD(bb, use_scm=md["scm"], prior=md["prior"], aux=md["aux"], c=md["channels"], audio=use_audio).to(device)
    if tr.get("init_from"):
        ck0 = torch.load(ROOT / tr["init_from"], map_location="cpu", weights_only=False)
        if use_audio and not any(k.startswith("av.") for k in ck0["decoder"]):  # visual checkpoint -> new audio modules
            r = model.decoder.load_state_dict(ck0["decoder"], strict=False)
            assert not r.unexpected_keys and all(k.startswith("av.") for k in r.missing_keys), r
            log(f"audio modules new ({len(r.missing_keys)} tensors); the rest from {tr['init_from']}")
        else:
            model.decoder.load_state_dict(ck0["decoder"])
        log(f"decoder initialized from {tr['init_from']}")
        if stage2 and "lora" in ck0:
            bb.load_state_dict(ck0["lora"], strict=False)
            log("LoRA weights initialized from the same checkpoint")

    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_lora = sum(p.numel() for p in lora_parameters(bb))
    log(f"params: decoder {n_dec / 1e6:.2f}M, LoRA {n_lora / 1e6:.2f}M, backbone {sum(p.numel() for p in bb.parameters()) / 1e6:.1f}M")

    size = tuple(d["size"])
    g = (size[0] // 16 * 4, size[1] // 16 * 4)
    # data.target: "central" (the thesis model) or "last" (causal variant, research/log.md section 12D)
    tmode = d.get("target", "central")
    causal = tmode == "last"
    sel = last_frame if causal else central
    if causal and d.get("salicon"):
        raise ValueError("static co-training is defined for central targets only")
    if causal:
        log("causal variant: each clip is supervised on its last sampled frame; holdout scored causally")
    train_ds = ClipDataset(videos_for(cfg, "train"), T=d["T"], stride=d["stride"], size=size, out_size=g,
                           clips_per_video=tr["clips_per_video"], hflip=d["hflip"], seed=tr["seed"], audio=use_audio,
                           target=tmode)
    ds = train_ds
    if d.get("salicon"):
        sal = SaliconDataset(d["salicon_root"], "train", T=d["T"], size=size, out_size=g, limit=d.get("salicon_limit"))
        ds = ConcatDataset([train_ds, sal])
        log(f"static co-training: {len(sal)} SALICON images per epoch")
    val_videos = videos_for(cfg, "holdout")
    loader = DataLoader(ds, batch_size=tr["batch"], shuffle=True, num_workers=tr["workers"], pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4,
                        generator=torch.Generator().manual_seed(tr["seed"]))

    groups = [{"params": [p for n, p in model.decoder.named_parameters() if not n.startswith("prior.")], "lr": tr["lr"]},
              {"params": list(model.decoder.prior.parameters()) if model.decoder.prior is not None else [], "lr": tr["lr_prior"],
               "weight_decay": 0.0}]
    if stage2:
        groups.append({"params": list(lora_parameters(bb)), "lr": tr["lr_lora"]})
    opt = torch.optim.AdamW(groups, weight_decay=tr["wd"])
    steps_per_epoch = len(loader)
    total = steps_per_epoch * tr["epochs"]
    warm = int(steps_per_epoch * tr["warmup_epochs"])
    base = [gr["lr"] for gr in opt.param_groups]

    def set_lr(step):
        f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for gr, b in zip(opt.param_groups, base):
            gr["lr"] = b * f

    start_epoch, step, best = 0, 0, -1.0
    last = run_dir / "last.pt"

    def trainable_state():
        st = {"decoder": model.decoder.state_dict()}
        if stage2:
            st["lora"] = {n: p.detach().cpu() for n, p in bb.named_parameters() if "lora_" in n}
        return st

    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(ck["decoder"])
        if stage2:
            bb.load_state_dict(ck["lora"], strict=False)
        opt.load_state_dict(ck["opt"])
        start_epoch, step, best = ck["epoch"], ck["step"], ck["best"]
        log(f"resumed from last.pt: epoch {start_epoch}, step {step}, best {best:.4f}")

    def save_last(epoch, step_in_epoch_done: bool):
        st = trainable_state()
        st.update(opt=opt.state_dict(), epoch=epoch, step=step, best=best)
        torch.save(st, run_dir / "last.tmp")
        os.replace(run_dir / "last.tmp", last)

    T = d["T"]
    for epoch in range(start_epoch, tr["epochs"]):
        train_ds.set_epoch(epoch)
        model.train()
        model.backbone.eval()  # no dropout / drop-path in the backbone
        t0, seen, run = time.time(), 0, {"loss": 0.0, "kl": 0.0, "cc": 0.0, "nss": 0.0}
        skip = step - epoch * steps_per_epoch  # mid-epoch resume: skip already-done batches
        for bi, batch in enumerate(loader):
            if bi < skip:
                continue
            why = stop_requested(run_dir)
            if why:
                save_last(epoch, False)
                log(f"STOP requested ({why}); saved last.pt at step {step}; exiting", event="stop", step=step)
                return 0
            x = batch["frames"].to(device, non_blocking=True)
            y = batch["density"].to(device, non_blocking=True)
            fx = batch["fixations"].to(device, non_blocking=True)
            set_lr(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, backbone_grad=stage2,
                            audio=batch["audio"].to(device, non_blocking=True) if use_audio else None)
            loss, parts = saliency_loss(sel(out["logits"], T), y, fx, tr["w_cc"], tr["w_nss"])
            if model.decoder.aux:
                for k in ("aux16", "aux8"):
                    la, _ = saliency_loss(sel(out[k], T), y, fx, tr["w_cc"], 0.0)
                    loss = loss + tr["w_aux"] * la
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for gr in opt.param_groups for p in gr["params"]], 1.0)
            opt.step()
            step += 1
            seen += x.shape[0]
            run["loss"] += loss.item()
            for k in ("kl", "cc", "nss"):
                run[k] += parts[k]
            nb = bi + 1 - skip
            if step % tr["log_every"] == 0:
                log(f"ep {epoch} step {step}/{total} loss {run['loss'] / nb:.4f} cc {run['cc'] / nb:.4f} "
                    f"nss {run['nss'] / nb:.3f} {seen / (time.time() - t0):.1f} clips/s "
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            if step % tr["ckpt_every"] == 0:
                save_last(epoch, False)
        nb = max(steps_per_epoch - skip, 1)
        log(f"epoch {epoch} done in {(time.time() - t0) / 60:.1f} min", event="train_epoch", epoch=epoch,
            **{k: v / nb for k, v in run.items()})
        # ---- internal validation
        t1 = time.time()
        mean, _ = evaluate(model, val_videos, T, d["stride"], size, device, sigma=cfg["eval"]["sigma"],
                           frame_step=cfg["eval"]["frame_step"], workers=cfg["eval"]["workers"], causal=causal)
        log(f"epoch {epoch} holdout " + " ".join(f"{k}={v:.4f}" for k, v in mean.items()) +
            f" ({(time.time() - t1) / 60:.1f} min)", event="holdout", epoch=epoch, **mean)
        st = trainable_state()
        torch.save(st, run_dir / f"epoch_{epoch}.pt")
        if mean["cc"] > best:
            best = mean["cc"]
            torch.save({**st, "epoch": epoch, "holdout": mean}, run_dir / "best.pt")
            log(f"new best holdout CC {best:.4f} at epoch {epoch}")
        save_last(epoch + 1, True)
    (run_dir / "DONE").write_text(json.dumps({"best_holdout_cc": best}), newline="\n")
    log("training finished", event="done", best=best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
