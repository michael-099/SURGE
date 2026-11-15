"""Train several SURGE-D decoders on one frozen-encoder stream (research/log.md sections 15B, 15C, 15D).

Each batch is encoded once; every decoder then takes its own forward pass, backward pass and optimiser step (one
after the other, so peak memory is that of a single decoder). Clips, order and augmentation are those of
scripts/train.py for the same seed. Each decoder is scored on the holdout after each of its epochs with the protocol
of train.py (every eval.frame_step-th frame, sigma eval.sigma), sharing the encoder pass; its best epoch is kept.

    python -u scripts/train_multi.py --config configs/g1_base.yaml --set run_dir=runs/inv_c_s0 \
        model.backbone=vjepa2_g train.epochs=3 train.warmup_epochs=0.25 train.seed=0 \
        --variants R,V1,V2,V3,V4,V5,V6,lr1e-4,lr9e-4,nocc,nonss,R6
    python -u scripts/train_multi.py ... --probes 1-40      (section 15B: a linear readout per block, plus P0)

Outputs per variant in <run_dir>/<variant>/: best.pt, log.jsonl, holdout_e<k>.json (per-video scores) and DONE.
Stops cleanly (saving every variant's state) if SAFETY_STOP.txt (repo root) or <run_dir>/STOP exists; resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from surge.backbones import build_backbone  # noqa: E402
from surge.config import load_config  # noqa: E402
from surge.data import ClipDataset, inference_windows, load_gray, load_rgb  # noqa: E402
from surge.infer import _IO_POOL, to_native  # noqa: E402
from surge.losses import log_softmax2d, saliency_loss  # noqa: E402
from surge.metrics import all_metrics  # noqa: E402
from surge.model import Decoder, LinearReadout, central  # noqa: E402

METRICS = ("cc", "nss", "sim", "kld")


def variant_specs(names: list[str], probes: list[int] | None):
    """name -> dict(kind, decoder kwargs, tap selection, training overrides)."""
    base = {
        "R": {},
        "V1": {"kind": "linear"},
        "V2": {"dec": {"tap_mode": "last"}},
        "V3": {"dec": {"temporal": False}},
        "V4": {"dec": {"topdown": False}},
        "V5": {"dec": {"c": 64}},
        "V6": {"dec": {"prior": False, "aux": False}},
        "lr1e-4": {"train": {"lr": 1e-4}},
        "lr9e-4": {"train": {"lr": 9e-4}},
        "nocc": {"train": {"w_cc": 0.0}},
        "nonss": {"train": {"w_nss": 0.0}},
        "R6": {"train": {"epochs": 6}},
    }
    specs = {}
    for n in names:
        if n not in base:
            raise ValueError(f"unknown variant {n}")
        specs[n] = {"kind": "decoder", "dec": {}, "train": {}, **base[n]}
    for layer in probes or []:
        specs[f"L{layer}"] = {"kind": "probe", "layer": layer, "dec": {}, "train": {}}
    if probes:
        specs["P0"] = {"kind": "prior_only", "dec": {}, "train": {}}
    return specs


class Variant:
    def __init__(self, name, spec, d, tr, device, seed, tap_index):
        self.name, self.spec = name, spec
        torch.manual_seed(seed)
        kind = spec["kind"]
        if kind == "decoder":
            kw = {"c": 256, "prior": True, "aux": True, **spec["dec"]}
            c = kw.pop("c")
            self.module = Decoder(d, c, **kw)
            self.sel = lambda taps: taps  # the four default taps
        elif kind == "linear":
            self.module = LinearReadout(4 * d)
            self.sel = lambda taps: taps
        elif kind == "probe":
            self.module = LinearReadout(d)
            i = tap_index[spec["layer"]]
            self.sel = lambda taps, i=i: [taps[i]]
        elif kind == "prior_only":
            self.module = LinearReadout(d, zero=True)
            self.sel = lambda taps: [taps[0]]
        else:
            raise ValueError(kind)
        self.module.to(device)
        t = {**tr, **spec["train"]}
        self.epochs, self.lr, self.w_cc, self.w_nss = t["epochs"], t["lr"], t["w_cc"], t["w_nss"]
        self.w_aux, self.warmup = t["w_aux"], t["warmup_epochs"]
        prior = getattr(self.module, "prior", None)
        groups = [{"params": [p for n, p in self.module.named_parameters() if not n.startswith("prior.")],
                   "lr": self.lr},
                  {"params": list(prior.parameters()) if prior is not None else [], "lr": t["lr_prior"],
                   "weight_decay": 0.0}]
        groups = [g for g in groups if g["params"]]
        self.opt = torch.optim.AdamW(groups, weight_decay=t["wd"])
        self.base = [g["lr"] for g in self.opt.param_groups]
        self.params = [p for g in self.opt.param_groups for p in g["params"]]
        self.best = -1.0
        self.n_params = sum(p.numel() for p in self.module.parameters())

    def set_lr(self, step, steps_per_epoch):
        total = steps_per_epoch * self.epochs
        warm = int(steps_per_epoch * self.warmup)
        f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * f


def _score_many(args):
    """Metrics of K predictions for one frame; the ground truth is read once."""
    preds, mp, fp, sigma, skip_empty = args
    gt = load_gray(mp).astype(np.float64)
    fix = load_gray(fp)
    if skip_empty and not (fix > 0).any():
        return [{k: float("nan") for k in METRICS} for _ in preds]
    return [all_metrics(to_native(p, gt.shape, sigma), gt, fix, None, METRICS) for p in preds]


@torch.no_grad()
def evaluate_multi(backbone, variants, videos, T, stride, size, device, frame_step, sigma, workers, batch=4):
    """Holdout scores for every variant, sharing the encoder pass (windows and batches as surge.infer.evaluate)."""
    for v in variants:
        v.module.eval()
    per_video = {v.name: {} for v in variants}
    with ProcessPoolExecutor(workers) as ex:
        for vid in videos:
            n = len(vid)
            wins = list(inference_windows(n, T, stride))
            cache, maps = {}, {v.name: {} for v in variants}
            for b0 in range(0, len(wins), batch):
                chunk = wins[b0:b0 + batch]
                need = sorted({j for idx, _ in chunk for j in idx if j not in cache})
                for j, im in zip(need, _IO_POOL.map(lambda j: load_rgb(vid.frames[j], size), need)):
                    cache[j] = im
                x = torch.from_numpy(np.stack([np.stack([cache[j] for j in idx]) for idx, _ in chunk]))
                x = x.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
                lo = min(min(i) for i, _ in chunk)
                for k in [k for k in cache if k < lo]:
                    del cache[k]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    taps = backbone(x)
                    taps = [t.float() if t.dtype != torch.bfloat16 else t for t in taps]
                    for v in variants:
                        p = log_softmax2d(v.module(v.sel(taps), T_out=T)["logits"].float()).exp().cpu().numpy()
                        for bi, (_, assign) in enumerate(chunk):
                            for f, q in assign:
                                if f % frame_step:
                                    continue
                                a = int(np.floor(q))
                                w = q - a
                                maps[v.name][f] = p[bi, a] if w == 0 else (1 - w) * p[bi, a] + w * p[bi, min(a + 1, T - 1)]
            frames = sorted(maps[variants[0].name])
            jobs = [([maps[v.name][f] for v in variants], vid.maps[f], vid.fixations[f], sigma, vid.skip_empty)
                    for f in frames]
            res = list(ex.map(_score_many, jobs, chunksize=4))
            for vi, v in enumerate(variants):
                per_video[v.name][vid.name] = {k: float(np.nanmean([r[vi][k] for r in res])) for k in METRICS}
    out = {}
    for v in variants:
        pv = per_video[v.name]
        out[v.name] = ({k: float(np.mean([x[k] for x in pv.values()])) for k in METRICS}, pv)
        v.module.train()
    return out


def parse_probes(s: str | None):
    if not s:
        return None
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--variants", default="")
    ap.add_argument("--probes", default=None, help="e.g. 1-40: one linear readout per block output, plus P0")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tr, md, d = cfg["train"], cfg["model"], cfg["data"]
    run_dir = ROOT / cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**cfg, "variants": args.variants, "probes": args.probes},
                                                     indent=1), newline="\n")
    logf = open(run_dir / "log.txt", "a")

    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    seed = tr["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    probes = parse_probes(args.probes)
    names = [n for n in args.variants.split(",") if n]
    specs = variant_specs(names, probes)
    if probes:
        layers = probes
    else:
        layers = None  # the backbone's default taps
    bb = build_backbone(md["backbone"], img=d["size"][0], taps=layers)
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.to(device).eval()
    tap_index = {l: i for i, l in enumerate(layers)} if layers else {}
    D = bb.embed_dim
    variants = [Variant(n, s, D, tr, device, seed * 1000 + i, tap_index) for i, (n, s) in enumerate(specs.items())]
    for v in variants:
        (run_dir / v.name).mkdir(exist_ok=True)
        log(f"variant {v.name}: {v.spec} params {v.n_params / 1e6:.3f}M epochs {v.epochs} lr {v.lr}")

    size = tuple(d["size"])
    g = (size[0] // 16 * 4, size[1] // 16 * 4)
    from scripts.train import videos_for
    train_ds = ClipDataset(videos_for(cfg, "train"), T=d["T"], stride=d["stride"], size=size, out_size=g,
                           clips_per_video=tr["clips_per_video"], hflip=d["hflip"], seed=seed)
    val_videos = videos_for(cfg, "holdout")
    loader = DataLoader(train_ds, batch_size=tr["batch"], shuffle=True, num_workers=tr["workers"], pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4,
                        generator=torch.Generator().manual_seed(seed))
    spe = len(loader)
    T = d["T"]
    max_epochs = max(v.epochs for v in variants)

    # ---- resume
    state_f = run_dir / "state.json"
    start_epoch, step = 0, 0
    if state_f.exists():
        st = json.loads(state_f.read_text())
        start_epoch, step = st["epoch"], st["step"]
        for v in variants:
            ck = torch.load(run_dir / v.name / "last.pt", map_location="cpu", weights_only=False)
            v.module.load_state_dict(ck["module"])
            v.opt.load_state_dict(ck["opt"])
            v.best = ck["best"]
        log(f"resumed: epoch {start_epoch}, step {step}")

    def save_state(epoch, step):
        for v in variants:
            torch.save({"module": v.module.state_dict(), "opt": v.opt.state_dict(), "best": v.best},
                       run_dir / v.name / "last.tmp")
            os.replace(run_dir / v.name / "last.tmp", run_dir / v.name / "last.pt")
        state_f.write_text(json.dumps({"epoch": epoch, "step": step}), newline="\n")

    def stop_requested():
        for p in (ROOT / "SAFETY_STOP.txt", run_dir / "STOP"):
            if p.exists():
                return str(p)
        return None

    for epoch in range(start_epoch, max_epochs):
        train_ds.set_epoch(epoch)
        active = [v for v in variants if epoch < v.epochs]
        for v in active:
            v.module.train()
        t0 = time.time()
        skip = step - epoch * spe
        sums = {v.name: 0.0 for v in active}
        nb = 0
        for bi, batch in enumerate(loader):
            if bi < skip:
                continue
            why = stop_requested()
            if why:
                save_state(epoch, step)
                log(f"STOP requested ({why}); state saved at step {step}; exiting")
                return 0
            x = batch["frames"].to(device, non_blocking=True)
            y = batch["density"].to(device, non_blocking=True)
            fx = batch["fixations"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                taps = bb(x)
            taps = [t.float() if t.dtype != torch.bfloat16 else t for t in taps]
            for v in active:
                v.set_lr(step, spe)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = v.module(v.sel(taps), T_out=T)
                loss, _ = saliency_loss(central(out["logits"], T), y, fx, v.w_cc, v.w_nss)
                for k in [k for k in out if k.startswith("aux")]:
                    la, _ = saliency_loss(central(out[k], T), y, fx, v.w_cc, 0.0)
                    loss = loss + v.w_aux * la
                v.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(v.params, 1.0)
                v.opt.step()
                sums[v.name] += loss.item()
            step += 1
            nb += 1
            if step % tr["log_every"] == 0:
                log(f"ep {epoch} step {step}/{spe * max_epochs} " +
                    " ".join(f"{k}={s / nb:.4f}" for k, s in list(sums.items())[:6]) +
                    f" {nb * x.shape[0] / (time.time() - t0):.1f} clips/s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            if step % tr["ckpt_every"] == 0:
                save_state(epoch, step)
        log(f"epoch {epoch} trained in {(time.time() - t0) / 60:.1f} min ({len(active)} decoders)")
        t1 = time.time()
        res = evaluate_multi(bb, active, val_videos, T, d["stride"], size, device, cfg["eval"]["frame_step"],
                             cfg["eval"]["sigma"], cfg["eval"]["workers"])
        for v in active:
            mean, pv = res[v.name]
            rec = {"epoch": epoch, "mean": mean, "per_video": pv, "train_loss": sums[v.name] / max(nb, 1)}
            (run_dir / v.name / f"holdout_e{epoch}.json").write_text(json.dumps(rec), newline="\n")
            with open(run_dir / v.name / "log.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch, **mean, "train_loss": rec["train_loss"]}) + "\n")
            if mean["cc"] > v.best:
                v.best = mean["cc"]
                torch.save({"module": v.module.state_dict(), "spec": v.spec, "epoch": epoch, "holdout": mean},
                           run_dir / v.name / "best.pt")
                (run_dir / v.name / "best.json").write_text(json.dumps(rec), newline="\n")
            if epoch + 1 == v.epochs:
                (run_dir / v.name / "DONE").write_text(json.dumps({"best_holdout_cc": v.best}), newline="\n")
        log(f"epoch {epoch} holdout ({(time.time() - t1) / 60:.1f} min): " +
            " ".join(f"{v.name}={res[v.name][0]['cc']:.4f}" for v in active))
        save_state(epoch + 1, step)
    (run_dir / "DONE").write_text(json.dumps({v.name: v.best for v in variants}), newline="\n")
    log("all variants finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
