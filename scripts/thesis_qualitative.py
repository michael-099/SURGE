"""Thesis material (GPU): qualitative DHF1K-val examples and the inference cost of SURGE-D.

    python -u scripts/thesis_qualitative.py --out runs/thesis

1. Qualitative examples. Selection rule, fixed before any map is looked at: rank the 100 DHF1K-val videos by the
   per-video CC of the ViT-g 3-seed ensemble (runs/vitg/val_ens3.json); take the videos at the 10th, 50th and 90th
   percentile, plus the video with the largest and the one with the smallest per-video CC difference to SalFoM
   (runs/metric_check/salfom_val_all.json). Frame: the middle frame of each video (index n // 2). Maps: the ViT-g
   3-seed ensemble at sigma 8 (its holdout-tuned value, runs/vitg/sigma_ens3.json) and SalFoM's released val maps.
2. Cost. FLOPs of one forward pass on a 16 x 256 x 256 clip (torch.utils.flop_counter) for the ViT-L and ViT-g
   models, split into encoder and decoder; wall-clock throughput on one val video with frames preloaded in memory
   (bf16 autocast, batch 4 and batch 1, first window excluded as warm-up). Stops if SAFETY_STOP.txt exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from scripts.evaluate import get_videos, load_model  # noqa: E402
from surge.data import inference_windows, load_gray, load_rgb  # noqa: E402
from surge.infer import predict_video, to_native  # noqa: E402
from surge.metrics import cc as cc_metric  # noqa: E402

SALFOM = Path("D:/Research/baselines/salfom_val/maps")
VITG = [ROOT / f"runs/vitg_s{k}" for k in range(3)]
VITL = ROOT / "runs/main_b1_s0"


def stop():
    if (ROOT / "SAFETY_STOP.txt").exists():
        print("SAFETY_STOP.txt present; exiting", flush=True)
        sys.exit(2)


def select_videos():
    ours = json.load(open(ROOT / "runs/vitg/val_ens3.json"))["per_video"]
    sal = json.load(open(ROOT / "runs/metric_check/salfom_val_all.json"))["per_video"]
    vids = sorted(ours, key=lambda v: ours[v]["cc"])
    picks = []
    for q in (0.10, 0.50, 0.90):
        picks.append(("p%d" % round(q * 100), vids[int(round(q * (len(vids) - 1)))]))
    d = {v: ours[v]["cc"] - sal[v]["cc"] for v in ours}
    picks.append(("best_vs_salfom", max(d, key=d.get)))
    picks.append(("worst_vs_salfom", min(d, key=d.get)))
    seen, out = set(), []
    for tag, v in picks:
        if v not in seen:
            seen.add(v)
            out.append((tag, v, ours[v]["cc"], sal[v]["cc"]))
    return out


def qualitative(out_dir: Path, device):
    stop()
    picks = select_videos()
    print("selected:", picks, flush=True)
    sigma = json.load(open(ROOT / "runs/vitg/sigma_ens3.json"))["best_sigma"]
    models = [load_model(r, "best.pt", device) for r in VITG]
    vids = {v.name: v for v in get_videos("dhf1k", "val", None)}
    rec = {}
    for tag, name, cc_ours_video, cc_sal_video in picks:
        stop()
        v = vids[name]
        n = len(v)
        f = n // 2
        acc = None
        for m, cfg in models:
            d = cfg["data"]
            o = dict(predict_video(m, v, d["T"], d["stride"], tuple(d["size"]), device))
            acc = o[f].astype(np.float64) if acc is None else acc + o[f]
        gt = load_gray(v.maps[f]).astype(np.float64)
        fix = load_gray(v.fixations[f])
        ours = to_native(acc / len(models), gt.shape, sigma)
        sal = cv2.imread(str(SALFOM / name / f"{f + 1:04d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float64)
        if sal.shape != gt.shape:
            sal = cv2.resize(sal, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(cv2.imread(v.frames[f]), cv2.COLOR_BGR2RGB)
        rec[name] = dict(tag=tag, frame=f + 1, n_frames=n, img=img, gt=gt, fix=(fix > 127).astype(np.uint8),
                         ours=ours.astype(np.float32), salfom=sal.astype(np.float32),
                         cc_ours_frame=cc_metric(ours, gt), cc_salfom_frame=cc_metric(sal, gt),
                         cc_ours_video=cc_ours_video, cc_salfom_video=cc_sal_video)
        print(f"{tag} {name} frame {f + 1}/{n}: CC ours {rec[name]['cc_ours_frame']:.3f} "
              f"SalFoM {rec[name]['cc_salfom_frame']:.3f} (video means {cc_ours_video:.3f} / {cc_sal_video:.3f})",
              flush=True)
    np.savez_compressed(out_dir / "qualitative.npz", **{f"{k}__{kk}": vv for k, r in rec.items() for kk, vv in r.items()})
    (out_dir / "qualitative.json").write_text(json.dumps(
        {k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in r.items()
             if kk not in ("img", "gt", "fix", "ours", "salfom")} for k, r in rec.items()}, indent=1), newline="\n")
    del models
    torch.cuda.empty_cache()


def flops(model, device):
    x = torch.randn(1, 16, 3, 256, 256, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with FlopCounterMode(display=False) as fc:
            taps = model.backbone(x)
        enc = fc.get_total_flops()
        taps = [t.float() for t in taps]
        with FlopCounterMode(display=False) as fc:
            model.decoder(taps, None, T_out=16)
        dec = fc.get_total_flops()
    return enc / 1e12, dec / 1e12


def timing(model, video, device, batch):
    n = len(video)
    wins = list(inference_windows(n, 16, 2))
    frames = {j: load_rgb(video.frames[j], (256, 256)) for j in range(n)}
    xs = [torch.from_numpy(np.stack([frames[j] for j in idx])).permute(0, 3, 1, 2) for idx, _ in wins]
    times, outs = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(xs), batch):
            x = torch.stack(xs[b0:b0 + batch]).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if b0 > 0:  # first call is warm-up
                times.append(dt)
                outs += sum(len(a) for _, a in wins[b0:b0 + batch])
    return {"batch": batch, "windows": len(wins), "seconds": float(sum(times)), "output_frames": outs,
            "frames_per_second": float(outs / sum(times)), "ms_per_window": float(1000 * sum(times) / max(1, len(times) * batch))}


def cost(out_dir: Path, device):
    res = {}
    video = get_videos("dhf1k", "val", None)[0]
    for name, run in (("vit_l", VITL), ("vit_g", VITG[0])):
        stop()
        m, _ = load_model(run, "best.pt", device)
        enc, dec = flops(m, device)
        torch.cuda.reset_peak_memory_stats()
        t4 = timing(m, video, device, 4)
        mem4 = torch.cuda.max_memory_allocated() / 2**30
        t1 = timing(m, video, device, 1)
        res[name] = {"encoder_tflops_per_clip": enc, "decoder_tflops_per_clip": dec,
                     "tflops_per_output_frame": (enc + dec) / 16, "decoder_params_M": sum(p.numel() for p in m.decoder.parameters()) / 1e6,
                     "backbone_params_M": sum(p.numel() for p in m.backbone.parameters()) / 1e6,
                     "encoder_params_M": sum(p.numel() for n_, p in m.backbone.named_parameters() if "predictor" not in n_) / 1e6,
                     "batch4": t4, "batch1": t1, "peak_mem_GB_batch4": mem4, "video": video.name, "video_frames": len(video)}
        print(name, json.dumps(res[name]), flush=True)
        del m
        torch.cuda.empty_cache()
    res["gpu"] = torch.cuda.get_device_name(0)
    res["torch"] = torch.__version__
    (out_dir / "cost.json").write_text(json.dumps(res, indent=1), newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/thesis")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    qualitative(out, device)
    cost(out, device)
    print("done", flush=True)


if __name__ == "__main__":
    main()
