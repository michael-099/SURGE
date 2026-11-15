"""Ground-truth statistics and resolution ceilings per scored frame of DHF1K (research/log.md sections 15F-F1, 15J).

    python scripts/inv_gt_stats.py --split holdout --out runs/review/inv/gt_stats_holdout.json
    python scripts/inv_gt_stats.py --split val --out runs/review/inv/gt_stats_val.json

Per scored frame f (every --frame-step-th frame), no model involved:
  ceil{16,32,64}_{cc,nss,sim,kld}  the ground-truth map area-downsampled to r x r (the model's square output grids)
                                   and resized back, scored against the map and fixations (an upper bound for a
                                   prediction at that resolution)
  interp_cc     CC of the mean of the maps at f-1 and f+1 against the map at f (the cost of temporal interpolation)
  entropy       entropy (bits) of the 64x64 map, normalised to a distribution
  centre        CC of the 64x64 map with the mean training map (how central the frame's gaze is)
  change5       1 - CC of the 64x64 maps at f and f-5 (how much gaze moved in five frames)
  motion        mean absolute difference of the 64x64 grey frames f and f-2, in [0, 1]
  since_cut     frames since the last scene cut before or at f (TransNet V2 scenes; -1 if no cut before f)
The mean training map (540 training videos, every 10th frame, 64x64) is cached in runs/inv/train_mean_map.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from surge.data import dhf1k_videos, load_gray, read_split_file  # noqa: E402
from surge.metrics import all_metrics, cc  # noqa: E402

G = 64
SCENES = Path("D:/DHF1K/transnet_scenes")


def small(m: np.ndarray, r: int = G) -> np.ndarray:
    s = cv2.resize(m.astype(np.float32), (r, r), interpolation=cv2.INTER_AREA)
    t = s.sum()
    return s / t if t > 0 else np.full((r, r), 1.0 / (r * r), np.float32)


def grey(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (G, G), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def cuts_of(name: str) -> list[int]:
    p = SCENES / f"{int(name):03d}.AVI.scenes.txt"
    if not p.exists():
        return []
    starts = [int(line.split()[0]) for line in open(p) if line.strip()]
    return sorted(s for s in starts if s > 0)


def _train_mean_part(paths):
    acc = np.zeros((G, G), np.float64)
    for p in paths:
        acc += small(load_gray(p))
    return acc, len(paths)


def train_mean(workers: int) -> np.ndarray:
    f = ROOT / "runs/inv/train_mean_map.npy"
    if f.exists():
        return np.load(f)
    vids = dhf1k_videos("D:/DHF1K", read_split_file(ROOT / "configs/splits/dhf1k_train540.txt"))
    paths = [m for v in vids for m in v.maps[::10]]
    chunks = [paths[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_train_mean_part, chunks))
    acc = sum(p[0] for p in parts) / sum(p[1] for p in parts)
    acc = acc / acc.sum()
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, acc)
    return acc


def frame_stats(args):
    v, f, mean_map = args
    n = len(v)
    gt = load_gray(v.maps[f]).astype(np.float64)
    fix = load_gray(v.fixations[f])
    H, W = gt.shape
    out = {"f": f}
    for r in (16, 32, 64):
        up = cv2.resize(small(gt, r), (W, H), interpolation=cv2.INTER_LINEAR)
        for k, x in all_metrics(up, gt, fix, None, ("cc", "nss", "sim", "kld")).items():
            out[f"ceil{r}_{k}"] = x
    if 0 < f < n - 1:
        nb = load_gray(v.maps[f - 1]).astype(np.float64) + load_gray(v.maps[f + 1]).astype(np.float64)
        out["interp_cc"] = cc(nb, gt)
    else:
        out["interp_cc"] = float("nan")
    g64 = small(gt)
    p = g64[g64 > 0]
    out["entropy"] = float(-(p * np.log2(p)).sum())
    out["centre"] = cc(g64, mean_map)
    out["change5"] = 1.0 - cc(g64, small(load_gray(v.maps[f - 5]))) if f >= 5 else float("nan")
    out["motion"] = float(np.abs(grey(v.frames[f]) - grey(v.frames[f - 2])).mean()) if f >= 2 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["holdout", "val"])
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    split = {"holdout": "dhf1k_holdout60", "val": "dhf1k_val100"}[a.split]
    videos = dhf1k_videos("D:/DHF1K", read_split_file(ROOT / "configs/splits" / f"{split}.txt"))
    mean_map = train_mean(a.workers)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for v in videos:
            rows = list(ex.map(frame_stats, [(v, f, mean_map) for f in range(0, len(v), a.frame_step)], chunksize=8))
            cuts = cuts_of(v.name)
            per = {k: [r[k] for r in rows] for k in rows[0]}
            per["since_cut"] = [(f - max([c for c in cuts if c <= f])) if any(c <= f for c in cuts) else -1
                                for f in per["f"]]
            res[v.name] = {"cuts": cuts, "n": len(v), **per}
            print(v.name, len(rows), "cuts", len(cuts), flush=True)
    Path(a.out).write_text(json.dumps(res), newline="\n")
    keys = [k for k in next(iter(res.values())) if k not in ("cuts", "n", "f", "since_cut")]
    for k in keys:
        vals = np.array([np.nanmean(r[k]) for r in res.values()])
        print(f"{k:12s} mean over videos {np.nanmean(vals):.4f}")


if __name__ == "__main__":
    main()
