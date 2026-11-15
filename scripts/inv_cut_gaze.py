"""Gaze persistence across scene cuts, from the ground truth alone (research/log.md section 15L; descriptive
follow-up, specified after the 15L result; CPU, no model).

    python scripts/inv_cut_gaze.py --out runs/review/inv/summary_cut_gaze.json

For every cut c (a TransNet V2 shot start > 0) whose previous shot has >= 15 frames and whose new shot has >= 45
frames, the ground-truth density of frame c+k (k = 0..29) is compared by CC (maps area-downsampled to 64 px width)
with
  pre:  the ground truth of the last frame before the cut (c-1), and
  post: the mean ground truth of frames c+30..c+44 (gaze settled in the new shot).
Control: the same profiles around pseudo-cuts in the middle of shots with >= 60 frames on both sides, i.e. how fast
gaze maps change without a cut. Profiles are averaged per video, then over videos; 95 % intervals by video bootstrap
(10,000 resamples, seed 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surge.data import load_gray  # noqa: E402

K = 30          # frames after the cut that are profiled
POST = (30, 45)  # frames after the cut averaged as the settled post-cut gaze
MIN_PREV = 15
MIN_NEW = POST[1]
WIDTH = 64
REPORT_K = (0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 29)


def cc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def small(path: str) -> np.ndarray:
    m = load_gray(path).astype(np.float32)
    h = max(1, round(m.shape[0] * WIDTH / m.shape[1]))
    return cv2.resize(m, (WIDTH, h), interpolation=cv2.INTER_AREA)


def events(shots, n):
    """(cuts, pseudo-cuts) as frame indices satisfying the length conditions."""
    cuts, pseudo = [], []
    for i, (a, b) in enumerate(shots):
        if i > 0:
            pa, pb = shots[i - 1]
            if pb - pa + 1 >= MIN_PREV and b - a + 1 >= MIN_NEW and a + MIN_NEW <= n:
                cuts.append(a)
        if b - a + 1 >= 120 + MIN_NEW:  # a pseudo-cut with >= 60 frames before it and >= 60 + 45 after it
            pseudo.append(a + 60)
    return cuts, pseudo


def profile(get, c):
    """CC of frames c..c+K-1 with the pre-cut frame and with the settled post-cut mean."""
    pre = get(c - 1)
    post = np.mean([get(c + j) for j in range(*POST)], axis=0)
    return [cc(get(c + k), pre) for k in range(K)], [cc(get(c + k), post) for k in range(K)]


def video_profiles(v, kind, pool):
    from scripts.probe_eval import load_shots
    n = len(v)
    shots = load_shots(kind, v.name, n)
    cuts, pseudo = events(shots, n)
    need = sorted({f for c in cuts + pseudo for f in [c - 1, *range(c, c + POST[1])] if 0 <= f < n})
    maps = dict(zip(need, pool.map(lambda f: small(v.maps[f]), need)))
    get = maps.__getitem__
    out = {}
    for name, ev in (("cut", cuts), ("pseudo", pseudo)):
        if ev:
            pr = [profile(get, c) for c in ev]
            out[name] = (np.nanmean([p for p, _ in pr], 0), np.nanmean([q for _, q in pr], 0), len(ev))
    return out


def summarise(per_video: list[dict], name: str, rng_seed=0, n_boot=10000) -> dict | None:
    rows = [pv[name] for pv in per_video if name in pv]
    if not rows:
        return None
    pre = np.array([r[0] for r in rows])
    post = np.array([r[1] for r in rows])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(rows), (n_boot, len(rows)))

    def ci(x):
        b = np.nanmean(x[idx], axis=1)  # (n_boot, K)
        return np.nanpercentile(b, [2.5, 97.5], axis=0)
    cpre, cpost = ci(pre), ci(post)
    mpre, mpost = np.nanmean(pre, 0), np.nanmean(post, 0)
    cross = next((k for k in range(K) if mpost[k] > mpre[k]), None)
    return {"videos": len(rows), "events": int(sum(r[2] for r in rows)),
            "pre": [float(x) for x in mpre], "post": [float(x) for x in mpost],
            "pre_ci95": [[float(a), float(b)] for a, b in zip(*cpre)],
            "post_ci95": [[float(a), float(b)] for a, b in zip(*cpost)],
            "first_k_post_above_pre": cross}


def groups():
    from scripts.evaluate import get_videos
    from surge.data import stavis_videos
    yield "diem_f1_all", "diem", stavis_videos("diem", 1, "test") + stavis_videos("diem", 1, "train")
    yield "dhf1k_holdout_val", "dhf1k", get_videos("dhf1k", "holdout", None) + get_videos("dhf1k", "val", None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-videos", type=int, default=None)
    a = ap.parse_args(argv)
    res = {}
    with ThreadPoolExecutor(8) as pool:
        for gname, kind, vids in groups():
            vids = vids[: a.max_videos]
            per_video = []
            for v in vids:
                try:
                    per_video.append(video_profiles(v, kind, pool))
                except FileNotFoundError as e:  # no shot file: the video is skipped and counted
                    print(f"  {v.name}: {e}")
            res[gname] = {"cut": summarise(per_video, "cut"), "pseudo": summarise(per_video, "pseudo"),
                          "videos_total": len(vids)}
            print(f"== {gname} ({len(vids)} videos)")
            for ev in ("cut", "pseudo"):
                r = res[gname][ev]
                if r is None:
                    print(f"  {ev}: none")
                    continue
                print(f"  {ev}: {r['events']} events in {r['videos']} videos; first k with CC(post) > CC(pre): "
                      f"{r['first_k_post_above_pre']}")
                for k in REPORT_K:
                    print(f"    k={k:2d}  CC with pre-cut frame {r['pre'][k]:.3f} "
                          f"[{r['pre_ci95'][k][0]:.3f}, {r['pre_ci95'][k][1]:.3f}]   "
                          f"CC with settled post-cut gaze {r['post'][k]:.3f} "
                          f"[{r['post_ci95'][k][0]:.3f}, {r['post_ci95'][k][1]:.3f}]")
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
