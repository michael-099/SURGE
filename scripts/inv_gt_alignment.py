"""Agreement of each video's ground-truth map with its own fixations (research/log.md section 15I; data check, no
model). For every 10th frame with fixations, the NSS of the ground-truth map at the fixated pixels; the mean per
video. A video whose two ground-truth files disagree (mean NSS < 1) cannot be scored meaningfully.

    python scripts/inv_gt_alignment.py --dataset mvva --out runs/review/inv/gt_alignment_mvva.json
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

THRESHOLD = 1.0


def gt_nss(v, step=10):
    vals = []
    for i in range(0, len(v), step):
        m = cv2.imread(v.maps[i], cv2.IMREAD_GRAYSCALE).astype(np.float64)
        f = cv2.imread(v.fixations[i], cv2.IMREAD_GRAYSCALE)
        if m.std() == 0 or not (f > 0).any():
            continue
        vals.append(float(((m - m.mean()) / m.std())[f > 0].mean()))
    return v.name, (float(np.mean(vals)) if vals else float("nan")), len(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from scripts.evaluate import get_videos
    if a.dataset == "mvva":  # all laid-out videos, before the exclusion this check motivates
        from surge.data import mvva_videos
        vids = mvva_videos(exclude=())
    else:
        vids = get_videos(a.dataset, a.split, None)
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(gt_nss, vids))
    per = {n: {"nss": s, "frames": k} for n, s, k in res}
    low = sorted(n for n, s, _ in res if not s >= THRESHOLD)
    vals = np.array([s for _, s, _ in res])
    out = {"dataset": a.dataset, "videos": len(res), "threshold": THRESHOLD, "below_threshold": low,
           "median": float(np.nanmedian(vals)),
           "lowest_above_threshold": float(np.nanmin(vals[vals >= THRESHOLD])) if (vals >= THRESHOLD).any() else None,
           "per_video": per}
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
    print(f"{a.dataset}: {len(res)} videos, median NSS {out['median']:.2f}; below {THRESHOLD}: "
          f"{[(n, round(per[n]['nss'], 2)) for n in low]}; lowest at or above: {out['lowest_above_threshold']}")


if __name__ == "__main__":
    main()
