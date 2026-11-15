"""AViMoS fixation maps at the resolution of its saliency maps (research/log.md section 15I).

The GRAFT-A layout of AViMoS (D:/Datasets/AViMoS, CC-BY-4.0) has frames and maps at 540x960 but fixation maps at
1080x1920. The metrics need fixations on the map's grid, so each 2x2 block is max-pooled (a block is fixated if any
of its pixels is). Reads in place; writes only under --out.

    python -I scripts/prep_avimos_fix.py --root D:/Datasets/AViMoS --out D:/Datasets/AViMoS/surge_fix540 \
        --first 1001 --last 1150 --workers 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def one(args):
    vid, root, out = args
    src = Path(root) / "fixations" / vid
    dst = Path(out) / vid
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.png")):
        f = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if f is None:
            raise IOError(p)
        h, w = f.shape[0] // 2 * 2, f.shape[1] // 2 * 2
        m = (f[:h, :w] > 0).reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))
        cv2.imwrite(str(dst / p.name), m.astype(np.uint8) * 255)
        n += 1
    return vid, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    vids = [f"{i:04d}" for i in range(a.first, a.last + 1)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(v, a.root, a.out) for v in vids]))
    print(f"{len(res)} videos, {sum(n for _, n in res)} fixation maps")


if __name__ == "__main__":
    main()
