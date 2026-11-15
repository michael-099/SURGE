"""AViMoS fixation maps at the resolution of its saliency maps (research/log.md section 15I).

The GRAFT-A layout of AViMoS (D:/Datasets/AViMoS, CC-BY-4.0) has frames and maps at 540x960 but fixation maps at the
native video size. The metrics need fixations on the map's grid, so each fixated pixel (y, x) of an H x W fixation
map marks cell (floor(y h / H), floor(x w / W)) of the h x w saliency-map grid. For the landscape 1080x1920 videos
this equals max-pooling each 2x2 block (the first version of this script). Two public-test videos (1142, 1143) are
portrait (fixations 1920x1080) while their frames and maps were stored squeezed to 540x960; per-axis scaling aligns
their fixations with the maps (checked against the release maps: NSS 6.7 and 3.5 against 0.6 and 0.5 transposed).
Reads in place; writes only under --out.

    python -I scripts/prep_avimos_fix.py --root D:/Datasets/AViMoS --out D:/Datasets/AViMoS/surge_fix540 \
        --first 1001 --last 1150 --workers 6 [--ids 1142 1143]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def to_grid(f: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary h x w map: cell (floor(y h / H), floor(x w / W)) is set for every fixated pixel (y, x) of f."""
    ys, xs = np.nonzero(f)
    m = np.zeros((h, w), np.uint8)
    m[np.minimum(ys * h // f.shape[0], h - 1), np.minimum(xs * w // f.shape[1], w - 1)] = 255
    return m


def one(args):
    vid, root, out = args
    src = Path(root) / "fixations" / vid
    dst = Path(out) / vid
    dst.mkdir(parents=True, exist_ok=True)
    mp = sorted((Path(root) / "maps" / vid).glob("*.png"))
    n = 0
    hw = None
    for p in sorted(src.glob("*.png")):
        f = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if f is None:
            raise IOError(p)
        if hw is None:
            hw = cv2.imread(str(mp[0]), cv2.IMREAD_GRAYSCALE).shape if mp else (f.shape[0] // 2, f.shape[1] // 2)
        cv2.imwrite(str(dst / p.name), to_grid(f, *hw))
        n += 1
    return vid, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--ids", nargs="*", default=None, help="only these video ids (within first..last)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    vids = [f"{i:04d}" for i in range(a.first, a.last + 1)]
    if a.ids:
        vids = [v for v in vids if v in {f"{int(i):04d}" for i in a.ids}]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [(v, a.root, a.out) for v in vids]))
    print(f"{len(res)} videos, {sum(n for _, n in res)} fixation maps")


if __name__ == "__main__":
    main()
