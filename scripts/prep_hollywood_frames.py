"""Decode the Hollywood-2 action clips (823 train + 884 test; not the auto-train set) to JPEG frames.

    python -u scripts/prep_hollywood_frames.py [--workers 6]
Writes D:/Datasets/Hollywood2/surge/frames/<clip>/NNNN.jpg (1-based, every decoded frame, quality 95) and
D:/Datasets/Hollywood2/surge/clips.csv (clip, split, frames, fps, width, height). Resumable per clip.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

SRC = Path("D:/Datasets/Hollywood2/extracted/Hollywood2")
DST = Path("D:/Datasets/Hollywood2/surge")


def clip_list():
    out = []
    for split in ("train", "test"):
        for line in open(SRC / "ClipSets" / f"actions_{split}.txt"):
            t = line.split()
            if t:
                out.append((t[0], split))
    return sorted(set(out))


def decode(args):
    clip, split = args
    od = DST / "frames" / clip
    done = od / ".done"
    v = cv2.VideoCapture(str(SRC / "AVIClips" / f"{clip}.avi"))
    fps, w, h = v.get(cv2.CAP_PROP_FPS), int(v.get(3)), int(v.get(4))
    if done.exists():
        n = int(done.read_text())
        return clip, split, n, fps, w, h
    od.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        ok, im = v.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(str(od / f"{n:04d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    done.write_text(str(n))
    return clip, split, n, fps, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    clips = clip_list()
    print(len(clips), "clips", flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(decode, clips, chunksize=4)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(i + 1, r, flush=True)
    with open(DST / "clips.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "split", "frames", "fps", "width", "height"])
        wr.writerows(rows)
    print("done", len(rows), "clips,", sum(r[2] for r in rows), "frames", flush=True)


if __name__ == "__main__":
    main()
