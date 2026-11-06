"""Build per-frame fixation and density maps for Hollywood-2 from the Vig, Dorr & Cox (2012) gaze data
(research/log.md section 9).

    python -u scripts/prep_hollywood_gaze.py [--workers 8]
Input:  D:/Datasets/Hollywood2_VigGaze/raw/subject{0..4}.zip  (s<k>_<clip>.coord / .sacc)
        D:/Datasets/Hollywood2/surge/clips.csv and frames/ (from prep_hollywood_frames.py)
Output: D:/Datasets/Hollywood2/surge/{fixation,maps}/<clip>/NNNN.png, points.csv (clip, frames, frames with a point,
        mean points per frame). Resumable per clip.
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAW = Path("D:/Datasets/Hollywood2_VigGaze/raw")
DST = Path("D:/Datasets/Hollywood2/surge")
SIGMA = 20.0


def parse_coord(text: str):
    """-> (W, H, samples array (n,4): t_us, x, y, conf)."""
    W = H = None
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        t = line.split()
        if t[0] == "gaze":
            W, H = int(t[1]), int(t[2])
        elif t[0] == "geometry":
            continue
        else:
            rows.append([float(x) for x in t[:4]])
    return W, H, np.array(rows, dtype=np.float64).reshape(-1, 4)


def parse_sacc(text: str):
    """-> array (n,2): onset_us, offset_us."""
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        t = line.split()
        if t[0] == "saccades":
            continue
        rows.append([float(t[0]), float(t[3])])
    return np.array(rows, dtype=np.float64).reshape(-1, 2)


def frame_points(samples, saccs, n, fps, W, H):
    """One point per frame (median of valid non-saccade samples in the frame's interval), or None."""
    if len(samples) == 0:
        return [None] * n
    t, x, y, c = samples.T
    ok = c >= 1.0
    if len(saccs):
        # sample is inside a saccade if onset <= t <= offset for some saccade
        idx = np.searchsorted(saccs[:, 0], t, side="right") - 1
        inside = (idx >= 0) & (t <= saccs[np.clip(idx, 0, None), 1])
        ok &= ~inside
    t, x, y = t[ok], x[ok], y[ok]
    f = np.floor(t / 1e6 * fps).astype(np.int64)  # 0-based frame index
    pts = [None] * n
    for k in range(n):
        m = f == k
        if m.any():
            px = int(round(float(np.median(x[m]))))
            py = int(round(float(np.median(y[m]))))
            if 0 <= px < W and 0 <= py < H:
                pts[k] = (px, py)
    return pts


def build(args):
    clip, n, fps, w, h = args
    fd, md = DST / "fixation" / clip, DST / "maps" / clip
    done = md / ".done"
    if done.exists():
        return clip, n, *map(float, done.read_text().split())
    per_frame = [[] for _ in range(n)]
    for s in range(5):
        with zipfile.ZipFile(RAW / f"subject{s}.zip") as z:
            names = set(z.namelist())
            cn, sn = f"s{s}_{clip}.coord", f"s{s}_{clip}.sacc"
            if cn not in names:
                continue
            W, H, samples = parse_coord(z.read(cn).decode("latin-1"))
            saccs = parse_sacc(z.read(sn).decode("latin-1")) if sn in names else np.zeros((0, 2))
        if (W, H) != (w, h):  # gaze coordinates are in the stated resolution; rescale if it differs
            samples[:, 1] *= w / W
            samples[:, 2] *= h / H
        for k, p in enumerate(frame_points(samples, saccs, n, fps, w, h)):
            if p is not None:
                per_frame[k].append(p)
    fd.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)
    with_pts, tot = 0, 0
    for k in range(n):
        fix = np.zeros((h, w), np.uint8)
        for px, py in per_frame[k]:
            fix[py, px] = 255
        dens = cv2.GaussianBlur(fix.astype(np.float32), (0, 0), SIGMA, borderType=cv2.BORDER_REPLICATE)
        dens = np.round(255 * dens / dens.max()).astype(np.uint8) if dens.max() > 0 else np.zeros((h, w), np.uint8)
        cv2.imwrite(str(fd / f"{k + 1:04d}.png"), fix)
        cv2.imwrite(str(md / f"{k + 1:04d}.png"), dens)
        with_pts += bool(per_frame[k])
        tot += len(per_frame[k])
    stats = (with_pts, tot / max(n, 1))
    done.write_text(f"{stats[0]} {stats[1]}")
    return clip, n, *stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    clips = []
    for r in csv.DictReader(open(DST / "clips.csv")):
        clips.append((r["clip"], int(r["frames"]), float(r["fps"]), int(r["width"]), int(r["height"])))
    clips = clips[: a.limit]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(build, clips, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(i + 1, r, flush=True)
    if a.limit is None:
        with open(DST / "points.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["clip", "frames", "frames_with_point", "mean_points_per_frame"])
            wr.writerows(rows)
    n = sum(r[1] for r in rows)
    print(f"done {len(rows)} clips, {n} frames, {sum(r[2] for r in rows) / n:.3f} with a point, "
          f"{sum(r[3] * r[1] for r in rows) / n:.2f} points per frame", flush=True)


if __name__ == "__main__":
    main()
