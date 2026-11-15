"""Lay out MVVA (Liu et al., ECCV 2020; 300 multi-face videos, eye tracking of 34 observers) for SURGE-D
(research/log.md section 15I/15K). Reads the release in place; writes only under --out.

    python -I scripts/prep_mvva.py --src D:/Datasets/MVVA/extracted --out D:/Datasets/MVVA/surge --workers 6

Per video <id> with n = min(video frames, heatmaps, fixation frames) frames, aligned at frame 0:
  frames/<id>/NNNN.jpg     the video frames resized to 640x360, JPEG quality 95
  fixation/<id>/NNNN.png   binary map at 640x360 from the 34 observers' fixations, scaled per axis from the native
                           video size (points outside the frame dropped)
  maps/<id>/NNNN.png       only where the release heatmaps are not 640x360: the heatmaps resized to 640x360
  index.csv                id, n, video frames, heatmaps, fixation frames, fps, faces, native width and height,
                           ground-truth source ('release' or 'resized')
Elsewhere the ground-truth maps are the release's heatmaps (save_our_hmap/video_<id>/<k>.jpg), used in place.

The release's heatmaps are at half the native video size. 291 videos are 1280x720 (heatmaps 640x360); the first
layout scaled every fixation by 1/2, which is exact for them and is kept (no rewrite). Eight videos have another
native size (five 1920x1080, 1280x658, 1280x672, 1282x720); they are laid out again with per-axis scaling and
resized heatmaps (marker fixation/<id>/.v2). Video 343 has no heatmaps (n = 0).
Downloaded data: numpy arrays are read with allow_pickle=False; no code from the release is run.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360


def resize_map(m: np.ndarray) -> np.ndarray:
    """A heatmap to 640x360: area interpolation when shrinking, linear when an axis grows."""
    grows = m.shape[0] < H or m.shape[1] < W
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR if grows else cv2.INTER_AREA)


def fixation_map(p: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Binary 640x360 map from fixations p (2, observers): x, y in native video pixels."""
    m = np.zeros((H, W), np.uint8)
    x = np.floor(p[0] * sx)
    y = np.floor(p[1] * sy)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    m[y[keep].astype(int), x[keep].astype(int)] = 255
    return m


def one(args):
    vid, src, out = args
    src, out = Path(src), Path(out)
    vpath = next((src / "mvva_database_v1" / "our_database_video").glob(f"{vid}.*"))
    # heatmap folders are zero-padded to 3 digits (video_084) while video and .npy names are not (84.avi, 84.npy)
    hm = sorted((src / "mvva_database_v1" / "save_our_hmap" / f"video_{int(vid):03d}").glob("*.jpg"),
                key=lambda p: int(p.stem))
    fx = np.load(src / "mvva_database_v1" / "all_videos_all_frames_fixation" / f"{vid}.npy", allow_pickle=False)
    faces = np.load(src / "all_face_tracking_release" / f"{vid}.npy", allow_pickle=False)
    cap = cv2.VideoCapture(str(vpath))
    nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = min(nv, len(hm), fx.shape[0])
    hshape = cv2.imread(str(hm[0]), cv2.IMREAD_GRAYSCALE).shape if hm else (H, W)
    resized = hshape != (H, W)
    row = {"id": vid, "n": n, "video_frames": nv, "heatmaps": len(hm), "fixation_frames": fx.shape[0],
           "fps": round(fps, 3), "faces": faces.shape[1] if faces.ndim == 3 else 0, "width": w0, "height": h0,
           "gt": "resized" if resized else "release"}
    fd, xd, md = out / "frames" / vid, out / "fixation" / vid, out / "maps" / vid
    fd.mkdir(parents=True, exist_ok=True)
    xd.mkdir(parents=True, exist_ok=True)
    complete = n > 0 and len(list(fd.glob("*.jpg"))) == n and len(list(xd.glob("*.png"))) == n
    first_layout_exact = (w0, h0) == (2 * W, 2 * H) and not resized
    if complete and (first_layout_exact or (xd / ".v2").exists()):
        cap.release()
        return row
    if resized:
        md.mkdir(parents=True, exist_ok=True)
    sx, sy = W / w0, H / h0
    got = 0
    for i in range(n):
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(fd / f"{i:04d}.jpg"), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(xd / f"{i:04d}.png"), fixation_map(fx[i], sx, sy))
        if resized:
            cv2.imwrite(str(md / f"{i:04d}.png"), resize_map(cv2.imread(str(hm[i]), cv2.IMREAD_GRAYSCALE)))
        got += 1
    cap.release()
    if got:
        (xd / ".v2").write_text("per-axis scaling from the native size\n")
    row["n"] = got
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    vids = sorted(p.stem for p in (Path(a.src) / "mvva_database_v1" / "our_database_video").iterdir())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, [(v, a.src, a.out) for v in vids]))
    with open(Path(a.out) / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mism = [r for r in rows if len({r["video_frames"], r["heatmaps"], r["fixation_frames"]}) > 1]
    other = [r["id"] for r in rows if (r["width"], r["height"]) != (2 * W, 2 * H)]
    print(f"{len(rows)} videos, {sum(r['n'] for r in rows)} frames; {len(mism)} with unequal frame counts "
          f"(aligned at frame 0, cut to the shortest); native size other than 1280x720: {other}; "
          f"heatmaps resized: {[r['id'] for r in rows if r['gt'] == 'resized']}; no frames: "
          f"{[r['id'] for r in rows if r['n'] == 0]}")


if __name__ == "__main__":
    main()
