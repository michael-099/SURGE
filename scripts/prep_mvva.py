"""Lay out MVVA (Liu et al., ECCV 2020; 300 multi-face videos, eye tracking of 34 observers) for SURGE-D
(research/log.md section 15I/15K). Reads the release in place; writes only under --out.

    python -I scripts/prep_mvva.py --src D:/Datasets/MVVA/extracted --out D:/Datasets/MVVA/surge --workers 6

Per video <id> with n = min(video frames, heatmaps, fixation frames) frames, aligned at frame 0:
  frames/<id>/NNNN.jpg     the video frames resized to the heatmap size (640x360), JPEG quality 95
  fixation/<id>/NNNN.png   binary map at 640x360 from the 34 observers' fixations (video pixels / 2; points outside
                           the frame dropped)
  index.csv                id, n, video frames, heatmaps, fixation frames, fps, faces
The ground-truth maps are the release's heatmaps (save_our_hmap/video_<id>/NNN.jpg), used in place.
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


def one(args):
    vid, src, out = args
    src, out = Path(src), Path(out)
    vpath = next((src / "mvva_database_v1" / "our_database_video").glob(f"{vid}.*"))
    # heatmap folders are zero-padded to 3 digits (video_084) while video and .npy names are not (84.avi, 84.npy)
    hm = sorted((src / "mvva_database_v1" / "save_our_hmap" / f"video_{int(vid):03d}").glob("*.jpg"))
    fx = np.load(src / "mvva_database_v1" / "all_videos_all_frames_fixation" / f"{vid}.npy", allow_pickle=False)
    faces = np.load(src / "all_face_tracking_release" / f"{vid}.npy", allow_pickle=False)
    cap = cv2.VideoCapture(str(vpath))
    nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = min(nv, len(hm), fx.shape[0])
    fd, xd = out / "frames" / vid, out / "fixation" / vid
    fd.mkdir(parents=True, exist_ok=True)
    xd.mkdir(parents=True, exist_ok=True)
    if n > 0 and len(list(fd.glob("*.jpg"))) == n and len(list(xd.glob("*.png"))) == n:  # already laid out
        cap.release()
        return {"id": vid, "n": n, "video_frames": nv, "heatmaps": len(hm), "fixation_frames": fx.shape[0],
                "fps": round(fps, 3), "faces": faces.shape[1] if faces.ndim == 3 else 0}
    got = 0
    for i in range(n):
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(fd / f"{i:04d}.jpg"), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        m = np.zeros((H, W), np.uint8)
        p = fx[i]  # (2, 34): x, y in video pixels
        x = np.floor(p[0] / 2.0)
        y = np.floor(p[1] / 2.0)
        keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        m[y[keep].astype(int), x[keep].astype(int)] = 255
        cv2.imwrite(str(xd / f"{i:04d}.png"), m)
        got += 1
    cap.release()
    return {"id": vid, "n": got, "video_frames": nv, "heatmaps": len(hm), "fixation_frames": fx.shape[0],
            "fps": round(fps, 3), "faces": faces.shape[1] if faces.ndim == 3 else 0}


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
    print(f"{len(rows)} videos, {sum(r['n'] for r in rows)} frames; {len(mism)} with unequal frame counts "
          f"(aligned at frame 0, cut to the shortest)")


if __name__ == "__main__":
    main()
