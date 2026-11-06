"""Write SURGE-D saliency maps for the DHF1K test videos (701-1000) in the format the DHF1K authors ask for:
<out>/<video>/<frame>.png, 8-bit grayscale at the video's native frame size, frames named as in D:/DHF1K/frames
(0001.png ...). Frame counts were checked against the annotations for videos 1-700 (all 700 equal).

    python -u scripts/predict_dhf1k_test.py --run runs/main_b1_s0 --run runs/main_b1_s1 --run runs/main_b1_s2 \
        --sigma 8 --out D:/DHF1K_submission/SURGE-D

Resumable: a video whose folder already holds one PNG per frame is skipped. Polls SAFETY_STOP.txt between videos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.evaluate import load_model  # noqa: E402
from surge.data import dhf1k_videos  # noqa: E402
from surge.infer import predict_video, to_native  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--first", type=int, default=701)
    ap.add_argument("--last", type=int, default=1000)
    ap.add_argument("--root", default="D:/DHF1K")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    models = [load_model(ROOT / r, a.ckpt, device) for r in a.run]
    videos = dhf1k_videos(a.root, range(a.first, a.last + 1), with_labels=False)
    out = Path(a.out)
    for vi, v in enumerate(videos):
        if (ROOT / "SAFETY_STOP.txt").exists():
            print("SAFETY_STOP.txt present; stopping", flush=True)
            sys.exit(2)
        od = out / v.name
        n = len(v.frames)
        assert n > 0, v.name
        if od.is_dir() and len(list(od.glob("*.png"))) == n:
            continue
        hw = cv2.imread(v.frames[0]).shape[:2]
        acc = None
        for m, cfg in models:
            d = cfg["data"]
            o = dict(predict_video(m, v, d["T"], d["stride"], tuple(d["size"]), device))
            arr = np.stack([o[i] for i in range(n)]).astype(np.float64)
            acc = arr if acc is None else acc + arr
        od.mkdir(parents=True, exist_ok=True)
        for f in range(n):
            pm = to_native(acc[f], hw, a.sigma)
            name = Path(v.frames[f]).stem + ".png"
            cv2.imwrite(str(od / name), np.round(255 * pm / pm.max()).astype(np.uint8))
        print(f"[{vi + 1}/{len(videos)}] {v.name}: {n} frames {hw[1]}x{hw[0]}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
