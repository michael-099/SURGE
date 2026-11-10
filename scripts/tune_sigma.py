"""Tune the output blur sigma (native pixels) on the internal holdout; one map is then used for all metrics.

    python -u scripts/tune_sigma.py --run runs/x [--run runs/y ...] --out runs/x/sigma.json
Selection: highest holdout CC (ties -> smaller sigma).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.evaluate import get_videos, load_model  # noqa: E402
from surge.infer import predict_video, score_saved  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--dataset", default="dhf1k")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--sigmas", default="0,4,8,12,16,24")
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    vids = get_videos(a.dataset, a.split, None)
    models = [load_model(ROOT / r, a.ckpt, device) for r in a.run]
    # Causal models (data.target=last, research/log.md section 12D) predict only the scored frames; the other frames
    # stay zero and are never scored (score_saved uses the same frame step).
    modes = {cfg["data"].get("target", "central") for _, cfg in models}
    if len(modes) > 1:
        raise ValueError(f"ensemble mixes target modes {modes}")
    causal = modes == {"last"}
    small = {}
    for v in vids:
        acc = None
        for m, cfg in models:
            d = cfg["data"]
            sub = range(0, len(v), a.frame_step) if causal else None
            o = dict(predict_video(m, v, d["T"], d["stride"], tuple(d["size"]), device, causal=causal, frames=sub))
            first = next(iter(o.values()))
            arr = np.zeros((len(v), *first.shape), np.float64)
            for i, q in o.items():
                arr[i] = q
            if not causal:
                assert len(o) == len(v), (v.name, len(o), len(v))
            acc = arr if acc is None else acc + arr
        small[v.name] = (acc / len(models)).astype(np.float32)
        print("predicted", v.name, flush=True)
    res = {}
    for s in [float(x) for x in a.sigmas.split(",")]:
        mean, _ = score_saved(small, vids, s, frame_step=a.frame_step)
        res[s] = mean
        print(f"sigma {s}: " + " ".join(f"{k}={x:.4f}" for k, x in mean.items()), flush=True)
    best = max(res, key=lambda s: (round(res[s]["cc"], 4), -s))
    Path(a.out).write_text(json.dumps({"best_sigma": best, "scores": {str(k): v for k, v in res.items()}}, indent=1),
                           newline="\n")
    print("BEST SIGMA", best, flush=True)


if __name__ == "__main__":
    main()
