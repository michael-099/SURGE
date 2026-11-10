"""Gate G2 intervals (research/log.md section 12F): 95 % bootstrap intervals over the 60 holdout videos for the NSS
and CC of every G2 map at both frame rates, and for the paired differences surprise - frame difference and
surprise - token atypicality. Reads runs/g2/g2_results.json (per-video scores stored by scripts/g2_surprise.py).

    python scripts/g2_intervals.py --out runs/review/g2_intervals.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAPS = ["surprise_raw", "surprise_ctx", "atypicality", "frame_diff", "center_prior"]
RATES = {"stride_8": "about 4 fps", "stride_2": "about 15 fps"}


def boot(x: np.ndarray, idx: np.ndarray):
    b = x[idx].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g2 = json.load(open(ROOT / "runs/g2/g2_results.json"))
    out = {}
    for rate, label in RATES.items():
        pv = g2[rate]["per_video"]
        vids = sorted(pv)
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, len(vids), (a.n, len(vids)))
        res = {"videos": len(vids), "maps": {}, "differences": {}}
        for m in MAPS:
            res["maps"][m] = {}
            for k in ("nss", "cc"):
                x = np.array([pv[v][m][k] for v in vids], float)
                res["maps"][m][k] = boot(x, idx)
        for s in ("surprise_raw", "surprise_ctx"):
            for r in ("frame_diff", "atypicality"):
                for k in ("nss", "cc"):
                    d = np.array([pv[v][s][k] - pv[v][r][k] for v in vids], float)
                    res["differences"][f"{s}-{r}/{k}"] = boot(d, idx)
        out[rate] = res
        print(f"== {rate} ({label}), {len(vids)} videos")
        for m in MAPS:
            print(f"  {m:13s} " + "  ".join(
                f"{k.upper()} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in res["maps"][m].items()))
        for k, v in res["differences"].items():
            print(f"  {k:32s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")


if __name__ == "__main__":
    main()
