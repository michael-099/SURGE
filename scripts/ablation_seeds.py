"""Component ablations with three seeds (research/log.md section 12E).

For each factor, the per-video holdout CC difference to the B1 model of the same seed is averaged over videos and
seeds. A two-level bootstrap (seeds with replacement, then videos with replacement, shared across seeds because the
comparison is paired) gives 90 % and 95 % intervals. Outcome, with the margin delta = 0.003 fixed in advance:
  effect      the 95 % interval excludes 0 and |mean| >= delta
  equivalent  the 90 % interval lies within (-delta, +delta)
  inconclusive otherwise

    python scripts/ablation_seeds.py --out runs/review/ablation_seeds.json
Reads runs/review/holdout/<run>.json (scripts/evaluate.py on the holdout, every 5th frame, sigma 0).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FACTORS = {"noprior": "Without centre prior", "noaux": "Without auxiliary heads",
           "salicon": "With SALICON co-training", "stride4": "Stride 4"}
REF = ["main_b1_s0", "main_b1_s1", "main_b1_s2"]


def runs_for(f):
    return [f"abl_{f}", f"abl_{f}_s1", f"abl_{f}_s2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.003)
    ap.add_argument("--metric", default="cc")
    ap.add_argument("--dir", default=str(ROOT / "runs/review/holdout"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = Path(a.dir)
    load = lambda r: json.load(open(H / f"{r}.json"))["per_video"]  # noqa: E731
    ref = [load(r) for r in REF]
    vids = sorted(ref[0])
    rng = np.random.default_rng(a.seed)
    res = {}
    for f, label in FACTORS.items():
        ab = [load(r) for r in runs_for(f)]
        d = np.array([[ab[s][v][a.metric] - ref[s][v][a.metric] for v in vids] for s in range(3)])  # seeds x videos
        si = rng.integers(0, 3, (a.n, 3))
        vi = rng.integers(0, len(vids), (a.n, len(vids)))
        boots = np.array([d[si[b]][:, vi[b]].mean() for b in range(a.n)])
        mean = float(d.mean())
        c95 = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
        c90 = [float(x) for x in np.percentile(boots, [5, 95])]
        if (c95[0] > 0 or c95[1] < 0) and abs(mean) >= a.delta:
            outcome = "effect"
        elif -a.delta < c90[0] and c90[1] < a.delta:
            outcome = "equivalent"
        else:
            outcome = "inconclusive"
        per_seed = [float(x) for x in d.mean(1)]
        res[f] = {"label": label, "mean": mean, "ci95": c95, "ci90": c90, "per_seed": per_seed, "outcome": outcome,
                  "abl_cc_per_seed": [float(np.mean([ab[s][v][a.metric] for v in vids])) for s in range(3)],
                  "ref_cc_per_seed": [float(np.mean([ref[s][v][a.metric] for v in vids])) for s in range(3)]}
        print(f"{label:28s} {a.metric} diff {mean:+.4f}  95% [{c95[0]:+.4f}, {c95[1]:+.4f}]  "
              f"90% [{c90[0]:+.4f}, {c90[1]:+.4f}]  per seed {', '.join(f'{x:+.4f}' for x in per_seed)}  -> {outcome}")
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
