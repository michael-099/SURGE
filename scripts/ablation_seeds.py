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
    # Other comparisons with the same rule (research/log.md section 14A): --ref r0,r1,r2 and --variant name=r0,r1,r2
    ap.add_argument("--ref", default=None, help="comma-separated reference runs (default: the B1 ViT-L runs)")
    ap.add_argument("--variant", action="append", default=[], help="name=run0,run1,run2 (default: the 12E factors)")
    # research/log.md section 14D: any number of seeds (one run per seed, in the order of --ref), and the level of
    # the interval that decides an effect (default 95; 14D uses 97.5). Defaults reproduce section 12E exactly.
    ap.add_argument("--level", type=float, default=95.0)
    a = ap.parse_args()
    global REF, FACTORS, runs_for
    if a.ref:
        REF = a.ref.split(",")
    if a.variant:
        groups = dict(v.split("=", 1) for v in a.variant)
        FACTORS = {k: k for k in groups}
        runs_for = lambda f: groups[f].split(",")  # noqa: E731
    H = Path(a.dir)
    load = lambda r: json.load(open(H / f"{r}.json"))["per_video"]  # noqa: E731
    ref = [load(r) for r in REF]
    vids = sorted(ref[0])
    rng = np.random.default_rng(a.seed)
    res = {}
    S = len(REF)
    lo = (100.0 - a.level) / 2
    for f, label in FACTORS.items():
        runs = runs_for(f)
        assert len(runs) == S, f"{f}: {len(runs)} runs for {S} reference seeds"
        ab = [load(r) for r in runs]
        d = np.array([[ab[s][v][a.metric] - ref[s][v][a.metric] for v in vids] for s in range(S)])  # seeds x videos
        si = rng.integers(0, S, (a.n, S))
        vi = rng.integers(0, len(vids), (a.n, len(vids)))
        boots = np.array([d[si[b]][:, vi[b]].mean() for b in range(a.n)])
        mean = float(d.mean())
        c95 = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
        c90 = [float(x) for x in np.percentile(boots, [5, 95])]
        cmain = [float(x) for x in np.percentile(boots, [lo, 100.0 - lo])]
        if (cmain[0] > 0 or cmain[1] < 0) and abs(mean) >= a.delta:
            outcome = "effect"
        elif -a.delta < c90[0] and c90[1] < a.delta:
            outcome = "equivalent"
        else:
            outcome = "inconclusive"
        per_seed = [float(x) for x in d.mean(1)]
        res[f] = {"label": label, "mean": mean, "ci95": c95, "ci90": c90, "per_seed": per_seed, "outcome": outcome,
                  "level": a.level, "ci_main": cmain, "seeds": S,
                  "abl_cc_per_seed": [float(np.mean([ab[s][v][a.metric] for v in vids])) for s in range(S)],
                  "ref_cc_per_seed": [float(np.mean([ref[s][v][a.metric] for v in vids])) for s in range(S)]}
        extra = "" if a.level == 95.0 else f"  {a.level:g}% [{cmain[0]:+.4f}, {cmain[1]:+.4f}]"
        print(f"{label:28s} {a.metric} diff {mean:+.4f}  95% [{c95[0]:+.4f}, {c95[1]:+.4f}]  "
              f"90% [{c90[0]:+.4f}, {c90[1]:+.4f}]{extra}  per seed {', '.join(f'{x:+.4f}' for x in per_seed)}  -> {outcome}")
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
