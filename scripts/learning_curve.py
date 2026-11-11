"""Learning curve over the number of training videos (research/log.md section 13A).

Arms: 135, 270 and 540 training videos (ViT-L, seeds 0-2; the 540-video arm is main_b1_s{0,1,2}). For each pair of
arms the per-video holdout CC difference is averaged over videos and seeds; a two-level bootstrap (seeds with
replacement, then videos with replacement, shared across seeds because the comparison is paired) gives 90 % and 95 %
intervals. Outcome with the margin delta = 0.003, as in section 12E:
  effect      the 95 % interval excludes 0 and |mean| >= delta
  equivalent  the 90 % interval lies within (-delta, +delta)
  inconclusive otherwise

    python scripts/learning_curve.py --out runs/review/learning_curve.json
Reads runs/review/holdout/<run>.json (scripts/evaluate.py on the holdout, every 5th frame, sigma 0).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = {540: [f"main_b1_s{s}" for s in range(3)], 270: [f"lc270_s{s}" for s in range(3)],
        135: [f"lc135_s{s}" for s in range(3)]}
PAIRS = [(270, 540), (135, 270), (135, 540)]  # (smaller, larger): difference = smaller - larger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.003)
    ap.add_argument("--dir", default=str(ROOT / "runs/review/holdout"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = Path(a.dir)
    load = lambda r: json.load(open(H / f"{r}.json"))["per_video"]  # noqa: E731
    pv = {n: [load(r) for r in runs] for n, runs in ARMS.items()}
    vids = sorted(pv[540][0])
    assert all(sorted(p) == vids for arm in pv.values() for p in arm)
    res = {"videos": len(vids), "delta": a.delta, "arms": {}, "pairs": {}}
    for n in sorted(ARMS):
        arm = {}
        for m in ("cc", "nss", "sim", "kld"):
            per_seed = [float(np.mean([p[v][m] for v in vids])) for p in pv[n]]
            arm[m] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1))}
        res["arms"][str(n)] = arm
        print(f"{n:4d} videos: CC {arm['cc']['mean']:.4f} +- {arm['cc']['sd']:.4f}  per seed "
              f"{', '.join(f'{x:.4f}' for x in arm['cc']['per_seed'])}   NSS {arm['nss']['mean']:.3f}  "
              f"SIM {arm['sim']['mean']:.4f}  KLD {arm['kld']['mean']:.3f}")
    rng = np.random.default_rng(a.seed)
    for small, large in PAIRS:
        out = {}
        for m in ("cc", "nss", "sim", "kld"):
            d = np.array([[pv[small][s][v][m] - pv[large][s][v][m] for v in vids] for s in range(3)])  # seeds x videos
            si = rng.integers(0, 3, (a.n, 3))
            vi = rng.integers(0, len(vids), (a.n, len(vids)))
            boots = np.array([d[si[b]][:, vi[b]].mean() for b in range(a.n)])
            mean = float(d.mean())
            c95 = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
            c90 = [float(x) for x in np.percentile(boots, [5, 95])]
            out[m] = {"mean": mean, "ci95": c95, "ci90": c90, "per_seed": [float(x) for x in d.mean(1)]}
            if m == "cc":
                if (c95[0] > 0 or c95[1] < 0) and abs(mean) >= a.delta:
                    out[m]["outcome"] = "effect"
                elif -a.delta < c90[0] and c90[1] < a.delta:
                    out[m]["outcome"] = "equivalent"
                else:
                    out[m]["outcome"] = "inconclusive"
        res["pairs"][f"{small}-{large}"] = out
        c = out["cc"]
        print(f"CC({small}) - CC({large}): {c['mean']:+.4f}  95% [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
              f"90% [{c['ci90'][0]:+.4f}, {c['ci90'][1]:+.4f}]  per seed "
              f"{', '.join(f'{x:+.4f}' for x in c['per_seed'])}  -> {c['outcome']}")
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
