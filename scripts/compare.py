"""Paired bootstrap over videos between two evaluation JSONs (scripts/evaluate.py outputs).

    python scripts/compare.py A.json B.json [--n 10000]
Prints mean(A) - mean(B) with a 95% percentile CI per metric (resampling videos with replacement).
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = json.load(open(a.a))["per_video"]
    B = json.load(open(a.b))["per_video"]
    vids = sorted(set(A) & set(B))
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(vids), (a.n, len(vids)))
    res = {}
    for k in sorted(set(A[vids[0]]) & set(B[vids[0]])):
        d = np.array([A[v][k] - B[v][k] for v in vids])
        boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[k] = {"a": float(np.mean([A[v][k] for v in vids])), "b": float(np.mean([B[v][k] for v in vids])),
                  "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "a_wins": int((d > 0).sum()), "n": len(vids)}
        print(f"{k:6s} A {res[k]['a']:.4f}  B {res[k]['b']:.4f}  A-B {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"A>B in {(d > 0).sum()}/{len(vids)} videos")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
