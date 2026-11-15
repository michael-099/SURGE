"""Scene cuts by frames since the cut (research/log.md section 15L, descriptive addition; CPU).

    python scripts/inv_cuts_bins.py --out runs/review/inv/summary_cuts_bins.json

Same inputs and pairing as scripts/inv_cuts.py. For each bin of frames since the last cut (0-4, 5-9, 10-14, and
>= 60 for reference), per-video mean CC under each condition and the per-video difference clamped - unclamped, with
the same bootstrap (two-level over seeds and videos when several seeds exist). Descriptive: the 12E label is printed
for orientation only; the primary outcome of 15L is the pooled 0-14 bin of scripts/inv_cuts.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inv_cuts import frame_table, per_video  # noqa: E402
from scripts.inv_summary import boot  # noqa: E402

R = ROOT / "runs/review/inv"
BINS = {"0-4": (0, 5), "5-9": (5, 10), "10-14": (10, 15), ">=60": (60, 10**9)}


def compare_bins(pairs, kind):
    """pairs: list of (unclamped path, clamped path), one per seed -> {bin: result}."""
    tabs = [(frame_table(a, kind), frame_table(b, kind)) for a, b in pairs]
    res = {}
    for name, (lo, hi) in BINS.items():
        sel = lambda sc, lo=lo, hi=hi: lo <= sc < hi  # noqa: E731
        un = [per_video(t0, sel) for t0, _ in tabs]
        cl = [per_video(t1, sel) for _, t1 in tabs]
        vids = sorted(set.intersection(*[set(x) for x in un + cl]))
        if not vids:
            res[name] = None
            continue
        d = np.array([[c[v] - u[v] for v in vids] for u, c in zip(un, cl)])
        r = boot(d)
        r["unclamped"] = float(np.mean([[u[v] for v in vids] for u in un]))
        r["clamped"] = float(np.mean([[c[v] for v in vids] for c in cl]))
        r["frames_seed0"] = int(sum(1 for rows in tabs[0][0].values() for c, sc in rows.values()
                                    if lo <= sc < hi and np.isfinite(c)))
        res[name] = r
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="override: KIND UNCLAMPED CLAMPED [UNCLAMPED CLAMPED ...] (for tests)")
    a = ap.parse_args(argv)
    out = {}
    if a.pairs:
        kind, paths = a.pairs[0], a.pairs[1:]
        out[kind] = compare_bins([(Path(paths[i]), Path(paths[i + 1])) for i in range(0, len(paths), 2)], kind)
    else:
        diem = [(R / f"l_diem_dense_s{s}.json", R / f"l_diem_shot_s{s}.json") for s in range(3)]
        diem = [p for p in diem if p[0].exists() and p[1].exists()]
        if diem:
            out["diem"] = compare_bins(diem, "diem")
        if (R / "t_dense_s0.json").exists() and (R / "l_dhf1k_shot_s0.json").exists():
            out["dhf1k_holdout"] = compare_bins([(R / "t_dense_s0.json", R / "l_dhf1k_shot_s0.json")], "dhf1k")
    for k, res in out.items():
        print(f"== {k}")
        for name, x in res.items():
            if x is None:
                print(f"  {name:6s} no frames")
                continue
            print(f"  {name:6s} n={x['frames_seed0']:5d} videos={x['videos']:3d}  unclamped {x['unclamped']:.4f} "
                  f"clamped {x['clamped']:.4f}  diff {x['mean']:+.4f} 95% [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                  f"  ({x['outcome']}, descriptive)")
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")


if __name__ == "__main__":
    main()
