"""Research log section 16 results, post hoc and descriptive: the deficit of positions 0-1 against 8-9 on the all-16
tiling, with and without each video's first window (frames 0-31, which have no previous window), from the probe rows
(video, position, frame, cc, nss). Same definition as inv_summary._positions: the mean of the per-position means.
Usage: python -I scripts/p1_first_window.py D:/Research/SURGE  ->  runs/review/p1/posthoc_first_window.json"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
P = root / "runs/review/p1"
FIRST = 32  # all-16 tiling: windows start at frames 0, 32, 64, ... (T 16 x stride 2)


def deficit(rows, keep):
    r = [x for x in rows if keep(x)]
    by = {k: np.nanmean([x[3] for x in r if x[1] == k]) for k in range(16)}
    p01, p89 = np.mean([by[0], by[1]]), np.mean([by[8], by[9]])
    return {"pos01": float(p01), "pos89": float(p89), "deficit": float(p89 - p01),
            "frames_pos0": sum(1 for x in r if x[1] == 0)}


sets = {"step0 plain (s0)": [P / "step0_all16.json"], "step0 stream (s0)": [P / "step0_all16_stream.json"]}
for t in ("S_stream", "S_nostream", "Rall", "R_all16"):
    sets[t] = [P / "probe" / f"p1_s{s}_{t}.json" for s in range(3)]
out = {}
for name, files in sets.items():
    res = {"all": [], "without_first_window": [], "first_window_only": []}
    for f in files:
        rows = json.load(open(f))["positions"]["rows"]
        res["all"].append(deficit(rows, lambda x: True))
        res["without_first_window"].append(deficit(rows, lambda x: x[2] >= FIRST))
        res["first_window_only"].append(deficit(rows, lambda x: x[2] < FIRST))
    out[name] = {k: {m: float(np.mean([d[m] for d in v])) for m in ("pos01", "pos89", "deficit", "frames_pos0")}
                 for k, v in res.items()}
    out[name]["per_seed_deficit_without_first"] = [d["deficit"] for d in res["without_first_window"]]
    o = out[name]
    print(f"{name:18s} all {o['all']['pos01']:.4f}/{o['all']['pos89']:.4f} def {o['all']['deficit']:+.4f} | "
          f"w/o first {o['without_first_window']['pos01']:.4f}/{o['without_first_window']['pos89']:.4f} "
          f"def {o['without_first_window']['deficit']:+.4f} (n0 {o['without_first_window']['frames_pos0']:.0f}) | "
          f"first only def {o['first_window_only']['deficit']:+.4f} (n0 {o['first_window_only']['frames_pos0']:.0f})"
          f" | per seed w/o first {', '.join(f'{x:+.4f}' for x in o['per_seed_deficit_without_first'])}")
(root / "runs/review/p1/posthoc_first_window.json").write_text(json.dumps(out, indent=1), newline="\n")
