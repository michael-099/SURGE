"""Scene-cut hypothesis test of research/log.md section 15L (CPU).

    python scripts/inv_cuts.py --out runs/review/inv/summary_cuts.json

Reads per-frame scores of dense windows without (`*_dense_*`) and with (`*_shot_*`) clamping to the frame's shot:
  DIEM (primary):   runs/review/inv/l_diem_dense_s{0,1,2}.json, l_diem_shot_s{0,1,2}.json
  DHF1K holdout:    runs/review/inv/t_dense_s0.json (15G), l_dhf1k_shot_s0.json
Post-cut frames: 0-14 frames after a cut (a shot start > 0). Per video, mean CC over its post-cut frames per condition;
outcome by the rule of 12E on the per-video difference clamped - unclamped (two-level bootstrap over seeds and videos,
10,000 resamples, seed 0). Descriptive: all frames, and the post-cut dip (0-14 against >= 60 frames after a cut).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inv_summary import boot  # noqa: E402
from scripts.probe_eval import load_shots  # noqa: E402

R = ROOT / "runs/review/inv"


def frame_table(path, kind):
    """{video: {frame: (cc, frames since the last cut or -1)}}"""
    d = json.load(open(path))
    out = {}
    for v, pf in d["per_frame"].items():
        n = max(pf["frames"]) + 1
        if kind == "dhf1k":
            from surge.data import dhf1k_videos
            n = len(dhf1k_videos("D:/DHF1K", [int(v)])[0])
        shots = load_shots(kind, v, n if kind != "diem" else 10**7)
        starts = [a for a, _ in shots if a > 0]
        rows = {}
        for f, c in zip(pf["frames"], pf["cc"]):
            prev = [s for s in starts if s <= f]
            rows[f] = (c, f - max(prev) if prev else -1)
        out[v] = rows
    return out


def per_video(tab, sel):
    out = {}
    for v, rows in tab.items():
        xs = [c for f, (c, sc) in rows.items() if sel(sc) and np.isfinite(c)]
        if xs:
            out[v] = float(np.mean(xs))
    return out


POST = lambda sc: 0 <= sc < 15  # noqa: E731
LATE = lambda sc: sc >= 60  # noqa: E731
ALL = lambda sc: True  # noqa: E731


def compare(pairs, kind):
    """pairs: list of (unclamped path, clamped path), one per seed."""
    res = {}
    tabs = [(frame_table(a, kind), frame_table(b, kind)) for a, b in pairs]
    for name, sel in (("post_cut", POST), ("all_frames", ALL)):
        un = [per_video(t0, sel) for t0, _ in tabs]
        cl = [per_video(t1, sel) for _, t1 in tabs]
        vids = sorted(set.intersection(*[set(x) for x in un + cl]))
        d = np.array([[c[v] - u[v] for v in vids] for u, c in zip(un, cl)])
        r = boot(d)
        r["unclamped"] = float(np.mean([[u[v] for v in vids] for u in un]))
        r["clamped"] = float(np.mean([[c[v] for v in vids] for c in cl]))
        res[name] = r
    for cond, idx in (("unclamped", 0), ("clamped", 1)):  # post-cut dip, descriptive
        post = [per_video(t[idx], POST) for t in tabs]
        late = [per_video(t[idx], LATE) for t in tabs]
        vids = sorted(set.intersection(*[set(x) for x in post + late]))
        d = np.array([[p[v] - l[v] for v in vids] for p, l in zip(post, late)])
        res[f"dip_{cond}"] = boot(d)
    n_post = sum(1 for t0, _ in tabs[:1] for rows in t0.values() for c, sc in rows.values() if POST(sc))
    res["post_cut_frames_seed0"] = n_post
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    diem = [(R / f"l_diem_dense_s{s}.json", R / f"l_diem_shot_s{s}.json") for s in range(3)]
    diem = [p for p in diem if p[0].exists() and p[1].exists()]
    if diem:
        out["diem"] = compare(diem, "diem")
    if (R / "t_dense_s0.json").exists() and (R / "l_dhf1k_shot_s0.json").exists():
        out["dhf1k_holdout"] = compare([(R / "t_dense_s0.json", R / "l_dhf1k_shot_s0.json")], "dhf1k")
    for k, r in out.items():
        print(f"== {k} ({r['post_cut']['seeds']} seeds, {r['post_cut']['videos']} videos, "
              f"{r['post_cut_frames_seed0']} post-cut frames in seed 0)")
        for part in ("post_cut", "all_frames"):
            x = r[part]
            print(f"  {part:10s} unclamped {x['unclamped']:.4f} clamped {x['clamped']:.4f}  diff {x['mean']:+.4f} "
                  f"95% [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] 90% [{x['ci90'][0]:+.4f}, {x['ci90'][1]:+.4f}] "
                  f"-> {x['outcome']}")
        for cond in ("unclamped", "clamped"):
            x = r[f"dip_{cond}"]
            print(f"  post-cut dip ({cond}): {x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]")
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")


if __name__ == "__main__":
    main()
