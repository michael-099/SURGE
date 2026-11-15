"""Summaries and tests of research/log.md section 15 from the result files (CPU only).

    python scripts/inv_summary.py --part testtime   (15E1/E3, 15F2, 15G, 15H: conditions vs the unperturbed run)
    python scripts/inv_summary.py --part static     (15E2: video model vs static model, seeds 0-2, rule of 12E)
    python scripts/inv_summary.py --part ablation   (15C/D: variants vs R per seed, rule of 12E)
    python scripts/inv_summary.py --part probes     (15B: holdout CC per encoder block)
    python scripts/inv_summary.py --part failure    (15J: attributes vs per-video CC; cuts)
    python scripts/inv_summary.py --part zeroshot   (15I)
Each part writes runs/review/inv/summary_<part>.json and prints a table.

Rule of 12E (fixed in advance): effect if the 95 % interval excludes 0 and |mean| >= 0.003; equivalent if the 90 %
interval lies within (-0.003, +0.003); otherwise inconclusive. One seed: bootstrap over videos; several seeds:
two-level bootstrap (seeds with replacement, then videos with replacement, shared across seeds). 10,000 resamples,
bootstrap seed 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "runs/review/inv"
DELTA = 0.003


def outcome(mean, c95, c90, delta=DELTA):
    if (c95[0] > 0 or c95[1] < 0) and abs(mean) >= delta:
        return "effect"
    if -delta < c90[0] and c90[1] < delta:
        return "equivalent"
    return "inconclusive"


def boot(diffs: np.ndarray, n=10000, seed=0) -> dict:
    """diffs: seeds x videos (paired differences). Two-level bootstrap (one level if one seed)."""
    diffs = np.atleast_2d(diffs)
    S, V = diffs.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, S, (n, S))
    vi = rng.integers(0, V, (n, V))
    b = np.array([diffs[si[k]][:, vi[k]].mean() for k in range(n)])
    mean = float(diffs.mean())
    c95 = [float(x) for x in np.percentile(b, [2.5, 97.5])]
    c90 = [float(x) for x in np.percentile(b, [5, 95])]
    return {"mean": mean, "ci95": c95, "ci90": c90, "outcome": outcome(mean, c95, c90),
            "per_seed": [float(x) for x in diffs.mean(1)], "seeds": S, "videos": V}


def pv(path, metric="cc"):
    d = json.load(open(path))
    return {k: v[metric] for k, v in d["per_video"].items()}, d


def paired(ref_paths, var_paths, metric="cc"):
    refs = [pv(p, metric)[0] for p in ref_paths]
    vars_ = [pv(p, metric)[0] for p in var_paths]
    vids = sorted(set.intersection(*[set(r) for r in refs + vars_]))
    d = np.array([[v[x] - r[x] for x in vids] for r, v in zip(refs, vars_)])
    out = boot(d)
    out["ref_mean"] = [float(np.mean([r[x] for x in vids])) for r in refs]
    out["var_mean"] = [float(np.mean([v[x] for x in vids])) for v in vars_]
    return out


def fmt(name, r, metric="cc"):
    return (f"{name:28s} {metric} {r['var_mean'][0] if len(r['var_mean']) == 1 else np.mean(r['var_mean']):.4f}  "
            f"diff {r['mean']:+.4f}  95% [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
            f"90% [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]  -> {r['outcome']}")


def part_testtime():
    ref = R / "t_orig_s0.json"
    conds = ["t_reverse_s0", "t_shuffle_s0", "t_static_s0", "t_stride1_s0", "t_stride4_s0", "t_all16_s0",
             "t_central4_s0", "t_overlap_s0", "t_dense_s0", "t_size192_s0", "t_size224_s0", "t_size320_s0",
             "t_size384_s0", "t_noop_s0", "t_noise04_s0", "t_noise16_s0", "t_blur2_s0", "t_blur8_s0", "t_jpeg10_s0",
             "t_contrast05_s0", "t_bright05_s0", "t_gray_s0", "t_dup_s0"]
    res = {"reference": json.load(open(ref))["mean"]}
    print(f"reference t_orig_s0: " + " ".join(f"{k}={v:.4f}" for k, v in res["reference"].items()))
    for c in conds:
        p = R / f"{c}.json"
        if not p.exists():
            continue
        row = {}
        for m in ("cc", "nss", "sim", "kld"):
            row[m] = paired([ref], [p], m)
        d = json.load(open(p))
        row["passes_per_video_frame"] = d.get("passes_per_video_frame")
        res[c] = row
        print(fmt(c, row["cc"]) + f"  NSS {row['nss']['mean']:+.3f}  passes/frame {row['passes_per_video_frame']:.4f}")
    o = json.load(open(ref))
    if "positions" in o:
        bp = o["positions"]["by_position"]
        res["positions"] = bp
        print("CC by output position:", " ".join(f"{k}:{bp[k]['cc']:.3f}" for k in sorted(bp, key=int)))
    if "flip" in o:
        res["flip"] = o["flip"]["mean"]
        print("flip consistency:", {k: round(v, 4) for k, v in o["flip"]["mean"].items()})
    return res


def part_static():
    res = {}
    for m in ("cc", "nss", "sim", "kld"):
        refs = [ROOT / f"runs/review/holdout/vitg_s{s}.json" for s in range(3)]
        st = [R / f"static_model_s{s}.json" for s in range(3)]
        if not all(p.exists() for p in st):
            print("static models not all scored yet")
            return res
        r = paired(refs, st, m)  # static - video
        res[m] = r
        print(fmt(f"static model - video ({m})", r, m))
    return res


C_VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6", "lr1e-4", "lr9e-4", "nocc", "nonss"]


def part_ablation():
    res = {}
    H = R / "holdout"
    H.mkdir(exist_ok=True)
    seeds = [s for s in range(3) if (ROOT / f"runs/inv_c_s{s}/DONE").exists()]
    if not seeds:
        print("no finished 15C run")
        return res
    for s in seeds:  # copy each variant's best per-video scores in the format of runs/review/holdout
        for v in ["R"] + C_VARIANTS + (["R6"] if s == 0 else []):
            src = ROOT / f"runs/inv_c_s{s}/{v}/best.json"
            if src.exists():
                d = json.load(open(src))
                (H / f"c_s{s}_{v}.json").write_text(json.dumps({"mean": d["mean"], "per_video": d["per_video"],
                                                                 "epoch": d["epoch"]}), newline="\n")
    print(f"seeds {seeds}")
    for v in C_VARIANTS:
        row = {}
        for m in ("cc", "nss", "sim", "kld"):
            row[m] = paired([H / f"c_s{s}_R.json" for s in seeds], [H / f"c_s{s}_{v}.json" for s in seeds], m)
        row["best_epoch"] = [json.load(open(H / f"c_s{s}_{v}.json"))["epoch"] for s in seeds]
        res[v] = row
        print(fmt(v, row["cc"]) + f"  per seed {', '.join(f'{x:+.4f}' for x in row['cc']['per_seed'])}  "
              f"NSS {row['nss']['mean']:+.3f}  best epochs {row['best_epoch']}")
    # R against the standard runs of train.py (descriptive)
    rr = paired([ROOT / f"runs/review/holdout/vitg_s{s}.json" for s in seeds], [H / f"c_s{s}_R.json" for s in seeds])
    res["R_vs_standard"] = rr
    print(fmt("R (multi) - vitg_s (standard)", rr))
    if (H / "c_s0_R6.json").exists():
        res["R6_seed0"] = paired([H / "c_s0_R.json"], [H / "c_s0_R6.json"])
        print(fmt("R6 - R (seed 0)", res["R6_seed0"]))
    return res


def part_probes():
    d = ROOT / "runs/inv_b_s0"
    res = {}
    for p in sorted(d.glob("*/best.json"), key=lambda p: (p.parent.name[0], int(p.parent.name[1:]))):
        b = json.load(open(p))
        res[p.parent.name] = {"cc": b["mean"]["cc"], "nss": b["mean"]["nss"], "epoch": b["epoch"]}
    for k, v in res.items():
        print(f"{k:5s} CC {v['cc']:.4f} NSS {v['nss']:.3f} epoch {v['epoch']}")
    return res


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def boot_spearman(x, y, n=10000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    r, k = spearman(x, y)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(spearman(x[i], y[i])[0])
    return {"rho": r, "n": k, "ci95": [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]}


def part_failure():
    res = {}
    for split, perf in (("holdout", R / "t_orig_s0.json"), ("val", ROOT / "runs/vitg/val_single_s0.json")):
        gs = json.load(open(R / f"gt_stats_{split}.json"))
        cc_v, d = pv(perf)
        prior = None
        if split == "holdout" and (ROOT / "runs/inv_b_s0/P0/best.json").exists():
            prior = {k: v["cc"] for k, v in json.load(open(ROOT / "runs/inv_b_s0/P0/best.json"))["per_video"].items()}
        vids = sorted(set(cc_v) & set(gs))
        attrs = {
            "motion": [np.nanmean(gs[v]["motion"]) for v in vids],
            "cuts_per_100f": [100.0 * len(gs[v]["cuts"]) / gs[v]["n"] for v in vids],
            "gt_entropy": [np.nanmean(gs[v]["entropy"]) for v in vids],
            "gt_centre": [np.nanmean(gs[v]["centre"]) for v in vids],
            "gt_change5": [np.nanmean(gs[v]["change5"]) for v in vids],
            "ceil16_cc": [np.nanmean(gs[v]["ceil16_cc"]) for v in vids],
        }
        y = [cc_v[v] for v in vids]
        out = {"videos": len(vids), "cc": {a: boot_spearman(x, y) for a, x in attrs.items()}}
        if prior:
            gain = [cc_v[v] - prior[v] for v in vids]
            out["gain_over_prior"] = {a: boot_spearman(x, gain) for a, x in attrs.items()}
            out["prior_cc_mean"] = float(np.mean([prior[v] for v in vids]))
        order = np.argsort(y)
        out["worst10"] = [{"video": vids[i], "cc": y[i], **{a: float(attrs[a][i]) for a in attrs}} for i in order[:10]]
        out["best10"] = [{"video": vids[i], "cc": y[i], **{a: float(attrs[a][i]) for a in attrs}} for i in order[-10:]]
        res[split] = out
        print(f"== {split} ({len(vids)} videos): Spearman rho with per-video CC")
        for a, r in out["cc"].items():
            g = out.get("gain_over_prior", {}).get(a)
            print(f"  {a:14s} rho {r['rho']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" +
                  (f"   gain over prior: {g['rho']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]" if g else ""))
    # cuts, frame level (holdout, per-frame scores of t_orig_s0)
    o = json.load(open(R / "t_orig_s0.json"))
    gs = json.load(open(R / "gt_stats_holdout.json"))
    near, far = [], []
    for v, pf in o.get("per_frame", {}).items():
        g = gs[v]
        sc = dict(zip(g["f"], g["since_cut"]))
        cc_f = dict(zip(pf["frames"], pf["cc"]))
        a = [cc_f[f] for f in pf["frames"] if 0 <= sc.get(f, -1) < 15 and np.isfinite(cc_f[f])]
        b = [cc_f[f] for f in pf["frames"] if not (0 <= sc.get(f, -1) < 15) and np.isfinite(cc_f[f])]
        if a and b:
            near.append(np.mean(a))
            far.append(np.mean(b))
    if near:
        dd = np.array(near) - np.array(far)
        r = boot(dd[None, :])
        res["cuts_frame_level"] = {"videos_with_cuts": len(near), "near_cut_cc": float(np.mean(near)),
                                   "other_cc": float(np.mean(far)), **r}
        print(f"frames < 15 after a cut vs other frames (holdout, {len(near)} videos with cuts): "
              f"{np.mean(near):.4f} vs {np.mean(far):.4f}, diff {r['mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    return res


def part_zeroshot():
    res = {}
    for p in sorted(R.glob("zs_*.json")):
        d = json.load(open(p))
        res[p.stem] = {**d["mean"], "videos": d["videos"], "frames": d["frames"], "sigma": d["args"]["sigma"]}
        print(f"{p.stem:22s} " + " ".join(f"{k}={v:.4f}" for k, v in d["mean"].items()) + f"  ({d['videos']} videos)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["testtime", "static", "ablation", "probes", "failure", "zeroshot"])
    a = ap.parse_args()
    res = globals()[f"part_{a.part}"]()
    (R / f"summary_{a.part}.json").write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
