"""Section 14F (research/log.md): fold-mean scores of the localisation-map models against the ViT-g visual fold models.

    python scripts/summarize_ssl.py --stage 1     (Coutrot2 only: the stage-1 decision)
    python scripts/summarize_ssl.py --stage 2     (all six datasets: the adoption rule)
Inputs: runs/ssl/stavis_<fold>.json (matched maps), runs/ssl/stavis_rot_<fold>.json (rotated soundtrack),
        runs/g_other/stavis_s0_<fold>.json (visual: runs/g_stavis_<fold>_s0, section 11E).
Rule (fixed before the runs): on a dataset the maps help if fold-mean ΔCC (maps - visual) >= +0.010 and matched -
rotated >= +0.005. Stage 2 runs only if they help on Coutrot2; adopted if they also lower fold-mean CC by no more than
0.005 on every other dataset. Intervals (paired bootstrap over test videos, 95 %) are descriptive only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDS = {"diem": [1], "coutrot1": [1, 2, 3], "coutrot2": [1, 2, 3], "etmd": [1, 2, 3], "avad": [1, 2, 3],
         "summe": [1, 2, 3]}
METRICS = ("cc", "nss", "sim", "kld", "auc_j", "sauc")


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def boot(d: np.ndarray, n: int = 100000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def dataset(key: str) -> dict:
    out = {"folds": {}}
    per = {"visual": {}, "maps": {}, "rotated": {}}
    fm = {c: {k: [] for k in METRICS} for c in per}
    for f in FOLDS[key]:
        F = f"{key}_f{f}"
        src = {"visual": ROOT / f"runs/g_other/stavis_s0_{F}.json", "maps": ROOT / f"runs/ssl/stavis_{F}.json",
               "rotated": ROOT / f"runs/ssl/stavis_rot_{F}.json"}
        js = {c: load(p) for c, p in src.items()}
        assert set(js["visual"]["per_video"]) == set(js["maps"]["per_video"]) == set(js["rotated"]["per_video"])
        for c, j in js.items():
            for k in METRICS:
                fm[c][k].append(j["mean"][k])
            for v, s in j["per_video"].items():
                per[c][v] = s["cc"]
        out["folds"][F] = {c: js[c]["mean"]["cc"] for c in js}
    for c in fm:
        out[c] = {k: float(np.mean(x)) for k, x in fm[c].items()}
    vids = sorted(per["visual"])
    a = {c: np.array([per[c][v] for v in vids]) for c in per}
    out["d_maps_visual"] = out["maps"]["cc"] - out["visual"]["cc"]
    out["d_matched_rotated"] = out["maps"]["cc"] - out["rotated"]["cc"]
    out["ci_maps_visual_videos"] = boot(a["maps"] - a["visual"])
    out["ci_matched_rotated_videos"] = boot(a["maps"] - a["rotated"])
    out["videos"] = len(vids)
    out["helps"] = bool(out["d_maps_visual"] >= 0.010 and out["d_matched_rotated"] >= 0.005)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    keys = ["coutrot2"] if a.stage == 1 else list(FOLDS)
    res = {k: dataset(k) for k in keys}
    print("| Dataset | Visual | Maps (matched) | Maps (rotated) | Maps - visual [95 % CI, videos] | Matched - rotated "
          "[95 % CI] | Helps? |")
    print("|---|---|---|---|---|---|---|")
    for k, r in res.items():
        print(f"| {k} | {r['visual']['cc']:.4f} | {r['maps']['cc']:.4f} | {r['rotated']['cc']:.4f} | "
              f"{r['d_maps_visual']:+.4f} [{r['ci_maps_visual_videos'][0]:+.4f}, {r['ci_maps_visual_videos'][1]:+.4f}] | "
              f"{r['d_matched_rotated']:+.4f} [{r['ci_matched_rotated_videos'][0]:+.4f}, "
              f"{r['ci_matched_rotated_videos'][1]:+.4f}] | {'yes' if r['helps'] else 'no'} |")
    if a.stage == 1:
        res["stage1_pass"] = res["coutrot2"]["helps"]
        print("STAGE1", "PASS" if res["stage1_pass"] else "FAIL")
    else:
        others_ok = all(r["d_maps_visual"] >= -0.005 for k, r in res.items() if k != "coutrot2")
        res["adopted"] = bool(res["coutrot2"]["helps"] and others_ok)
        print("ADOPTED" if res["adopted"] else "NOT ADOPTED", "(no other dataset below -0.005:", others_ok, ")")
    out = ROOT / f"runs/ssl/summary_stage{a.stage}.json"
    out.write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
