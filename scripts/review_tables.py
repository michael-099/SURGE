"""Print the numbers and LaTeX rows of the review follow-up (research/log.md section 12) from the result files, so no
number is typed by hand into the thesis.

    python scripts/review_tables.py causal g      (12D, ViT-g; 'l' for ViT-L)
    python scripts/review_tables.py ablations     (12E)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs/review"
MET = ["cc", "nss", "sim", "kld", "auc_j", "sauc"]


def load(p):
    return json.load(open(p))


def sgn(x, nd=3):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def ci(c, nd=3):
    d, (lo, hi) = c["diff"], c["ci95"]
    bold = lo > 0 or hi < 0
    v = f"\\mathbf{{{sgn(d, nd).replace('$-$', '-')}}}" if bold else sgn(d, nd).replace("$-$", "-")
    return f"${v}$ [{sgn(lo, nd)}, {sgn(hi, nd)}]"


def best_epoch(run):
    rows = [json.loads(line) for line in open(ROOT / "runs" / run / "log.jsonl") if line.strip()]
    hold = {r["epoch"]: r["cc"] for r in rows if r.get("event") == "holdout"}
    b = max(hold, key=hold.get)
    return b, hold[b], hold


def causal(tag):
    name = {"g": "ViT-g", "l": "ViT-L"}[tag]
    print(f"== 12D causal {name}")
    for k in range(3):
        b, cc, hold = best_epoch(f"causal_{tag}_s{k}")
        print(f"  seed {k}: holdout CC by epoch {', '.join(f'{e}:{c:.4f}' for e, c in sorted(hold.items()))}; best epoch {b}")
    s1, se = load(R / f"causal_{tag}_sigma_s0.json")["best_sigma"], load(R / f"causal_{tag}_sigma_ens3.json")["best_sigma"]
    print(f"  sigma: single {s1}, ensemble {se}")
    sal0, salb = load(R / "salfom_val_step10_sigma0.json")["mean"], load(R / "salfom_val_step10_best.json")["mean"]
    print("  SalFoM every 10th frame: released " + " ".join(f"{m}={sal0[m]:.4f}" for m in MET))
    print("                           best blur " + " ".join(f"{m}={salb[m]:.4f}" for m in MET))
    rows = []
    vals = {}
    for x in ("s0", "s1", "s2", "ens3"):
        v = load(R / f"causal_{tag}_val_{x}.json")["mean"]
        vals[x] = v
        cb = load(R / f"cmp_causal_{tag}_{x}_vs_salfom_best.json")
        c0 = load(R / f"cmp_causal_{tag}_{x}_vs_salfom.json")
        print(f"  causal {x}: " + " ".join(f"{m}={v[m]:.4f}" for m in MET))
        print(f"     vs SalFoM best blur: " + "  ".join(f"{m} {cb[m]['diff']:+.4f} [{cb[m]['ci95'][0]:+.4f}, {cb[m]['ci95'][1]:+.4f}]" for m in MET))
        print(f"     vs SalFoM released : CC {c0['cc']['diff']:+.4f} [{c0['cc']['ci95'][0]:+.4f}, {c0['cc']['ci95'][1]:+.4f}]")
        rows.append((x, v, cb, c0))
    import numpy as np
    ccs = [vals[x]["cc"] for x in ("s0", "s1", "s2")]
    print(f"  causal single seeds CC mean {np.mean(ccs):.4f} SD {np.std(ccs, ddof=1):.4f}")
    off = {}
    if tag == "g" and (R / "offline_g_val_step10_ens3.json").exists():
        for x in ("s0", "ens3"):
            off[x] = load(R / f"offline_g_val_step10_{x}.json")["mean"]
            co = load(R / f"cmp_causal_vs_offline_g_{x}.json")
            print(f"  offline {x} (same frames): " + " ".join(f"{m}={off[x][m]:.4f}" for m in MET))
            print(f"     causal - offline: " + "  ".join(f"{m} {co[m]['diff']:+.4f} [{co[m]['ci95'][0]:+.4f}, {co[m]['ci95'][1]:+.4f}]" for m in MET))
        ob = load(R / "cmp_offline_g_ens3_step10_vs_salfom_best.json")
        print(f"  offline ens3 vs SalFoM best (same frames): CC {ob['cc']['diff']:+.4f} [{ob['cc']['ci95'][0]:+.4f}, {ob['cc']['ci95'][1]:+.4f}]")
    lab = {"s0": "seed 0", "s1": "seed 1", "s2": "seed 2", "ens3": "three-seed ensemble"}
    print("% LaTeX rows (tab:causal): model & CC & NSS & dCC vs SalFoM released & dCC vs SalFoM best blur")
    for x, v, cb, c0 in rows:
        print(f"    Causal, {lab[x]} & {v['cc']:.4f} & {v['nss']:.3f} & {ci(c0['cc'])} & {ci(cb['cc'])} \\\\")
    for x, v in off.items():
        f0, fb = R / f"cmp_offline_g_{x}_step10_vs_salfom.json", R / f"cmp_offline_g_{x}_step10_vs_salfom_best.json"
        d0 = ci(load(f0)["cc"]) if f0.exists() else "--"
        db = ci(load(fb)["cc"]) if fb.exists() else "--"
        print(f"    Offline, {lab[x]} & {v['cc']:.4f} & {v['nss']:.3f} & {d0} & {db} \\\\")
    mname = {"cc": "CC", "nss": "NSS", "sim": "SIM", "kld": "KLD", "auc_j": "AUC-J", "sauc": "sAUC"}

    def dci(c):
        return f"\\dci{{{sgn(c['diff'], 4)}}}{{{sgn(c['ci95'][0], 4)}, {sgn(c['ci95'][1], 4)}}}"

    print("% appendix rows (tab:app-causal): metric & causal - SalFoM best blur for seed 0, 1, 2, ensemble")
    for i, m in enumerate(MET):
        end = " \\\\[5pt]" if i < len(MET) - 1 else " \\\\"
        print(f"    \\acrshort{{{mname[m]}}} & " + " & ".join(dci(cb[m]) for _, _, cb, _ in rows) + end)
    if off:
        print("% appendix rows (tab:app-causal-offline): metric & causal - offline, seed 0 & ensemble")
        co = {x: load(R / f"cmp_causal_vs_offline_g_{x}.json") for x in off}
        for i, m in enumerate(MET):
            end = " \\\\[5pt]" if i < len(MET) - 1 else " \\\\"
            print(f"    \\acrshort{{{mname[m]}}} & " + " & ".join(dci(co[x][m]) for x in off) + end)


def ablations():
    a = load(R / "ablation_seeds.json")
    print("== 12E ablations with three seeds (holdout CC)")
    for f, r in a.items():
        print(f"  {r['label']:28s} {r['mean']:+.4f}  95% [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"90% [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]  per seed {', '.join(f'{x:+.4f}' for x in r['per_seed'])}"
              f"  -> {r['outcome']}")
    print("% LaTeX rows: factor & per-seed dCC & mean [95%] & 90% & outcome")
    for f, r in a.items():
        ps = " / ".join(sgn(x, 4) for x in r["per_seed"])
        print(f"    {r['label']} & {ps} & {sgn(r['mean'], 4)} [{sgn(r['ci95'][0], 4)}, {sgn(r['ci95'][1], 4)}] & "
              f"[{sgn(r['ci90'][0], 4)}, {sgn(r['ci90'][1], 4)}] & {r['outcome']} \\\\")


if __name__ == "__main__":
    if sys.argv[1] == "causal":
        causal(sys.argv[2])
    else:
        ablations()
