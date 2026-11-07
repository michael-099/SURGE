"""Fold means for the STAViS seed ensembles (research/log.md section 10A) next to the seed-0 single models.
    python scripts/summarize_stavis_ens.py  -> prints a markdown table, writes runs/ens/stavis_summary.json"""
import json
from pathlib import Path

DS = {"diem": 1, "coutrot1": 3, "coutrot2": 3, "etmd": 3, "avad": 3, "summe": 3}
MET = ["cc", "nss", "sim", "kld", "auc_j", "sauc"]
out = {}
print("| Dataset | Model | " + " | ".join(MET) + " |\n|---|---|" + "---|" * len(MET))
for d, n in DS.items():
    for tag, pat in (("single", "runs/stavis/ft_{}.json"), ("ensemble", "runs/ens/stavis_{}.json")):
        fs = [Path(pat.format(f"{d}_f{k}")) for k in range(1, n + 1)]
        if not all(f.exists() for f in fs):
            continue
        m = {k: sum(json.load(open(f))["mean"][k] for f in fs) / n for k in MET}
        out.setdefault(d, {})[tag] = m
        print(f"| {d} | {tag} | " + " | ".join(f"{m[k]:.3f}" for k in MET) + " |")
    if len(out.get(d, {})) == 2:
        print(f"| {d} | Δ | " + " | ".join(f"{out[d]['ensemble'][k] - out[d]['single'][k]:+.3f}" for k in MET) + " |")
json.dump(out, open("runs/ens/stavis_summary.json", "w"), indent=1)
