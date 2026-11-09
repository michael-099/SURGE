"""Section 11E: ViT-g on the STAViS datasets, next to ViT-L (sections 7 and 10A).
    python scripts/summarize_stavis_vitg.py
Fold means per dataset (as in sections 7/10A), and per-dataset pooled test videos (each video is in exactly one test
fold) written to runs/g_other/pooled_<model>_<dataset>.json so that scripts/compare.py can bootstrap over videos.
Writes runs/g_other/stavis_summary.json."""
import json
from pathlib import Path

DS = {"diem": 1, "coutrot1": 3, "coutrot2": 3, "etmd": 3, "avad": 3, "summe": 3}
MET = ["cc", "nss", "sim", "kld", "auc_j", "sauc"]
SRC = {"L_single": "runs/stavis/ft_{}.json", "L_ens": "runs/ens/stavis_{}.json",
       "g_single": "runs/g_other/stavis_s0_{}.json", "g_ens": "runs/g_other/stavis_ens_{}.json"}
out = {}
print("| Dataset | Model | " + " | ".join(MET) + " |\n|---|---|" + "---|" * len(MET))
for d, n in DS.items():
    for tag, pat in SRC.items():
        fs = [Path(pat.format(f"{d}_f{k}")) for k in range(1, n + 1)]
        if not all(f.exists() for f in fs):
            continue
        js = [json.load(open(f)) for f in fs]
        m = {k: sum(j["mean"][k] for j in js) / n for k in MET}
        out.setdefault(d, {})[tag] = m
        pv = {}
        for j in js:
            assert not set(pv) & set(j["per_video"])
            pv.update(j["per_video"])
        Path(f"runs/g_other/pooled_{tag}_{d}.json").write_text(json.dumps({"mean": m, "per_video": pv}))
        print(f"| {d} | {tag} | " + " | ".join(f"{m[k]:.3f}" for k in MET) + " |")
json.dump(out, open("runs/g_other/stavis_summary.json", "w"), indent=1)
