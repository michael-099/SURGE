"""Fold means for section 11D: visual (section 7 single), audio (matched), audio (rotated control).
    python scripts/summarize_audio.py  -> markdown table + claim-rule outcome; writes runs/av/summary.json"""
import json
from pathlib import Path

DS = {"diem": 1, "coutrot1": 3, "coutrot2": 3, "etmd": 3, "avad": 3, "summe": 3}
MET = ["cc", "nss", "sim", "kld", "auc_j", "sauc"]
SRC = {"visual": "runs/stavis/ft_{}.json", "audio": "runs/av/stavis_{}.json", "rotated": "runs/av/stavis_rot_{}.json"}
out = {}
print("| Dataset | Model | " + " | ".join(MET) + " |\n|---|---|" + "---|" * len(MET))
for d, n in DS.items():
    for tag, pat in SRC.items():
        fs = [Path(pat.format(f"{d}_f{k}")) for k in range(1, n + 1)]
        if all(f.exists() for f in fs):
            m = {k: sum(json.load(open(f))["mean"][k] for f in fs) / n for k in MET}
            out.setdefault(d, {})[tag] = m
            print(f"| {d} | {tag} | " + " | ".join(f"{m[k]:.3f}" for k in MET) + " |")
print()
for d, r in out.items():
    if len(r) == 3:
        dv = r["audio"]["cc"] - r["visual"]["cc"]
        dr = r["audio"]["cc"] - r["rotated"]["cc"]
        r["claim"] = bool(dv >= 0.010 and dr >= 0.005)
        print(f"{d}: audio - visual CC {dv:+.4f}, matched - rotated {dr:+.4f} -> audio helps: {r['claim']}")
json.dump(out, open("runs/av/summary.json", "w"), indent=1)
