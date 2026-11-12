"""Print the numbers and LaTeX rows of the review follow-ups (research/log.md sections 12 and 13) from the result
files, so no number is typed by hand into the thesis.

    python scripts/review_tables.py causal g      (12D, ViT-g; 'l' for ViT-L)
    python scripts/review_tables.py ablations     (12E)
    python scripts/review_tables.py g2 [file]     (13B, 13C and the mask-token addendum; default
                                                   runs/review/g2_incremental.json)
    python scripts/review_tables.py timing [file] (13D; default runs/review/timing.json)
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


ERR = {"surprise_raw": "Raw surprise", "surprise_ctx": "Contextual surprise", "masked_tube": "Masked-tube error"}
REF = {"atypicality": "Token atypicality", "frame_diff": "Frame difference", "center_prior": "Empirical centre prior"}


def g2(path=None):
    """Sections 13B and 13C with the mask-token addendum: gate G2 re-scored, incremental information."""
    d = load(path or R / "g2_incremental.json")
    S = {8: d["stride_8"], 2: d["stride_2"]}
    print(f"== 13B/13C: {S[8]['videos']} videos; scored frames {S[8]['frames']} (4 fps), {S[2]['frames']} (15 fps); "
          f"smoke={d['smoke']}; reproduction_passed={d['reproduction_passed']}")
    c = d["checks"]
    print(f"  encoder checks passed={c.get('passed')}: all-visible {c.get('all_visible_max_abs_diff')}, first-half "
          f"{c.get('first_half_max_abs_diff')} (tolerance {c.get('tolerance')}, {c.get('dtype')})")
    print(f"  mask-token norms {[round(x, 4) for x in c.get('mask_token_l2_norm', [])]}")
    md = c.get("mask_token_diagnostic", {})
    if md:
        e = md["mean_error_per_token"]
        print(f"  masked-tube error per token on {len(md['videos'])} videos: token 0 {e[0]:.4f}, tokens 1-9 "
              f"{min(e[1:]):.4f} to {max(e[1:]):.4f}")
    worst = 0.0
    for s in (8, 2):
        for k, v in S[s]["reproduction"].items():
            dn = v["nss_stored_maps"] - v["nss_g2"]
            worst = max(worst, abs(dn)) if "passed" in v else worst
            ig = (f"  IG(mix) {v['ig_mix_stored_maps']:+.4f} vs {v['ig_mix_g2']:+.4f}" if "ig_mix_g2" in v else "")
            print(f"  stride {s} {k:14s} NSS {v['nss_stored_maps']:+.4f} vs G2 {v['nss_g2']:+.4f} (diff {dn:+.5f})"
                  f"{'  rule' if 'passed' in v else ''}{ig}")
    print(f"  largest |NSS difference| among the three maps of the rule: {worst:.5f}")
    for tok, g in d["gate"].items():
        print(f"  GATE {tok}: {'GO' if g['go'] else 'NO-GO'} {g['variants']}")
    for s in (8, 2):
        m = S[s]["standalone"]["maps"]
        for sfx in ("", "_t0"):
            for var in ("surprise_raw", "surprise_ctx"):
                k = var + sfx
                print(f"    stride {s} {k:16s} NSS {m[k]['nss']['mean']:+.4f} (atypicality "
                      f"{m['atypicality']['nss']['mean']:+.4f}, frame difference {m['frame_diff']['nss']['mean']:+.4f}); "
                      f"IG(mix) {m[k]['ig_mix']:+.4f}")
    num = lambda x, nd=3: "$" + f"{x:.{nd}f}" + "$"  # noqa: E731
    print("% LaTeX rows (tab:g2): map & NSS 4 fps & NSS 15 fps & CC 4 fps & CC 15 fps")
    for title, sfx, names in (("trained mask token", "_t0", ERR), ("untrained mask token", "", ERR),
                              ("reference maps", "", REF)):
        print(f"    % {title}")
        for k, lab in names.items():
            v8, v2 = S[8]["standalone"]["maps"][k + sfx], S[2]["standalone"]["maps"][k + sfx]
            print(f"    {lab} & {num(v8['nss']['mean'])} & {num(v2['nss']['mean'])} & {num(v8['cc']['mean'])} & "
                  f"{num(v2['cc']['mean'])} \\\\")

    def dci(r, nd=3):
        return f"\\dci{{{sgn(r['mean'], nd)}}}{{{sgn(r['ci95'][0], nd)}, {sgn(r['ci95'][1], nd)}}}"

    print("% appendix rows (tab:app-g2): map & NSS 4 fps & NSS 15 fps & CC 4 fps & CC 15 fps, 95 % intervals")
    for title, sfx in (("trained mask token", "_t0"), ("untrained mask token", "")):
        print(f"    % {title}")
        for k, lab in ERR.items():
            v8, v2 = S[8]["standalone"]["maps"][k + sfx], S[2]["standalone"]["maps"][k + sfx]
            print(f"    {lab} & {dci(v8['nss'])} & {dci(v2['nss'])} & {dci(v8['cc'])} & {dci(v2['cc'])} \\\\[5pt]")
        for k, lab in ERR.items():
            for ref, rl in (("frame_diff", "frame difference"), ("atypicality", "atypicality")):
                q = lambda s, key: S[s]["standalone"]["differences"][f"{k + sfx}-{ref}/{key}"]  # noqa: E731
                print(f"    {lab} $-$ {rl} & {dci(q(8, 'nss'))} & {dci(q(2, 'nss'))} & {dci(q(8, 'cc'))} & "
                      f"{dci(q(2, 'cc'))} \\\\[5pt]")
    print("    % reference maps")
    for k, lab in REF.items():
        v8, v2 = S[8]["standalone"]["maps"][k], S[2]["standalone"]["maps"][k]
        print(f"    {lab} & {dci(v8['nss'])} & {dci(v2['nss'])} & {dci(v8['cc'])} & {dci(v2['cc'])} \\\\[5pt]")
    below = [k for s in (8, 2) for k, v in S[s]["standalone"]["differences"].items() if v["ci95"][1] >= 0]
    print(f"  paired differences whose 95 % interval does NOT lie below zero: {below or 'none'}")
    if "incremental" not in S[8]:
        print("  no information-gain analysis in this file")
        return
    print("== incremental information U = IG(B + error) - IG(B), bits per fixation, out of fold")
    for s in (8, 2):
        mo = S[s]["models"]
        print(f"  stride {s}: IG(P) {mo['P']['ig']['mean']:+.4f} [{mo['P']['ig']['ci95'][0]:+.4f}, "
              f"{mo['P']['ig']['ci95'][1]:+.4f}], prior weight {mo['P']['weights_fit_on_A'][0]:.3f} / "
              f"{mo['P']['weights_fit_on_B'][0]:.3f};  IG(B) {mo['B']['ig']['mean']:+.4f} "
              f"[{mo['B']['ig']['ci95'][0]:+.4f}, {mo['B']['ig']['ci95'][1]:+.4f}], weights A "
              f"{[round(x, 3) for x in mo['B']['weights_fit_on_A']]} B {[round(x, 3) for x in mo['B']['weights_fit_on_B']]}")
        for k, u in S[s]["incremental"].items():
            print(f"    U({k:16s}) {u['mean']:+.4f} 95% [{u['ci95'][0]:+.4f}, {u['ci95'][1]:+.4f}] 90% "
                  f"[{u['ci90'][0]:+.4f}, {u['ci90'][1]:+.4f}] -> {u['reading']}; weight {u['error_weight'][0]:+.3f} / "
                  f"{u['error_weight'][1]:+.3f}{'  PRIMARY ' + u['primary'] if u['primary'] else ''}")
            print(f"        leave-one-out: frame difference {u['loo_frame_diff']['mean']:+.4f} "
                  f"[{u['loo_frame_diff']['ci95'][0]:+.4f}, {u['loo_frame_diff']['ci95'][1]:+.4f}], atypicality "
                  f"{u['loo_atypicality']['mean']:+.4f} [{u['loo_atypicality']['ci95'][0]:+.4f}, "
                  f"{u['loo_atypicality']['ci95'][1]:+.4f}]")
        for k, v in S[s]["single_cue_gain_over_prior"].items():
            print(f"    single cue over the fitted prior: {k:16s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, "
                  f"{v['ci95'][1]:+.4f}]")
        print("    mean map value: " + ", ".join(f"{k} {v:.4f}" for k, v in S[s]["mean_map_value"].items()))
        print("    token-0 ~ token-1 correlation: " + ", ".join(
            f"{k} {v:.3f}" for k, v in S[s]["token_map_correlation"].items()))
    w = lambda u: f"{sgn(u['error_weight'][0], 2)} / {sgn(u['error_weight'][1], 2)}"  # noqa: E731
    print("% LaTeX rows (tab:g2-incremental): error map & U 4 fps & weights & U 15 fps & weights")
    for title, sfx in (("trained mask token", "_t0"), ("untrained mask token", "")):
        print(f"    % {title}")
        for k, lab in ERR.items():
            u8, u2 = S[8]["incremental"][k + sfx], S[2]["incremental"][k + sfx]
            star = "$^{\\star}$" if u8["primary"] else ""
            print(f"    {lab}{star} & {dci(u8, 4)} & {w(u8)} & {dci(u2, 4)} & {w(u2)} \\\\[5pt]")
    print("    % for scale: what each simple cue adds to the full model of the primary comparison (13B), and IG(B)")
    for s in (8, 2):
        u, mo = S[s]["incremental"]["surprise_raw_t0"], S[s]["models"]
        print(f"    stride {s}: frame difference {dci(u['loo_frame_diff'], 4)}; atypicality "
              f"{dci(u['loo_atypicality'], 4)}; IG(B) {dci(mo['B']['ig'], 4)}; IG(P) {dci(mo['P']['ig'], 4)}")
    print("% appendix rows (tab:app-g2-single): cue & gain over the fitted prior, 4 fps & 15 fps")
    for k in S[8]["single_cue_gain_over_prior"]:
        lab = ERR.get(k.replace("_t0", ""), REF.get(k, k)) + (" (trained token)" if k.endswith("_t0") else "")
        print(f"    {lab} & {dci(S[8]['single_cue_gain_over_prior'][k], 4)} & "
              f"{dci(S[2]['single_cue_gain_over_prior'][k], 4)} \\\\[5pt]")


def timing(path=None):
    """Section 13D: measured inference time."""
    t = load(path or R / "timing.json")
    print(f"== 13D timing on {t['device']}, torch {t['torch']}, autocast {t['autocast']}, {t['warm_up']} warm-up and "
          f"{t['timed']} timed passes")
    lab = {"salfom": "SalFoM", "surge_l": "SURGE-D ViT-L", "surge_g": "SURGE-D ViT-g"}
    for k, m in t["models"].items():
        print(f"  {lab[k]}: {m['parameters']:,} parameters"
              + (f" (decoder {m['decoder_parameters']:,})" if "decoder_parameters" in m else "")
              + (f"; missing/unexpected keys {m['missing_keys']}/{m['unexpected_keys']}" if "missing_keys" in m else "")
              + f"; input {m['input']}; maps per pass {m['maps_per_pass']}")
        for B, r in m["batch"].items():
            if r.get("out_of_memory"):
                print(f"    batch {B}: out of memory")
                continue
            print(f"    batch {B}: pass {r['median_ms']:.1f} ms [{r['q1_ms']:.1f}, {r['q3_ms']:.1f}], encoder "
                  f"{r['encoder']['median_ms']:.1f} ms, per output frame {r['ms_per_output_frame']:.2f} ms"
                  + (f" (causal {r['ms_per_output_frame_causal']:.1f} ms)" if "ms_per_output_frame_causal" in r else "")
                  + f", peak {r['peak_mem_gb']:.2f} GB, output {r['output_shape']}")
    for B, row in t.get("ratios", {}).items():
        print(f"  ratios at batch {B}: " + "; ".join(f"{k} {v:.2f}" for k, v in row.items()))
    print(f"  offline statement refuted: {t.get('offline_statement_refuted')} "
          f"(batch sizes measured: {t.get('offline_statement_batches_measured')})")
    print("% LaTeX rows (tab:timing): model & ms per pass b1 & ms per frame b1 & ms per pass b8 & ms per frame b8")
    for k, m in t["models"].items():
        b = m["batch"]
        cell = lambda B, key, nd: (f"{b[B][key]:.{nd}f}" if key in b.get(B, {}) else "--")  # noqa: E731
        if k == "salfom":
            print(f"    SalFoM (one map per pass) & {cell('1', 'median_ms', 1)} & {cell('1', 'ms_per_output_frame', 1)}"
                  f" & {cell('8', 'median_ms', 1)} & {cell('8', 'ms_per_output_frame', 1)} \\\\")
        else:
            print(f"    {lab[k]}, offline (16 maps per pass) & {cell('1', 'median_ms', 1)} & "
                  f"{cell('1', 'ms_per_output_frame', 1)} & {cell('8', 'median_ms', 1)} & "
                  f"{cell('8', 'ms_per_output_frame', 1)} \\\\")
            print(f"    {lab[k]}, causal (one map per pass) & {cell('1', 'median_ms', 1)} & "
                  f"{cell('1', 'ms_per_output_frame_causal', 1)} & {cell('8', 'median_ms', 1)} & "
                  f"{cell('8', 'ms_per_output_frame_causal', 1)} \\\\")


if __name__ == "__main__":
    if sys.argv[1] == "causal":
        causal(sys.argv[2])
    elif sys.argv[1] == "g2":
        g2(*sys.argv[2:3])
    elif sys.argv[1] == "timing":
        timing(*sys.argv[2:3])
    else:
        ablations()
