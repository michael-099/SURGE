"""Incremental information of the prediction error beyond simple cues (research/log.md sections 13B and 13C, with the
mask-token addendum of 2026-11-11).

Reads the token maps of scripts/g2_maps.py and the fixations of the 60 holdout videos. Prediction-error maps exist
for two mask tokens: token 1 (the HF port's default, an untrained token; the readout of gate G2; keys without
suffix) and token 0 (the trained token; suffix `_t0`; the primary readout).

Model family, on a grid of 45x80 cells (8x8 px):  p(cell) = softmax(w_c * log prior + sum_k w_k * z_k)
  prior  empirical centre prior of the 540 training videos (runs/g2/center_prior.npy), summed per cell
  z_k    cue k upsampled to 360x640 and blurred (sigma 16 px) as in G2, averaged per cell, standardised over the
         cells of the frame
Score: information gain over the centre prior in bits per fixation (mean over a frame's fixations of
log2 p(cell) - log2 prior(cell)); per video the mean over its scored frames; overall the mean over videos.
Fit: the weights maximise the mean per-video information gain of the fitting half (concave; L-BFGS from w_c = 1,
other weights 0). Cross-fitting: videos sorted by id, assigned alternately to halves A and B; weights fitted on one
half are scored on the other, so every video has an out-of-fold score for every model.

  B = {prior, frame difference, atypicality}      F = B + error map
  U = IG(F) - IG(B): mean over videos with 95 % and 90 % video-bootstrap intervals (10,000 resamples, weights fixed)
  reading: "incremental information" if the 95 % interval lies above 0; "negligible" if the 90 % interval lies
  within +-0.01 bits; otherwise "inconclusive".
  primary: surprise_raw_t0 at stride 8 (13B) and masked_tube_t0 at stride 8 (13C); everything else is descriptive.
For scale: the leave-one-out gains of frame difference and atypicality from F, IG(B), and (descriptive) the gain of
each single cue over the fitted prior.

Also:
  - the check that the stored token-1 maps reproduce the NSS of runs/g2/g2_results.json (tolerance 0.01);
  - standalone NSS and CC of every map with intervals and paired differences, as in section 12F;
  - gate G2 re-scored by its own rule (scripts/g2_surprise.py: mean NSS above both atypicality and frame difference,
    and positive information gain of the 50/50 mixture with the prior), for token 1 and token 0;
  - descriptive: mean value of each stored map, mean per-frame correlation between the token-0 and token-1 maps.

    python scripts/g2_incremental.py --out runs/review/g2_incremental.json
`--smoke` uses only the videos whose maps exist and does not enforce the reproduction check (tests only).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from surge.data import dhf1k_videos, load_gray, read_split_file  # noqa: E402
from surge.metrics import cc, nss  # noqa: E402

H, W, CELL = 360, 640, 8
GH, GW = H // CELL, W // CELL
LN2 = float(np.log(2.0))
RATES = {8: "about 4 fps", 2: "about 15 fps"}
BASE = ["frame_diff", "atypicality"]
READOUTS = ["surprise_raw", "surprise_ctx", "masked_tube"]  # token 1: no suffix; token 0: suffix _t0
PRIMARY = {(8, "surprise_raw_t0"): "13B", (8, "masked_tube_t0"): "13C"}
VERSION = 2  # of the feature cache


def up(m16: np.ndarray, sigma: float) -> np.ndarray:  # as scripts/g2_surprise.py
    m = cv2.resize(m16.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return m


def cells(m: np.ndarray, how="mean") -> np.ndarray:
    r = m.reshape(GH, CELL, GW, CELL)
    return (r.mean((1, 3)) if how == "mean" else r.sum((1, 3))).ravel()


def info_gain_mix(m: np.ndarray, prior: np.ndarray, sel: np.ndarray) -> float:  # as scripts/g2_surprise.py
    p = m - m.min()
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / p.size)
    mix = 0.5 * p + 0.5 * prior
    if not sel.any():
        return float("nan")
    return float(np.mean(np.log2(mix[sel] + 1e-12) - np.log2(prior[sel] + 1e-12)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def video_features(job):
    """Per scored frame of one video: standardised cell maps of every cue, fixation counts per cell, NSS, CC and the
    mixture information gain of G2, the mean of each stored map and the token-0 / token-1 map correlations."""
    name, npz, map_paths, fix_paths, cues, pairs, sigma, prior_path = job
    z = np.load(npz)
    prior = np.load(prior_path)
    Z, C, S, M, R, frames = [], [], [], [], [], []
    for i, f in enumerate(z["frames"]):
        gt = load_gray(map_paths[f])
        if gt.shape != (H, W):  # as G2
            continue
        fix = load_gray(fix_paths[f]) > 0
        zs, sc = [], []
        for c in cues:
            m = up(z[c][i], sigma)
            sc.append((nss(m, fix), cc(m, gt.astype(np.float64)), info_gain_mix(m, prior, fix)))
            v = cells(m).astype(np.float64)
            sd = v.std()
            zs.append((v - v.mean()) / sd if sd > 0 else np.zeros_like(v))
        sc.append((nss(prior, fix), cc(prior, gt.astype(np.float64)), 0.0))  # centre prior, last column
        Z.append(np.stack(zs, 1).astype(np.float32))                          # (cells, cues)
        C.append(cells(fix.astype(np.float64), "sum").astype(np.float32))
        S.append(sc)
        M.append([float(z[c][i].mean()) for c in cues])
        R.append([corr(z[a][i].astype(np.float64), z[b][i].astype(np.float64)) for a, b in pairs])
        frames.append(int(f))
    return (name, np.asarray(frames), np.asarray(Z), np.asarray(C), np.asarray(S, np.float64),
            np.asarray(M, np.float64), np.asarray(R, np.float64).reshape(len(frames), len(pairs)))


def objective(w, X, Q, a):
    """Negative weighted mean log-likelihood of the fixations (nats) and its gradient."""
    lg = X @ w
    m = lg.max(1, keepdims=True)
    ex = np.exp(lg - m)
    s = ex.sum(1, keepdims=True)
    ll = (Q * lg).sum(1) - (m + np.log(s))[:, 0]
    g = np.einsum("fc,fck->k", (Q - ex / s) * a[:, None], X)
    return -(a @ ll), -g


def frame_ll(w, X, Q):
    lg = X @ w
    m = lg.max(1, keepdims=True)
    return (Q * lg).sum(1) - (m + np.log(np.exp(lg - m).sum(1, keepdims=True)))[:, 0]


def boot(x: np.ndarray, idx: np.ndarray) -> dict:
    b = x[idx].mean(1)
    return {"mean": float(x.mean()), "ci95": [float(v) for v in np.percentile(b, [2.5, 97.5])],
            "ci90": [float(v) for v in np.percentile(b, [5, 95])]}


def reading(r: dict, margin: float) -> dict:
    inside = bool(-margin < r["ci90"][0] and r["ci90"][1] < margin)
    r["within_margin"] = inside
    r["reading"] = "incremental information" if r["ci95"][0] > 0 else "negligible" if inside else "inconclusive"
    return r


def fmt(r: dict) -> str:
    return (f"{r['mean']:+.4f}  95% [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
            f"90% [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="runs/g2/maps")
    ap.add_argument("--split", default="configs/splits/dhf1k_holdout60.txt")
    ap.add_argument("--strides", default="8,2")
    ap.add_argument("--sigma", type=float, default=16.0)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--tol", type=float, default=0.01, help="tolerance of the G2 reproduction check (NSS)")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tests only: videos with maps, no reproduction rule")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    maps = Path(a.maps) if Path(a.maps).is_absolute() else ROOT / a.maps
    all_vids = sorted(dhf1k_videos("D:/DHF1K", read_split_file(ROOT / a.split)), key=lambda v: v.name)
    prior_path = str(ROOT / "runs/g2/center_prior.npy")
    pc = cells(np.load(prior_path), "sum")
    assert pc.min() > 0
    logp = np.log(pc / pc.sum())
    g2 = json.load(open(ROOT / "runs/g2/g2_results.json"))
    checks = json.load(open(maps / "checks.json")) if (maps / "checks.json").exists() else {}
    res = {"cells": [GH, GW], "sigma": a.sigma, "margin_bits": a.margin, "smoke": bool(a.smoke), "checks": checks}
    ok = True
    gate = {"token1": {}, "token0": {}}
    for s in [int(x) for x in a.strides.split(",")]:
        d = maps / f"stride_{s}"
        vids = [v for v in all_vids if (d / f"{v.name}.npz").exists()]
        if not a.smoke:
            assert len(vids) == len(all_vids), f"stride {s}: maps of {len(vids)} of {len(all_vids)} videos"
        keys = [k for k in np.load(d / f"{vids[0].name}.npz").files if k != "frames"]
        errors = [k for k in keys if k not in BASE]
        cues = BASE + errors
        pairs = [(r, r + "_t0") for r in READOUTS if r in cues and r + "_t0" in cues]
        cache = maps / f"features_stride_{s}.npz"
        z = np.load(cache) if cache.exists() and not a.recompute else None
        if z is not None and "version" in z.files and int(z["version"]) == VERSION and list(z["cues"]) == cues \
                and list(z["videos"]) == [v.name for v in vids]:
            Z, C, S, M, R, vid = z["Z"], z["C"], z["S"], z["M"], z["R"], z["vid"]
        else:
            jobs = [(v.name, str(d / f"{v.name}.npz"), v.maps, v.fixations, cues, pairs, a.sigma, prior_path)
                    for v in vids]
            with ProcessPoolExecutor(a.workers) as ex:
                out = list(ex.map(video_features, jobs))
            assert [o[0] for o in out] == [v.name for v in vids]
            Z, C, S, M, R = (np.concatenate([o[j] for o in out]) for j in (2, 3, 4, 5, 6))
            vid = np.concatenate([np.full(len(o[1]), i) for i, o in enumerate(out)])
            np.savez(cache, Z=Z, C=C, S=S, M=M, R=R, vid=vid, cues=np.array(cues), version=VERSION,
                     videos=np.array([v.name for v in vids]))
        nv = len(vids)
        r = {"rate": RATES.get(s, ""), "videos": nv, "frames": int(len(vid)), "cues": cues}
        print(f"== stride {s} ({r['rate']}): {nv} videos, {len(vid)} scored frames, cues {cues}")

        # ---- standalone NSS / CC / mixture IG per video (nanmean over frames, as G2) and the reproduction check
        names = cues + ["center_prior"]
        pv = np.stack([np.nanmean(S[vid == i], 0) for i in range(nv)])  # (videos, maps, 3)
        mean = lambda k, q: float(pv[:, names.index(k), q].mean())  # noqa: E731
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, nv, (a.n, nv))
        rep = {}
        for k in ("surprise_raw", "atypicality", "frame_diff"):
            new, old = mean(k, 0), g2[f"stride_{s}"]["mean"][k]["nss"]
            rep[k] = {"nss_stored_maps": new, "nss_g2": old, "passed": bool(abs(new - old) <= a.tol)}
            ok &= rep[k]["passed"] or a.smoke
        for k in ("surprise_ctx", "center_prior"):  # reported, not part of the rule
            rep[k] = {"nss_stored_maps": mean(k, 0), "nss_g2": g2[f"stride_{s}"]["mean"][k]["nss"]}
        for k in ("surprise_raw", "surprise_ctx", "atypicality", "frame_diff"):
            rep[k]["ig_mix_stored_maps"], rep[k]["ig_mix_g2"] = mean(k, 2), g2[f"stride_{s}"]["mean"][k]["ig"]
        r["reproduction"] = rep
        print("  reproduction of G2 (NSS, stored token-1 maps vs g2_results.json): " + "; ".join(
            f"{k} {v['nss_stored_maps']:+.4f} vs {v['nss_g2']:+.4f}" + ("" if v.get("passed", True) else " FAILED")
            for k, v in rep.items()))
        st = {"maps": {}, "differences": {}}
        for j, m in enumerate(names):
            st["maps"][m] = {"nss": boot(pv[:, j, 0], idx), "cc": boot(pv[:, j, 1], idx), "ig_mix": mean(m, 2)}
        for m in errors:
            for ref in BASE:
                for q, key in enumerate(("nss", "cc")):
                    st["differences"][f"{m}-{ref}/{key}"] = boot(
                        pv[:, names.index(m), q] - pv[:, names.index(ref), q], idx)
        r["standalone"] = st
        for m in names:
            v = st["maps"][m]
            print(f"  {m:18s} NSS {v['nss']['mean']:+.3f} [{v['nss']['ci95'][0]:+.3f}, {v['nss']['ci95'][1]:+.3f}]  "
                  f"CC {v['cc']['mean']:+.3f} [{v['cc']['ci95'][0]:+.3f}, {v['cc']['ci95'][1]:+.3f}]  "
                  f"IG(mix) {v['ig_mix']:+.3f}")
        for k, v in st["differences"].items():
            print(f"  {k:34s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

        # ---- gate G2 by its own rule, per token (log section 3; scripts/g2_surprise.py)
        for tok, sfx in (("token1", ""), ("token0", "_t0")):
            for var in ("surprise_raw", "surprise_ctx"):
                k = var + sfx
                if k not in cues:
                    continue
                gate[tok][f"stride_{s}/{var}"] = bool(mean(k, 0) > mean("atypicality", 0)
                                                      and mean(k, 0) > mean("frame_diff", 0) and mean(k, 2) > 0)
        # ---- descriptive: mean of each stored map; correlation between the token-0 and token-1 maps
        r["mean_map_value"] = {c: float(np.mean([M[vid == i, j].mean() for i in range(nv)])) for j, c in enumerate(cues)}
        r["token_map_correlation"] = {f"{p[0]}~{p[1]}": float(np.mean([np.nanmean(R[vid == i, j]) for i in range(nv)]))
                                      for j, p in enumerate(pairs)}
        print("  mean map value: " + "  ".join(f"{c} {v:.4f}" for c, v in r["mean_map_value"].items()))
        print("  token-0 ~ token-1 map correlation: " + "  ".join(
            f"{c} {v:.3f}" for c, v in r["token_map_correlation"].items()))
        if not ok:
            res[f"stride_{s}"] = r
            continue

        # ---- information gain models
        keep = C.sum(1) > 0                               # frames with at least one fixation
        Q = (C[keep] / C[keep].sum(1, keepdims=True)).astype(np.float64)
        fv = vid[keep]
        X_all = np.concatenate([np.broadcast_to(logp[None, :, None], (int(keep.sum()), logp.size, 1)),
                                Z[keep].astype(np.float64)], 2)  # column 0 = log prior, then the cues
        lp_q = Q @ logp
        per_video = np.bincount(fv, minlength=nv)
        assert per_video.min() > 0
        half = np.arange(nv) % 2                          # videos sorted by id, alternately A (0) and B (1)
        col = {c: i + 1 for i, c in enumerate(cues)}

        def run(model_cues):
            """Out-of-fold per-video information gain (bits per fixation) and the weights of both folds."""
            cols = [0] + [col[c] for c in model_cues]
            ig = np.zeros(nv)
            ws = []
            for h in (0, 1):
                fit = half[fv] == h
                Xf = np.ascontiguousarray(X_all[fit][:, :, cols])
                af = 1.0 / (per_video[fv[fit]] * (half == h).sum())
                w0 = np.zeros(len(cols))
                w0[0] = 1.0
                o = minimize(objective, w0, args=(Xf, Q[fit], af), jac=True, method="L-BFGS-B",
                             options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9})
                ws.append([float(x) for x in o.x])
                te = ~fit
                g = (frame_ll(o.x, np.ascontiguousarray(X_all[te][:, :, cols]), Q[te]) - lp_q[te]) / LN2
                sums = np.bincount(fv[te], weights=g, minlength=nv)
                ig[half != h] = (sums / per_video)[half != h]
            return ig, ws

        models = {"P": [], "B": list(BASE)}
        for c in cues:
            models[f"P+{c}"] = [c]
        for c in errors:
            models[f"F[{c}]"] = BASE + [c]
            models[f"F[{c}]-frame_diff"] = ["atypicality", c]
            models[f"F[{c}]-atypicality"] = ["frame_diff", c]
        ig, wts = {}, {}
        for name, mc in models.items():
            ig[name], wts[name] = run(mc)
        r["models"] = {name: {"cues": ["log_prior"] + models[name], "ig": boot(ig[name], idx),
                              "weights_fit_on_A": wts[name][0], "weights_fit_on_B": wts[name][1]} for name in models}
        print("  information gain over the centre prior, bits per fixation, out of fold:")
        for name in models:
            print(f"    {name:34s} {fmt(r['models'][name]['ig'])}   w(A) "
                  + " ".join(f"{x:+.3f}" for x in wts[name][0]) + "   w(B) " + " ".join(f"{x:+.3f}" for x in wts[name][1]))
        r["single_cue_gain_over_prior"] = {c: boot(ig[f"P+{c}"] - ig["P"], idx) for c in cues}
        for c, v in r["single_cue_gain_over_prior"].items():
            print(f"    gain of {c:18s} over the fitted prior alone   {fmt(v)}")
        r["incremental"] = {}
        for c in errors:
            F_ = f"F[{c}]"
            u = reading(boot(ig[F_] - ig["B"], idx), a.margin)
            u["primary"] = PRIMARY.get((s, c), "")
            u["error_weight"] = [wts[F_][0][-1], wts[F_][1][-1]]
            u["loo_frame_diff"] = boot(ig[F_] - ig[f"{F_}-frame_diff"], idx)
            u["loo_atypicality"] = boot(ig[F_] - ig[f"{F_}-atypicality"], idx)
            r["incremental"][c] = u
            print(f"  U({c}) = IG(F) - IG(B): {fmt(u)}  -> {u['reading']}"
                  f"{' (inside the margin)' if u['within_margin'] else ''}; weight {u['error_weight'][0]:+.3f} / "
                  f"{u['error_weight'][1]:+.3f}" + (f"   [PRIMARY {u['primary']}]" if u["primary"] else ""))
            print(f"      leave-one-out gain of frame_diff  {fmt(u['loo_frame_diff'])}")
            print(f"      leave-one-out gain of atypicality {fmt(u['loo_atypicality'])}")
        res[f"stride_{s}"] = r
    res["reproduction_passed"] = bool(ok)
    res["gate"] = {tok: {"variants": v, "go": bool(any(v.values()))} for tok, v in gate.items() if v}
    for tok, v in res["gate"].items():
        print(f"GATE G2 with {tok}: " + ("GO" if v["go"] else "NO-GO") + "  " + json.dumps(v["variants"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")
    if not ok:
        print("REPRODUCTION CHECK FAILED: the information-gain analysis was not run")
        sys.exit(4)


if __name__ == "__main__":
    main()
