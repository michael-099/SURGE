"""Section 15M (research/log.md): how much of SURGE-D's zero-shot transfer gap is a low-dimensional dataset bias?

For each dataset of 15I, the zero-shot maps of runs/vitg_s0, saved at decoder resolution during the 15I runs
(scripts/evaluate.py --save-small; scripts/probe_eval.py --save-small for MVVA), are transformed with three
dataset-level parameters:
    blur      sigma = m * sigma0 (sigma0: the pre-registered blur of 15I), m in M_GRID
    sharpness the blurred, normalised map raised to the power gamma, gamma in G_GRID
    centre    multiplied by exp(-kappa * r^2), r the distance to the frame centre in units of width and height,
              kappa in K_GRID (negative: away from the centre)
(m, gamma, kappa) = (1, 1, 0) is the identity: the 15I scoring. CC and NSS are invariant to adding a constant or a
uniform floor, so no floor parameter is fitted.

Two-fold cross-fitting over videos (fixed split, seed 0): the parameters are chosen on one half by mean per-video CC
(at most FIT_FRAMES_PER_VIDEO frames per video and about FIT_FRAMES_TOTAL in all, evenly spaced; maps and ground
truth at FIT_WIDTH pixels wide) and scored on the other half at native resolution with the evaluator's metric
functions (CC, NSS, SIM, KLD).

Check first: the identity transform re-scored at native resolution must reproduce the 15I per-video CC (largest
difference 1e-3, mean difference 2e-4); otherwise the dataset is reported as failed and not analysed.

Output per dataset: zero-shot, adapted and (where it exists) fine-tuned CC on the same videos; adapted - zero-shot with
a 95 % video bootstrap interval; the share of the fine-tuning gain recovered, (adapted - zs) / (ft - zs), reported
only where ft - zs >= 0.02, with a bootstrap interval. Descriptive (rule in section 15M). CPU only.

    python scripts/inv_bias.py [--datasets ucf,diem,...] [--workers 16]
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

from surge.data import load_gray  # noqa: E402
from surge.metrics import all_metrics  # noqa: E402

M_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
G_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
K_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
FIT_WIDTH = 160
FIT_FRAMES_PER_VIDEO = 40
FIT_FRAMES_TOTAL = 4000  # per fitting half; fewer frames per video for datasets with many videos
METRICS = ("cc", "nss", "sim", "kld")
R = ROOT / "runs/review/inv"
SMALL = ROOT / "runs/inv/zs_small"
STAVIS_FOLDS = {"diem": ["diem_f1"], **{d: [f"{d}_f{k}" for k in (1, 2, 3)]
                                        for d in ("coutrot1", "coutrot2", "etmd", "avad", "summe")}}


def sources(name: str) -> list[dict]:
    """The 15I runs that make up one dataset: zero-shot score file, saved maps, dataset key and split."""
    if name in STAVIS_FOLDS:
        return [{"zs": R / f"zs_stavis_{f}.json", "small": SMALL / f"stavis_{f}", "dataset": f, "split": "val"}
                for f in STAVIS_FOLDS[name]]
    if name == "mvva":
        return [{"zs": R / "zs_mvva.json", "small": ROOT / "runs/inv/mvva_small_s0", "dataset": "mvva", "split": "val"}]
    key = {"ucf": "ucf", "hollywood": "hollywood_vig", "avimos": "avimos"}[name]
    return [{"zs": R / f"zs_{name}.json", "small": SMALL / name, "dataset": key, "split": "val"}]


FT = {  # fine-tuned ViT-g seed 0 (thesis), per video; None where no fine-tuned model exists
    "ucf": ROOT / "runs/g_other/ucf_s0.json", "hollywood": ROOT / "runs/g_other/hollywood_s0.json",
    **{d: ROOT / f"runs/g_other/pooled_g_single_{d}.json" for d in STAVIS_FOLDS},
    "avimos": None, "mvva": None,
}


def centre_r2(h: int, w: int) -> np.ndarray:
    y = (np.arange(h) + 0.5) / h - 0.5
    x = (np.arange(w) + 0.5) / w - 0.5
    return (y[:, None] ** 2 + x[None, :] ** 2).astype(np.float64)


def transform(small: np.ndarray, hw, sigma: float, gamma: float, kappa: float) -> np.ndarray:
    """small (h, w) map -> native (H, W) map: resize, blur, normalise (as surge.infer.to_native), then sharpness and
    centre. With gamma 1 and kappa 0 this equals to_native(small, hw, sigma)."""
    out = cv2.resize(small.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out = out / max(out.sum(), 1e-12)
    if gamma == 1.0 and kappa == 0.0:
        return out
    p = np.power(np.maximum(out.astype(np.float64), 0.0), gamma)
    if kappa != 0.0:
        p = p * np.exp(-kappa * centre_r2(*hw))
    return (p / max(p.sum(), 1e-300)).astype(np.float32)


def _score(args):
    small, map_path, fix_path, skip_empty, params = args
    gt = load_gray(map_path).astype(np.float64)
    fix = load_gray(fix_path)
    if skip_empty and not (fix > 0).any():
        return None
    out = []
    for sigma, gamma, kappa in params:
        out.append(all_metrics(transform(small, gt.shape, sigma, gamma, kappa), gt, fix, None, METRICS))
    return out


def _fit_item(args):
    """One fitting frame at FIT_WIDTH: base map (resized, unblurred) and ground-truth density, or None."""
    small, map_path, fix_path, skip_empty = args
    gt = load_gray(map_path).astype(np.float32)
    if skip_empty and not (load_gray(fix_path) > 0).any():
        return None
    H, W = gt.shape
    w = FIT_WIDTH
    h = max(8, int(round(H * w / W)))
    g = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return b, g, W


def _zs(x: np.ndarray) -> np.ndarray:
    """z-score each map of a stack (N, h, w) over its pixels."""
    x = x.reshape(len(x), -1).astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    s = x.std(1, keepdims=True)
    return x / np.where(s > 0, s, 1.0)


def fit(items: list[tuple], vid_of: np.ndarray, sigma0: float) -> tuple[tuple, dict]:
    """Grid search on the fitting frames (grouped by shape). Returns (m, gamma, kappa) maximising mean per-video CC."""
    groups: dict[tuple, list[int]] = {}
    for i, (b, g, W) in enumerate(items):
        groups.setdefault((b.shape, W), []).append(i)
    nv = int(vid_of.max()) + 1
    scores = {}
    per_frame_cc = {}
    for (shape, W), idx in groups.items():
        h, w = shape
        B = np.stack([items[i][0] for i in idx])
        Gz = _zs(np.stack([items[i][1] for i in idx]))
        r2 = centre_r2(h, w)
        for m in M_GRID:
            s = m * sigma0 * w / W  # the native blur, scaled to the fitting width
            Bb = np.stack([cv2.GaussianBlur(x, (0, 0), s) if s > 0 else x for x in B]).astype(np.float64)
            Bb = Bb / np.maximum(Bb.sum((1, 2), keepdims=True), 1e-12)
            for gm in G_GRID:
                P = np.power(np.maximum(Bb, 0.0), gm)
                for k in K_GRID:
                    Q = P * np.exp(-k * r2)[None] if k != 0 else P
                    c = (_zs(Q) * Gz).mean(1)
                    per_frame_cc.setdefault((m, gm, k), {})[tuple(idx)] = c
    for key, parts in per_frame_cc.items():
        cc_all = np.empty(len(items))
        for idx, c in parts.items():
            cc_all[list(idx)] = c
        sums = np.bincount(vid_of, weights=cc_all, minlength=nv)
        cnts = np.bincount(vid_of, minlength=nv)
        ok = cnts > 0
        scores[key] = float((sums[ok] / cnts[ok]).mean())
    best = max(scores, key=scores.get)
    return best, {"fit_cc_best": scores[best], "fit_cc_identity": scores[(1.0, 1.0, 0.0)]}


def boot_mean(d: np.ndarray, n=10000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    b = d[idx].mean(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def boot_ratio(num: np.ndarray, den: np.ndarray, n=10000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return [float(x) for x in np.percentile(r, [2.5, 97.5])]


def analyse(name: str, workers: int) -> dict:
    from scripts.evaluate import get_videos
    vids, zs_cc, sigma0 = [], {}, None
    for src in sources(name):
        if not src["zs"].exists():
            return {"dataset": name, "status": f"missing {src['zs'].name}"}
        z = json.loads(src["zs"].read_text())
        s0 = float(z["args"]["sigma"])
        if sigma0 is None:
            sigma0 = s0
        elif s0 != sigma0:
            raise ValueError(f"{name}: sigma differs between folds")
        fs = int(z["args"]["frame_step"])
        for v in get_videos(src["dataset"], src["split"], None):
            f = src["small"] / f"{v.name}.npz"
            if v.name not in z["per_video"]:
                continue
            if not f.exists():
                return {"dataset": name, "status": f"missing maps {f}"}
            zs_cc[v.name] = z["per_video"][v.name]["cc"]
            vids.append((v, f, fs))
    names = sorted(zs_cc)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(names))
    half = {names[i]: int(j >= len(names) // 2) for j, i in enumerate(perm)}
    by_name = {v.name: (v, f, fs) for v, f, fs in vids}

    def load(nm):
        v, f, fs = by_name[nm]
        d = np.load(f)
        fr = [int(x) for x in d["frames"]]
        maps = d["maps"]
        keep = [i for i, x in enumerate(fr) if x % fs == 0 and x < len(v)]
        return v, [fr[i] for i in keep], maps[keep]

    result = {"dataset": name, "sigma0": sigma0, "videos": len(names), "folds": {}}
    adapted, identity = {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for fold in (0, 1):
            fit_names = [n for n in names if half[n] == fold]
            test_names = [n for n in names if half[n] != fold]
            items, vid_of = [], []
            cap = max(1, min(FIT_FRAMES_PER_VIDEO, FIT_FRAMES_TOTAL // max(len(fit_names), 1)))
            for vi, nm in enumerate(fit_names):
                v, fr, maps = load(nm)
                sel = np.unique(np.linspace(0, len(fr) - 1, min(cap, len(fr))).round().astype(int))
                jobs = [(maps[i], v.maps[fr[i]], v.fixations[fr[i]], v.skip_empty) for i in sel]
                for r in ex.map(_fit_item, jobs, chunksize=8):
                    if r is not None:
                        items.append(r)
                        vid_of.append(vi)
            (m, gm, k), info = fit(items, np.array(vid_of), sigma0)
            params = [(sigma0, 1.0, 0.0), (m * sigma0, gm, k)]
            for nm in test_names:
                v, fr, maps = load(nm)
                jobs = [(maps[i], v.maps[f], v.fixations[f], v.skip_empty, params) for i, f in enumerate(fr)]
                res = [r for r in ex.map(_score, jobs, chunksize=8) if r is not None]
                identity[nm] = {k_: float(np.nanmean([r[0][k_] for r in res])) for k_ in METRICS}
                adapted[nm] = {k_: float(np.nanmean([r[1][k_] for r in res])) for k_ in METRICS}
            result["folds"][fold] = {"fit_videos": len(fit_names), "fit_frames": len(items),
                                     "params": {"m": m, "sigma": m * sigma0, "gamma": gm, "kappa": k}, **info}
            print(f"  {name} fold {fold}: m={m} gamma={gm} kappa={k}  fit CC {info['fit_cc_identity']:.4f} -> "
                  f"{info['fit_cc_best']:.4f}", flush=True)
    diff = np.array([identity[n]["cc"] - zs_cc[n] for n in names])
    check = {"max_abs": float(np.abs(diff).max()), "mean": float(diff.mean())}
    result["check"] = check
    if check["max_abs"] > 1e-3 or abs(check["mean"]) > 2e-4:
        result["status"] = "check failed: identity re-score does not reproduce the zero-shot scores"
        return result
    zs = np.array([zs_cc[n] for n in names])
    ad = np.array([adapted[n]["cc"] for n in names])
    result.update({"status": "ok",
                   "zs": {k_: float(np.mean([identity[n][k_] for n in names])) for k_ in METRICS},
                   "adapted": {k_: float(np.mean([adapted[n][k_] for n in names])) for k_ in METRICS},
                   "adapted_minus_zs_cc": {"mean": float((ad - zs).mean()), "ci95": boot_mean(ad - zs)},
                   "per_video": {n: {"zs": zs_cc[n], "adapted": adapted[n]} for n in names}})
    if FT.get(name) is not None and FT[name].exists():
        ftv = json.loads(FT[name].read_text())["per_video"]
        common = [n for n in names if n in ftv]
        i = [names.index(n) for n in common]
        ft = np.array([ftv[n]["cc"] for n in common])
        gain, rec = ft - zs[i], ad[i] - zs[i]
        result["ft"] = {"videos": len(common), "cc": float(ft.mean()), "ft_minus_zs": float(gain.mean()),
                        "ft_minus_zs_ci95": boot_mean(gain), "ft_minus_adapted": float((ft - ad[i]).mean()),
                        "ft_minus_adapted_ci95": boot_mean(ft - ad[i])}
        if gain.mean() >= 0.02:
            result["ft"]["share_recovered"] = float(rec.mean() / gain.mean())
            result["ft"]["share_recovered_ci95"] = boot_ratio(rec, gain)
    return result


def selfcheck(zs_json: str, small_dir: str, dataset: str, split: str, tol: float = 1e-5) -> None:
    """Smoke test of the saved maps: the identity transform must reproduce the evaluator's per-video CC."""
    from scripts.evaluate import get_videos
    z = json.loads(Path(zs_json).read_text())
    fs, sigma = int(z["args"]["frame_step"]), float(z["args"]["sigma"])
    worst = 0.0
    for v in get_videos(dataset, split, None):
        if v.name not in z["per_video"]:
            continue
        d = np.load(Path(small_dir) / f"{v.name}.npz")
        res = [_score((m, v.maps[f], v.fixations[f], v.skip_empty, [(sigma, 1.0, 0.0)]))
               for f, m in zip(d["frames"], d["maps"]) if f % fs == 0]
        c = float(np.nanmean([r[0]["cc"] for r in res if r is not None]))
        worst = max(worst, abs(c - z["per_video"][v.name]["cc"]))
        print(f"{v.name}: evaluator {z['per_video'][v.name]['cc']:.6f}  saved maps {c:.6f}", flush=True)
    print(f"largest difference {worst:.2e}", flush=True)
    if worst > tol:
        raise SystemExit(f"saved maps do not reproduce the evaluator (tolerance {tol})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ucf,diem,coutrot1,coutrot2,etmd,avad,summe,hollywood,avimos,mvva")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(R / "summary_bias.json"))
    ap.add_argument("--selfcheck", nargs=4, metavar=("ZS_JSON", "SMALL_DIR", "DATASET", "SPLIT"), default=None)
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck(*a.selfcheck)
        return
    rows = {}
    for name in a.datasets.split(","):
        r = analyse(name, a.workers)
        (R / f"m_bias_{name}.json").write_text(json.dumps(r, indent=1), newline="\n")
        rows[name] = {k: v for k, v in r.items() if k != "per_video"}
        if r.get("status") != "ok":
            print(f"{name:10s} {r.get('status')}", flush=True)
            continue
        ft = r.get("ft", {})
        sh = (f"share {ft['share_recovered']:+.2f} [{ft['share_recovered_ci95'][0]:+.2f}, "
              f"{ft['share_recovered_ci95'][1]:+.2f}]" if "share_recovered" in ft else "")
        d = r["adapted_minus_zs_cc"]
        print(f"{name:10s} zs {r['zs']['cc']:.4f} adapted {r['adapted']['cc']:.4f} "
              f"diff {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
              + (f"  ft {ft['cc']:.4f} (ft-zs {ft['ft_minus_zs']:+.4f}) {sh}" if ft else ""), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1), newline="\n")


if __name__ == "__main__":
    main()
