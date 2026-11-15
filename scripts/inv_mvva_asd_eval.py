"""Section 15N (research/log.md): can Light-ASD supply the speaker on MVVA? CPU.

Part 1 (labels only): ROC AUC of the Light-ASD score (scripts/inv_mvva_asd.py) for talking label 1 against all other
labels over face-frames, pooled and as the mean of per-video AUCs; in frames with exactly one talking face and at
least one silent face (faces with a box), how often the talking face has the highest score (a face without a score
ranks lowest), against chance (mean of 1/faces).

Part 2 (gaze, descriptive): the 15K speaker oracle p * (1 + a * 1[N] + b * 1[T]) on the 15K frames, with T and N from
the detector (score > 0: talking; no score: silent) instead of the labels. Weights on the 15K grid, chosen on one half
of the videos and scored on the other (the 15K halves). The unchanged map, the face oracle and the label-based speaker
oracle come from the 15K rows (runs/inv/k_mvva_faces_rows.json); the same frames are rebuilt here, and the unchanged
map's CC must reproduce the 15K value (|difference| < 1e-9) and the label masks must equal those of 15K.
CC and NSS of the 49 re-weighted maps are computed from sufficient statistics (q is linear in p, p*1[N], p*1[T]),
checked against the direct metric functions on the first frame of every video (|difference| < 1e-9).

    python scripts/inv_mvva_asd_eval.py --asd runs/inv/mvva_asd --small runs/inv/mvva_small_s0 \
        --rows15k runs/inv/k_mvva_faces_rows.json --out runs/review/inv/n_mvva_asd.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.inv_mvva_faces import GRID, H, SRC, W, boxes_for, masks, native_sizes  # noqa: E402
from surge.data import load_gray, mvva_videos  # noqa: E402
from surge.metrics import cc, nss  # noqa: E402

TOL = 1e-9


def auc(s: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC with average ranks for ties (Mann-Whitney)."""
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores_by_frame(npz) -> dict:
    """{frame: {face: score}}."""
    out = {}
    for f, j, s in zip(npz["frame"], npz["face"], npz["score"]):
        out.setdefault(int(f), {})[int(j)] = float(s)
    return out


def face_masks(t, f, size, talking):
    """Talking / other masks at 640x360 with the box geometry of scripts/inv_mvva_faces.masks; talking(j) decides
    each face. Returns (mt, mn) with mn excluding mt, or None if no face has a usable box."""
    if f >= t.shape[0] or t.ndim != 3:
        return None
    scale = np.array([W / size[0], H / size[1], W / size[0], H / size[1]])
    mt, mn = np.zeros((H, W), bool), np.zeros((H, W), bool)
    any_face = False
    for j in range(t.shape[1]):
        b = t[f, j] * scale
        if not np.all(np.isfinite(b)):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        any_face = True
        if talking(j):
            mt[y1:y2, x1:x2] = True
        else:
            mn[y1:y2, x1:x2] = True
    mn &= ~mt
    return (mt, mn) if any_face else None


def stats(p, gt, fx, mt, mn):
    """Sufficient statistics of q = p (1 + a 1[N] + b 1[T]) for CC against gt and NSS at fx."""
    U = np.stack([p.ravel(), (p * mn).ravel(), (p * mt).ravel()])
    g = gt.ravel()
    return {"mu": U.mean(1), "gram": U @ U.T / U.shape[1], "cg": U @ g / U.shape[1], "gm": g.mean(), "gsd": g.std(),
            "uf": U[:, fx.ravel()].mean(1)}


def cc_nss_closed(st, a, b):
    w = np.array([1.0, a, b])
    mq = st["mu"] @ w
    sq = np.sqrt(max(w @ st["gram"] @ w - mq * mq, 0.0))
    sq = sq if sq > 0 else 1.0
    return float((st["cg"] @ w - mq * st["gm"]) / (sq * (st["gsd"] if st["gsd"] > 0 else 1.0))), \
        float((st["uf"] @ w - mq) / sq)


def one_video(args):
    vid, small, sigma, asd_dir, frames15k, base15k = args
    v = {x.name: x for x in mvva_videos()}[vid]
    z = np.load(Path(small) / f"{vid}.npz")
    pred = dict(zip((int(f) for f in z["frames"]), z["maps"]))
    t, k = boxes_for(vid)
    size = native_sizes()[vid]
    sc = scores_by_frame(np.load(Path(asd_dir) / f"{vid}.npz"))
    rows, worst_base, worst_closed, mask_mismatch = [], 0.0, 0.0, 0
    for i, f in enumerate(frames15k):
        lab = masks(t, k, f, size)
        mine = face_masks(t, f, size, lambda j: f < k.shape[1] and j < k.shape[0] and k[j, f] == 1)
        if lab is None or mine is None or not (np.array_equal(lab[0], mine[0]) and np.array_equal(lab[1], mine[1])):
            mask_mismatch += 1
        det = face_masks(t, f, size, lambda j: sc.get(f, {}).get(j, -np.inf) > 0)
        mt, mn = det
        gt = load_gray(v.maps[f]).astype(np.float64)
        fix = load_gray(v.fixations[f])
        p = cv2.resize(pred[f].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        if sigma > 0:
            p = cv2.GaussianBlur(p, (0, 0), sigma)
        p = (p / p.sum()).astype(np.float64)
        fx = fix > 0
        base = cc(p, gt)
        worst_base = max(worst_base, abs(base - base15k[i]))
        st = stats(p, gt, fx, mt, mn)
        r = {"f": f, "n_det_talking_faces": int(sum(1 for j, s in sc.get(f, {}).items() if s > 0)),
             "det_T_area": float(mt.mean()), "det_N_area": float(mn.mean())}
        for a, b in itertools.product(GRID, GRID):
            r[f"asd_{a}_{b}"] = cc_nss_closed(st, a, b)
        if i == 0:  # closed form against the metric functions
            for a, b in ((0.0, 0.0), (1.0, 2.0), (16.0, 0.5)):
                q = p * (1 + a * mn + b * mt)
                c1, n1 = cc(q, gt), nss(q, fix)
                c2, n2 = r[f"asd_{a}_{b}"]
                worst_closed = max(worst_closed, abs(c1 - c2), abs(n1 - n2))
        rows.append(r)
    return vid, rows, worst_base, worst_closed, mask_mismatch


def part1(asd_dir: Path) -> dict:
    pooled_s, pooled_y, per_video, top1, chance = [], [], {}, [], []
    for p in sorted(asd_dir.glob("*.npz"), key=lambda p: int(p.stem)):
        vid = p.stem
        d = np.load(p)
        t, k = boxes_for(vid)
        if k.ndim != 2 or t.ndim != 3:
            continue
        f, j, s = d["frame"], d["face"], d["score"]
        m = (f < k.shape[1]) & (j < k.shape[0])
        y = k[j[m], f[m]] == 1
        pooled_s.append(s[m])
        pooled_y.append(y)
        per_video[vid] = {"auc": auc(s[m], y), "face_frames": int(m.sum()), "talking_share": float(y.mean()) if m.any()
                          else float("nan"), "score_above_0_share": float((s[m] > 0).mean()) if m.any() else float("nan")}
        sc = scores_by_frame(d)
        for fr in range(min(t.shape[0], k.shape[1])):
            faces = [jj for jj in range(min(t.shape[1], k.shape[0])) if np.all(np.isfinite(t[fr, jj]))
                     and abs(t[fr, jj, 2] - t[fr, jj, 0]) >= 2 and abs(t[fr, jj, 3] - t[fr, jj, 1]) >= 2]
            talk = [jj for jj in faces if k[jj, fr] == 1]
            if len(talk) != 1 or len(faces) < 2:
                continue
            ss = {jj: sc.get(fr, {}).get(jj, -np.inf) for jj in faces}
            best = max(ss.values())
            winners = [jj for jj, v in ss.items() if v == best]
            top1.append(float(winners == talk))
            chance.append(1.0 / len(faces))
    s, y = np.concatenate(pooled_s), np.concatenate(pooled_y)
    vals = np.array([v["auc"] for v in per_video.values()])
    return {"videos": len(per_video), "face_frames": int(len(s)), "talking_share": float(y.mean()),
            "score_above_0_share": float((s > 0).mean()), "auc_pooled": auc(s, y),
            "auc_mean_per_video": float(np.nanmean(vals)), "auc_videos": int(np.isfinite(vals).sum()),
            "top1_frames": len(top1), "top1": float(np.mean(top1)), "top1_chance": float(np.mean(chance)),
            "per_video": per_video}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asd", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--rows15k", required=True)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rows", default=None, help="write the per-frame detector-oracle rows here")
    ap.add_argument("--limit", type=int, default=None, help="first N videos of part 2 only (smoke test)")
    ap.add_argument("--only", nargs="*", default=None, help="these videos of part 2 only (smoke test)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"part1": part1(Path(a.asd))}
    p1 = out["part1"]
    print(f"part 1: {p1['videos']} videos, {p1['face_frames']} face-frames, talking {p1['talking_share']:.3f}, "
          f"score>0 {p1['score_above_0_share']:.3f}; AUC pooled {p1['auc_pooled']:.3f}, per-video mean "
          f"{p1['auc_mean_per_video']:.3f}; top-1 {p1['top1']:.3f} vs chance {p1['top1_chance']:.3f} "
          f"({p1['top1_frames']} frames)", flush=True)
    r15 = {v: r for v, r in json.loads(Path(a.rows15k).read_text()).items() if r}
    vids = sorted(r15, key=lambda s: int(s))
    halves = [vids[0::2], vids[1::2]]  # as in 15K
    todo = [v for v in vids if a.only is None or v in a.only][:a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_video, [(v, a.small, a.sigma, a.asd, [r["f"] for r in r15[v]], [r["cc"] for r in r15[v]])
                                      for v in todo]))
    det = {v: rows for v, rows, _, _, _ in res}
    checks = {"max_base_cc_diff": max(x[2] for x in res), "max_closed_form_diff": max(x[3] for x in res),
              "label_mask_mismatches": int(sum(x[4] for x in res))}
    out["checks"] = checks
    print("checks:", checks, flush=True)
    if a.rows:
        Path(a.rows).write_text(json.dumps(det), newline="\n")
    if checks["max_base_cc_diff"] > TOL or checks["max_closed_form_diff"] > TOL or checks["label_mask_mismatches"]:
        out["status"] = "check failed; part 2 not reported"
        Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
        print(out["status"])
        return
    halves = [[v for v in h if v in det] for h in halves]

    def vmean(vs, src, key, idx=0):
        return float(np.mean([np.nanmean([r[key][idx] if isinstance(r[key], (list, tuple)) else r[key] for r in src[v]])
                              for v in vs]))

    folds, per = [], []
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        fw = max(GRID, key=lambda x: vmean(fit, r15, f"face_{x}"))  # 15K's face oracle choice
        sw = max(itertools.product(GRID, GRID), key=lambda ab: vmean(fit, r15, f"spk_{ab[0]}_{ab[1]}"))
        dw = max(itertools.product(GRID, GRID), key=lambda ab: vmean(fit, det, f"asd_{ab[0]}_{ab[1]}"))
        folds.append({"face_weight": fw, "label_speaker_weights": sw, "detector_weights": dw,
                      "test_videos": len(test)})
        for v in test:
            per.append([np.nanmean([r["cc"] for r in r15[v]]),
                        np.nanmean([r[f"face_{fw}"][0] for r in r15[v]]),
                        np.nanmean([r[f"spk_{sw[0]}_{sw[1]}"][0] for r in r15[v]]),
                        np.nanmean([r[f"asd_{dw[0]}_{dw[1]}"][0] for r in det[v]])])
    per = np.array(per)  # base, face, label speaker, detector speaker
    m = per.mean(0)
    rng = np.random.default_rng(0)
    bs = np.array([per[i].mean(0) for i in (rng.integers(0, len(per), len(per)) for _ in range(10000))])

    def ci(x):
        return [float(v) for v in np.percentile(x, [2.5, 97.5])]
    share = (bs[:, 3] - bs[:, 1]) / (bs[:, 2] - bs[:, 1])
    out["part2"] = {"videos": len(per), "frames": int(sum(len(det[v]) for v in det)), "folds": folds,
                    "base": float(m[0]), "face": float(m[1]), "label_speaker": float(m[2]), "detector_speaker": float(m[3]),
                    "detector_minus_base": {"mean": float(m[3] - m[0]), "ci95": ci(bs[:, 3] - bs[:, 0])},
                    "detector_minus_face": {"mean": float(m[3] - m[1]), "ci95": ci(bs[:, 3] - bs[:, 1])},
                    "label_minus_face": {"mean": float(m[2] - m[1]), "ci95": ci(bs[:, 2] - bs[:, 1])},
                    "share_realised": {"mean": float((m[3] - m[1]) / (m[2] - m[1])), "ci95": ci(share)},
                    "det_talking_faces_per_frame": float(np.mean([r["n_det_talking_faces"] for v in det
                                                                  for r in det[v]]))}
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
    print(json.dumps({k: v for k, v in out["part2"].items() if k != "folds"}, indent=1))
    for f in folds:
        print(f)


if __name__ == "__main__":
    main()
