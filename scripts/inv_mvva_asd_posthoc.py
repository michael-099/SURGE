"""Section 15N, post hoc (research/log.md): is Light-ASD's decision threshold what limits part 2? CPU, descriptive.

Part 2 (scripts/inv_mvva_asd_eval.py) counts a face as talking when its Light-ASD score is above 0. This script
repeats part 2 on the same frames (those of 15K), halves, weight grid and bootstrap with other splits of the faces:
  thr_t      score > t, for t in THRESHOLDS; t = 0 is part 2 and must reproduce it (|difference| < 1e-9).
             "calibrated" chooses t together with the weights on the fit half.
  soft       each face weighted by Light-ASD's speaking probability sigmoid(score), with no threshold; a face without
             a score has probability 0.
  top1       the face with the highest score in the frame is talking (all tied faces). Every 15K frame contains a
             talking face by selection, so top1 uses label information that a deployed model would not have; it is a
             reference for the ranking only.
Masks: mt = max over faces of the face's talking weight inside its box; mn = (union of the boxes) * (1 - mt). For
weights in {0, 1} this is the mask pair of part 2 and 15K. The map is q = p * (1 + a * mn + b * mt), (a, b) on the
15K grid, CC and NSS from the closed form of part 2.

    python scripts/inv_mvva_asd_posthoc.py --asd runs/inv/mvva_asd --small runs/inv/mvva_small_s0 \
        --rows15k runs/inv/k_mvva_faces_rows.json --part2 runs/review/inv/n_mvva_asd.json \
        --out runs/review/inv/n_mvva_asd_posthoc.json
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

from scripts.inv_mvva_asd_eval import cc_nss_closed, scores_by_frame, stats  # noqa: E402
from scripts.inv_mvva_faces import GRID, H, W, boxes_for, native_sizes  # noqa: E402
from surge.data import load_gray, mvva_videos  # noqa: E402

THRESHOLDS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
PAIRS = list(itertools.product(GRID, GRID))
TOL = 1e-9


def usable(t, f, size):
    """[(j, (x1, y1, x2, y2))] for the faces of frame f with a usable box at 640x360 (the geometry of
    scripts/inv_mvva_faces.masks)."""
    if t.ndim != 3 or f >= t.shape[0]:
        return []
    scale = np.array([W / size[0], H / size[1], W / size[0], H / size[1]])
    out = []
    for j in range(t.shape[1]):
        b = t[f, j] * scale
        if not np.all(np.isfinite(b)):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        out.append((j, (x1, y1, x2, y2)))
    return out


def soft_masks(boxes, weight):
    """(mt, mn): mt = max over faces of weight(j) inside the face's box, mn = union of the boxes * (1 - mt)."""
    mt = np.zeros((H, W))
    face = np.zeros((H, W), bool)
    for j, (x1, y1, x2, y2) in boxes:
        face[y1:y2, x1:x2] = True
        w = weight(j)
        if w > 0:
            sub = mt[y1:y2, x1:x2]
            np.maximum(sub, w, out=sub)
    return mt, face * (1.0 - mt)


def variants(s: dict, faces: list) -> dict:
    """{name: weight function} for the scores s = {face: score} of one frame; top1 ranks the usable faces."""
    ranked = [s[j] for j in faces if j in s]
    best = max(ranked) if ranked else None
    out = {f"thr_{th}": (lambda j, th=th: 1.0 if s.get(j, -np.inf) > th else 0.0) for th in THRESHOLDS}
    out["soft"] = lambda j: float(1.0 / (1.0 + np.exp(-s[j]))) if j in s else 0.0
    out["top1"] = lambda j: 1.0 if best is not None and j in s and s[j] == best else 0.0
    return out


def one_video(args):
    vid, small, sigma, asd_dir, frames15k = args
    v = {x.name: x for x in mvva_videos()}[vid]
    z = np.load(Path(small) / f"{vid}.npz")
    pred = dict(zip((int(f) for f in z["frames"]), z["maps"]))
    t, k = boxes_for(vid)
    size = native_sizes()[vid]
    sc = scores_by_frame(np.load(Path(asd_dir) / f"{vid}.npz"))
    vals = {}
    counts = {"faces": 0, "scored": 0, "label_talking": 0, "soft_sum": 0.0,
              **{f"above_{th}": 0 for th in THRESHOLDS}}
    for f in frames15k:
        boxes = usable(t, f, size)
        s = sc.get(f, {})
        gt = load_gray(v.maps[f]).astype(np.float64)
        fix = load_gray(v.fixations[f])
        p = cv2.resize(pred[f].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        if sigma > 0:
            p = cv2.GaussianBlur(p, (0, 0), sigma)
        p = (p / p.sum()).astype(np.float64)
        fx = fix > 0
        for name, fn in variants(s, [j for j, _ in boxes]).items():
            mt, mn = soft_masks(boxes, fn)
            st = stats(p, gt, fx, mt, mn)
            vals.setdefault(name, []).append([cc_nss_closed(st, a, b) for a, b in PAIRS])
        for j, _ in boxes:
            counts["faces"] += 1
            counts["label_talking"] += int(j < k.shape[0] and f < k.shape[1] and k[j, f] == 1)
            if j in s:
                counts["scored"] += 1
                counts["soft_sum"] += float(1.0 / (1.0 + np.exp(-s[j])))
                for th in THRESHOLDS:
                    counts[f"above_{th}"] += int(s[j] > th)
    means = {name: np.nanmean(np.array(x), axis=0).tolist() for name, x in vals.items()}  # (pairs, 2): CC, NSS
    return vid, means, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asd", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--rows15k", required=True)
    ap.add_argument("--part2", required=True, help="part 2's json, for the t = 0 check")
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N videos only (smoke test; no check, no output)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r15 = {v: r for v, r in json.loads(Path(a.rows15k).read_text()).items() if r}
    vids = sorted(r15, key=lambda s: int(s))
    halves = [vids[0::2], vids[1::2]]  # as in 15K and part 2
    todo = vids[:a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one_video, [(v, a.small, a.sigma, a.asd, [r["f"] for r in r15[v]]) for v in todo]))
    det = {v: m for v, m, _ in res}
    counts = {key: sum(c[key] for _, _, c in res) for key in res[0][2]}
    halves = [[v for v in h if v in det] for h in halves]
    names = list(res[0][1])

    def vmean15(vs, key):
        return float(np.mean([np.nanmean([r[key][0] for r in r15[v]]) for v in vs]))

    def vmean(vs, name, i):
        return float(np.mean([det[v][name][i][0] for v in vs]))

    folds, per = [], []  # per test video: base, face, label speaker, then each variant and "calibrated" (CC, NSS)
    cols = names + ["calibrated"]
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        fw = max(GRID, key=lambda x: vmean15(fit, f"face_{x}"))
        sw = max(PAIRS, key=lambda ab: vmean15(fit, f"spk_{ab[0]}_{ab[1]}"))
        choice = {n: max(range(len(PAIRS)), key=lambda i: vmean(fit, n, i)) for n in names}
        cal = max(((f"thr_{th}", i) for th in THRESHOLDS for i in range(len(PAIRS))),
                  key=lambda ni: vmean(fit, ni[0], ni[1]))
        folds.append({"face_weight": fw, "label_speaker_weights": sw, "test_videos": len(test),
                      **{n: PAIRS[i] for n, i in choice.items()},
                      "calibrated": {"variant": cal[0], "weights": PAIRS[cal[1]]}})
        for v in test:
            row = [np.nanmean([r["cc"] for r in r15[v]]), np.nanmean([r[f"face_{fw}"][0] for r in r15[v]]),
                   np.nanmean([r[f"spk_{sw[0]}_{sw[1]}"][0] for r in r15[v]])]
            row += [det[v][n][choice[n]][0] for n in names] + [det[v][cal[0]][cal[1]][0]]
            row += [np.nanmean([r["nss"] for r in r15[v]]), np.nanmean([r[f"face_{fw}"][1] for r in r15[v]]),
                    np.nanmean([r[f"spk_{sw[0]}_{sw[1]}"][1] for r in r15[v]])]
            row += [det[v][n][choice[n]][1] for n in names] + [det[v][cal[0]][cal[1]][1]]
            per.append(row)
    per = np.array(per)
    nc = 3 + len(cols)  # CC columns; NSS columns follow
    m = per.mean(0)
    rng = np.random.default_rng(0)  # the draws of part 2
    bs = np.array([per[i].mean(0) for i in (rng.integers(0, len(per), len(per)) for _ in range(10000))])

    def ci(x):
        return [float(q) for q in np.percentile(x, [2.5, 97.5])]

    out = {"videos": len(per), "frames": int(sum(len(r15[v]) for v in det)), "folds": folds,
           "face_counts": {**counts, "label_talking_share": counts["label_talking"] / counts["faces"],
                           "soft_mean": counts["soft_sum"] / max(counts["scored"], 1),
                           **{f"above_{th}_share": counts[f"above_{th}"] / counts["faces"] for th in THRESHOLDS}},
           "base": float(m[0]), "face": float(m[1]), "label_speaker": float(m[2]),
           "label_minus_face": {"mean": float(m[2] - m[1]), "ci95": ci(bs[:, 2] - bs[:, 1])}, "variants": {}}
    for c, n in enumerate(cols):
        i = 3 + c
        share = (bs[:, i] - bs[:, 1]) / (bs[:, 2] - bs[:, 1])
        out["variants"][n] = {"cc": float(m[i]),
                              "minus_base": {"mean": float(m[i] - m[0]), "ci95": ci(bs[:, i] - bs[:, 0])},
                              "minus_face": {"mean": float(m[i] - m[1]), "ci95": ci(bs[:, i] - bs[:, 1])},
                              "share_realised": {"mean": float((m[i] - m[1]) / (m[2] - m[1])), "ci95": ci(share)},
                              "nss_minus_base": float(m[nc + i] - m[nc]),
                              "nss_minus_face": float(m[nc + i] - m[nc + 1])}
    if a.limit is not None:
        print(json.dumps(out["variants"], indent=1))
        return
    p2 = json.loads(Path(a.part2).read_text())["part2"]
    diff = abs(out["variants"]["thr_0.0"]["cc"] - p2["detector_speaker"])
    out["check_thr0_vs_part2"] = diff
    out["status"] = "ok" if diff < TOL else "check failed: t = 0 does not reproduce part 2"
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
    print(out["status"], f"(t = 0 vs part 2: {diff:.2e})")
    print(json.dumps(out["face_counts"], indent=1))
    for n, r in out["variants"].items():
        print(f"{n:12s} CC {r['cc']:.4f}  -base {r['minus_base']['mean']:+.4f} {r['minus_base']['ci95']}  "
              f"-face {r['minus_face']['mean']:+.4f} {r['minus_face']['ci95']}  share {r['share_realised']['mean']:.2f}"
              f" {r['share_realised']['ci95']}")
    for f in folds:
        print(f)


if __name__ == "__main__":
    main()
