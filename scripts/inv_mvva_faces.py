"""Multi-face scenes on MVVA (research/log.md section 15K), from saved predictions (CPU).

    python scripts/inv_mvva_faces.py --small runs/inv/mvva_small_s0 --out runs/review/inv/k_mvva_faces.json

Frames used: scored frames (as saved by scripts/probe_eval.py --save-small) with at least one talking and one
non-talking face box (MVVA face tracks and per-face talking labels; label 1 = talking, other values = not talking or not
annotated, boxes in native video pixels, scaled per axis to the 640x360 maps with the native size from the layout
index). The prediction is resized to 640x360, blurred with the
sigma of the zero-shot run (8) and normalised.
  shares     ground-truth mass, fixations and predicted mass inside talking faces (T), other faces (N) and elsewhere,
             and the area of each
  oracles    face:    p * (1 + a * 1[T or N])      speaker: p * (1 + a * 1[N] + b * 1[T])
             weights chosen on a grid to maximise mean CC on one half of the videos (sorted ids, alternating), scored
             on the other half, and the reverse; reported: CC and NSS change against the unchanged map on held-out
             videos. speaker - face bounds what knowing who speaks could add to this model's map.
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

from surge.data import load_gray, mvva_videos  # noqa: E402
from surge.metrics import cc, nss  # noqa: E402

SRC = Path("D:/Datasets/MVVA/extracted")
W, H = 640, 360
GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def native_sizes() -> dict:
    """{video id: (width, height)} of the source videos, from the layout index (scripts/prep_mvva.py)."""
    import csv
    return {r["id"]: (int(r["width"]), int(r["height"]))
            for r in csv.DictReader(open(Path("D:/Datasets/MVVA/surge") / "index.csv"))}


def boxes_for(vid: str):
    t = np.load(SRC / "all_face_tracking_release" / f"{vid}.npy", allow_pickle=False)
    k = np.load(SRC / "all_face_talking_release" / f"{vid}.npy", allow_pickle=False)
    return t, k


def masks(t, k, f, size=(2 * W, 2 * H)):
    """Talking and other-face masks at 640x360 for frame f (boxes x1, y1, x2, y2 in native pixels of a video of
    `size` = (width, height), scaled per axis), or None if the frame lacks either kind."""
    if f >= t.shape[0] or t.ndim != 3:
        return None
    scale = np.array([W / size[0], H / size[1], W / size[0], H / size[1]])
    mt, mn = np.zeros((H, W), bool), np.zeros((H, W), bool)
    nt = nn = 0
    for j in range(t.shape[1]):
        b = t[f, j] * scale
        if not np.all(np.isfinite(b)):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        talking = f < k.shape[1] and j < k.shape[0] and k[j, f] == 1
        if talking:
            mt[y1:y2, x1:x2] = True
            nt += 1
        else:
            mn[y1:y2, x1:x2] = True
            nn += 1
    mn &= ~mt
    if nt == 0 or nn == 0:
        return None
    return mt, mn


def one_video(args):
    vid, small, sigma = args
    v = {x.name: x for x in mvva_videos()}[vid]
    z = np.load(Path(small) / f"{vid}.npz")
    t, k = boxes_for(vid)
    size = native_sizes()[vid]
    rows = []
    for f, m in zip(z["frames"], z["maps"]):
        f = int(f)
        mk = masks(t, k, f, size)
        if mk is None:
            continue
        mt, mn = mk
        gt = load_gray(v.maps[f]).astype(np.float64)
        fix = load_gray(v.fixations[f])
        if gt.shape != (H, W) or not (fix > 0).any():
            continue
        p = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        if sigma > 0:
            p = cv2.GaussianBlur(p, (0, 0), sigma)
        p = p / p.sum()
        g = gt / gt.sum()
        fx = fix > 0
        r = {"f": f,
             "gt_T": float(g[mt].sum()), "gt_N": float(g[mn].sum()),
             "fix_T": float(fx[mt].sum() / fx.sum()), "fix_N": float(fx[mn].sum() / fx.sum()),
             "pred_T": float(p[mt].sum()), "pred_N": float(p[mn].sum()),
             "area_T": float(mt.mean()), "area_N": float(mn.mean()),
             "cc": cc(p, gt), "nss": nss(p, fix)}
        face = (mt | mn).astype(np.float64)
        for a in GRID:  # face oracle
            q = p * (1 + a * face)
            r[f"face_{a}"] = (cc(q, gt), nss(q, fix))
        for a, b in itertools.product(GRID, GRID):  # speaker oracle
            q = p * (1 + a * mn + b * mt)
            r[f"spk_{a}_{b}"] = (cc(q, gt), nss(q, fix))
        rows.append(r)
    return vid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vids = sorted(p.stem for p in Path(a.small).glob("*.npz"))
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one_video, [(v, a.small, a.sigma) for v in vids]))
    res = {v: r for v, r in res.items() if r}
    vids = sorted(res, key=lambda s: int(s))
    halves = [vids[0::2], vids[1::2]]

    def vmean(vs, key, idx=None):
        out = []
        for v in vs:
            xs = [r[key] if idx is None else r[key][idx] for r in res[v]]
            out.append(np.nanmean(xs))
        return float(np.mean(out))

    out = {"videos": len(vids), "frames": sum(len(r) for r in res.values())}
    for k in ("gt_T", "gt_N", "fix_T", "fix_N", "pred_T", "pred_N", "area_T", "area_N", "cc", "nss"):
        out[k] = vmean(vids, k)
    # cross-fitted oracles
    fold = []
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        best_face = max(GRID, key=lambda x: vmean(fit, f"face_{x}", 0))
        best_spk = max(itertools.product(GRID, GRID), key=lambda ab: vmean(fit, f"spk_{ab[0]}_{ab[1]}", 0))
        base = (vmean(test, "cc"), vmean(test, "nss"))
        face = (vmean(test, f"face_{best_face}", 0), vmean(test, f"face_{best_face}", 1))
        spk = (vmean(test, f"spk_{best_spk[0]}_{best_spk[1]}", 0), vmean(test, f"spk_{best_spk[0]}_{best_spk[1]}", 1))
        fold.append({"face_weight": best_face, "speaker_weights": {"other": best_spk[0], "talking": best_spk[1]},
                     "test_videos": len(test), "base": base, "face": face, "speaker": spk})
    out["oracles"] = fold
    for name in ("face", "speaker"):
        out[f"{name}_dcc"] = float(np.mean([f[name][0] - f["base"][0] for f in fold]))
        out[f"{name}_dnss"] = float(np.mean([f[name][1] - f["base"][1] for f in fold]))
    out["speaker_minus_face_dcc"] = out["speaker_dcc"] - out["face_dcc"]
    out["speaker_minus_face_dnss"] = out["speaker_dnss"] - out["face_dnss"]
    # per-video values for a bootstrap of the speaker-over-face gain (held-out halves only)
    per = []
    for fd, (fit, test) in zip(fold, ((halves[0], halves[1]), (halves[1], halves[0]))):
        a = fd["face_weight"]
        sa, sb = fd["speaker_weights"]["other"], fd["speaker_weights"]["talking"]
        for v in test:
            per.append([np.nanmean([r["cc"] for r in res[v]]),
                        np.nanmean([r[f"face_{a}"][0] for r in res[v]]),
                        np.nanmean([r[f"spk_{sa}_{sb}"][0] for r in res[v]])])
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = [per[i].mean(0) for i in (rng.integers(0, len(per), len(per)) for _ in range(10000))]
    bs = np.array(bs)
    out["boot_dcc_95"] = {"face": [float(x) for x in np.percentile(bs[:, 1] - bs[:, 0], [2.5, 97.5])],
                          "speaker": [float(x) for x in np.percentile(bs[:, 2] - bs[:, 0], [2.5, 97.5])],
                          "speaker_minus_face": [float(x) for x in np.percentile(bs[:, 2] - bs[:, 1], [2.5, 97.5])]}
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "oracles"}, indent=1))
    for f in fold:
        print(f)


if __name__ == "__main__":
    main()
