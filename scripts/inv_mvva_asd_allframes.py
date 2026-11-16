"""Section 15N, post hoc (research/log.md): the detector re-weighting on every scored MVVA frame. CPU, descriptive.

The 15K frames were selected by the labels to show a talking and a silent face. Here the re-weighting that both halves
chose in part 2 and in the post-hoc analysis, q = p * (1 + 1[T]) (faces called talking x2, other faces unchanged), is
applied to every frame scored by the zero-shot run of 15I (`runs/review/inv/zs_mvva.json`, 294 videos), with
  det_0      T = faces with a Light-ASD score above 0 (the registered threshold)
  det_-0.5   T = faces with a score above -0.5 (the calibrated threshold)
  label      T = faces with talking label 1 (reference; uses the labels)
No weights are fitted. Masks have the geometry of scripts/inv_mvva_faces.masks (boxes scaled per axis to the 640x360
maps). The unchanged map p is built as in 15I (surge.infer.to_native, sigma 8) and must reproduce the 15I per-video CC.
Frame types: "k15" (the 15K frames), "face" (other frames with at least one usable face box), "noface" (the rest).

    python scripts/inv_mvva_asd_allframes.py --asd runs/inv/mvva_asd --small runs/inv/mvva_small_s0 \
        --zs runs/review/inv/zs_mvva.json --rows15k runs/inv/k_mvva_faces_rows.json \
        --out runs/review/inv/n_mvva_asd_allframes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.inv_mvva_asd_eval import scores_by_frame  # noqa: E402
from scripts.inv_mvva_asd_posthoc import soft_masks, usable  # noqa: E402
from scripts.inv_mvva_faces import H, W, boxes_for, native_sizes  # noqa: E402
from surge.data import load_gray, mvva_videos  # noqa: E402
from surge.infer import to_native  # noqa: E402
from surge.metrics import cc, nss  # noqa: E402

TOL = 1e-5  # the saved maps are float16 (research/log.md 15N, amendment)
VARIANTS = ["det_0", "det_-0.5", "label"]
TYPES = ["k15", "face", "noface"]


def one_video(args):
    vid, small, sigma, asd_dir, k15 = args
    v = {x.name: x for x in mvva_videos()}[vid]
    z = np.load(Path(small) / f"{vid}.npz")
    t, k = boxes_for(vid)
    size = native_sizes()[vid]
    p_asd = Path(asd_dir) / f"{vid}.npz"
    sc = scores_by_frame(np.load(p_asd)) if p_asd.exists() else {}
    rows = []
    for f, m in zip(z["frames"], z["maps"]):
        f = int(f)
        gt = load_gray(v.maps[f]).astype(np.float64)
        fix = load_gray(v.fixations[f])
        if v.skip_empty and not (fix > 0).any():
            continue
        p = to_native(m, gt.shape, sigma)
        r = {"f": f, "cc": cc(p, gt), "nss": nss(p, fix)}
        boxes = usable(t, f, size) if gt.shape == (H, W) else []
        r["type"] = "k15" if f in k15 else ("face" if boxes else "noface")
        s = sc.get(f, {})
        rules = {"det_0": lambda j: 1.0 if s.get(j, -np.inf) > 0.0 else 0.0,
                 "det_-0.5": lambda j: 1.0 if s.get(j, -np.inf) > -0.5 else 0.0,
                 "label": lambda j: 1.0 if j < k.shape[0] and f < k.shape[1] and k[j, f] == 1 else 0.0}
        for name, rule in rules.items():
            if boxes:
                mt, _ = soft_masks(boxes, rule)
                q = p.astype(np.float64) * (1.0 + mt)
                r[name] = (cc(q, gt), nss(q, fix), float(mt.mean()))
            else:
                r[name] = (r["cc"], r["nss"], 0.0)
        rows.append(r)
    return vid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asd", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--zs", required=True, help="15I per-video results, for the check")
    ap.add_argument("--rows15k", required=True)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N videos only (smoke test; no output)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zs = json.loads(Path(a.zs).read_text())["per_video"]
    r15 = json.loads(Path(a.rows15k).read_text())
    vids = sorted(zs, key=lambda s: int(s))[:a.limit]
    jobs = [(v, a.small, a.sigma, a.asd, {r["f"] for r in (r15.get(v) or [])}) for v in vids]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one_video, jobs))
    worst = max(abs(np.nanmean([r["cc"] for r in res[v]]) - zs[v]["cc"]) for v in vids)
    out = {"videos": len(vids), "frames": int(sum(len(res[v]) for v in vids)), "check_base_vs_15I": float(worst),
           "frames_by_type": {ty: int(sum(r["type"] == ty for v in vids for r in res[v])) for ty in TYPES}}
    print(f"check: largest per-video |CC - 15I| = {worst:.2e}", flush=True)
    if worst > TOL:
        out["status"] = "check failed; nothing reported"
        if a.limit is None:
            Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
        print(out["status"])
        return
    def summary(sel):
        """Per-video means over the frames that sel(row) keeps; videos without such frames are left out."""
        per = []
        for v in vids:
            rs = [r for r in res[v] if sel(r)]
            if rs:
                per.append([np.nanmean([r["cc"] for r in rs]), np.nanmean([r["nss"] for r in rs])]
                           + [np.nanmean([r[n][i] for r in rs]) for n in VARIANTS for i in (0, 1, 2)])
            else:
                per.append(None)
        keep = [i for i, x in enumerate(per) if x is not None]
        if not keep:
            return {"videos": 0, "frames": 0}
        arr = np.array([per[i] for i in keep])
        rng = np.random.default_rng(0)  # video bootstrap within the videos kept
        bs = np.array([arr[rng.integers(0, len(arr), len(arr))].mean(0) for _ in range(10000)])
        m = arr.mean(0)
        o = {"videos": len(keep), "frames": int(sum(sel(r) for v in vids for r in res[v])), "cc": float(m[0]),
             "nss": float(m[1])}
        for c, n in enumerate(VARIANTS):
            j = 2 + 3 * c
            o[n] = {"dcc": float(m[j] - m[0]), "dcc_ci95": [float(x) for x in np.percentile(bs[:, j] - bs[:, 0], [2.5, 97.5])],
                    "dnss": float(m[j + 1] - m[1]),
                    "dnss_ci95": [float(x) for x in np.percentile(bs[:, j + 1] - bs[:, 1], [2.5, 97.5])],
                    "talking_area": float(m[j + 2])}
        return o

    out["all"] = summary(lambda r: True)
    for ty in TYPES:
        out[ty] = summary(lambda r, ty=ty: r["type"] == ty)
    out["status"] = "ok"
    if a.limit is None:
        Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
    for key in ["all"] + TYPES:
        o = out[key]
        if not o["videos"]:
            print(f"{key:7s} no frames")
            continue
        print(f"{key:7s} {o['videos']} videos, {o['frames']} frames, CC {o['cc']:.4f}: " + "; ".join(
            f"{n} {o[n]['dcc']:+.4f} [{o[n]['dcc_ci95'][0]:+.4f}, {o[n]['dcc_ci95'][1]:+.4f}] NSS {o[n]['dnss']:+.3f}"
            for n in VARIANTS))


if __name__ == "__main__":
    main()
