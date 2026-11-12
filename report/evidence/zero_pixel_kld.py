"""Diagnostic for the review: how much of SalFoM's KLD (and SIM) on DHF1K val comes from pixels that its 8-bit,
min-max-scaled maps set exactly to zero? Every 10th frame of the 100 validation videos.

    python -I zero_pixel_kld.py <repo root> <salfom maps dir> [sigma]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(sys.argv[1])
MAPS = Path(sys.argv[2])
SIGMA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
sys.path.insert(0, str(ROOT))
from surge.data import load_gray  # noqa: E402
from surge.metrics import EPS, kld, sim  # noqa: E402


def one(job):
    mp_png, gt_path = job
    p = cv2.imread(str(mp_png), cv2.IMREAD_GRAYSCALE)
    q = load_gray(gt_path).astype(np.float64)
    if SIGMA > 0:
        p = cv2.GaussianBlur(p.astype(np.float32), (0, 0), SIGMA)
    p = p.astype(np.float64)
    if p.shape != q.shape:
        p = cv2.resize(p, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR)
    zero = p == 0
    pn = p / (p.sum() + EPS)
    qn = q / (q.sum() + EPS)
    terms = qn * np.log(EPS + qn / (pn + EPS))
    k_all = float(terms.sum())
    k_zero = float(terms[zero].sum())
    # the same map with a floor of one grey level added before normalisation (zeros removed)
    pf = p + 1.0
    return dict(zero_frac=float(zero.mean()), gt_mass_on_zero=float(qn[zero].sum()), kld=k_all, kld_from_zero=k_zero,
                kld_floor=kld(pf, q), sim=sim(p, q), sim_floor=sim(pf, q), check=kld(p, q))


def main():
    from surge.data import dhf1k_videos, read_split_file
    vids = dhf1k_videos("D:/DHF1K", read_split_file(ROOT / "configs/splits" / "dhf1k_val100.txt"))
    jobs, owner = [], []
    for v in vids:
        for f in range(0, len(v), 10):
            jobs.append((MAPS / v.name / f"{f + 1:04d}.png", v.maps[f]))
            owner.append(v.name)
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs, chunksize=16))
    per = {}
    for name, r in zip(owner, res):
        per.setdefault(name, []).append(r)
    keys = res[0].keys()
    vm = {k: np.mean([np.mean([r[k] for r in rs]) for rs in per.values()]) for k in keys}
    print(f"videos {len(per)}, frames {len(res)}, sigma {SIGMA}")
    for k in keys:
        print(f"  {k:16s} {vm[k]:.4f}")


if __name__ == "__main__":
    main()
