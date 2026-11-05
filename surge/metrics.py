"""Saliency metrics at native resolution, following the standard MIT / DHF1K definitions.

All functions take a predicted saliency map `pred` (H, W, float, non-negative), and either a continuous
ground-truth density `gt` (H, W) or a binary fixation map `fix` (H, W, bool / {0, 1}).
They are written with numpy so they run on the CPU in DataLoader workers or evaluation threads.
"""
from __future__ import annotations

import numpy as np

EPS = 2.220446049250313e-16  # MATLAB eps, as in the MIT benchmark code


def _z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def cc(pred: np.ndarray, gt: np.ndarray) -> float:
    a, b = _z(pred).ravel(), _z(gt).ravel()
    return float((a * b).mean())


def nss(pred: np.ndarray, fix: np.ndarray) -> float:
    m = fix > 0
    if not m.any():
        return float("nan")
    return float(_z(pred)[m].mean())


def sim(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(np.float64)
    q = gt.astype(np.float64)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    return float(np.minimum(p, q).sum())


def kld(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(np.float64)
    q = gt.astype(np.float64)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    return float((q * np.log(EPS + q / (p + EPS))).sum())


def _auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the ROC curve for positive vs negative scores (Mann-Whitney U, ties count 1/2)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sv = allv[order]
    # average ranks for ties
    i = 0
    n = len(sv)
    r = np.arange(1, n + 1, dtype=np.float64)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = r
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auc_judd(pred: np.ndarray, fix: np.ndarray) -> float:
    """Judd AUC (MIT benchmark AUC_Judd): thresholds at the saliency values of fixated pixels.

    TP rate at threshold t = fraction of fixations with s >= t; FP rate = fraction of non-fixated pixels
    with s >= t. The curve is built from the sorted fixation values, as in AUC_Judd.m.
    """
    s = pred.astype(np.float64).ravel()
    s = s + np.random.default_rng(0).random(s.size) / 1e7  # AUC_Judd.m jitter: breaks ties of quantized maps
    s = (s - s.min()) / (s.max() - s.min() + EPS)
    f = (fix.ravel() > 0)
    nfix = int(f.sum())
    if nfix == 0:
        return float("nan")
    # small jitter as in the reference implementation is omitted; ties are rare for float maps
    sth = np.sort(s[f])[::-1]
    npix = s.size
    sorted_all = np.sort(s)[::-1]
    # number of pixels with value >= each threshold
    above = np.searchsorted(-sorted_all, -sth, side="right")
    tp = np.concatenate([[0.0], np.arange(1, nfix + 1) / nfix, [1.0]])
    fp = np.concatenate([[0.0], (above - np.arange(1, nfix + 1)) / (npix - nfix), [1.0]])
    return float(np.trapezoid(tp, fp))


def auc_shuffled(pred: np.ndarray, fix: np.ndarray, other_fix: np.ndarray, n_splits: int = 100,
                 step: float = 0.1, seed: int = 0) -> float:
    """Shuffled AUC, following AUC_shuffled.m (MIT saliency benchmark / DHF1K evaluation code).

    Saliency is rescaled to [0, 1]. Negatives: for each of `n_splits` splits, as many locations as there are
    fixations, drawn without replacement from the fixated pixels of `other_fix` (fixations of other videos).
    ROC from thresholds 0:step:max, area by the trapezoid rule; the result is the mean over splits.
    """
    s = pred.astype(np.float64).ravel()
    s = (s - s.min()) / (s.max() - s.min() + EPS)
    sth = s[fix.ravel() > 0]
    ind = np.flatnonzero(other_fix.ravel() > 0)
    nf = len(sth)
    if nf == 0 or len(ind) == 0:
        return float("nan")
    no = min(nf, len(ind))
    rng = np.random.default_rng(seed)
    sth_sorted = np.sort(sth)
    aucs = []
    for _ in range(n_splits):
        cur = np.sort(s[rng.permutation(ind)[:no]])
        th = np.arange(0, max(sth.max(), cur.max()) + 1e-12, step)[::-1]
        tp = np.concatenate([[0.0], (nf - np.searchsorted(sth_sorted, th, side="left")) / nf, [1.0]])
        fp = np.concatenate([[0.0], (no - np.searchsorted(cur, th, side="left")) / no, [1.0]])
        aucs.append(np.trapezoid(tp, fp))
    return float(np.mean(aucs))


def all_metrics(pred: np.ndarray, gt: np.ndarray, fix: np.ndarray, other_fix: np.ndarray | None = None,
                which=("cc", "nss", "sim", "kld", "auc_j", "sauc")) -> dict:
    out = {}
    if "cc" in which:
        out["cc"] = cc(pred, gt)
    if "nss" in which:
        out["nss"] = nss(pred, fix)
    if "sim" in which:
        out["sim"] = sim(pred, gt)
    if "kld" in which:
        out["kld"] = kld(pred, gt)
    if "auc_j" in which:
        out["auc_j"] = auc_judd(pred, fix)
    if "sauc" in which and other_fix is not None:
        out["sauc"] = auc_shuffled(pred, fix, other_fix)
    return out
