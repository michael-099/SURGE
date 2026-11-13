"""Whole-video inference and scoring at native resolution."""
from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch

from .data import Video, audio_features, causal_windows, inference_windows, load_gray, load_rgb, ssl_condition, ssl_maps
from .losses import log_softmax2d
from .metrics import all_metrics
from concurrent.futures import ThreadPoolExecutor

_IO_POOL = ThreadPoolExecutor(8)  # frame decoding (OpenCV releases the GIL)


@torch.no_grad()
def predict_video(model, video: Video, T: int, stride: int, size, device, batch: int = 4, amp=torch.bfloat16,
                  surprise_fn=None, causal: bool = False, frames=None):
    """Yield (frame_index, probability map at decoder resolution) for every frame of the video.

    causal=True (research/log.md section 12D): one window per frame, ending at that frame, and only for `frames`
    (default: every frame); the map is the output at the last sampled position."""
    n = len(video)
    wins = list(causal_windows(n, T, stride, frames) if causal else inference_windows(n, T, stride))
    cache = {}
    A = audio_features(video, n) if getattr(model, "uses_audio", False) else None
    use_ssl = getattr(model, "ssl", False)  # section 14F: localisation maps as the modulation input
    if use_ssl and surprise_fn is not None:
        raise ValueError("a localisation-map model takes no surprise function")
    C = ssl_maps(video, n) if use_ssl else None
    hw = (size[0] // 16, size[1] // 16)
    for b0 in range(0, len(wins), batch):
        chunk = wins[b0:b0 + batch]
        need = sorted({j for idx, _ in chunk for j in idx if j not in cache})
        for j, im in zip(need, _IO_POOL.map(lambda j: load_rgb(video.frames[j], size), need)):
            cache[j] = im
        x = [np.stack([cache[j] for j in idx]) for idx, _ in chunk]
        lo = min(min(i) for i, _ in chunk)
        for k in [k for k in cache if k < lo]:
            del cache[k]
        x = torch.from_numpy(np.stack(x)).permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
            s = surprise_fn(x) if surprise_fn is not None else None
            if use_ssl:
                s = torch.from_numpy(np.stack([ssl_condition(C, idx, hw) for idx, _ in chunk])).to(device)
            if A is not None:
                a = torch.from_numpy(np.stack([A[idx] for idx, _ in chunk])).to(device)
                out = model(x, surprise=s, audio=a)
            else:
                out = model(x, surprise=s)
        p = log_softmax2d(out["logits"].float()).exp().cpu().numpy()  # B,T,h,w
        for bi, (_, assign) in enumerate(chunk):
            for f, q in assign:
                a = int(np.floor(q))
                w = q - a
                m = p[bi, a] if w == 0 else (1 - w) * p[bi, a] + w * p[bi, min(a + 1, T - 1)]
                yield f, m


def to_native(m: np.ndarray, hw, sigma: float = 0.0) -> np.ndarray:
    out = cv2.resize(m.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    return out / max(out.sum(), 1e-12)


def _score_one(args):
    pred_small, map_path, fix_path, other_fix, sigma, which, skip_empty = args
    gt = load_gray(map_path).astype(np.float64)
    fix = load_gray(fix_path)
    if skip_empty and not (fix > 0).any():
        return {k: float("nan") for k in which if k != "sauc" or other_fix is not None}
    pred = to_native(pred_small, gt.shape, sigma)
    of = None
    if other_fix is not None and other_fix.shape != gt.shape:
        of = cv2.resize(other_fix, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        of = other_fix
    return all_metrics(pred, gt, fix, of, which)


def other_fixations(videos: list[Video], k_videos: int = 10, frames_per_video: int = 1, seed: int = 0):
    """For shuffled AUC: per video, a binary map pooling fixations from frames of k other videos."""
    rng = random.Random(seed)
    out = {}
    for v in videos:
        others = [o for o in videos if o.name != v.name]
        acc = None
        for o in rng.sample(others, min(k_videos, len(others))):
            for _ in range(frames_per_video):
                f = load_gray(o.fixations[rng.randrange(len(o))]) > 127
                if acc is None:
                    acc = f.copy()
                else:
                    if f.shape != acc.shape:
                        f = cv2.resize(f.astype(np.uint8), (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    acc |= f
        out[v.name] = acc.astype(np.uint8) * 255
    return out


def evaluate(model, videos: list[Video], T, stride, size, device, sigma: float = 0.0, frame_step: int = 1,
             which=("cc", "nss", "sim", "kld"), workers: int = 12, batch: int = 4, sauc_maps=None,
             surprise_fn=None, save_small: dict | None = None, log=None, causal: bool = False):
    """Score every `frame_step`-th frame. Returns (mean over videos of per-video means, per-video dict).
    causal=True predicts only the scored frames, each from the window that ends at it (section 12D)."""
    model.eval()
    per_video = {}
    with ProcessPoolExecutor(workers) as ex:
        for vi, v in enumerate(videos):
            jobs, frames = [], []
            preds = {}
            sub = range(0, len(v), frame_step) if causal else None
            for f, m in predict_video(model, v, T, stride, size, device, batch=batch, surprise_fn=surprise_fn,
                                      causal=causal, frames=sub):
                if save_small is not None:
                    preds[f] = m.astype(np.float16)
                if f % frame_step:
                    continue
                of = sauc_maps.get(v.name) if sauc_maps is not None else None
                jobs.append((m, v.maps[f], v.fixations[f], of, sigma, which, v.skip_empty))
                frames.append(f)
            if save_small is not None:  # frames that were not predicted (causal subsets) stay zero
                first = next(iter(preds.values()))
                arr = np.zeros((len(v), *first.shape), first.dtype)
                for i, q in preds.items():
                    arr[i] = q
                save_small[v.name] = arr
            res = list(ex.map(_score_one, jobs, chunksize=8))
            per_video[v.name] = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
            if log:
                log(f"  [{vi + 1}/{len(videos)}] {v.name} " + " ".join(f"{k}={x:.4f}" for k, x in per_video[v.name].items()))
    keys = next(iter(per_video.values())).keys()
    mean = {k: float(np.mean([pv[k] for pv in per_video.values()])) for k in keys}
    return mean, per_video


def score_saved(small: dict, videos: list[Video], sigma: float, frame_step: int = 1, which=("cc", "nss", "sim", "kld"),
                workers: int = 12, sauc_maps=None):
    """Re-score saved low-resolution predictions (e.g. to tune the blur sigma) without re-running the model."""
    per_video = {}
    with ProcessPoolExecutor(workers) as ex:
        for v in videos:
            jobs = [(small[v.name][f].astype(np.float32), v.maps[f], v.fixations[f],
                     sauc_maps.get(v.name) if sauc_maps else None, sigma, which, v.skip_empty)
                    for f in range(0, len(v), frame_step)]
            res = list(ex.map(_score_one, jobs, chunksize=8))
            per_video[v.name] = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
    keys = next(iter(per_video.values())).keys()
    return {k: float(np.mean([pv[k] for pv in per_video.values()])) for k in keys}, per_video
