"""Gate G2: zero-shot test of V-JEPA 2 predictor surprise as a fixation cue (rules: research/log.md section 3).

For every holdout video and frame rate, windows of 16 sampled frames are split into 8 context + 8 target frames.
For each target tubelet, the first frame is scored. Maps compared (all 16x16 token grids, upsampled to 360x640 and
blurred with the same sigma): raw surprise, contextual surprise (true past minus another video's past), token
atypicality, frame difference (pixel |I_t - I_t-1| pooled to the token grid), and the empirical center prior.

Scores: NSS, CC, and information gain (bits/fixation) of 0.5*map + 0.5*prior over the prior alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from surge.data import dhf1k_videos, load_gray, load_rgb, read_split_file  # noqa: E402
from surge.metrics import cc, nss  # noqa: E402

H, W = 360, 640


def center_prior(train_videos, step=30, cache=None):
    if cache and Path(cache).exists():
        return np.load(cache)
    acc = np.zeros((H, W), np.float64)
    for v in train_videos:
        for f in range(0, len(v), step):
            m = load_gray(v.maps[f]).astype(np.float64)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H))
            s = m.sum()
            if s > 0:
                acc += m / s
    acc /= acc.sum()
    if cache:
        np.save(cache, acc)
    return acc


def up(m16: np.ndarray, sigma: float) -> np.ndarray:
    m = cv2.resize(m16.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return m


def info_gain(m: np.ndarray, prior: np.ndarray, fix: np.ndarray) -> float:
    p = m - m.min()
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / p.size)
    mix = 0.5 * p + 0.5 * prior
    sel = fix > 127
    if not sel.any():
        return float("nan")
    return float(np.mean(np.log2(mix[sel] + 1e-12) - np.log2(prior[sel] + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="configs/splits/dhf1k_holdout60.txt")
    ap.add_argument("--strides", default="8,2", help="frame strides (30 fps video): 8 ~ 4 fps, 2 = 15 fps")
    ap.add_argument("--sigma", type=float, default=16.0)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--out", default="runs/g2/g2_results.json")
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from surge.backbones import VJEPA2
    bb = VJEPA2().to(device).eval()
    vids = dhf1k_videos("D:/DHF1K", read_split_file(ROOT / a.split))[: a.max_videos]
    train = dhf1k_videos("D:/DHF1K", read_split_file(ROOT / "configs/splits/dhf1k_train540.txt"))
    prior = center_prior(train, cache=out.parent / "center_prior.npy")
    names = ["surprise_raw", "surprise_ctx", "atypicality", "frame_diff", "center_prior"]
    results = {}
    t0 = time.time()
    for s in [int(x) for x in a.strides.split(",")]:
        per_video = {}
        prev_ctx = None
        for vi, v in enumerate(vids):
            n = len(v)
            span = 15 * s
            starts = list(range(0, max(n - span, 1), 8 * s))
            rows = {k: {"nss": [], "cc": [], "ig": []} for k in names}
            for st in starts:
                idx = [min(st + k * s, n - 1) for k in range(16)]
                x = torch.from_numpy(np.stack([load_rgb(v.frames[j], (256, 256)) for j in idx])).permute(0, 3, 1, 2)
                x = x[None].to(device)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    full = bb.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state.float()
                    ctx = bb.model(pixel_values_videos=x[:, :8], skip_predictor=True).last_hidden_state
                    ids_c = [torch.arange(1024, device=device)[None]]
                    ids_t = [torch.arange(1024, 2048, device=device)[None]]
                    ln = lambda t: F.layer_norm(t.float(), t.shape[-1:])
                    tgt = ln(full[:, 1024:])
                    e = (ln(bb.model.predictor(encoder_hidden_states=ctx, context_mask=ids_c, target_mask=ids_t)
                            .last_hidden_state) - tgt).abs().mean(-1)
                    wrong = prev_ctx if prev_ctx is not None else ctx.flip(1)
                    e_w = (ln(bb.model.predictor(encoder_hidden_states=wrong, context_mask=ids_c, target_mask=ids_t)
                              .last_hidden_state) - tgt).abs().mean(-1)
                e = e.view(4, 16, 16).cpu().numpy()
                e_w = e_w.view(4, 16, 16).cpu().numpy()
                zt = tgt.view(4, 256, 1024)
                atyp = (zt - zt.mean(1, keepdim=True)).abs().mean(-1).view(4, 16, 16).cpu().numpy()
                for k in range(4):
                    f = idx[8 + 2 * k]
                    if f >= n or (k and f == idx[8 + 2 * (k - 1)]):
                        continue
                    gt = load_gray(v.maps[f]).astype(np.float64)
                    fix = load_gray(v.fixations[f])
                    if gt.shape != (H, W):
                        continue
                    g1 = cv2.imread(v.frames[f], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    g0 = cv2.imread(v.frames[max(f - 1, 0)], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    fd = cv2.resize(np.abs(g1 - g0), (16, 16), interpolation=cv2.INTER_AREA)
                    maps = {"surprise_raw": up(e[k], a.sigma), "surprise_ctx": up(e[k] - e_w[k], a.sigma),
                            "atypicality": up(atyp[k], a.sigma), "frame_diff": up(fd, a.sigma), "center_prior": prior}
                    for name, m in maps.items():
                        rows[name]["nss"].append(nss(m, fix))
                        rows[name]["cc"].append(cc(m, gt))
                        rows[name]["ig"].append(info_gain(m, prior, fix) if name != "center_prior" else 0.0)
            prev_ctx = ctx
            per_video[v.name] = {k: {m: float(np.nanmean(vals)) for m, vals in r.items()} for k, r in rows.items()}
            print(f"stride {s} [{vi + 1}/{len(vids)}] {v.name} " +
                  " ".join(f"{k}:nss={per_video[v.name][k]['nss']:.3f}" for k in names), flush=True)
        mean = {k: {m: float(np.mean([pv[k][m] for pv in per_video.values()])) for m in ("nss", "cc", "ig")} for k in names}
        results[f"stride_{s}"] = {"mean": mean, "per_video": per_video}
        print(f"STRIDE {s} MEAN " + json.dumps(mean), flush=True)
    # decision (research/log.md section 3)
    verdict = {}
    for key, r in results.items():
        m = r["mean"]
        for var in ("surprise_raw", "surprise_ctx"):
            ok = m[var]["nss"] > m["atypicality"]["nss"] and m[var]["nss"] > m["frame_diff"]["nss"] and m[var]["ig"] > 0
            verdict[f"{key}/{var}"] = bool(ok)
    results["verdict"] = verdict
    results["go"] = any(verdict.values())
    results["seconds"] = time.time() - t0
    out.write_text(json.dumps(results, indent=1), newline="\n")
    print("VERDICT", json.dumps(verdict), "GO" if results["go"] else "NO-GO", flush=True)


if __name__ == "__main__":
    main()
