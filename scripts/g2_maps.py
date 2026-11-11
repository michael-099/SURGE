"""Token maps for the incremental-information test and the masked-tube readout (research/log.md sections 13B, 13C).

13B. The maps of gate G2 are recomputed exactly as in scripts/g2_surprise.py (same model, windows, scored frames,
wrong-context rule and bfloat16 autocast) and stored at token resolution (16x16) instead of being scored: raw
surprise, contextual surprise, token atypicality and frame difference.

13C. The masked-tube error. The 16x16 token grid is cut into sixteen 4x4 blocks. In each of four passes the four
blocks with the same (row parity, column parity) are masked in all 8 tubelets (512 of 2,048 tokens). The encoder runs
on the 1,536 visible tokens only, the predictor predicts the masked tokens, and the error is the mean absolute
difference to the layer-normalised features of the full clip, as in G2. Four passes give one error per token; the map
of a scored frame is the error at its tubelet.

Before the videos are processed (float32, first window of the first video, stride 8):
  check (i)   all 2,048 tokens visible: the masked encoder reproduces the model's own encoder output;
  check (ii)  the first 1,024 tokens visible: it reproduces the encoding of the first 8 frames alone.
Tolerance 1e-3 (maximum absolute difference). If a check fails, the masked-tube error is not computed.

Mask tokens (log section 13, addendum of 2026-11-11). The released predictor has 10 mask tokens, of which only token 0
was trained (tokens 1-9 are exactly zero). The HF port always reads token 1, so gate G2 used an untrained token.
Every prediction-error map is therefore stored twice: with token 1 (the readout of G2, used for the reproduction
check) and with token 0 (the trained token, suffix `_t0`, the primary readout of 13B and 13C). Encodings, targets and
windows are shared; only the predictor call differs. A diagnostic without gaze data records the mean masked-tube
error of all 10 tokens on the first stride-8 window of every 6th video.

    python scripts/g2_maps.py --out runs/g2/maps
Output: <out>/stride_<s>/<video>.npz with `frames` (n,) and (n, 16, 16) float32 maps `atypicality`, `frame_diff`,
`surprise_raw`, `surprise_ctx`, `masked_tube` (token 1) and `surprise_raw_t0`, `surprise_ctx_t0`, `masked_tube_t0`
(token 0), and <out>/checks.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from surge.data import dhf1k_videos, load_rgb, read_split_file  # noqa: E402

G = 16           # token grid (256 / 16)
TUB = 8          # tubelets in a 16-frame clip
N = TUB * G * G  # 2,048 tokens
TOKENS = {1: "", 0: "_t0"}  # mask token -> key suffix; 1 = the HF port's default (as G2), 0 = the trained token
_IO = ThreadPoolExecutor(8)


def ln(t: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(t.float(), t.shape[-1:])


def masked_encode(model, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Encoder output for the tokens `ids` (B, n) only; all other tokens are removed before the first layer."""
    enc = model.encoder
    h = enc.embeddings(x)
    h = torch.gather(h, 1, ids[..., None].expand(-1, -1, h.shape[-1]))
    for layer in enc.layer:
        h = layer(h, ids)[0]
    return enc.layernorm(h)


def tube_ids(device):
    """Visible (4, 1536) and masked (4, 512) token ids of the four passes."""
    r = torch.arange(G)
    blk = ((r[:, None] // 4) % 2) * 2 + ((r[None, :] // 4) % 2)  # pass in which the token's 4x4 block is masked
    which = blk.flatten().repeat(TUB)                             # the same spatial mask in every tubelet
    ids = torch.arange(N)
    ids_m = torch.stack([ids[which == p] for p in range(4)])
    ids_v = torch.stack([ids[which != p] for p in range(4)])
    assert ids_m.shape == (4, N // 4) and ids_v.shape == (4, 3 * N // 4)
    assert torch.equal(ids_m.flatten().sort().values, ids)       # every token is masked in exactly one pass
    return ids_v.to(device), ids_m.to(device)


@contextmanager
def use_mask_token(model, j: int):
    """The HF predictor always reads mask token 1 (`mask_index=1`); put token j there for the duration of a call."""
    mt = model.predictor.embeddings.mask_tokens
    d = 1 % mt.shape[0]
    if j == d:
        yield
        return
    keep = mt.data[d].clone()
    mt.data[d] = mt.data[j]
    try:
        yield
    finally:
        mt.data[d] = keep


def tube_context(model, x: torch.Tensor, ids_v: torch.Tensor) -> torch.Tensor:
    """Encodings of the visible tokens of the four passes, scattered into full-length (4, N, D) buffers (the
    predictor gathers its context from a full-length tensor with the context ids)."""
    hv = masked_encode(model, x.expand(ids_v.shape[0], -1, -1, -1, -1), ids_v)
    buf = hv.new_zeros(ids_v.shape[0], N, hv.shape[-1])
    buf.scatter_(1, ids_v[..., None].expand(-1, -1, hv.shape[-1]), hv)
    return buf


def tube_error(model, buf: torch.Tensor, tgt_full: torch.Tensor, ids_v, ids_m, token: int = 1) -> torch.Tensor:
    """Per-token masked-tube error (N,). tgt_full: layer-normalised full-clip features (1, N, D), float32."""
    with use_mask_token(model, token):
        pred = model.predictor(encoder_hidden_states=buf, context_mask=[ids_v], target_mask=[ids_m]).last_hidden_state
    tgt = tgt_full.expand(ids_m.shape[0], -1, -1).gather(1, ids_m[..., None].expand(-1, -1, tgt_full.shape[-1]))
    err = (ln(pred) - tgt).abs().mean(-1)  # (4, 512), in the order of ids_m
    out = torch.zeros(N, device=err.device, dtype=torch.float32)
    out[ids_m.flatten()] = err.flatten().float()
    return out


def window_starts(n: int, s: int) -> list[int]:
    return list(range(0, max(n - 15 * s, 1), 8 * s))  # as scripts/g2_surprise.py


def window_idx(n: int, st: int, s: int) -> list[int]:
    return [min(st + k * s, n - 1) for k in range(16)]


def load_clip(v, idx, device, cache=None) -> torch.Tensor:
    cache = {} if cache is None else cache
    need = [j for j in dict.fromkeys(idx) if j not in cache]
    for j, im in zip(need, _IO.map(lambda j: load_rgb(v.frames[j], (256, 256)), need)):
        cache[j] = im
    x = torch.from_numpy(np.stack([cache[j] for j in idx])).permute(0, 3, 1, 2)
    return x[None].to(device)


@torch.no_grad()
def encoder_checks(model, x: torch.Tensor, tol: float) -> dict:
    ids = torch.arange(N, device=x.device)[None]
    ref = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
    d1 = (masked_encode(model, x, ids) - ref).abs().max().item()
    ref8 = model(pixel_values_videos=x[:, :8], skip_predictor=True).last_hidden_state
    d2 = (masked_encode(model, x, ids[:, : N // 2]) - ref8).abs().max().item()
    return {"dtype": str(ref.dtype), "feature_mean_abs": ref.abs().mean().item(), "tolerance": tol,
            "all_visible_max_abs_diff": d1, "first_half_max_abs_diff": d2, "passed": bool(d1 <= tol and d2 <= tol)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="configs/splits/dhf1k_holdout60.txt")
    ap.add_argument("--strides", default="8,2", help="frame strides (30 fps video): 8 ~ 4 fps, 2 = 15 fps")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None, help="smoke tests only")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/g2/maps")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    from surge.backbones import VJEPA2
    model = VJEPA2().to(device).eval().model
    vids = dhf1k_videos("D:/DHF1K", read_split_file(ROOT / a.split))[: a.max_videos]
    ids_v, ids_m = tube_ids(device)
    t0 = time.time()

    # ---- checks (float32) and mask-token diagnostic (no gaze data)
    x0 = load_clip(vids[0], window_idx(len(vids[0]), 0, 8), device)
    checks = encoder_checks(model, x0, a.tol)
    print("CHECKS", json.dumps(checks), flush=True)
    tube = checks["passed"]
    mt = model.predictor.embeddings.mask_tokens
    K = mt.shape[0]
    assert K > 1, "the token-1 readout of G2 needs at least two mask tokens"
    checks["mask_token_l2_norm"] = [float(mt[j].float().norm()) for j in range(K)]
    print("MASK TOKEN NORMS", json.dumps(checks["mask_token_l2_norm"]), flush=True)
    if tube:
        diag_v = vids[::6]
        err = np.zeros(K)
        for v in diag_v:
            x = load_clip(v, window_idx(len(v), 0, 8), device)
            with torch.no_grad(), torch.autocast(**amp):
                tgt = ln(model(pixel_values_videos=x, skip_predictor=True).last_hidden_state)
                buf = tube_context(model, x, ids_v)
                for j in range(K):
                    err[j] += tube_error(model, buf, tgt, ids_v, ids_m, token=j).mean().item()
        err /= len(diag_v)
        checks["mask_token_diagnostic"] = {"videos": [v.name for v in diag_v], "mean_error_per_token": err.tolist()}
        print("MASK TOKENS", json.dumps(checks["mask_token_diagnostic"]), flush=True)
    else:
        print("a check failed: the masked-tube error is NOT computed (section 13C)", flush=True)
    (out / "checks.json").write_text(json.dumps(checks, indent=1), newline="\n")

    # ---- maps
    ids_c = [torch.arange(N // 2, device=device)[None]]
    ids_t = [torch.arange(N // 2, N, device=device)[None]]

    def context(v, st, s):
        x = load_clip(v, window_idx(len(v), st, s), device)
        with torch.no_grad(), torch.autocast(**amp):
            return model(pixel_values_videos=x[:, :8], skip_predictor=True).last_hidden_state

    for s in [int(x) for x in a.strides.split(",")]:
        d = out / f"stride_{s}"
        d.mkdir(exist_ok=True)
        prev_ctx = None
        for vi, v in enumerate(vids):
            n = len(v)
            starts = window_starts(n, s)[: a.max_windows]
            f_out = d / f"{v.name}.npz"
            if f_out.exists():  # resume: only the last context is needed (wrong context of the next video)
                nxt = vi + 1 < len(vids) and not (d / f"{vids[vi + 1].name}.npz").exists()
                prev_ctx = context(v, starts[-1], s) if nxt else None
                continue
            rows = {"frames": [], "atypicality": [], "frame_diff": []}
            for sfx in TOKENS.values():
                for key in ("surprise_raw", "surprise_ctx") + (("masked_tube",) if tube else ()):
                    rows[key + sfx] = []
            cache = {}
            for st in starts:
                idx = window_idx(n, st, s)
                for k in [k for k in cache if k < st]:
                    del cache[k]
                x = load_clip(v, idx, device, cache)
                with torch.no_grad(), torch.autocast(**amp):
                    full = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state.float()
                    ctx = model(pixel_values_videos=x[:, :8], skip_predictor=True).last_hidden_state
                    tgt = ln(full[:, N // 2:])
                    wrong = prev_ctx if prev_ctx is not None else ctx.flip(1)
                    if tube:
                        buf = tube_context(model, x, ids_v)
                        tgt_full = ln(full)
                    e, e_w, e_t = {}, {}, {}
                    for j in TOKENS:
                        with use_mask_token(model, j):
                            e[j] = (ln(model.predictor(encoder_hidden_states=ctx, context_mask=ids_c,
                                                       target_mask=ids_t).last_hidden_state) - tgt).abs().mean(-1)
                            e_w[j] = (ln(model.predictor(encoder_hidden_states=wrong, context_mask=ids_c,
                                                         target_mask=ids_t).last_hidden_state) - tgt).abs().mean(-1)
                        if tube:
                            e_t[j] = tube_error(model, buf, tgt_full, ids_v, ids_m, token=j).view(TUB, G, G)[TUB // 2:]
                e = {j: m.float().view(4, G, G).cpu().numpy() for j, m in e.items()}
                e_w = {j: m.float().view(4, G, G).cpu().numpy() for j, m in e_w.items()}
                e_t = {j: m.cpu().numpy() for j, m in e_t.items()}
                zt = tgt.view(4, G * G, -1)
                atyp = (zt - zt.mean(1, keepdim=True)).abs().mean(-1).view(4, G, G).cpu().numpy()
                for k in range(4):
                    f = idx[8 + 2 * k]
                    if f >= n or (k and f == idx[8 + 2 * (k - 1)]):
                        continue
                    g1 = cv2.imread(v.frames[f], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    g0 = cv2.imread(v.frames[max(f - 1, 0)], cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    rows["frames"].append(f)
                    rows["atypicality"].append(atyp[k])
                    rows["frame_diff"].append(cv2.resize(np.abs(g1 - g0), (G, G), interpolation=cv2.INTER_AREA))
                    for j, sfx in TOKENS.items():
                        rows["surprise_raw" + sfx].append(e[j][k])
                        rows["surprise_ctx" + sfx].append(e[j][k] - e_w[j][k])
                        if tube:
                            rows["masked_tube" + sfx].append(e_t[j][k])
            prev_ctx = ctx
            arrs = {k: np.asarray(r, np.int32 if k == "frames" else np.float32) for k, r in rows.items()}
            tmp = d / f"{v.name}.tmp.npz"
            np.savez(tmp, **arrs)
            tmp.replace(f_out)
            print(f"stride {s} [{vi + 1}/{len(vids)}] {v.name} windows {len(starts)} frames {len(rows['frames'])} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
