"""Test-time probes of SURGE-D (research/log.md section 15): temporal order, temporal stride, input size, tiling of
the multi-frame output, input perturbations, scores per output position and flip consistency.

With its defaults it reproduces scripts/evaluate.py (holdout, every 5th frame, sigma 0); that is checked before any
other result of it is used (section 15A).

    python -u scripts/probe_eval.py --run runs/vitg_s0 --split holdout --out runs/review/inv/x.json
        [--temporal orig|reverse|shuffle|static] [--stride 2] [--size 256]
        [--tiling default|all16|central4|overlap|dense]
        [--perturb none|noop|noise:S|blur:S|jpeg:Q|contrast:F|bright:F|gray|dup]
        [--positions] [--flip-consistency] [--per-frame]
        [--stream] [--variant NAME]     (section 16: the decoder's stream buffer; a scripts/train_multi.py decoder)

Only frames on the scoring grid (every --frame-step-th frame) are predicted and scored. `noop` runs the float
perturbation path without changing the frame (it measures that path's own effect against `none`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from surge.data import IMAGENET_MEAN, IMAGENET_STD, load_gray, load_rgb  # noqa: E402
from surge.infer import _score_one  # noqa: E402
from surge.losses import log_softmax2d  # noqa: E402
from surge.metrics import cc as cc_metric, nss as nss_metric  # noqa: E402

_IO = ThreadPoolExecutor(8)
METRICS = ("cc", "nss", "sim", "kld")


# ------------------------------------------------------------------------------------------------ model
def load_model(run: Path, ckpt: str, device, img: int | None, variant: str | None = None):
    """variant (section 16): load <run>/<variant>/best.pt of a scripts/train_multi.py run, with its decoder options."""
    from surge.backbones import build_backbone
    from surge.model import Decoder, SurgeD
    cfg = json.loads((run / "config.json").read_text())
    md = cfg["model"]
    if md.get("expert") or md.get("audio") or md.get("pgate") or md.get("ssl") or md.get("scm"):
        raise ValueError("probe_eval supports the visual SURGE-D models only")
    size = img or cfg["data"]["size"][0]
    bb = build_backbone(md["backbone"], img=size)
    if variant:
        st = torch.load(run / variant / "best.pt", map_location="cpu", weights_only=False)
        if st["spec"]["kind"] != "decoder":
            raise ValueError("only decoder variants are supported")
        kw = {"c": 256, "prior": True, "aux": True, **st["spec"]["dec"]}
        c = kw.pop("c")
        m = SurgeD(bb, prior=kw["prior"], aux=kw["aux"], c=c)
        m.decoder = Decoder(bb.embed_dim, c, **kw)
        m.decoder.load_state_dict(st["module"])
        return m.to(device).eval(), cfg, size
    st = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    if "lora" in st:
        raise ValueError("LoRA checkpoints are not supported here")
    m = SurgeD(bb, prior=md["prior"], aux=md["aux"], c=md["channels"])
    m.decoder.load_state_dict(st["decoder"])
    return m.to(device).eval(), cfg, size


# ------------------------------------------------------------------------------------------------ frames
def make_loader(perturb: str, size: int):
    """Frame loader with an optional perturbation applied at native resolution before resizing."""
    kind, _, val = perturb.partition(":")
    if kind in ("none", "dup"):  # dup acts on the window indices, not on the pixels
        return lambda path: load_rgb(path, (size, size))
    v = float(val) if val else None

    def load(path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(path)
        if kind == "jpeg":
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(v)])
            if not ok:
                raise IOError(f"jpeg encode failed: {path}")
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if kind == "noise":  # independent per frame, reproducible from the path
            seed = int.from_bytes(hashlib.md5(path.encode()).digest()[:4], "little")
            x = np.clip(x + np.random.default_rng(seed).normal(0.0, v, x.shape).astype(np.float32), 0.0, 1.0)
        elif kind == "blur":
            x = cv2.GaussianBlur(x, (0, 0), v)
        elif kind == "contrast":
            m = float(x.mean())
            x = np.clip((x - m) * v + m, 0.0, 1.0)
        elif kind == "bright":
            x = np.clip(x * v, 0.0, 1.0)
        elif kind == "gray":
            g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            x = np.repeat(g[..., None], 3, -1)
        elif kind not in ("jpeg", "noop"):
            raise ValueError(perturb)
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
        return ((x - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
    return load


# ------------------------------------------------------------------------------------------------ shots
def load_shots(kind: str, video_name: str, n: int) -> list[tuple[int, int]]:
    """Shots (first, last frame, inclusive) of a video from TransNet V2 scene files; one shot if none is found."""
    import csv
    if kind == "dhf1k":
        p = Path("D:/DHF1K/transnet_scenes") / f"{int(video_name):03d}.AVI.scenes.txt"
    elif kind == "diem":
        ids = {r["name"]: int(r["id"]) for r in csv.DictReader(open("D:/Datasets/DIEM/graft/ids.csv"))}
        p = Path("D:/Datasets/DIEM/graft/transnet_scenes") / f"{ids[video_name]:03d}.scenes.txt"
    elif kind == "hollywood":
        p = Path("D:/Datasets/Hollywood2/surge/transnet_scenes") / f"{video_name}.scenes.txt"
    elif kind == "avimos":
        p = Path("D:/Datasets/AViMoS/transnet_scenes") / f"{int(video_name):04d}.scenes.txt"
    else:
        raise ValueError(kind)
    if not p.exists():
        raise FileNotFoundError(p)
    sh = sorted(tuple(int(x) for x in line.split()[:2]) for line in open(p) if line.strip())
    out = []
    for i, (a, b) in enumerate(sh):  # make the shots tile [0, n-1]
        a = 0 if i == 0 else a
        b = n - 1 if i == len(sh) - 1 else min(b, sh[i + 1][0] - 1)
        if a <= b and a < n:
            out.append((a, min(b, n - 1)))
    return out or [(0, n - 1)]


def shot_of(shots, f):
    for a, b in shots:
        if a <= f <= b:
            return a, b
    return shots[-1]


# ------------------------------------------------------------------------------------------------ windows
def windows(n: int, T: int, stride: int, tiling: str, scored: list[int], shots=None):
    """Yield (clamped indices, raw indices, [(frame, fractional output position)]). 'default' is exactly
    surge.data.inference_windows; only frames in `scored` are assigned. With `shots` (dense tiling only), each
    window is clamped to the shot of its frame instead of to the video."""
    want = set(scored)
    if tiling == "dense":  # one window per scored frame, which sits at position T/2
        for f in scored:
            raw = [f + (k - T // 2) * stride for k in range(T)]
            lo, hi = shot_of(shots, f) if shots else (0, n - 1)
            yield [min(max(j, lo), hi) for j in raw], raw, [(f, float(T // 2))]
        return
    if shots:
        raise ValueError("shot clamping is defined for the dense tiling")
    k0, k1, hop = {
        "default": (T // 4, 3 * T // 4, (T // 2) * stride),
        "all16": (0, T, T * stride),
        "central4": (3 * T // 8, 5 * T // 8, (T // 4) * stride),
        "overlap": (T // 4, 3 * T // 4, (T // 4) * stride),
    }[tiling]
    span = (k1 - k0) * stride
    start = -k0 * stride
    while start + k0 * stride < n:
        raw = [start + k * stride for k in range(T)]
        assign = [(f, (f - start) / stride) for f in range(start + k0 * stride, min(start + k0 * stride + span, n))
                  if f in want]
        yield [min(max(j, 0), n - 1) for j in raw], raw, assign
        start += hop


def static_windows(n: int, T: int, scored: list[int]):
    for f in scored:
        yield [f] * T, [f] * T, [(f, -1.0)]


# ------------------------------------------------------------------------------------------------ scoring
def _score_pos(args):
    """CC and NSS of one output position's map against the frame it shows."""
    m, mp, fp, skip_empty = args
    gt = load_gray(mp).astype(np.float64)
    fix = load_gray(fp)
    pred = cv2.resize(m.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = {"cc": cc_metric(pred, gt)}
    out["nss"] = nss_metric(pred, fix) if (fix > 0).any() else float("nan")
    if skip_empty and not (fix > 0).any():
        out = {"cc": float("nan"), "nss": float("nan")}
    return out


@torch.no_grad()
def run_video(model, v, T, stride, size, device, a, loader, rng_tag: str):
    """Returns ({frame: summed map}, {frame: count}, positions list, {frame: flipped map sum})."""
    n = len(v)
    scored = list(range(0, n, a.frame_step))
    shots = load_shots(a.shots, v.name, n) if a.shots else None
    if a.temporal == "static":
        wins = list(static_windows(n, T, scored))
    else:
        wins = list(windows(n, T, stride, a.tiling, scored, shots))
    acc, cnt, accf = {}, {}, {}
    positions = []
    cache = {}
    state = {}  # --stream (section 16): the decoder's buffers from the previous window of this video
    for b0 in range(0, len(wins), a.batch):
        chunk = wins[b0:b0 + a.batch]
        idxs = []
        for idx, raw, _ in chunk:
            if a.perturb == "dup":  # every other sampled frame repeats the one before it
                idx = [idx[k - (k % 2)] for k in range(T)]
            idxs.append(idx)
        need = sorted({j for idx in idxs for j in idx if j not in cache})
        for j, im in zip(need, _IO.map(lambda j: loader(v.frames[j]), need)):
            cache[j] = im
        lo = min(min(i) for i in idxs)
        for k in [k for k in cache if k < lo]:
            del cache[k]
        perms = []
        xs = []
        for wi, idx in enumerate(idxs):
            clip = np.stack([cache[j] for j in idx])
            if a.temporal == "reverse":
                perm = np.arange(T)[::-1].copy()
            elif a.temporal == "shuffle":
                h = hashlib.md5(f"{rng_tag}/{v.name}/{chunk[wi][1][0]}".encode()).digest()
                perm = np.random.default_rng(int.from_bytes(h[:4], "little")).permutation(T)
            else:
                perm = np.arange(T)
            perms.append(perm)
            xs.append(clip[perm])
        x = torch.from_numpy(np.stack(xs)).permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
        outs = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if a.stream:  # windows in order, one at a time, each with the previous window's buffers
                taps = model.backbone(x)
                taps = [t.float() if t.dtype != torch.bfloat16 else t for t in taps]
                lgs = []
                for bi in range(x.shape[0]):
                    st = {"in": state, "out": {}}
                    lgs.append(model.decoder([t[bi:bi + 1] for t in taps], T_out=T, stream=st)["logits"])
                    state = st["out"]
                outs.append(torch.cat(lgs))
            else:
                outs.append(model(x)["logits"])
            if a.flip_consistency:
                outs.append(model(x.flip(-1))["logits"].flip(-1))
        ps = [log_softmax2d(o.float()).exp().cpu().numpy() for o in outs]  # B,T,h,w
        for bi, (idx, raw, assign) in enumerate(chunk):
            inv = np.argsort(perms[bi])  # output position k shows frame idx[perm[k]]: back to the original order
            for pi, p in enumerate(ps):
                po = p[bi][inv]
                tgt_acc = acc if pi == 0 else accf
                for f, q in assign:
                    if q < 0:
                        m = po[T // 4: T // 4 + T // 2].mean(0)
                    else:
                        a0 = int(np.floor(q))
                        w = q - a0
                        m = po[a0] if w == 0 else (1 - w) * po[a0] + w * po[min(a0 + 1, T - 1)]
                    if f in tgt_acc:
                        tgt_acc[f] = tgt_acc[f] + m
                    else:
                        tgt_acc[f] = m.astype(np.float64)
                    if pi == 0:
                        cnt[f] = cnt.get(f, 0) + 1
            if a.positions:
                po = ps[0][bi][inv]
                for k in range(T):
                    j = raw[k]
                    if 0 <= j < n:
                        positions.append((k, j, po[k].astype(np.float32)))
    return acc, cnt, positions, accf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--dataset", default="dhf1k")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--temporal", default="orig", choices=["orig", "reverse", "shuffle", "static"])
    ap.add_argument("--stride", type=int, default=None, help="temporal stride (default: the model's)")
    ap.add_argument("--size", type=int, default=None, help="input size (default: the model's)")
    ap.add_argument("--tiling", default="default", choices=["default", "all16", "central4", "overlap", "dense"])
    ap.add_argument("--perturb", default="none")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--flip-consistency", action="store_true")
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--shots", default=None, choices=["dhf1k", "diem", "hollywood", "avimos"],
                    help="dense tiling only: clamp each window to the shot of its frame (TransNet V2 scenes)")
    ap.add_argument("--save-small", default=None,
                    help="folder: per video an .npz with the scored frames and their maps at decoder resolution "
                         "(float16), for analyses that need the maps (section 15K)")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stream", action="store_true",
                    help="section 16: all16 tiling only; the decoder carries its stream buffers from window to window")
    ap.add_argument("--variant", default=None,
                    help="section 16: score <run>/<variant>/best.pt of a scripts/train_multi.py run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.temporal == "static" and (a.tiling != "default" or a.positions or a.flip_consistency):
        raise ValueError("static clips have their own windows")
    if a.positions and a.tiling not in ("default", "all16"):
        raise ValueError("--positions is defined for the default and all16 tilings")
    if a.stream and (a.tiling != "all16" or a.temporal != "orig" or a.flip_consistency or a.perturb == "dup"):
        raise ValueError("--stream is defined for the all16 tiling in the original order")
    from scripts.evaluate import get_videos
    videos = get_videos(a.dataset, a.split, a.root)[: a.max_videos]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, size = load_model(ROOT / a.run, a.ckpt, device, a.size, a.variant)
    T = cfg["data"]["T"]
    stride = a.stride or cfg["data"]["stride"]
    loader = make_loader(a.perturb, size)
    t0 = time.time()
    per_video, per_frame, pos_rows, flip_rows = {}, {}, [], {}
    passes = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for vi, v in enumerate(videos):
            acc, cnt, positions, accf = run_video(model, v, T, stride, size, device, a, loader, "shuffle0")
            frames = sorted(acc)
            if a.save_small:
                Path(a.save_small).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(a.save_small) / f"{v.name}.npz", frames=np.array(frames),
                                    maps=np.stack([acc[f] / cnt[f] for f in frames]).astype(np.float16))
            jobs = [(acc[f] / cnt[f], v.maps[f], v.fixations[f], None, a.sigma, METRICS, v.skip_empty) for f in frames]
            res = list(ex.map(_score_one, jobs, chunksize=8))
            per_video[v.name] = {k: float(np.nanmean([r[k] for r in res])) for k in METRICS}
            if a.per_frame:
                per_frame[v.name] = {"frames": frames, **{k: [r[k] for r in res] for k in METRICS}}
            if a.flip_consistency:
                fj = [(accf[f] / cnt[f], v.maps[f], v.fixations[f], None, a.sigma, METRICS, v.skip_empty) for f in frames]
                fres = list(ex.map(_score_one, fj, chunksize=8))
                between = [cc_metric(acc[f] / cnt[f], accf[f] / cnt[f]) for f in frames]
                flip_rows[v.name] = {"cc_between": float(np.mean(between)),
                                     **{f"flipped_{k}": float(np.nanmean([r[k] for r in fres])) for k in METRICS}}
            if a.positions:
                pj = [(m, v.maps[j], v.fixations[j], v.skip_empty) for _, j, m in positions]
                pres = list(ex.map(_score_pos, pj, chunksize=32))
                pos_rows += [(v.name, k, j, r["cc"], r["nss"]) for (k, j, _), r in zip(positions, pres)]
            passes += sum(1 for _ in (static_windows(len(v), T, list(range(0, len(v), a.frame_step)))
                                      if a.temporal == "static" else
                                      windows(len(v), T, stride, a.tiling, list(range(0, len(v), a.frame_step)),
                                              load_shots(a.shots, v.name, len(v)) if a.shots else None)))
            print(f"[{vi + 1}/{len(videos)}] {v.name} " + " ".join(f"{k}={x:.4f}" for k, x in per_video[v.name].items()),
                  flush=True)
    mean = {k: float(np.mean([pv[k] for pv in per_video.values()])) for k in METRICS}
    nframes_total = sum(len(v) for v in videos)
    out = {"mean": mean, "per_video": per_video, "frames": sum(len(range(0, len(v), a.frame_step)) for v in videos),
           "videos": len(videos), "seconds": time.time() - t0, "args": vars(a),
           "encoder_passes": passes, "passes_per_video_frame": passes / max(nframes_total, 1)}
    if a.per_frame:
        out["per_frame"] = per_frame
    if a.flip_consistency:
        out["flip"] = {"per_video": flip_rows,
                       "mean": {k: float(np.mean([r[k] for r in flip_rows.values()])) for k in next(iter(flip_rows.values()))}}
    if a.positions:
        rows = np.array([(k, c, s) for _, k, _, c, s in pos_rows], dtype=np.float64)
        out["positions"] = {
            "by_position": {int(k): {"cc": float(np.nanmean(rows[rows[:, 0] == k, 1])),
                                     "nss": float(np.nanmean(rows[rows[:, 0] == k, 2])),
                                     "n": int((rows[:, 0] == k).sum())} for k in range(T)},
            "rows": [[vn, int(k), int(j), float(c), float(s)] for vn, k, j, c, s in pos_rows],
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), newline="\n")
    print("MEAN " + " ".join(f"{k}={x:.4f}" for k, x in mean.items()), f"passes/frame={out['passes_per_video_frame']:.4f}",
          f"({out['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
