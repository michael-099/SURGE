"""Evaluate SURGE-D checkpoints, or score saved prediction maps.

Model:   python -u scripts/evaluate.py --run runs/g1_vjepa21 [--ckpt best.pt] --split val --metrics all --sigma 0
Maps:    python -u scripts/evaluate.py --maps D:/some/maps --split val       (maps/<video>/<frame>.png)
Ensemble:python -u scripts/evaluate.py --run runs/a --run runs/b ...           (averages probabilities)
Writes a JSON with the mean (over videos of per-video means) and per-video scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from surge.data import dhf1k_videos, parse_stavis, read_split_file, stavis_split_videos, ucf_videos  # noqa: E402
from surge.infer import other_fixations, predict_video, to_native  # noqa: E402
from surge.metrics import all_metrics  # noqa: E402
from surge.data import load_gray  # noqa: E402

ALL = ("cc", "nss", "sim", "kld", "auc_j", "sauc")


def get_videos(dataset: str, split: str, root: str | None):
    if dataset == "dhf1k":
        f = {"val": "dhf1k_val100", "holdout": "dhf1k_holdout60", "train": "dhf1k_train540"}[split]
        return dhf1k_videos(root or "D:/DHF1K", read_split_file(ROOT / "configs/splits" / f"{f}.txt"))
    if dataset == "ucf":
        vids = ucf_videos(root or "D:/Datasets/UCF_Sports", "test" if split == "val" else "train")
        if split in ("holdout", "train"):
            hold = set(open(ROOT / "configs/splits/ucf_holdout10.txt").read().split())
            vids = [v for v in vids if (v.name in hold) == (split == "holdout")]
        return vids
    if parse_stavis(dataset):
        return stavis_split_videos(dataset, split, ROOT / "configs/splits")
    if dataset == "hollywood_vig":
        from surge.data import hollywood_vig_videos
        return hollywood_vig_videos(split, ROOT / "configs/splits")
    raise ValueError(dataset)


def score_frame(args):
    pred, mp, fp, of, which, skip_empty = args
    gt = load_gray(mp).astype(np.float64)
    fix = load_gray(fp)
    if skip_empty and not (fix > 0).any():
        return {k: float("nan") for k in which if k != "sauc" or of is not None}
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    if of is not None and of.shape != gt.shape:
        of = cv2.resize(of, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return all_metrics(pred, gt, fix, of, which)


def load_model(run: Path, ckpt: str, device):
    from surge.backbones import add_lora, build_backbone
    from surge.model import SurgeD
    cfg = json.loads((run / "config.json").read_text())
    bb = build_backbone(cfg["model"]["backbone"])
    st = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    if "lora" in st:
        add_lora(bb, r=cfg["train"]["lora_r"], alpha=cfg["train"]["lora_alpha"])
        bb.load_state_dict(st["lora"], strict=False)
    m = SurgeD(bb, use_scm=cfg["model"]["scm"], prior=cfg["model"]["prior"], aux=cfg["model"]["aux"],
               c=cfg["model"]["channels"], audio=cfg["model"].get("audio", False))
    m.decoder.load_state_dict(st["decoder"])
    return m.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--maps", default=None)
    ap.add_argument("--dataset", default="dhf1k")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--metrics", default="all")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--audio-rotate", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--save-maps", default=None, help="folder to write 8-bit PNG maps at native resolution")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    which = ALL if a.metrics == "all" else tuple(a.metrics.split(","))
    videos = get_videos(a.dataset, a.split, a.root)[: a.max_videos]
    if a.audio_rotate:  # control (section 11D): video i gets the audio of video i+1
        aud = [v.audio for v in videos]
        for i, v in enumerate(videos):
            v.audio = aud[(i + 1) % len(videos)]
        print("audio rotated:", [(v.name, Path(v.audio).stem if v.audio else None) for v in videos], flush=True)
    sauc = other_fixations(videos) if "sauc" in which else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(ROOT / r, a.ckpt, device) for r in a.run]
    t0 = time.time()
    per_video, nframes = {}, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for vi, v in enumerate(videos):
            n = len(v)
            if a.maps:
                preds = {}
                for f in range(0, n, a.frame_step):
                    p = cv2.imread(str(Path(a.maps) / v.name / f"{f + 1:04d}.png"), cv2.IMREAD_GRAYSCALE)
                    if p is None:
                        raise IOError(f"missing map {v.name}/{f + 1:04d}.png")
                    preds[f] = p.astype(np.float64)
            else:
                acc = None
                for m, cfg in models:
                    d = cfg["data"]
                    outs = [dict(predict_video(m, v, d["T"], d["stride"], tuple(d["size"]), device, batch=a.batch))]
                    if a.flip_tta:
                        outs.append({f: mm[:, ::-1] for f, mm in predict_video(_FlipModel(m), v, d["T"], d["stride"],
                                                                              tuple(d["size"]), device, batch=a.batch)})
                    for o in outs:
                        if acc is None:
                            acc = {f: o[f].astype(np.float64) for f in o}
                        else:
                            for f in acc:
                                acc[f] += o[f]
                hw = load_gray(v.maps[0]).shape if v.maps else None
                preds = {f: to_native(acc[f], hw, a.sigma) for f in range(0, n, a.frame_step)}
                if a.save_maps:
                    od = Path(a.save_maps) / v.name
                    od.mkdir(parents=True, exist_ok=True)
                    for f in range(n):
                        pm = to_native(acc[f], hw, a.sigma)
                        cv2.imwrite(str(od / f"{f + 1:04d}.png"), np.round(255 * pm / pm.max()).astype(np.uint8))
            jobs = [(preds[f], v.maps[f], v.fixations[f], sauc[v.name] if sauc else None, which, v.skip_empty)
                    for f in sorted(preds)]
            res = list(ex.map(score_frame, jobs, chunksize=8))
            nframes += len(res)
            per_video[v.name] = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
            print(f"[{vi + 1}/{len(videos)}] {v.name} " + " ".join(f"{k}={x:.4f}" for k, x in per_video[v.name].items()),
                  flush=True)
    keys = next(iter(per_video.values())).keys()
    mean = {k: float(np.mean([pv[k] for pv in per_video.values()])) for k in keys}
    out = {"mean": mean, "per_video": per_video, "frames": nframes, "videos": len(videos), "seconds": time.time() - t0,
           "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), newline="\n")
    print("MEAN " + " ".join(f"{k}={x:.4f}" for k, x in mean.items()), flush=True)


class _FlipModel(torch.nn.Module):
    """Runs the model on horizontally flipped frames (the caller flips the output back)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    @property
    def uses_audio(self) -> bool:
        return getattr(self.m, "uses_audio", False)

    def forward(self, x, surprise=None, audio=None):
        if audio is not None:
            return self.m(x.flip(-1), surprise=surprise, audio=audio)
        return self.m(x.flip(-1), surprise=surprise)


if __name__ == "__main__":
    main()
