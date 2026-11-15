"""Cost profile of SURGE-D (research/log.md section 15A, A3).

Protocol of 13D (scripts/time_inference.py): random input already on the GPU, bfloat16 autocast, 10 warm-up passes,
then 50 timed passes, median; peak GPU memory. FLOPs with torch.utils.flop_counter.FlopCounterMode (as
runs/thesis/cost.json), one clip.
  encoder x input:  ViT-g at 192, 256, 384 pixels with 16 frames, and at 256 pixels with 32 frames; ViT-L at 256
  decoders:         R and the variants of 15C (V1-V6) on ViT-g taps: parameters, FLOPs, forward time (batch 8),
                    training step (forward + backward, batch 8) time and peak memory
  training step:    encoder forward (no gradient) + decoder R forward and backward, batch 8, at 256 and 384 pixels
Per output frame (offline): pass / (B x T/2 x stride) at the default tiling (hop = T/2 x stride frames per window).

    python -u scripts/inv_cost.py --out runs/review/inv/a3_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from scripts.time_inference import bench  # noqa: E402
from surge.backbones import build_backbone  # noqa: E402
from surge.losses import saliency_loss  # noqa: E402
from surge.model import Decoder, LinearReadout, central  # noqa: E402


def flops(fn, *args) -> float:
    with FlopCounterMode(display=False) as fc:
        fn(*args)
    return fc.get_total_flops() / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "encoders": {}, "decoders": {},
           "train_step": {}}

    def enc_fn(bb):
        def f(x):
            with torch.no_grad(), torch.autocast(**amp):
                return bb(x)
        return f

    # ---------------------------------------------------------------- encoders at several input sizes
    for name, img, T in [("vjepa2_g", 192, 16), ("vjepa2_g", 256, 16), ("vjepa2_g", 384, 16), ("vjepa2_g", 256, 32),
                         ("vjepa2_l", 256, 16)]:
        bb = build_backbone(name, img=img).to(dev).eval()
        f = enc_fn(bb)
        key = f"{name}_{img}px_{T}f"
        r = {"params_M": sum(p.numel() for p in bb.parameters()) / 1e6, "img": img, "T": T,
             "tflops_per_clip": flops(f, torch.randn(1, T, 3, img, img, generator=g).to(dev)),
             "output_frames_per_pass": T // 2 * 2}
        for B in (1, 8):
            x = torch.randn(B, T, 3, img, img, generator=g).to(dev)
            try:
                b = bench(f, x, dev, a.warm, a.n)
                b["ms_per_output_frame"] = b["median_ms"] / (B * r["output_frames_per_pass"])
                r[f"batch{B}"] = b
            except torch.OutOfMemoryError:
                r[f"batch{B}"] = {"out_of_memory": True}
                torch.cuda.empty_cache()
            del x
        r["tflops_per_output_frame"] = r["tflops_per_clip"] / r["output_frames_per_pass"]
        res["encoders"][key] = r
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("batch")},
              {k: round(v["median_ms"], 1) for k, v in r.items() if k.startswith("batch") and "median_ms" in v}, flush=True)
        del bb, f
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- decoders on ViT-g taps (random features)
    D, T = 1408, 16
    variants = {"R": {}, "V1": None, "V2": {"tap_mode": "last"}, "V3": {"temporal": False}, "V4": {"topdown": False},
                "V5": {"c": 64}, "V6": {"prior": False, "aux": False}}
    for B in (1, 8):
        pass
    for name, kw in variants.items():
        if kw is None:
            m = LinearReadout(4 * D).to(dev)
        else:
            kw = {"c": 256, **kw}
            c = kw.pop("c")
            m = Decoder(D, c, **kw).to(dev)
        r = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6}
        taps1 = [torch.randn(1, D, 8, 16, 16, generator=g).to(dev).bfloat16() for _ in range(4)]
        m.eval()

        def fwd(t, m=m):
            with torch.no_grad(), torch.autocast(**amp):
                return m(t, T_out=T)
        r["tflops_per_clip"] = flops(fwd, taps1)
        taps8 = [torch.randn(8, D, 8, 16, 16, generator=g).to(dev).bfloat16() for _ in range(4)]
        r["forward_batch8"] = bench(fwd, taps8, dev, a.warm, a.n)
        m.train()
        y = torch.softmax(torch.randn(8, T // 2, 64 * 64, generator=g), -1).reshape(8, T // 2, 64, 64).to(dev)
        fx = (torch.rand(8, T // 2, 64, 64, generator=g) > 0.999).float().to(dev)

        def step(t, m=m):
            with torch.autocast(**amp):
                out = m(t, T_out=T)
            loss, _ = saliency_loss(central(out["logits"], T), y, fx)
            for k in [k for k in out if k.startswith("aux")]:
                loss = loss + 0.25 * saliency_loss(central(out[k], T), y, fx, 1.0, 0.0)[0]
            loss.backward()
            m.zero_grad(set_to_none=True)
        r["train_step_batch8"] = bench(step, taps8, dev, a.warm, a.n)
        res["decoders"][name] = r
        print(name, round(r["params_M"], 3), "M", round(r["tflops_per_clip"], 4), "TFLOPs/clip",
              round(r["forward_batch8"]["median_ms"], 1), "ms fwd", round(r["train_step_batch8"]["median_ms"], 1),
              "ms step", round(r["train_step_batch8"]["peak_mem_gb"], 2), "GB", flush=True)
        del m, taps1, taps8
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- full training step (frozen encoder + R)
    for img in (256, 384):
        bb = build_backbone("vjepa2_g", img=img).to(dev).eval()
        for p in bb.parameters():
            p.requires_grad_(False)
        m = Decoder(D, 256).to(dev).train()
        go = img // 16 * 4
        y = torch.softmax(torch.randn(8, T // 2, go * go, generator=g), -1).reshape(8, T // 2, go, go).to(dev)
        fx = (torch.rand(8, T // 2, go, go, generator=g) > 0.999).float().to(dev)

        def step(x, bb=bb, m=m, y=y, fx=fx):
            with torch.no_grad(), torch.autocast(**amp):
                taps = bb(x)
            with torch.autocast(**amp):
                out = m(taps, T_out=T)
            loss, _ = saliency_loss(central(out["logits"], T), y, fx)
            for k in [k for k in out if k.startswith("aux")]:
                loss = loss + 0.25 * saliency_loss(central(out[k], T), y, fx, 1.0, 0.0)[0]
            loss.backward()
            m.zero_grad(set_to_none=True)
        x = torch.randn(8, T, 3, img, img, generator=g).to(dev)
        try:
            res["train_step"][f"{img}px"] = bench(step, x, dev, 3, 20)
        except torch.OutOfMemoryError:
            res["train_step"][f"{img}px"] = {"out_of_memory": True}
        print("train step", img, res["train_step"][f"{img}px"], flush=True)
        del bb, m, x
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
