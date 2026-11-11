"""Measured inference time of SalFoM and SURGE-D (research/log.md section 13D).

Models, each as it was evaluated:
  salfom    the released Model.py (github.com/mr17m/SalFoM---Video-Saliency-Prediction) with the released DHF1K
            weights; 16 frames at 224x224 in, one saliency map out
  surge_l   SURGE-D with the frozen V-JEPA 2 ViT-L encoder (runs/main_b1_s0); 16 frames at 256x256 in, 16 maps out
  surge_g   the same with the ViT-g encoder (runs/vitg_s0)
Protocol: batch 1 and batch 8; bfloat16 autocast; random input already on the GPU (frame decoding, resizing and the
final blur are excluded for every model); 10 warm-up passes, then 50 timed passes, each closed by
torch.cuda.synchronize(); median and inter-quartile range; peak GPU memory of the timed model.
Per output frame: SalFoM = pass / B; SURGE-D offline = pass / (16 B); SURGE-D causal = pass / B.
The encoder alone is timed the same way (descriptive).

SalFoM's repository is appended to the END of sys.path, so that nothing in it can shadow an installed module, and its
weights are read with weights_only=True.

    python -I scripts/time_inference.py --out runs/review/timing.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surge.losses import log_softmax2d  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, x, device, warm: int, n: int) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warm):
        fn(x)
    sync(device)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(x)
        sync(device)
        ts.append((time.perf_counter() - t) * 1000.0)
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
    return {"median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3), "peak_mem_gb": float(mem), "passes": n}


def load_salfom(repo: str, weights: str, device):
    sys.path.append(repo)
    from Model import net  # SalFoM's released model definition
    m = net()
    st = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(st, strict=False)
    info = {"parameters": sum(p.numel() for p in m.parameters()), "missing_keys": len(missing),
            "unexpected_keys": len(unexpected)}
    return m.to(device).eval(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salfom-repo", default="D:/Research/baselines/SalFoM")
    ap.add_argument("--salfom-weights", default="D:/Research/baselines/weights/SalFoM_DHF1K.pth")
    ap.add_argument("--models", default="salfom,surge_l,surge_g")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    runs = {"surge_l": "runs/main_b1_s0", "surge_g": "runs/vitg_s0"}
    res = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__,
           "autocast": "bfloat16" if amp["enabled"] else "off", "warm_up": a.warm, "timed": a.n, "models": {}}
    g = torch.Generator(device="cpu").manual_seed(0)
    for name in a.models.split(","):
        if name == "salfom":
            model, info = load_salfom(a.salfom_repo, a.salfom_weights, device)
            shape = (3, 16, 224, 224)  # B, C, T, H, W
            enc = model.encoder
            per_pass = 1

            def full(x, model=model):
                with torch.no_grad(), torch.autocast(**amp):
                    return model(x)
        else:
            from evaluate import load_model
            model, cfg = load_model(ROOT / runs[name], "best.pt", device)
            info = {"run": runs[name], "backbone": cfg["model"]["backbone"],
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "decoder_parameters": sum(p.numel() for p in model.decoder.parameters())}
            shape = (16, 3, 256, 256)  # B, T, C, H, W
            enc = model.backbone
            per_pass = 16

            def full(x, model=model):
                with torch.no_grad(), torch.autocast(**amp):
                    return log_softmax2d(model(x)["logits"].float()).exp()

        def encoder(x, enc=enc):
            with torch.no_grad(), torch.autocast(**amp):
                return enc(x)

        info["input"] = list(shape)
        info["maps_per_pass"] = per_pass
        info["batch"] = {}
        for B in [int(b) for b in a.batches.split(",")]:
            x = torch.randn((B, *shape), generator=g).to(device)
            try:
                y = full(x)
                y = y[0] if isinstance(y, (tuple, list)) else y
                r = bench(full, x, device, a.warm, a.n)
                r["output_shape"] = list(y.shape)
                r["ms_per_output_frame"] = r["median_ms"] / (B * per_pass)
                if per_pass > 1:
                    r["ms_per_output_frame_causal"] = r["median_ms"] / B
                r["encoder"] = bench(encoder, x, device, a.warm, a.n)
            except torch.OutOfMemoryError:  # reported, not hidden: this batch size has no timing for this model
                info["batch"][str(B)] = {"out_of_memory": True}
                print(f"{name:8s} B={B}: out of memory", flush=True)
                del x
                torch.cuda.empty_cache()
                continue
            info["batch"][str(B)] = r
            print(f"{name:8s} B={B}: pass {r['median_ms']:8.1f} ms [{r['q1_ms']:.1f}, {r['q3_ms']:.1f}]  "
                  f"encoder {r['encoder']['median_ms']:8.1f} ms  per output frame {r['ms_per_output_frame']:8.2f} ms"
                  + (f" (causal {r['ms_per_output_frame_causal']:.1f} ms)" if per_pass > 1 else "")
                  + f"  peak {r['peak_mem_gb']:.2f} GB  out {r['output_shape']}", flush=True)
            del x, y
        res["models"][name] = info
        del model, enc, full, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    m = res["models"]
    if "salfom" in m:
        res["ratios"] = {}
        for B in m["salfom"]["batch"]:
            s = m["salfom"]["batch"][B].get("ms_per_output_frame")
            row = {}
            for k in ("surge_l", "surge_g"):
                if s is not None and k in m and "ms_per_output_frame" in m[k]["batch"].get(B, {}):
                    row[f"salfom/{k}_offline"] = s / m[k]["batch"][B]["ms_per_output_frame"]
                    row[f"{k}_causal/salfom"] = m[k]["batch"][B]["ms_per_output_frame_causal"] / s
            res["ratios"][B] = row
            print(f"B={B}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()), flush=True)
        # reading of section 13D and its addendum: refuted if the ratio is below 2 at either batch size
        r = [row["salfom/surge_g_offline"] for row in res["ratios"].values() if "salfom/surge_g_offline" in row]
        if r:
            res["offline_statement_refuted"] = bool(min(r) < 2.0)
            res["offline_statement_batches_measured"] = len(r)
            print("offline efficiency statement (SalFoM per output frame at least 2x SURGE-D ViT-g offline): "
                  + ("REFUTED" if res["offline_statement_refuted"] else "holds") + f" (ratios {r})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")


if __name__ == "__main__":
    main()
