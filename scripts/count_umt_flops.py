"""FLOPs of the UMT-L/16 encoder (the backbone family of SalFoM) on one 16 x 224 x 224 clip, on the CPU.

SalFoM predicts one frame per encoder pass (sliding window), so this is its encoder cost per output frame.

Correction (research/log.md, 13D results): the first version of this script reported 1.899 TFLOPs. The counter has
no rule for the CPU kernel of F.scaled_dot_product_attention, which surge/umt_vit.py calls, so the two matrix
products of every attention layer were missing from that number, while the counts of the V-JEPA 2 encoders
(scripts/thesis_qualitative.py) include them. The attention is now written out for the count; the old number is
stored beside the corrected one.
    python scripts/count_umt_flops.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from surge.backbones import UMT  # noqa: E402


def explicit_attention(q, k, v):
    """The function of F.scaled_dot_product_attention (no mask, no dropout), written out so that the counter sees
    both matrix products."""
    a = (q * q.shape[-1] ** -0.5) @ k.transpose(-2, -1)
    return a.softmax(-1) @ v


torch.set_grad_enabled(False)
m = UMT().eval()
x = torch.randn(1, 16, 3, 224, 224)
with FlopCounterMode(display=False) as fc:
    ref = m(x)[-1]
fused = fc.get_total_flops()  # attention products not counted

F.scaled_dot_product_attention = explicit_attention
with FlopCounterMode(display=False) as fc:
    taps = m(x)
total = fc.get_total_flops()

n = int(taps[-1].shape[2] * taps[-1].shape[3] * taps[-1].shape[4])
depth = len(m.enc.blocks)
dim = int(taps[-1].shape[1])
# closed form of the two products (query-key and attention-value), 2 FLOPs per multiply-add as the counter counts
analytic = depth * 2 * (2 * n * n * dim)
res = {"umt_l_encoder_tflops_per_clip_16x224": total / 1e12,
       "attention_products_tflops": (total - fused) / 1e12,
       "attention_products_tflops_closed_form": analytic / 1e12,
       "first_version_without_attention_products_tflops": fused / 1e12,
       "max_abs_diff_explicit_vs_fused_attention": float((taps[-1] - ref).abs().max()),
       "tokens": n,
       "encoder_params_M": sum(p.numel() for p in m.parameters()) / 1e6}
print(json.dumps(res))
(ROOT / "runs/thesis/umt_flops.json").write_text(json.dumps(res, indent=1), newline="\n")
