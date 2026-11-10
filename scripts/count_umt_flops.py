"""FLOPs of the UMT-L/16 encoder (the backbone family of SalFoM) on one 16 x 224 x 224 clip, on the CPU.

SalFoM predicts one frame per encoder pass (sliding window), so this is its encoder cost per output frame.
    python scripts/count_umt_flops.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from surge.backbones import UMT  # noqa: E402

torch.set_grad_enabled(False)
m = UMT().eval()
x = torch.randn(1, 16, 3, 224, 224)
with FlopCounterMode(display=False) as fc:
    taps = m(x)
res = {"umt_l_encoder_tflops_per_clip_16x224": fc.get_total_flops() / 1e12,
       "tokens": int(taps[-1].shape[2] * taps[-1].shape[3] * taps[-1].shape[4]),
       "encoder_params_M": sum(p.numel() for p in m.parameters()) / 1e6}
print(json.dumps(res))
(ROOT / "runs/thesis/umt_flops.json").write_text(json.dumps(res, indent=1), newline="\n")
