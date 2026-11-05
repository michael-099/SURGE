"""CPU check of the V-JEPA 2.1 ViT-L wrapper: strict-ish load, tap shapes at 256x256 and 16 frames, finite outputs,
and that a non-square input works (16 x 28 tokens at 256 x 448)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from surge.backbones import VJEPA21  # noqa: E402

torch.set_grad_enabled(False)
t = time.time()
m = VJEPA21().eval()
print(f"loaded in {time.time() - t:.0f}s; params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
for H, W in [(256, 256), (256, 448)]:
    x = torch.randn(1, 16, 3, H, W)
    outs = m(x)
    print(H, W, [tuple(o.shape) for o in outs], "finite", all(torch.isfinite(o).all().item() for o in outs),
          "std", [round(o.std().item(), 3) for o in outs])
