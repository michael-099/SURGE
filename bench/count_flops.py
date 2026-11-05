"""Count forward FLOPs of V-JEPA 2 ViT-L encoder/predictor (random weights, config from the hub)."""
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoConfig, VJEPA2Model

torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
cfg._attn_implementation = "eager"
m = VJEPA2Model(cfg).eval()
print("encoder params %.1fM" % (sum(p.numel() for n, p in m.named_parameters() if not n.startswith("predictor")) / 1e6))
print("predictor params %.1fM" % (sum(p.numel() for n, p in m.named_parameters() if n.startswith("predictor")) / 1e6))


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        out = fn()
    return fc.get_total_flops() / 1e12, out


for T, H, W in [(16, 256, 256), (8, 256, 256), (16, 224, 224)]:
    x = torch.randn(1, T, 3, H, W)
    f, out = flops(lambda: m(pixel_values_videos=x, skip_predictor=True))
    print(f"encoder {T}x{H}x{W}: {f:.3f} TFLOPs, tokens {out.last_hidden_state.shape[1]}")

x = torch.randn(1, 8, 3, 256, 256)
c = m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
f, out = flops(lambda: m.predictor(encoder_hidden_states=c, context_mask=[torch.arange(1024)[None]],
                                    target_mask=[torch.arange(1024, 2048)[None]]))
print(f"predictor 1024 ctx -> 1024 tgt: {f:.3f} TFLOPs, out {tuple(out.last_hidden_state.shape)}")
