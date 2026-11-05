"""Video backbones for SURGE-D.

Every backbone maps frames (B, T, 3, H, W), normalized with ImageNet statistics, to a list of 4 feature
volumes (B, D, T', h, w) taken from 4 depths of the network ("taps"), plus metadata.

  vjepa21_l  V-JEPA 2.1 ViT-L/16 (distilled from ViT-G), official code (facebookresearch/vjepa2, MIT)
  vjepa2_l   V-JEPA 2 ViT-L/16 (HF transformers port, facebook/vjepa2-vitl-fpc64-256, MIT); also exposes the
             pretrained predictor for the surprise gate (G2)
  umt_l      UMT-L/16 (Unmasked Teacher, K710 -> K400 fine-tuned), the backbone family used by SalFoM
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .lora import inject_lora

VJEPA21_SRC = "D:/models/vjepa2_1/src"
VJEPA21_CKPT = "D:/models/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
VJEPA2_HF = "facebook/vjepa2-vitl-fpc64-256"
UMT_CKPT = "D:/models/umt/single_modality/l16_ptk710_ftk710_ftk400_f16_res224.pth"


def _tokens_to_volume(x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    B, N, D = x.shape
    assert N == t * h * w, (N, t, h, w)
    return x.transpose(1, 2).reshape(B, D, t, h, w)


class Backbone(nn.Module):
    embed_dim: int = 1024
    tubelet: int = 2
    patch: int = 16

    def grid(self, T: int, H: int, W: int):
        return T // self.tubelet, H // self.patch, W // self.patch


# ------------------------------------------------------------------------------------------------ V-JEPA 2.1
class VJEPA21(Backbone):
    def __init__(self, ckpt: str = VJEPA21_CKPT, src: str = VJEPA21_SRC, taps=(5, 11, 17, 23), grad_ckpt=False):
        super().__init__()
        if src not in sys.path:
            sys.path.insert(0, src)
        from app.vjepa_2_1.models import vision_transformer as vit  # noqa: E402

        self.enc = vit.vit_large(
            patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
            wide_SiLU=True, uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
            out_layers=list(taps), use_activation_checkpointing=grad_ckpt,
        )
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["ema_encoder"]
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        missing, unexpected = self.enc.load_state_dict(sd, strict=False)
        bad_missing = [k for k in missing if "pos_embed" not in k]
        if bad_missing or unexpected:
            raise RuntimeError(f"V-JEPA 2.1 load: missing={bad_missing[:5]} unexpected={unexpected[:5]}")
        self.lora_pattern = r"enc\.blocks\.(1[2-9]|2[0-3])\.attn\.(qkv|proj)"

    def forward(self, x):  # x: B,T,3,H,W
        B, T, C, H, W = x.shape
        outs = self.enc(x.permute(0, 2, 1, 3, 4))  # B,C,T,H,W -> list of (B,N,D)
        t, h, w = self.grid(T, H, W)
        return [_tokens_to_volume(o, t, h, w) for o in outs]


# ------------------------------------------------------------------------------------------------ V-JEPA 2 (HF)
class VJEPA2(Backbone):
    def __init__(self, name: str = VJEPA2_HF, taps=(6, 12, 18, 24), grad_ckpt=False, attn="sdpa"):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name, attn_implementation=attn)
        self.taps = taps
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()
        self.lora_pattern = r"model\.encoder\.layer\.(1[2-9]|2[0-3])\.attention\.(query|key|value|proj)"

    def forward(self, x):
        B, T, C, H, W = x.shape
        assert H == W, "the HF V-JEPA 2 port assumes a square token grid"
        out = self.model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
        t, h, w = self.grid(T, H, W)
        return [_tokens_to_volume(out.hidden_states[l], t, h, w) for l in self.taps]

    @torch.no_grad()
    def surprise(self, x, n_ctx_frames: int = 8):
        """Leak-free per-token predictor error for the target tubelets (B, T'_tgt, h, w), plus the
        'contextual' variant: error with the true past minus error with a past rolled across the batch."""
        import torch.nn.functional as F

        B, T, C, H, W = x.shape
        t, h, w = self.grid(T, H, W)
        full = self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        ctx = self.model(pixel_values_videos=x[:, :n_ctx_frames], skip_predictor=True).last_hidden_state
        n_ctx = ctx.shape[1]
        ids_c = [torch.arange(n_ctx, device=x.device).unsqueeze(0).repeat(B, 1)]
        ids_t = [torch.arange(n_ctx, full.shape[1], device=x.device).unsqueeze(0).repeat(B, 1)]
        ln = lambda a: F.layer_norm(a.float(), a.shape[-1:])
        tgt = ln(full[:, n_ctx:])

        def err(c):
            p = self.model.predictor(encoder_hidden_states=c, context_mask=ids_c, target_mask=ids_t).last_hidden_state
            return (ln(p) - tgt).abs().mean(-1)

        e = err(ctx)
        e_wrong = err(ctx.roll(1, dims=0)) if B > 1 else e
        nt = t - n_ctx // (h * w)
        return e.view(B, nt, h, w), (e - e_wrong).view(B, nt, h, w)


# ------------------------------------------------------------------------------------------------ UMT-L
class UMT(Backbone):
    """UMT-L/16 (OpenGVLab unmasked_teacher, single_modality ViT), loaded from the K710->K400 fine-tuned checkpoint.

    Architecture re-implemented here from the checkpoint's tensor shapes: patch embed Conv3d(3, 1024, (1,16,16)),
    learned/sinusoidal position table for 16 x 14 x 14 tokens, 24 pre-norm blocks with fused qkv (q/v bias).
    """
    tubelet = 1

    def __init__(self, ckpt: str = UMT_CKPT, taps=(6, 12, 18, 24), num_frames=16, img=224, grad_ckpt=False):
        super().__init__()
        from .umt_vit import build_umt_large
        self.enc, self.tubelet = build_umt_large(ckpt, num_frames=num_frames, img_size=img)
        self.taps = taps
        self.grad_ckpt = grad_ckpt
        self.lora_pattern = r"enc\.blocks\.(1[2-9]|2[0-3])\.attn\.(qkv|proj)"

    def forward(self, x):
        B, T, C, H, W = x.shape
        outs = self.enc.forward_taps(x.permute(0, 2, 1, 3, 4), self.taps, self.grad_ckpt)
        t, h, w = self.grid(T, H, W)
        return [_tokens_to_volume(o, t, h, w) for o in outs]


def build_backbone(name: str, grad_ckpt: bool = False) -> Backbone:
    if name == "vjepa21_l":
        return VJEPA21(grad_ckpt=grad_ckpt)
    if name == "vjepa2_l":
        return VJEPA2(grad_ckpt=grad_ckpt)
    if name == "umt_l":
        return UMT(grad_ckpt=grad_ckpt)
    raise ValueError(name)


def add_lora(bb: Backbone, r: int = 16, alpha: float = 16.0) -> list[str]:
    names = inject_lora(bb, bb.lora_pattern, r=r, alpha=alpha)
    if not names:
        raise RuntimeError(f"LoRA pattern matched nothing: {bb.lora_pattern}")
    return names
