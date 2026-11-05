"""UMT-L/16 video ViT (Li et al., "Unmasked Teacher", ICCV 2023; single-modality model), re-implemented from the
released checkpoint's tensors: Conv3d patch embed (1,16,16), fixed sinusoidal position table (VideoMAE style),
pre-norm blocks with fused qkv weights and separate q/v biases, LayerNorm eps 1e-6, GELU MLP."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoid_table(n: int, d: int) -> torch.Tensor:
    pos = np.arange(n)[:, None] / np.power(10000, 2 * (np.arange(d) // 2) / d)[None]
    pos[:, 0::2] = np.sin(pos[:, 0::2])
    pos[:, 1::2] = np.cos(pos[:, 1::2])
    return torch.tensor(pos, dtype=torch.float32)[None]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias), self.v_bias])
        w = self.qkv.weight if not hasattr(self.qkv, "base") else None
        if w is not None:
            qkv = F.linear(x, w, bias)
        else:  # LoRA-wrapped
            qkv = self.qkv(x) + bias
        q, k, v = qkv.reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class UMTViT(nn.Module):
    def __init__(self, dim=1024, depth=24, heads=16, num_frames=16, img_size=224, patch=16, tubelet=1):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(3, dim, (tubelet, patch, patch), stride=(tubelet, patch, patch))
        n = (num_frames // tubelet) * (img_size // patch) ** 2
        self.register_buffer("pos_embed", sinusoid_table(n, dim), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])

    def forward_taps(self, x, taps, grad_ckpt=False):  # x: B,C,T,H,W
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(x.dtype)
        outs = []
        for i, blk in enumerate(self.blocks, start=1):
            x = checkpoint(blk, x, use_reentrant=False) if grad_ckpt and self.training else blk(x)
            if i in taps:
                outs.append(x)
        return outs


def build_umt_large(ckpt: str, num_frames: int = 16, img_size: int = 224):
    m = UMTViT(num_frames=num_frames, img_size=img_size)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc_norm."))}
    m.load_state_dict(sd, strict=True)
    return m, 1
