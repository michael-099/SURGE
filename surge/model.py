"""SURGE-D decoder: SimpleFPN-3D, optional surprise-conditioned modulation (SCM), top-down (2+1)D decoder with
auxiliary heads, and a learned center prior (product of experts)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import Backbone


def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c)


class R21(nn.Module):
    """(2+1)D residual block: 1x3x3 spatial conv -> GN -> GELU -> 3x1x1 temporal conv -> GN, + skip, GELU."""

    def __init__(self, c: int):
        super().__init__()
        self.s = nn.Conv3d(c, c, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.n1 = gn(c)
        self.t = nn.Conv3d(c, c, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.n2 = gn(c)

    def forward(self, x):
        y = F.gelu(self.n1(self.s(x)))
        y = self.n2(self.t(y))
        return F.gelu(x + y)


class Up2(nn.Module):
    """Spatial x2 upsampling with a transposed conv (1,2,2)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, (1, 2, 2), stride=(1, 2, 2))
        self.n = gn(cout)

    def forward(self, x):
        return F.gelu(self.n(self.up(x)))


class SimpleFPN3D(nn.Module):
    """ViTDet-style pyramid from 4 plain-ViT taps: P4 (x4), P8 (x2), P16a, P16b (x1)."""

    def __init__(self, d: int, c: int = 256):
        super().__init__()
        self.p4 = nn.Sequential(nn.Conv3d(d, c, 1), gn(c), nn.GELU(), Up2(c, c), Up2(c, c))
        self.p8 = nn.Sequential(nn.Conv3d(d, c, 1), gn(c), nn.GELU(), Up2(c, c))
        self.p16a = nn.Sequential(nn.Conv3d(d, c, 1), gn(c), nn.GELU())
        self.p16b = nn.Sequential(nn.Conv3d(d, c, 1), gn(c), nn.GELU())

    def forward(self, taps):
        return [self.p4(taps[0]), self.p8(taps[1]), self.p16a(taps[2]), self.p16b(taps[3])]


class SurpriseEncoder(nn.Module):
    """Phi: [surprise r (zero on context tubelets); availability mask] (B,2,T',h,w) -> U (B,64,T',h,w)."""

    def __init__(self, c: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(2, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, c, 3, padding=1), nn.GELU())

    def forward(self, s):
        return self.net(s)


class SCM(nn.Module):
    """F~ = F * (1 + gamma(U)) + beta(U); gamma, beta are zero-initialized 1x1x1 convs (exact no-op at init)."""

    def __init__(self, cu: int, c: int):
        super().__init__()
        self.g = nn.Conv3d(cu, c, 1)
        self.b = nn.Conv3d(cu, c, 1)
        for m in (self.g, self.b):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, f, u):
        u = F.interpolate(u, size=f.shape[2:], mode="trilinear", align_corners=False)
        return f * (1 + self.g(u)) + self.b(u)


class CenterPrior(nn.Module):
    """Mixture of K diagonal Gaussians in normalized coordinates; returns log-density over the output grid."""

    def __init__(self, k: int = 8):
        super().__init__()
        g = torch.linspace(0.35, 0.65, 3)
        mu = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)[:k]
        self.mu = nn.Parameter(mu.clone())
        self.log_sigma = nn.Parameter(torch.full((k, 2), math.log(0.25)))
        self.w = nn.Parameter(torch.zeros(k))
        self.strength = nn.Parameter(torch.ones(()))

    def forward(self, h: int, w: int, device):
        ys = (torch.arange(h, device=device) + 0.5) / h
        xs = (torch.arange(w, device=device) + 0.5) / w
        p = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1)  # h,w,2
        d = (p[None] - self.mu[:, None, None]) / self.log_sigma.exp()[:, None, None]
        logn = -0.5 * (d ** 2).sum(-1) - self.log_sigma.sum(-1)[:, None, None]
        logmix = torch.logsumexp(logn + F.log_softmax(self.w, 0)[:, None, None], 0)
        logmix = logmix - torch.logsumexp(logmix.flatten(), 0)
        return self.strength * logmix  # h,w


class AudioFusion(nn.Module):
    """Audio conditioning of one pyramid level (research/log.md section 11D).

    a (B, T', da) per tubelet -> e = GELU(Linear(LayerNorm(a))) (B, T', d).
    F~ = F * (1 + gamma(e)) + beta(e) + R(F * s),  s = sigmoid(<Wq F(x), Wk e> / sqrt(d))  (audio-visual correlation map).
    gamma/beta and R are zero-initialized, so the module is an exact no-op at initialization.
    """

    def __init__(self, c: int, da: int = 768, d: int = 256):
        super().__init__()
        self.d = d
        self.norm = nn.LayerNorm(da)
        self.proj = nn.Linear(da, d)
        self.film = nn.Linear(d, 2 * c)
        self.q = nn.Conv3d(c, d, 1)
        self.k = nn.Linear(d, d)
        self.res = nn.Conv3d(c, c, 1)
        for m in (self.film, self.res):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, f, a):
        if a.shape[1] != f.shape[2]:  # align audio steps to the level's temporal size
            a = F.interpolate(a.transpose(1, 2), size=f.shape[2], mode="linear", align_corners=False).transpose(1, 2)
        e = F.gelu(self.proj(self.norm(a.to(f.dtype))))
        g, b = self.film(e).chunk(2, -1)
        g = g.transpose(1, 2)[..., None, None]
        b = b.transpose(1, 2)[..., None, None]
        k = self.k(e).transpose(1, 2)[..., None, None]  # B,d,T',1,1
        s = torch.sigmoid((self.q(f) * k).sum(1, keepdim=True) / math.sqrt(self.d))
        return f * (1 + g) + b + self.res(f * s)


class Decoder(nn.Module):
    def __init__(self, d: int = 1024, c: int = 256, use_scm: bool = False, prior: bool = True, aux: bool = True,
                 audio: bool = False):
        super().__init__()
        self.fpn = SimpleFPN3D(d, c)
        self.audio = audio
        if audio:
            self.av = nn.ModuleList([AudioFusion(c) for _ in range(3)])  # P8, P16a, P16b
        self.use_scm = use_scm
        if use_scm:
            self.phi = SurpriseEncoder(64)
            self.scm = nn.ModuleList([SCM(64, c) for _ in range(4)])
        self.fuse16 = nn.Sequential(nn.Conv3d(2 * c, c, 1), gn(c), nn.GELU())
        self.r16 = nn.Sequential(R21(c), R21(c))
        self.r8 = nn.Sequential(R21(c), R21(c))
        self.r4 = nn.Sequential(R21(c), R21(c))
        self.head = nn.Sequential(nn.Conv3d(c, c // 2, (1, 3, 3), padding=(0, 1, 1)), nn.GELU(), nn.Conv3d(c // 2, 1, 1))
        self.aux = aux
        if aux:
            self.aux16 = nn.Conv3d(c, 1, 1)
            self.aux8 = nn.Conv3d(c, 1, 1)
        self.prior = CenterPrior() if prior else None

    def forward(self, taps, surprise=None, T_out: int | None = None, audio=None):
        p4, p8, p16a, p16b = self.fpn(taps)
        if self.audio:
            if audio is None:
                raise ValueError("audio enabled but no audio given")
            B, Ta, Da = audio.shape  # per sampled frame -> per tubelet (mean of consecutive pairs)
            a = audio[:, : Ta // 2 * 2].reshape(B, Ta // 2, 2, Da).mean(2) if Ta > 1 else audio
            p8, p16a, p16b = self.av[0](p8, a), self.av[1](p16a, a), self.av[2](p16b, a)
        if self.use_scm:
            if surprise is None:
                raise ValueError("SCM enabled but no surprise given")
            u = self.phi(surprise)
            p4, p8, p16a, p16b = [m(f, u) for m, f in zip(self.scm, (p4, p8, p16a, p16b))]
        d16 = self.r16(self.fuse16(torch.cat([p16a, p16b], 1)))
        d8 = self.r8(p8 + F.interpolate(d16, size=p8.shape[2:], mode="nearest"))
        d4 = self.r4(p4 + F.interpolate(d8, size=p4.shape[2:], mode="nearest"))
        out = {"logits": self.head(d4)[:, 0]}  # B,T',H4,W4
        if self.aux and self.training:
            out["aux16"] = self.aux16(d16)[:, 0]
            out["aux8"] = self.aux8(d8)[:, 0]
        for k in list(out):
            x = out[k]
            if k != "logits":
                x = F.interpolate(x[:, None], size=out["logits"].shape[1:], mode="trilinear", align_corners=False)[:, 0]
            if T_out is not None and x.shape[1] != T_out:  # tubelets -> sampled frames
                x = F.interpolate(x[:, None], size=(T_out, *x.shape[2:]), mode="trilinear", align_corners=False)[:, 0]
            if self.prior is not None:
                x = x + self.prior(x.shape[-2], x.shape[-1], x.device)
            out[k] = x
        return out


class SurgeD(nn.Module):
    def __init__(self, backbone: Backbone, use_scm: bool = False, prior: bool = True, aux: bool = True, c: int = 256,
                 audio: bool = False):
        super().__init__()
        self.backbone = backbone
        self.decoder = Decoder(backbone.embed_dim, c, use_scm=use_scm, prior=prior, aux=aux, audio=audio)

    @property
    def uses_audio(self) -> bool:
        return self.decoder.audio

    def forward(self, frames, surprise=None, backbone_grad: bool = False, audio=None):
        T = frames.shape[1]
        if backbone_grad:
            taps = self.backbone(frames)
        else:
            with torch.no_grad():
                taps = self.backbone(frames)
        taps = [t.float() if t.dtype != torch.bfloat16 else t for t in taps]
        return self.decoder(taps, surprise, T_out=T, audio=audio)


def central(x: torch.Tensor, T: int) -> torch.Tensor:
    """Select the central T/2 sampled frames of a (B, T, ...) tensor."""
    c0 = T // 4
    return x[:, c0:c0 + T // 2]
