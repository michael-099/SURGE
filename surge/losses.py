"""Training losses on log-probability maps: KL(Y||S) + (1 - CC) - lambda_nss * NSS."""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def log_softmax2d(logits: torch.Tensor) -> torch.Tensor:
    s = logits.shape
    return F.log_softmax(logits.reshape(*s[:-2], -1).float(), -1).reshape(s)


def saliency_loss(logits, density, fixations, w_cc: float = 1.0, w_nss: float = 0.1):
    """logits, density, fixations: (B, T, h, w). density sums to 1 per frame. Returns (loss, parts)."""
    logp = log_softmax2d(logits)
    p = logp.exp()
    B, T = p.shape[:2]
    pf, yf, ff = p.reshape(B * T, -1), density.reshape(B * T, -1).float(), fixations.reshape(B * T, -1).float()
    lp = logp.reshape(B * T, -1)
    kl = (yf * (torch.log(yf + EPS) - lp)).sum(-1)
    pc, yc = pf - pf.mean(-1, keepdim=True), yf - yf.mean(-1, keepdim=True)
    cc = (pc * yc).sum(-1) / (pc.norm(dim=-1) * yc.norm(dim=-1) + EPS)
    z = (pf - pf.mean(-1, keepdim=True)) / (pf.std(-1, keepdim=True) + EPS)
    nf = ff.sum(-1)
    has = nf > 0
    nss = torch.where(has, (z * ff).sum(-1) / nf.clamp_min(1), torch.zeros_like(nf))
    loss = kl.mean() + w_cc * (1 - cc).mean() - w_nss * (nss.sum() / has.sum().clamp_min(1))
    return loss, {"kl": kl.mean().item(), "cc": cc.mean().item(), "nss": (nss.sum() / has.sum().clamp_min(1)).item()}
