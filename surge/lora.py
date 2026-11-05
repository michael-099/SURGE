"""Minimal LoRA for nn.Linear layers (Hu et al., 2022)."""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    @property
    def weight(self):  # some modules read .weight directly
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (self.drop(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scale
        return y


def inject_lora(model: nn.Module, pattern: str, r: int = 16, alpha: float = 16.0, dropout: float = 0.0) -> list[str]:
    """Replace every nn.Linear whose qualified name matches `pattern` (regex, fullmatch) with LoRALinear."""
    rx = re.compile(pattern)
    replaced = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and rx.fullmatch(full):
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout))
                replaced.append(full)
    return replaced


def set_lora_enabled(model: nn.Module, enabled: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def lora_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_a
            yield m.lora_b
