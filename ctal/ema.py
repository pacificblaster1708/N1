"""Exponential moving average of the student (Section 4.1)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def ema_decay(u: int, delta: float) -> float:
    """d_u = min(delta, (1 + u) / (10 + u))  -- warm-start rule of Sec. 4.1."""
    return min(delta, (1.0 + u) / (10.0 + u))


class ModelEMA:
    """Live EMA of every floating-point entry of the student's state dict.

    theta_bar_0 is a copy of the initial student.  After the u-th optimizer
    step (u = 1, 2, ...):  theta_bar_u = d_u theta_bar_{u-1} + (1 - d_u) theta_u.
    Floating buffers (e.g. BatchNorm running statistics) are averaged too;
    non-floating buffers (e.g. num_batches_tracked) are copied from the student.
    """

    def __init__(self, model: nn.Module, delta: float = 0.999):
        if not 0.0 <= delta < 1.0:
            raise ValueError("delta must lie in [0, 1)")
        self.delta = delta
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        self.num_updates += 1
        d = ema_decay(self.num_updates, self.delta)
        ema_state = self.module.state_dict()
        for name, value in model.state_dict().items():
            target = ema_state[name]
            if target.dtype.is_floating_point:
                target.mul_(d).add_(value.detach(), alpha=1.0 - d)
            else:
                target.copy_(value)
        return d

    def snapshot(self) -> nn.Module:
        """Frozen copy in evaluation mode, used as the refresh-t scorer."""
        scorer = copy.deepcopy(self.module).eval()
        for p in scorer.parameters():
            p.requires_grad_(False)
        return scorer
