"""Per-sample student losses that CTAL can wrap ("base vs base + CTAL").

CTAL only changes how per-sample losses are weighted (Eq. 16); the scoring
view h^L of Eq. (4) is always the frozen scorer's cross-entropy, independent
of the base loss, so the weighting rule is identical across bases.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BASE_LOSSES = ("ce", "gce", "sce")


def gce_loss(logits: torch.Tensor, y: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    """Generalized cross-entropy L_q = (1 - p_y^q) / q  (Zhang & Sabuncu, 2018)."""
    p_y = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    return (1.0 - p_y.clamp_min(1e-7).pow(q)) / q


def sce_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Symmetric cross-entropy alpha*CE + beta*RCE (Wang et al., 2019).

    Follows the official implementation: predictions clamped to [1e-7, 1],
    one-hot labels clamped to [1e-4, 1] before the log in the reverse term.
    """
    ce = F.cross_entropy(logits, y, reduction="none")
    pred = torch.softmax(logits, dim=1).clamp(1e-7, 1.0)
    onehot = F.one_hot(y, logits.shape[1]).to(pred.dtype).clamp(1e-4, 1.0)
    rce = -(pred * torch.log(onehot)).sum(dim=1)
    return alpha * ce + beta * rce


def sce_defaults(num_classes: int):
    """(alpha, beta) from the SCE paper: CIFAR-10 (0.1, 1.0), CIFAR-100 (6.0, 0.1)."""
    return (6.0, 0.1) if num_classes >= 100 else (0.1, 1.0)


def per_sample_loss(logits, y, kind: str = "ce", gce_q: float = 0.7,
                    sce_alpha: float = None, sce_beta: float = None) -> torch.Tensor:
    if kind == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    if kind == "gce":
        return gce_loss(logits, y, gce_q)
    if kind == "sce":
        a, b = sce_defaults(logits.shape[1])
        return sce_loss(logits, y, a if sce_alpha is None else sce_alpha,
                        b if sce_beta is None else sce_beta)
    raise ValueError(f"base loss must be one of {BASE_LOSSES}")
