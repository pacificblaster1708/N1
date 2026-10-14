"""Core CTAL computations (Section 4 of the paper).

Every function here is a pure, detached computation on tensors.  Equation
numbers refer to the paper.  Nothing in this module requires gradients; the
whole refresh is executed under ``torch.no_grad()`` by the trainer.

Pipeline at a refresh t (Algorithm 1, lines 8-16):

    logits_w, z_w, z_s, memory_{t-1}
        -> hardness views h^L, h^T, h^R                    Eqs. (4)-(6)
        -> observed-class mid-empirical coordinates U      Eq. (7)
        -> tail exceedances x                              Eq. (8)
        -> corroboration c, co-tail score tau              Eq. (9)
        -> same-label leave-one-out cosine kNN graph       Sec. 4.4
        -> observed local means s                          Eq. (10)
        -> class-preserving permutation null s^(p)         Eq. (11)
        -> classwise null quantiles L, H                   Eq. (12)
        -> collective activation a                         Eq. (13)
        -> risk r                                          Eq. (14)
        -> raw weights v, normalized weights w             Eq. (15)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CTALConfig:
    """Hyper-parameters of the weighting rule (defaults = paper, Sec. 6)."""

    q: float = 0.8          # tail threshold, Eq. (8)
    k: int = 32             # neighbours per sample, Sec. 4.4
    P: int = 20             # number of permutations, Eq. (11)
    eta0: float = 0.90      # lower null quantile, Eq. (12)
    eta1: float = 0.99      # upper null quantile, Eq. (12)
    rho: float = 1.0        # pointwise amplification, Eq. (14)
    kappa: float = 2.0      # collective amplification, Eq. (14)
    omega: float = 0.1      # weight floor, Eq. (15)
    lam: float = 2.0        # target suppression strength lambda, Eq. (15)
    beta: float = 0.8       # probability-memory decay, Eq. (3)
    eps_z: float = 1e-12    # feature-normalization floor, Sec. 4.2
    eps_a: float = 1e-6     # activation denominator floor, Eq. (13)
    cotail_form: str = "geometric"  # Eq. (9); others are ablations, see COTAIL_FORMS

    def __post_init__(self) -> None:
        if self.cotail_form not in COTAIL_FORMS:
            raise ValueError(f"cotail_form must be one of {COTAIL_FORMS}")
        if not 0.0 < self.q < 1.0:
            raise ValueError("q must lie in (0, 1)")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.P < 1:
            raise ValueError("P must be >= 1")
        if not 0.0 < self.eta0 < self.eta1 < 1.0:
            raise ValueError("need 0 < eta0 < eta1 < 1")
        if self.rho < 0 or self.kappa < 0:
            raise ValueError("rho and kappa must be >= 0")
        if not 0.0 < self.omega <= 1.0:
            raise ValueError("omega must lie in (0, 1]")
        if self.lam < 0:
            raise ValueError("lambda must be >= 0")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        if self.eps_z <= 0 or self.eps_a <= 0:
            raise ValueError("eps_z and eps_a must be > 0")


# Co-tail aggregations (reviewer ablation of the joint-tail statistic).
#   geometric : sqrt(x_L * (x_T + x_R)/2)        -- the paper's Eq. (9)
#   additive  : (x_L + x_T + x_R) / 3              -- soft additive tail activity
#   product   : (x_L x_T x_R)^(1/3)                -- all three views jointly extreme
#   min       : min(x_L, x_T, x_R)                 -- strict joint exceedance
#   indicator : 1{x_L > 0, x_T > 0, x_R > 0}       -- all views beyond threshold
COTAIL_FORMS = ("geometric", "additive", "product", "min", "indicator")

# Variants used in the paper's central decomposition (Sec. 6).
#   name -> (use rho?, use kappa?)
VARIANTS = {
    "ce": (False, False),          # ordinary training (lambda forced to 0)
    "base": (False, False),        # same as 'ce': base loss without reweighting
    "loss_tail": (False, False),   # r = x_L
    "pointwise": (True, False),    # r = x_L (1 + rho c)
    "collective": (False, True),   # r = x_L (1 + kappa a)
    "full": (True, True),          # r = x_L (1 + rho c + kappa a)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_indices(labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Map each observed class c to the sorted index set I_c (non-empty only)."""
    labels = labels.cpu()
    out: Dict[int, torch.Tensor] = {}
    for c in torch.unique(labels).tolist():
        out[int(c)] = torch.nonzero(labels == c, as_tuple=False).flatten()
    return out


def normalize_features(z: torch.Tensor, eps_z: float) -> torch.Tensor:
    """z / max(||z||_2, eps_z)   (Sec. 4.2)."""
    return z / z.norm(p=2, dim=1, keepdim=True).clamp_min(eps_z)


def linear_quantile(values: torch.Tensor, eta: float) -> torch.Tensor:
    """Empirical quantile with linear interpolation between order statistics.

    Identical to ``numpy.quantile(values, eta, method="linear")``: position
    eta * (n - 1) in the sorted sample, interpolated between its neighbours.
    Implemented by hand to avoid ``torch.quantile``'s input-size limit.
    """
    v = torch.sort(values.flatten()).values
    n = v.numel()
    if n == 0:
        raise ValueError("quantile of an empty set")
    pos = eta * (n - 1)
    lo = int(pos // 1)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return v[lo] + (v[hi] - v[lo]) * frac


# ---------------------------------------------------------------------------
# Section 4.2: detached hardness views
# ---------------------------------------------------------------------------

def hardness_views(
    logits_w: torch.Tensor,
    labels: torch.Tensor,
    memory_prev: torch.Tensor,
    z_w: torch.Tensor,
    z_s: torch.Tensor,
    eps_z: float,
):
    """Return (h_L, h_T, h_R, zhat_w) for one refresh.

    h_L : Eq. (4)  -log p_bar(y | weak view), computed stably from logits.
    h_T : Eq. (5)  [max_{k != y} m_k - m_y]_+ using the *previous* memory.
    h_R : Eq. (6)  (1 - <zhat_w, zhat_s>) / 2, in [0, 1].
    """
    labels = labels.long()
    h_L = F.cross_entropy(logits_w, labels, reduction="none")

    m_y = memory_prev.gather(1, labels[:, None]).squeeze(1)
    others = memory_prev.clone()
    others.scatter_(1, labels[:, None], float("-inf"))
    h_T = (others.max(dim=1).values - m_y).clamp_min(0.0)

    zh_w = normalize_features(z_w, eps_z)
    zh_s = normalize_features(z_s, eps_z)
    cos = (zh_w * zh_s).sum(dim=1)
    # Mathematically in [0, 1]; clamp only removes floating-point overshoot.
    h_R = (0.5 * (1.0 - cos)).clamp(0.0, 1.0)
    return h_L, h_T, h_R, zh_w


# ---------------------------------------------------------------------------
# Section 4.3: observed-class empirical coordinates
# ---------------------------------------------------------------------------

def class_midrank_coordinates(h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. (7): U_i = (#{j in I_c: h_j < h_i} + 0.5 #{j in I_c: h_j = h_i}) / n_c.

    The tie count includes j = i, so U_i lies in (0, 1).
    """
    U = torch.empty_like(h, dtype=torch.float64)
    for _, idx in class_indices(labels).items():
        idx = idx.to(h.device)
        v = h[idx]
        s = torch.sort(v).values
        less = torch.searchsorted(s, v, right=False)
        leq = torch.searchsorted(s, v, right=True)
        U[idx] = (less.double() + 0.5 * (leq - less).double()) / idx.numel()
    return U


def tail_exceedance(U: torch.Tensor, q: float) -> torch.Tensor:
    """Eq. (8): clip_[0,1]((U - q) / (1 - q))."""
    return ((U - q) / (1.0 - q)).clamp(0.0, 1.0)


def cotail_score(x_L: torch.Tensor, x_T: torch.Tensor, x_R: torch.Tensor,
                 form: str = "geometric"):
    """Eq. (9): c = (x_T + x_R)/2 and tau = sqrt(x_L * c).

    ``form`` selects an ablation of tau (see COTAIL_FORMS); c is always the
    pointwise corroboration of Eq. (9), so only the collective input changes.
    """
    c = 0.5 * (x_T + x_R)
    if form == "geometric":
        tau = torch.sqrt(x_L * c)
    elif form == "additive":
        tau = (x_L + x_T + x_R) / 3.0
    elif form == "product":
        tau = (x_L * x_T * x_R).pow(1.0 / 3.0)
    elif form == "min":
        tau = torch.minimum(torch.minimum(x_L, x_T), x_R)
    elif form == "indicator":
        tau = ((x_L > 0) & (x_T > 0) & (x_R > 0)).to(x_L.dtype)
    else:
        raise ValueError(f"unknown cotail form {form!r}")
    return c, tau


# ---------------------------------------------------------------------------
# Section 4.4: permutation-calibrated neighbourhood context
# ---------------------------------------------------------------------------

def knn_within_class(
    zhat: torch.Tensor, labels: torch.Tensor, k: int, block_size: int = 2048
) -> torch.Tensor:
    """Exact leave-one-out cosine kNN restricted to the same observed label.

    ``zhat`` must already be L2-normalized, so the dot product is the cosine.
    Returns an (N, k) LongTensor of global neighbour indices (on zhat.device).
    Blocked matrix multiplication bounds peak memory (Sec. 4.6).
    """
    N = zhat.shape[0]
    device = zhat.device
    nbr = torch.empty((N, k), dtype=torch.long, device=device)
    for c, idx in class_indices(labels).items():
        n_c = idx.numel()
        if n_c < k + 1:
            raise ValueError(
                f"observed class {c} has {n_c} examples; CTAL requires n_c >= k+1 = {k + 1}"
            )
        idx = idx.to(device)
        Z = zhat[idx]
        for start in range(0, n_c, block_size):
            end = min(start + block_size, n_c)
            S = Z[start:end] @ Z.T
            rows = torch.arange(end - start, device=device)
            S[rows, start + rows] = float("-inf")  # leave-one-out
            top = torch.topk(S, k, dim=1).indices
            nbr[idx[start:end]] = idx[top]
    return nbr


def class_permutations(
    labels: torch.Tensor, P: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Draw P independent permutations, each uniform within every I_c.

    Returns pi of shape (P, N) with pi[p, j] = pi_p(j); for j in I_c we have
    pi_p(j) in I_c.  Permuted scores are tau[pi[p]].
    """
    labels = labels.cpu()
    N = labels.numel()
    pi = torch.arange(N).repeat(P, 1)
    groups = class_indices(labels)
    for p in range(P):
        for _, idx in groups.items():
            pi[p, idx] = idx[torch.randperm(idx.numel(), generator=generator)]
    return pi


def local_means(tau: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
    """Eq. (10): s_i = (1/k) sum_{j in N_i} tau_j."""
    return tau[nbr].mean(dim=1)


def null_local_means(tau: torch.Tensor, nbr: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Eq. (11): s_i^(p) = (1/k) sum_{j in N_i} tau_{pi_p(j)}, shape (P, N)."""
    out = torch.empty((pi.shape[0], tau.numel()), dtype=tau.dtype, device=tau.device)
    for p in range(pi.shape[0]):
        out[p] = local_means(tau[pi[p]], nbr)
    return out


def null_thresholds(s_null: torch.Tensor, labels: torch.Tensor, eta0: float, eta1: float):
    """Eq. (12): classwise L_c = Q_eta0(S_c0), H_c = Q_eta1(S_c0).

    S_c0 pools the P * n_c null values of class c.  Returns per-sample
    tensors (L_{y_i}, H_{y_i}).
    """
    N = s_null.shape[1]
    L = torch.empty(N, dtype=s_null.dtype, device=s_null.device)
    H = torch.empty_like(L)
    for _, idx in class_indices(labels).items():
        idx = idx.to(s_null.device)
        pool = s_null[:, idx]
        L[idx] = linear_quantile(pool, eta0)
        H[idx] = linear_quantile(pool, eta1)
    return L, H


def collective_activation(
    s: torch.Tensor, L: torch.Tensor, H: torch.Tensor, eps_a: float
) -> torch.Tensor:
    """Eq. (13): a = clip_[0,1]((s - L) / max(H - L, eps_a))."""
    return ((s - L) / (H - L).clamp_min(eps_a)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Section 4.5: risk, weights, objective
# ---------------------------------------------------------------------------

def risk(x_L: torch.Tensor, c: torch.Tensor, a: torch.Tensor, rho: float, kappa: float):
    """Eq. (14): r = x_L (1 + rho c + kappa a)."""
    return x_L * (1.0 + rho * c + kappa * a)


def raw_and_normalized_weights(r: torch.Tensor, lam: float, omega: float):
    """Eq. (15): v = omega + (1-omega) exp(-lam r), w = v / mean(v)."""
    v = omega + (1.0 - omega) * torch.exp(-lam * r)
    w = v / v.mean()
    return v, w


def weighted_minibatch_loss(per_sample_loss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Eq. (16): sum_i sg(w_i) l_i / sum_i sg(w_i).  Gradients flow only via l_i."""
    w = w.detach()
    return (w * per_sample_loss).sum() / w.sum()


def update_memory(memory_prev: torch.Tensor, probs: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq. (3): m_t = beta m_{t-1} + (1 - beta) p_bar_t  (applied after scoring)."""
    return beta * memory_prev + (1.0 - beta) * probs


# ---------------------------------------------------------------------------
# Full refresh (Algorithm 1, lines 8-16)
# ---------------------------------------------------------------------------

def compute_refresh(
    logits_w: torch.Tensor,
    z_w: torch.Tensor,
    z_s: torch.Tensor,
    labels: torch.Tensor,
    memory_prev: torch.Tensor,
    cfg: CTALConfig,
    lam_t: float,
    variant: str = "full",
    perm_generator: Optional[torch.Generator] = None,
    activation_control: str = "none",
    control_generator: Optional[torch.Generator] = None,
    nbr: Optional[torch.Tensor] = None,
    pi: Optional[torch.Tensor] = None,
    knn_block_size: int = 2048,
) -> Dict[str, torch.Tensor]:
    """Compute every refresh-t quantity and the stored weights w^t.

    Statistics are computed in float64 on CPU (deterministic, N x small);
    the kNN search runs on the device of ``z_w``.

    variant            : one of VARIANTS; selects which of rho / kappa enter Eq. (14).
                         ``ce`` forces lambda = 0 (exact ERM, Prop. 2) but still
                         returns every diagnostic.
    activation_control : 'none'    -> genuine activation a_i.
                         'permute' -> negative control of Sec. 6: a_i is replaced by
                                      a_{pi(i)} for an independent class-preserving
                                      permutation pi (same classwise activation
                                      marginal, alignment with the sample broken).
    nbr, pi            : optional precomputed graph / permutations (used by tests).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")

    labels_cpu = labels.detach().cpu().long()
    logits_w = logits_w.detach().double().cpu()
    memory_prev = memory_prev.detach().double().cpu()

    h_L, h_T, h_R, _ = hardness_views(
        logits_w, labels_cpu, memory_prev, z_w.detach().double().cpu(),
        z_s.detach().double().cpu(), cfg.eps_z,
    )

    if nbr is None:
        # Graph on the original device / precision for speed.
        zh_dev = normalize_features(z_w.detach().float(), cfg.eps_z)
        nbr = knn_within_class(zh_dev, labels.to(zh_dev.device), cfg.k, knn_block_size)

    use_rho, use_kappa = VARIANTS[variant]
    return refresh_from_views(
        h_L, h_T, h_R, labels_cpu, nbr, cfg,
        lam_t=0.0 if variant in ("ce", "base") else lam_t,
        rho=cfg.rho if use_rho else 0.0,
        kappa=cfg.kappa if use_kappa else 0.0,
        perm_generator=perm_generator, pi=pi,
        activation_control=activation_control, control_generator=control_generator,
    )


def refresh_from_views(
    h_L: torch.Tensor, h_T: torch.Tensor, h_R: torch.Tensor,
    labels: torch.Tensor, nbr: torch.Tensor, cfg: CTALConfig,
    lam_t: float, rho: float, kappa: float,
    perm_generator: Optional[torch.Generator] = None,
    pi: Optional[torch.Tensor] = None,
    activation_control: str = "none",
    control_generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Eqs. (7)-(15) from given hardness views and a fixed feature graph."""
    if activation_control not in ("none", "permute"):
        raise ValueError("activation_control must be 'none' or 'permute'")
    labels_cpu = labels.detach().cpu().long()
    h_L, h_T, h_R = (h.detach().double().cpu() for h in (h_L, h_T, h_R))

    U_L = class_midrank_coordinates(h_L, labels_cpu)
    U_T = class_midrank_coordinates(h_T, labels_cpu)
    U_R = class_midrank_coordinates(h_R, labels_cpu)
    x_L = tail_exceedance(U_L, cfg.q)
    x_T = tail_exceedance(U_T, cfg.q)
    x_R = tail_exceedance(U_R, cfg.q)
    c, tau = cotail_score(x_L, x_T, x_R, cfg.cotail_form)

    nbr = nbr.cpu()

    s = local_means(tau, nbr)
    if pi is None:
        pi = class_permutations(labels_cpu, cfg.P, perm_generator)
    s_null = null_local_means(tau, nbr, pi)
    L, H = null_thresholds(s_null, labels_cpu, cfg.eta0, cfg.eta1)
    a = collective_activation(s, L, H, cfg.eps_a)

    a_used = a
    if activation_control == "permute":
        pi_ctrl = class_permutations(labels_cpu, 1, control_generator)[0]
        a_used = a[pi_ctrl]

    r = risk(x_L, c, a_used, rho, kappa)
    v, w = raw_and_normalized_weights(r, lam_t, cfg.omega)

    return {
        "h_L": h_L, "h_T": h_T, "h_R": h_R,
        "U_L": U_L, "U_T": U_T, "U_R": U_R,
        "x_L": x_L, "x_T": x_T, "x_R": x_R,
        "c": c, "tau": tau,
        "nbr": nbr, "s": s, "s_null": s_null, "L": L, "H": H,
        "a": a, "a_used": a_used,
        "r": r, "v": v, "w": w,
        "pi": pi,
        "lam_eff": torch.tensor(lam_t, dtype=torch.float64),
    }
