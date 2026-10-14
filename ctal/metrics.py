"""Mechanism diagnostics (Section 6).  Ground-truth corruption indicators and
subgroup ids are used here only for analysis; they never enter the weights."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def auroc(scores: torch.Tensor, positives: torch.Tensor) -> float:
    """AUROC of ``scores`` for detecting ``positives`` (Mann-Whitney, midranks for ties)."""
    scores = scores.detach().double().cpu().flatten()
    positives = positives.detach().bool().cpu().flatten()
    n_pos = int(positives.sum())
    n_neg = positives.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    s_sorted = scores[order]
    ranks = torch.empty_like(scores)
    _, inv, counts = torch.unique_consecutive(s_sorted, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    starts = ends - counts.double() + 1.0
    ranks[order] = ((starts + ends) / 2.0)[inv]
    rank_sum_pos = ranks[positives].sum().item()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _summary(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"n": 0}
    x = x.double()
    return {"n": int(x.numel()), "mean": x.mean().item(), "min": x.min().item(),
            "max": x.max().item(), "median": x.median().item()}


def excess_concentration(state: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """s_i - E_pi[s_i^pi] = s_i - mean_{j in I_c} tau_j  (exact null mean, Prop. 3).

    Its mean over affected examples is the left-hand side of Assumption 1.
    """
    tau, s = state["tau"].double(), state["s"].double()
    labels = labels.detach().cpu().long()
    K = int(labels.max()) + 1
    sums = torch.zeros(K, dtype=torch.float64).index_add_(0, labels, tau)
    counts = torch.bincount(labels, minlength=K).double().clamp_min(1)
    return s - (sums / counts)[labels]


def refresh_diagnostics(state: Dict[str, torch.Tensor], noisy: Optional[torch.Tensor], q: float,
                        group: Optional[torch.Tensor] = None,
                        group_names: Optional[dict] = None,
                        labels: Optional[torch.Tensor] = None) -> Dict:
    """Summaries recorded at every refresh (Appendix C + reviewer analyses).

    clean-hard := clean examples whose observed-class loss coordinate is in the
    loss tail (U_L > q, i.e. x_L > 0).
    tail_auroc := within the loss tail only, AUROC for noisy vs clean-hard.
                  'loss' uses h_L; 'risk' uses r.  risk > loss means the sample
                  weights separate the two beyond what loss rank alone can.
    """
    w, r, x_L = state["w"], state["r"], state["x_L"]
    out: Dict = {
        "lambda_eff": float(state["lam_eff"]),
        "loss_tail_frac": (x_L > 0).double().mean().item(),
        "cotail_pos_frac": (state["tau"] > 0).double().mean().item(),
        "activation_pos_frac": (state["a"] > 0).double().mean().item(),
        "tau": _summary(state["tau"]), "s": _summary(state["s"]), "a": _summary(state["a"]),
        "r": _summary(r), "w": _summary(w),
    }
    in_tail = x_L > 0
    if in_tail.any():
        # Weight spread unexplained by the loss coordinate: std of w within the tail
        # after removing the per-bin mean over 20 bins of U_L.
        bins = torch.clamp((state["U_L"][in_tail] * 20).long(), 0, 19)
        wt = w[in_tail].double()
        means = torch.zeros(20, dtype=torch.float64).index_add_(0, bins, wt)
        cnt = torch.bincount(bins, minlength=20).double().clamp_min(1)
        resid = wt - (means / cnt)[bins]
        out["tail_w_std"] = wt.std().item() if wt.numel() > 1 else 0.0
        out["tail_w_resid_std_given_loss"] = resid.std().item() if wt.numel() > 1 else 0.0
        # Exact: r / x_L = 1 + rho c + kappa a is the part of the risk NOT fixed by
        # the loss coordinate; its spread is 0 for loss-only weighting by construction.
        amp = (r[in_tail] / x_L[in_tail]).double()
        out["tail_amplification_std"] = amp.std().item() if amp.numel() > 1 else 0.0
    if noisy is not None:
        noisy = noisy.detach().bool().cpu()
        clean = ~noisy
        clean_hard = clean & in_tail
        out["auroc"] = {
            "loss": auroc(state["h_L"], noisy),
            "cotail": auroc(state["tau"], noisy),
            "context": auroc(state["s"], noisy),
            "activation": auroc(state["a"], noisy),
            "risk": auroc(r, noisy),
        }
        tail_noisy = noisy[in_tail]
        out["tail_auroc"] = {
            "loss": auroc(state["h_L"][in_tail], tail_noisy),
            "corroboration": auroc(state["c"][in_tail], tail_noisy),
            "activation": auroc(state["a"][in_tail], tail_noisy),
            "risk": auroc(r[in_tail], tail_noisy),
            "n_noisy": int(tail_noisy.sum()), "n_clean_hard": int((~tail_noisy).sum()),
        }
        out["w_clean"] = _summary(w[clean])
        out["w_noisy"] = _summary(w[noisy])
        out["w_clean_hard"] = _summary(w[clean_hard])
        out["w_noisy_tail"] = _summary(w[noisy & in_tail])
        out["noise_rate"] = noisy.double().mean().item()
        if labels is not None:
            ex = excess_concentration(state, labels)
            out["assumption1"] = {"noisy_mean_excess": ex[noisy].mean().item(),
                                  "clean_mean_excess": ex[clean].mean().item()}
    if group is not None:
        group = group.detach().cpu()
        names = group_names or {}
        out["w_by_group"] = {}
        for gid in torch.unique(group).tolist():
            m = group == gid
            key = names.get(int(gid), str(int(gid)))
            entry = _summary(w[m])
            if noisy is not None:
                entry["noise_rate"] = noisy[m].double().mean().item()
                entry["tail_frac"] = in_tail[m].double().mean().item()
            out["w_by_group"][key] = entry
    return out
