"""Tests for the CTAL weighting rule: reference equivalence and Propositions 1-5."""

import itertools
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ctal import core  # noqa: E402
from ctal.core import CTALConfig  # noqa: E402
from reference_impl import reference_refresh  # noqa: E402

torch.set_default_dtype(torch.float64)


def random_problem(seed, N=60, K=3, d=8, quantize=False):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(N) % K
    # shuffle so classes are interleaved irregularly
    labels = labels[torch.randperm(N, generator=g)]
    logits = 2.0 * torch.randn(N, K, generator=g)
    z_w = torch.randn(N, d, generator=g)
    z_s = z_w + 0.7 * torch.randn(N, d, generator=g)
    mem = torch.softmax(1.5 * torch.randn(N, K, generator=g), dim=1)
    if quantize:  # force many exact ties in every view
        logits = torch.round(logits)
        mem = torch.round(mem * 4) / 4
        z_s = z_w.clone()
        z_s[: N // 2] = -z_w[: N // 2]
    return labels, logits, z_w, z_s, mem


# ---------------------------------------------------------------------------
# 1. Equivalence with the naive reference implementation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("q", [0.5, 0.8])
def test_matches_reference(seed, quantize, q):
    labels, logits, z_w, z_s, mem = random_problem(seed, quantize=quantize)
    cfg = CTALConfig(q=q, k=5, P=7, rho=1.0, kappa=2.0, omega=0.1, lam=2.0)
    pi = core.class_permutations(labels, cfg.P, torch.Generator().manual_seed(seed))
    out = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, lam_t=1.3,
                               variant="full", pi=pi)
    # Graph from the reference (independent brute force) must agree as sets,
    # unless cosine ties make the top-k ambiguous (only in the quantized case).
    views = (out["h_L"].numpy(), out["h_T"].numpy(), out["h_R"].numpy()) if quantize else None
    ref = reference_refresh(logits.numpy(), z_w.numpy(), z_s.numpy(), labels.numpy(),
                            mem.numpy(), cfg, lam=1.3, rho=1.0, kappa=2.0, perms=pi.numpy(),
                            nbr=None if not quantize else out["nbr"].numpy(),
                            views_override=views)
    if quantize:  # the deliberately tied inputs must actually produce ties
        assert len(torch.unique(out["h_L"])) < len(labels) and (out["h_T"] == 0).sum() > 5
    if not quantize:
        for i in range(len(labels)):
            assert set(out["nbr"][i].tolist()) == set(ref["nbr"][i].tolist())
    for key in ["h_L", "h_T", "h_R", "U_L", "U_T", "U_R", "x_L", "x_T", "x_R", "c", "tau",
                "s", "s_null", "a", "r", "v", "w"]:
        np.testing.assert_allclose(out[key].numpy(), ref[key], rtol=0, atol=1e-10, err_msg=key)
    # the test is only meaningful if the collective part is exercised
    assert (out["tau"] > 0).any()


def test_reference_exercises_activation():
    """Make sure some test cases produce strictly interior and saturated activations."""
    seen_mid, seen_one = False, False
    for seed in range(10):
        labels, logits, z_w, z_s, mem = random_problem(seed)
        cfg = CTALConfig(q=0.5, k=5, P=7)
        out = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, lam_t=1.0,
                                   perm_generator=torch.Generator().manual_seed(seed))
        a = out["a"]
        seen_mid |= bool(((a > 0) & (a < 1)).any())
        seen_one |= bool((a == 1).any())
    assert seen_mid and seen_one


# ---------------------------------------------------------------------------
# 2. Individual building blocks
# ---------------------------------------------------------------------------

def test_midrank_with_ties():
    h = torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 5.0, 0.0])
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    U = core.class_midrank_coordinates(h, y)
    # class 0: values [0,0,1,2] -> 0: (0+0.5*2)/4=.25 ; 1: (2+.5)/4=.625 ; 2: (3+.5)/4=.875
    # class 1: values [2,2,5,0] -> 2: (1+.5*2)/4=.5 ; 5: (3+.5)/4=.875 ; 0: (0+.5)/4=.125
    expected = torch.tensor([0.25, 0.25, 0.625, 0.875, 0.5, 0.5, 0.875, 0.125])
    assert torch.allclose(U, expected)
    assert (U > 0).all() and (U < 1).all()


def test_hardness_views_values():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    y = torch.tensor([0, 2])
    mem = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    z_w = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    z_s = torch.tensor([[-3.0, 0.0], [0.0, 1.0]])
    hL, hT, hR, _ = core.hardness_views(logits, y, mem, z_w, z_s, 1e-12)
    p = torch.softmax(logits, 1)
    assert torch.allclose(hL, -torch.log(torch.stack([p[0, 0], p[1, 2]])))
    assert torch.allclose(hT, torch.tensor([0.3, 0.0]))       # 0.5-0.2 ; max(0.1,0.2)-0.7 < 0
    assert torch.allclose(hR, torch.tensor([1.0, 0.0]))       # opposite ; identical directions


def test_tail_exceedance_and_cotail_gate():
    U = torch.tensor([0.1, 0.8, 0.9, 1.0])
    x = core.tail_exceedance(U, 0.8)
    assert torch.allclose(x, torch.tensor([0.0, 0.0, 0.5, 1.0]))
    c, tau = core.cotail_score(torch.tensor([0.0, 1.0, 0.5]), torch.tensor([1.0, 0.0, 0.5]),
                               torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(c, torch.tensor([1.0, 0.0, 0.25]))
    assert torch.allclose(tau, torch.tensor([0.0, 0.0, math.sqrt(0.125)]))


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
def test_linear_quantile_matches_numpy(eta):
    x = torch.randn(1003, generator=torch.Generator().manual_seed(0))
    assert abs(core.linear_quantile(x, eta).item() - np.quantile(x.numpy(), eta, method="linear")) < 1e-12


def test_knn_bruteforce_and_constraints():
    g = torch.Generator().manual_seed(5)
    N, k = 200, 7
    labels = torch.randint(0, 4, (N,), generator=g)
    z = core.normalize_features(torch.randn(N, 16, generator=g), 1e-12)
    for block in (13, 4096):
        nbr = core.knn_within_class(z, labels, k, block_size=block)
        for i in range(N):
            assert i not in nbr[i].tolist()                        # leave-one-out
            assert (labels[nbr[i]] == labels[i]).all()             # same observed label
            cand = [j for j in range(N) if j != i and labels[j] == labels[i]]
            sims = (z[cand] @ z[i])
            top = {cand[j] for j in torch.topk(sims, k).indices.tolist()}
            assert set(nbr[i].tolist()) == top


def test_knn_small_class_raises():
    labels = torch.tensor([0] * 10 + [1] * 3)
    z = core.normalize_features(torch.randn(13, 4), 1e-12)
    with pytest.raises(ValueError, match="n_c >= k\\+1"):
        core.knn_within_class(z, labels, k=3)
    core.knn_within_class(z, labels, k=2)  # n_c = 3 = k+1 is allowed


def test_class_permutations_preserve_classes_and_multiset():
    labels = torch.tensor([0, 1, 2, 0, 1, 0, 2, 2, 2, 1])
    pi = core.class_permutations(labels, 50, torch.Generator().manual_seed(0))
    for p in range(50):
        assert sorted(pi[p].tolist()) == list(range(10))           # bijection
        assert (labels[pi[p]] == labels).all()                     # within-class


def test_class_permutations_are_uniform():
    labels = torch.zeros(4, dtype=torch.long)
    pi = core.class_permutations(labels, 24000, torch.Generator().manual_seed(1))
    counts = {}
    for row in pi.tolist():
        counts[tuple(row)] = counts.get(tuple(row), 0) + 1
    assert len(counts) == 24
    freq = np.array(list(counts.values())) / 24000
    assert np.all(np.abs(freq - 1 / 24) < 0.01)


# ---------------------------------------------------------------------------
# 3. Proposition 1: monotone-scale invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transform", [
    lambda h: torch.exp(h), lambda h: h ** 3 + 7.0, lambda h: torch.log1p(h) * 5 - 2,
    lambda h: torch.atan(h),
])
def test_prop1_monotone_invariance(transform):
    labels, logits, z_w, z_s, mem = random_problem(11, quantize=True)
    cfg = CTALConfig(q=0.6, k=5, P=9)
    hL, hT, hR, zh = core.hardness_views(logits, labels, mem, z_w, z_s, cfg.eps_z)
    # Snap to a grid so float rounding noise (~1e-17) cannot be merged into new
    # ties by the transform; ties that remain are genuine.
    hL, hT, hR = (torch.round(h * 1e6) / 1e6 for h in (hL, hT, hR))
    nbr = core.knn_within_class(zh, labels, cfg.k)
    pi = core.class_permutations(labels, cfg.P, torch.Generator().manual_seed(0))
    base = core.refresh_from_views(hL, hT, hR, labels, nbr, cfg, 2.0, 1.0, 2.0, pi=pi)
    for which in range(3):
        views = [hL, hT, hR]
        h, g = views[which], transform(views[which])
        # hypothesis of Prop. 1 holds on these floats: order and ties preserved
        assert torch.equal(h[:, None] < h[None, :], g[:, None] < g[None, :])
        assert torch.equal(h[:, None] == h[None, :], g[:, None] == g[None, :])
        views[which] = g
        out = core.refresh_from_views(*views, labels, nbr, cfg, 2.0, 1.0, 2.0, pi=pi)
        for key in ["U_L", "U_T", "U_R", "x_L", "c", "tau", "L", "H", "a", "r", "w"]:
            assert torch.equal(out[key], base[key]), (which, key)


# ---------------------------------------------------------------------------
# 4. Proposition 2: bounds and ERM recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_prop2_bounds(seed):
    labels, logits, z_w, z_s, mem = random_problem(seed, N=90)
    cfg = CTALConfig(q=0.5, k=6, P=10, rho=1.5, kappa=3.0, omega=0.2)
    lam = 3.0
    out = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, lam,
                               perm_generator=torch.Generator().manual_seed(seed))
    Rmax = 1 + cfg.rho + cfg.kappa
    vmin = cfg.omega + (1 - cfg.omega) * math.exp(-lam * Rmax)
    for key in ("c", "tau", "a"):
        assert (out[key] >= 0).all() and (out[key] <= 1).all()
    assert (out["r"] >= 0).all() and (out["r"] <= Rmax + 1e-12).all()
    assert (out["v"] >= vmin - 1e-12).all() and (out["v"] <= 1 + 1e-12).all()
    assert abs(out["w"].mean().item() - 1) < 1e-12
    assert (out["w"] >= vmin - 1e-12).all() and (out["w"] <= 1 / vmin + 1e-12).all()
    # loss-exceedance gate
    assert (out["r"][out["x_L"] == 0] == 0).all()


def test_prop2_erm_recovery():
    labels, logits, z_w, z_s, mem = random_problem(3)
    cfg = CTALConfig(q=0.5, k=5, P=5)
    for kwargs in ({"lam_t": 0.0, "variant": "full"}, {"lam_t": 2.0, "variant": "ce"}):
        out = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, **kwargs)
        assert torch.equal(out["w"], torch.ones_like(out["w"]))
    v, w = core.raw_and_normalized_weights(torch.full((17,), 0.37), 2.0, 0.1)
    assert torch.allclose(w, torch.ones(17), atol=1e-15)
    # Eq. (16) with unit weights is the ordinary minibatch mean
    l = torch.rand(33)
    assert torch.allclose(core.weighted_minibatch_loss(l, torch.ones(33)), l.mean())
    assert torch.allclose(core.weighted_minibatch_loss(l, 3.7 * torch.ones(33)), l.mean())


# ---------------------------------------------------------------------------
# 5. Proposition 3: exact permutation-null moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_c,k", [(6, 2), (7, 3), (5, 4)])
def test_prop3_exact_enumeration(n_c, k):
    g = torch.Generator().manual_seed(n_c * 10 + k)
    tau = torch.rand(n_c, generator=g)
    tau[0] = 0.0
    labels = torch.zeros(n_c, dtype=torch.long)
    z = core.normalize_features(torch.randn(n_c, 3, generator=g), 1e-12)
    nbr = core.knn_within_class(z, labels, k)
    vals = [[] for _ in range(n_c)]
    for perm in itertools.permutations(range(n_c)):
        s = core.local_means(tau[list(perm)], nbr)
        for i in range(n_c):
            vals[i].append(s[i].item())
    mean_c = tau.mean().item()
    S2 = tau.var(unbiased=True).item()
    var_theory = S2 / k * (1 - k / n_c)
    for i in range(n_c):
        v = np.array(vals[i])
        assert abs(v.mean() - mean_c) < 1e-12
        assert abs(v.var() - var_theory) < 1e-12   # population variance over all n_c! perms


def test_prop3_monte_carlo_with_sampler():
    """Same moments using the actual permutation sampler (multi-class)."""
    g = torch.Generator().manual_seed(0)
    labels = torch.tensor([0] * 40 + [1] * 25)
    tau = torch.rand(65, generator=g) ** 2
    z = core.normalize_features(torch.randn(65, 5, generator=g), 1e-12)
    k = 6
    nbr = core.knn_within_class(z, labels, k)
    pi = core.class_permutations(labels, 40000, g)
    s_null = core.null_local_means(tau, nbr, pi)
    for c, n in ((0, 40), (1, 25)):
        idx = (labels == c).nonzero().flatten()
        S2 = tau[idx].var(unbiased=True).item()
        i = idx[3].item()
        assert abs(s_null[:, i].mean().item() - tau[idx].mean().item()) < 4e-3
        assert abs(s_null[:, i].var().item() / (S2 / k * (1 - k / n)) - 1) < 0.04


# ---------------------------------------------------------------------------
# 6. Proposition 4: monotone contextual suppression
# ---------------------------------------------------------------------------

def test_prop4_monotone_and_ratio():
    g = torch.Generator().manual_seed(0)
    lam, omega = 2.0, 0.1
    r = torch.rand(50, generator=g) * 4
    _, w = core.raw_and_normalized_weights(r, lam, omega)
    order = torch.argsort(r)
    assert (torch.diff(w[order]) < 0).all()          # strictly decreasing in own risk
    for i, j in [(0, 1), (3, 17), (49, 2)]:
        ratio = (omega + (1 - omega) * math.exp(-lam * r[i])) / (omega + (1 - omega) * math.exp(-lam * r[j]))
        assert abs((w[i] / w[j]).item() - ratio) < 1e-12
    # raising only sample 0's risk lowers only its own weight (others rise)
    r2 = r.clone(); r2[0] += 0.5
    _, w2 = core.raw_and_normalized_weights(r2, lam, omega)
    assert w2[0] < w[0] and (w2[1:] > w[1:]).all()


def test_prop4_contextual_ordering():
    xL = torch.tensor([0.6, 0.6]); c = torch.tensor([0.3, 0.3]); a = torch.tensor([0.9, 0.2])
    r = core.risk(xL, c, a, rho=1.0, kappa=2.0)
    assert abs((r[0] - r[1]).item() - 0.6 * 2.0 * 0.7) < 1e-12
    _, w = core.raw_and_normalized_weights(torch.cat([r, torch.rand(5)]), 2.0, 0.1)
    assert w[0] < w[1]


# ---------------------------------------------------------------------------
# 7. Proposition 5: stability of the co-tail score
# ---------------------------------------------------------------------------

def _tau(u, q):
    x = core.tail_exceedance(u, q)
    return core.cotail_score(x[..., 0], x[..., 1], x[..., 2])[1]


@pytest.mark.parametrize("q", [0.0001, 0.5, 0.8, 0.95])
def test_prop5_bound(q):
    g = torch.Generator().manual_seed(int(q * 1000))
    u = torch.rand(200000, 3, generator=g)
    # bias toward the tail and toward tiny perturbations (worst case of the sqrt)
    v = (u + 0.002 * torch.randn(200000, 3, generator=g)).clamp(0, 1)
    u2 = q + (1 - q) * torch.rand(200000, 3, generator=g)
    for a, b in ((u, v), (u, u2), (u2, u2.flip(0))):
        lhs = (_tau(a, q) - _tau(b, q)).abs()
        d = (a - b).abs()
        rhs = torch.sqrt((d[:, 0] + 0.5 * d[:, 1] + 0.5 * d[:, 2]) / (1 - q))
        assert (lhs <= rhs + 1e-12).all()


# ---------------------------------------------------------------------------
# 8. Objective, memory, and the negative control
# ---------------------------------------------------------------------------

def test_weighted_loss_gradient_flows_only_through_losses():
    theta = torch.randn(5, requires_grad=True)
    w = (torch.rand(5) + 0.5).requires_grad_()
    l = theta ** 2
    core.weighted_minibatch_loss(l, w).backward()
    assert w.grad is None
    expected = (w.detach() * 2 * theta.detach()) / w.detach().sum()
    assert torch.allclose(theta.grad, expected)


def test_memory_update():
    m = torch.tensor([[0.5, 0.5]]); p = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(core.update_memory(m, p, 0.8), torch.tensor([[0.6, 0.4]]))


def test_permuted_activation_control():
    labels, logits, z_w, z_s, mem = random_problem(4)
    cfg = CTALConfig(q=0.5, k=5, P=7)
    kw = dict(pi=core.class_permutations(labels, 7, torch.Generator().manual_seed(0)))
    real = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, 2.0, **kw)
    ctrl = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, 2.0, activation_control="permute",
                                control_generator=torch.Generator().manual_seed(9), **kw)
    # Everything except the activation entering the risk is unchanged.
    for key in ("tau", "c", "x_L", "s", "a"):
        assert torch.equal(real[key], ctrl[key])
    for c in range(3):
        idx = labels == c
        assert torch.equal(torch.sort(ctrl["a_used"][idx]).values, torch.sort(real["a"][idx]).values)
    assert not torch.equal(ctrl["a_used"], real["a"])
    expected_r = real["x_L"] * (1 + cfg.rho * real["c"] + cfg.kappa * ctrl["a_used"])
    assert torch.allclose(ctrl["r"], expected_r)


@pytest.mark.parametrize("variant,rho_on,kappa_on", [
    ("loss_tail", 0, 0), ("pointwise", 1, 0), ("collective", 0, 1), ("full", 1, 1)])
def test_variant_risks(variant, rho_on, kappa_on):
    labels, logits, z_w, z_s, mem = random_problem(6)
    cfg = CTALConfig(q=0.5, k=5, P=7, rho=1.3, kappa=2.1)
    out = core.compute_refresh(logits, z_w, z_s, labels, mem, cfg, 2.0, variant=variant,
                               perm_generator=torch.Generator().manual_seed(0))
    expected = out["x_L"] * (1 + rho_on * cfg.rho * out["c"] + kappa_on * cfg.kappa * out["a"])
    assert torch.allclose(out["r"], expected)


def test_config_validation():
    with pytest.raises(ValueError):
        CTALConfig(eta0=0.99, eta1=0.9)
    with pytest.raises(ValueError):
        CTALConfig(q=1.0)
    with pytest.raises(ValueError):
        CTALConfig(omega=0.0)
