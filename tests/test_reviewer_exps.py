"""Tests for the code added for the reviewer experiments."""

import glob
import json
import os
import shlex
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ctal import core  # noqa: E402
from ctal.core import COTAIL_FORMS, CTALConfig  # noqa: E402
from ctal.data import (GROUP_CLUSTER, GROUP_RARE, ImageDataset, controlled_contamination,  # noqa: E402
                       make_long_tailed, make_synthetic_structured)
from ctal.losses import gce_loss, per_sample_loss, sce_loss  # noqa: E402
from ctal.metrics import excess_concentration, refresh_diagnostics  # noqa: E402
from ctal.trainer import CTALTrainer, TrainConfig, epoch_order  # noqa: E402

LABEL_DIR = os.environ.get("CIFARN_LABELS", "data")


# ---------------------------------------------------------------- losses

def test_base_losses_match_formulas():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 10, generator=g, dtype=torch.float64)
    y = torch.randint(0, 10, (64,), generator=g)
    p = torch.softmax(logits, 1)
    py = p[torch.arange(64), y]
    assert torch.allclose(per_sample_loss(logits, y, "ce"), -torch.log(py))
    assert torch.allclose(gce_loss(logits, y, 0.7), (1 - py ** 0.7) / 0.7)
    rce = -(p.clamp(1e-7, 1) * torch.log(F.one_hot(y, 10).double().clamp(1e-4, 1))).sum(1)
    assert torch.allclose(sce_loss(logits, y, 0.1, 1.0), 0.1 * -torch.log(py) + rce)
    # RCE = -log(1e-4) * (1 - p_y) (up to the 1e-7 clamp)
    assert torch.allclose(rce, -np.log(1e-4) * (1 - py), atol=1e-5)
    # defaults: CIFAR-10 (0.1, 1.0); CIFAR-100 (6.0, 0.1)
    assert torch.allclose(per_sample_loss(logits, y, "sce"), sce_loss(logits, y, 0.1, 1.0))
    l100 = torch.randn(4, 100, dtype=torch.float64); y100 = torch.tensor([0, 5, 99, 3])
    assert torch.allclose(per_sample_loss(l100, y100, "sce"), sce_loss(l100, y100, 6.0, 0.1))


@pytest.mark.parametrize("kind", ["ce", "gce", "sce"])
def test_base_losses_differentiable_and_finite(kind):
    logits = torch.tensor([[50.0, -50.0], [-50.0, 50.0]], requires_grad=True)
    loss = per_sample_loss(logits, torch.tensor([0, 0]), kind).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------- co-tail forms

def test_cotail_forms():
    xL = torch.tensor([0.0, 0.5, 1.0, 0.8, 0.4], dtype=torch.float64)
    xT = torch.tensor([1.0, 0.5, 0.0, 0.2, 0.9], dtype=torch.float64)
    xR = torch.tensor([1.0, 0.0, 0.0, 0.6, 0.1], dtype=torch.float64)
    exp = {
        "geometric": torch.sqrt(xL * (xT + xR) / 2),
        "additive": (xL + xT + xR) / 3,
        "product": (xL * xT * xR) ** (1 / 3),
        "min": torch.stack([xL, xT, xR]).min(0).values,
        "indicator": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float64),
    }
    for form in COTAIL_FORMS:
        c, tau = core.cotail_score(xL, xT, xR, form)
        assert torch.allclose(c, (xT + xR) / 2)
        assert torch.allclose(tau, exp[form]), form
        assert (tau >= 0).all() and (tau <= 1).all()
    with pytest.raises(ValueError):
        CTALConfig(cotail_form="max")


@pytest.mark.parametrize("form", COTAIL_FORMS)
def test_forms_keep_loss_gate_and_bounds(form):
    g = torch.Generator().manual_seed(1)
    N, K = 90, 3
    labels = torch.arange(N) % K
    cfg = CTALConfig(q=0.5, k=5, P=6, cotail_form=form)
    out = core.compute_refresh(torch.randn(N, K, generator=g, dtype=torch.float64),
                               torch.randn(N, 8, generator=g, dtype=torch.float64),
                               torch.randn(N, 8, generator=g, dtype=torch.float64), labels,
                               torch.softmax(torch.randn(N, K, generator=g, dtype=torch.float64), 1),
                               cfg, 2.0, perm_generator=g)
    assert (out["r"][out["x_L"] == 0] == 0).all()     # gate holds for every form
    assert abs(out["w"].mean().item() - 1) < 1e-12
    assert (out["a"] >= 0).all() and (out["a"] <= 1).all()


# ---------------------------------------------------------------- samplers

def test_class_blocked_order():
    labels = torch.arange(1000) % 10
    g = torch.Generator().manual_seed(0)
    o = epoch_order(labels, "class_blocked", 50, g)
    assert sorted(o.tolist()) == list(range(1000))
    pure = sum(len(torch.unique(labels[b])) == 1 for b in torch.split(o, 50))
    assert pure == 20                                   # 100 per class / 50 per batch
    r = epoch_order(labels, "random", 50, torch.Generator().manual_seed(0))
    assert sum(len(torch.unique(labels[b])) == 1 for b in torch.split(r, 50)) == 0
    o2 = epoch_order(labels, "class_blocked", 50, torch.Generator().manual_seed(0))
    assert torch.equal(o, o2)


# ---------------------------------------------------------------- controlled noise

def _toy_images(n_per=200, K=4, seed=0):
    d = make_synthetic_structured(n_per_class=n_per, n_test_per_class=50, num_classes=K,
                                  contaminated_fraction=0.0, seed=seed)
    return d.x_train, d.y_train_clean, d.x_test, d.y_test


@pytest.fixture(scope="module")
def controlled():
    x, y, xt, yt = _toy_images()
    return {n: controlled_contamination(x, y, xt, yt, n, rate=0.25, n_clusters=4, rare_keep=0.1,
                                        data_seed=3)
            for n in ("none", "cluster", "pair_random", "symmetric")}


def test_controlled_same_images_and_groups(controlled):
    ref = controlled["none"]
    assert (ref.y_train == ref.y_train_clean).all()
    for d in controlled.values():
        assert torch.equal(d.x_train, ref.x_train)
        assert torch.equal(d.y_train_clean, ref.y_train_clean)
        assert torch.equal(d.train_group, ref.train_group)
        assert torch.equal(d.test_group, ref.test_group)
        # rare subgroup is never relabelled
        rare = d.train_group == GROUP_RARE
        assert rare.any() and (d.y_train[rare] == d.y_train_clean[rare]).all()


def test_controlled_noise_placement_and_matched_counts(controlled):
    c, pr, sy = controlled["cluster"], controlled["pair_random"], controlled["symmetric"]
    K = 4
    pairs = {k: (k + 1) % K for k in range(K)}
    # cluster: exactly the contamination cluster is flipped, to the paired class
    flipped = c.y_train != c.y_train_clean
    assert torch.equal(flipped, c.train_group == GROUP_CLUSTER)
    for k in range(K):
        m = (c.y_train_clean == k) & flipped
        assert (c.y_train[m] == pairs[k]).all()
    # identical transition matrices for cluster vs pair_random (exact counts)
    def tm(d):
        M = torch.zeros(K, K, dtype=torch.long)
        M.index_put_((d.y_train_clean, d.y_train), torch.ones_like(d.y_train), accumulate=True)
        return M
    assert torch.equal(tm(c), tm(pr))
    # pair_random flips are spread out: much less overlap with the cluster
    fpr = pr.y_train != pr.y_train_clean
    assert (fpr & (pr.train_group == GROUP_CLUSTER)).sum() < 0.6 * fpr.sum()
    # symmetric: same per-class flip counts, never to the true class
    fsy = sy.y_train != sy.y_train_clean
    for k in range(K):
        assert int((flipped & (c.y_train_clean == k)).sum()) == int((fsy & (sy.y_train_clean == k)).sum())
    assert (sy.y_train[fsy] != sy.y_train_clean[fsy]).all()


def test_controlled_rare_subgroup_is_rare(controlled):
    d = controlled["none"]
    x, y, _, _ = _toy_images()
    n_rare = int((d.train_group == GROUP_RARE).sum())
    assert len(d.y_train) < len(y)                       # rare examples were removed
    assert n_rare < 0.05 * len(d.y_train)


def test_long_tailed():
    d = make_synthetic_structured(n_per_class=300, num_classes=10, seed=0)
    d = make_long_tailed(d, 10.0, data_seed=0)
    counts = torch.bincount(d.y_train_clean, minlength=10).tolist()
    assert counts[0] == 300 and counts[-1] == 30
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert torch.equal(d.train_group, d.y_train_clean)


# ---------------------------------------------------------------- diagnostics

def test_excess_concentration_and_diagnostics():
    g = torch.Generator().manual_seed(0)
    N, K = 120, 3
    labels = torch.arange(N) % K
    cfg = CTALConfig(q=0.5, k=6, P=5)
    st = core.compute_refresh(torch.randn(N, K, generator=g, dtype=torch.float64),
                              torch.randn(N, 8, generator=g, dtype=torch.float64),
                              torch.randn(N, 8, generator=g, dtype=torch.float64), labels,
                              torch.softmax(torch.randn(N, K, generator=g, dtype=torch.float64), 1),
                              cfg, 2.0, perm_generator=g)
    ex = excess_concentration(st, labels)
    for i in (0, 7, 50):
        cm = st["tau"][labels == labels[i]].mean()
        assert abs((st["s"][i] - cm).item() - ex[i].item()) < 1e-12
    noisy = torch.zeros(N, dtype=torch.bool); noisy[::4] = True
    grp = (torch.arange(N) % 2)
    d = refresh_diagnostics(st, noisy, 0.5, group=grp, group_names={0: "a", 1: "b"}, labels=labels)
    for key in ("auroc", "tail_auroc", "assumption1", "w_by_group", "w_noisy_tail",
                "tail_w_resid_std_given_loss"):
        assert key in d
    assert set(d["w_by_group"]) == {"a", "b"}
    json.dumps(d)  # serialisable


# ---------------------------------------------------------------- trainer + CLI

def tiny_cfg(**kw):
    from ctal.core import CTALConfig as C
    base = dict(arch="smallcnn", epochs=6, batch_size=64, lr=0.05, memory_init_after=2, warmup=3,
                ramp_epochs=2, refresh_every=2, seed=0, eval_ema=False, ctal=C(k=8, P=5))
    base.update(kw)
    return TrainConfig(**base)


@pytest.mark.parametrize("base_loss", ["gce", "sce"])
def test_trainer_base_losses_and_groups(tmp_path, base_loss, controlled):
    d = controlled["cluster"]
    data = ImageDataset(d.x_train.clone(), d.y_train.clone(), d.x_test.clone(), d.y_test.clone(),
                        d.num_classes, d.mean, d.std, d.y_train_clean.clone(), d.name,
                        d.train_group.clone(), d.test_group.clone(), dict(d.group_names))
    tr = CTALTrainer(data, tiny_cfg(base_loss=base_loss, batch_sampler="class_blocked"),
                     device="cpu", dump_dir=str(tmp_path / "states"))
    s = tr.train()
    assert set(s["final_group_acc"]) == {"other", "contamination_cluster", "rare_clean_subgroup"}
    assert len(s["final_per_class_acc"]) == 4
    assert s["timing"]["refresh_s"] > 0 and s["timing"]["train_s"] > 0
    files = sorted(os.listdir(tmp_path / "states"))
    assert len(files) == 2                              # first and last active refresh
    z = np.load(tmp_path / "states" / files[0])
    assert z["w"].shape == (len(data.y_train),) and "group" in z and "y_clean" in z
    ref = s["history"]["refreshes"][0]
    assert "rare_clean_subgroup" in ref["w_by_group"] and "tail_auroc" in ref


def test_gce_base_variant_is_unweighted(controlled):
    d = controlled["cluster"]
    tr = CTALTrainer(d, tiny_cfg(base_loss="gce", variant="base", epochs=5), device="cpu")
    tr.train()
    assert torch.equal(tr.weights, torch.ones_like(tr.weights))


def test_every_manifest_command_parses():
    sys.path.insert(0, ROOT)
    import experiments
    from train import build_parser
    jobs = experiments.experiments("gce")
    tags_methods = set()
    for j in jobs:
        args = [x.format(c10="a.pt", c100="b.pt") for x in j["args"]]
        a = build_parser().parse_args(args)
        from train import method_label
        key = (a.tag, method_label(a), a.seed)
        assert key not in tags_methods, f"duplicate run id {key}"   # no run overwrites another
        tags_methods.add(key)
    assert len(jobs) == len(tags_methods)
    ids = {j["exp"] for j in jobs}
    assert {"E1_decomposition", "E2_controlled", "E4_aggre", "E5_cotail_form", "E6_batch",
            "E7_longtail", "E8_sensitivity", "E9_ce_dropin"} <= ids


@pytest.mark.skipif(not os.path.exists(os.path.join(LABEL_DIR, "CIFAR-10_human.pt")),
                    reason="CIFAR-N label files not available")
def test_cifarn_pipeline_with_real_labels(tmp_path, monkeypatch):
    """Real CIFAR-N label files through the full CLI; images are random because
    the CIFAR download is unavailable in the test environment."""
    import torchvision
    from ctal.data import _load_label_file

    def fake(n_classes, file):
        lab = _load_label_file(os.path.join(LABEL_DIR, file))["clean_label"]
        class Fake:
            def __init__(self, root, train=True, download=False):
                n = len(lab) if train else 200
                rng = np.random.default_rng(0 if train else 1)
                self.data = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
                self.targets = lab.tolist() if train else (np.arange(n) % n_classes).tolist()
        return Fake

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", fake(10, "CIFAR-10_human.pt"))
    monkeypatch.setattr(torchvision.datasets, "CIFAR100", fake(100, "CIFAR-100_human.pt"))
    import train
    from ctal.data import load_cifar_n
    d = load_cifar_n("x", "cifar10", os.path.join(LABEL_DIR, "CIFAR-10_human.pt"), "worse_label")
    assert abs(d.noisy_mask.double().mean().item() - 0.4021) < 1e-3
    # subsample to keep the run short: patch load_data to keep 2,000 examples
    real_load = train.load_data

    def small(a):
        dd = real_load(a)
        return dd.subset_train(torch.randperm(len(dd.y_train), generator=torch.Generator().manual_seed(0))[:2000])

    monkeypatch.setattr(train, "load_data", small)
    for ds, lf, nt in (("cifar10", "CIFAR-10_human.pt", "worse_label"),
                       ("cifar100", "CIFAR-100_human.pt", "noisy_label")):
        train.main(["--dataset", ds, "--noise_type", nt, "--label_file", os.path.join(LABEL_DIR, lf),
                    "--arch", "smallcnn", "--epochs", "4", "--warmup", "2", "--memory_init_after", "1",
                    "--ramp_epochs", "1", "--refresh_every", "1", "--k", "4", "--P", "3",
                    "--base_loss", "sce", "--tag", f"t_{ds}", "--out_dir", str(tmp_path), "--device", "cpu"])
        s = glob.glob(str(tmp_path / f"t_{ds}" / "*_summary.json"))
        assert len(s) == 1


def test_amplification_spread_zero_iff_loss_rank_only():
    g = torch.Generator().manual_seed(2)
    N, K = 150, 3
    labels = torch.arange(N) % K
    args = (torch.randn(N, K, generator=g, dtype=torch.float64),
            torch.randn(N, 8, generator=g, dtype=torch.float64),
            torch.randn(N, 8, generator=g, dtype=torch.float64), labels,
            torch.softmax(torch.randn(N, K, generator=g, dtype=torch.float64), 1))
    cfg = CTALConfig(q=0.5, k=6, P=5)
    noisy = torch.zeros(N, dtype=torch.bool); noisy[::3] = True
    res = {}
    for v in ("loss_tail", "full"):
        st = core.compute_refresh(*args, cfg, 2.0, variant=v,
                                  pi=core.class_permutations(labels, 5, torch.Generator().manual_seed(0)))
        res[v] = refresh_diagnostics(st, noisy, 0.5, labels=labels)["tail_amplification_std"]
    assert res["loss_tail"] == 0.0 and res["full"] > 0.05
