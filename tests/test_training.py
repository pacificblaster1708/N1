"""Tests for the training loop: schedule, EMA, memory ordering, augmentation,
paired-run determinism and an end-to-end run."""

import copy
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctal import core  # noqa: E402
from ctal import trainer as trainer_mod  # noqa: E402
from ctal.core import CTALConfig  # noqa: E402
from ctal.data import make_synthetic_structured, random_crop_flip  # noqa: E402
from ctal.ema import ModelEMA, ema_decay  # noqa: E402
from ctal.models import build_model  # noqa: E402
from ctal.trainer import CTALTrainer, TrainConfig, lambda_at, lr_at, refresh_epochs  # noqa: E402


# ---------------------------------------------------------------- schedule

def test_paper_schedule():
    cfg = TrainConfig()
    assert refresh_epochs(cfg) == list(range(11, 21)) + list(range(21, 97, 5))
    assert refresh_epochs(cfg)[-1] == 96 and 26 in refresh_epochs(cfg)
    assert [lambda_at(e, cfg) for e in (1, 10)] == [0.0, 0.0]
    assert abs(lambda_at(11, cfg) - 0.2) < 1e-12
    assert lambda_at(20, cfg) == 2.0 and lambda_at(96, cfg) == 2.0
    lams = [lambda_at(e, cfg) for e in range(11, 21)]
    assert all(b > a for a, b in zip(lams, lams[1:]))
    assert lr_at(1, cfg) == 0.1 and lr_at(60, cfg) == 0.1
    assert abs(lr_at(61, cfg) - 0.01) < 1e-15 and abs(lr_at(100, cfg) - 0.01) < 1e-15


def test_memory_must_precede_first_refresh():
    with pytest.raises(ValueError):
        TrainConfig(memory_init_after=10, warmup=10)


# ---------------------------------------------------------------- EMA

def test_ema_decay_rule():
    assert abs(ema_decay(1, 0.999) - 2 / 11) < 1e-15
    assert abs(ema_decay(100, 0.999) - 101 / 110) < 1e-15
    assert ema_decay(10 ** 6, 0.999) == 0.999


def test_ema_update_matches_manual():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    ema = ModelEMA(model, 0.9)
    manual = {k: v.clone().double() for k, v in model.state_dict().items()}
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for u in range(1, 30):
        model.train()
        loss = model(torch.randn(8, 4)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(model)
        d = min(0.9, (1 + u) / (10 + u))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                manual[k] = d * manual[k] + (1 - d) * v.double()
            else:
                manual[k] = v.clone()
    for k, v in ema.module.state_dict().items():
        if v.dtype.is_floating_point:
            assert torch.allclose(v.double(), manual[k], atol=1e-6), k
        else:
            assert torch.equal(v, manual[k].to(v.dtype)), k  # num_batches_tracked copied
    assert ema.module.state_dict()["1.num_batches_tracked"].item() == 29


def test_snapshot_is_frozen_copy():
    model = nn.Linear(3, 2)
    ema = ModelEMA(model)
    snap = ema.snapshot()
    assert not snap.training and all(not p.requires_grad for p in snap.parameters())
    with torch.no_grad():
        ema.module.weight.add_(1.0)
    assert not torch.equal(snap.weight, ema.module.weight)


# ---------------------------------------------------------------- augmentation

def test_random_crop_flip_matches_slicing():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, 3, 32, 32)
    out = random_crop_flip(x, g)
    g2 = torch.Generator().manual_seed(3)
    oy = torch.randint(0, 9, (64,), generator=g2)
    ox = torch.randint(0, 9, (64,), generator=g2)
    flip = torch.rand(64, generator=g2) < 0.5
    xp = F.pad(x, (4, 4, 4, 4))
    for b in range(64):
        crop = xp[b, :, oy[b]:oy[b] + 32, ox[b]:ox[b] + 32]
        if flip[b]:
            crop = crop.flip(-1)
        assert torch.equal(out[b], crop)
    assert flip.any() and (~flip).any()


def test_crop_on_uint8_pads_black_like_torchvision():
    import torchvision.transforms.v2 as T
    x = torch.randint(1, 256, (1, 3, 32, 32), dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    outs = [random_crop_flip(x, g) for _ in range(200)]
    assert all(o.dtype == torch.uint8 for o in outs)
    # padded border pixels are exactly 0, as torchvision's RandomCrop(padding=4) fill=0
    tv = T.RandomCrop(32, padding=4)
    torch.manual_seed(0)
    tv_zero_frac = sum((tv(x[0]) == 0).float().mean().item() for _ in range(2000)) / 2000
    ours_zero_frac = sum((o == 0).float().mean().item() for o in outs) / 200
    assert abs(tv_zero_frac - ours_zero_frac) < 0.02


def test_models_expose_features():
    for arch, dim in (("smallcnn", 128), ("resnet18", 512), ("resnet34", 512)):
        m = build_model(arch, 7).eval()
        logits, feat = m(torch.randn(2, 3, 32, 32), return_features=True)
        assert logits.shape == (2, 7) and feat.shape == (2, dim)
        assert torch.allclose(m.linear(feat), logits, atol=1e-5)


# ---------------------------------------------------------------- trainer

def tiny_cfg(**kw):
    base = dict(arch="smallcnn", epochs=8, batch_size=64, lr=0.05, memory_init_after=2,
                warmup=3, ramp_epochs=2, refresh_every=2, seed=0, eval_ema=False,
                ctal=CTALConfig(k=8, P=5))
    base.update(kw)
    return TrainConfig(**base)


def tiny_data():
    return make_synthetic_structured(n_per_class=40, n_test_per_class=10, num_classes=4, seed=1)


def test_memory_read_before_update(monkeypatch):
    """Refresh-t scores use m^{t-1}; the memory changes only after scoring."""
    seen = []
    real = trainer_mod.compute_refresh

    def spy(logits_w, z_w, z_s, labels, memory_prev, *a, **k):
        seen.append((memory_prev.clone(), logits_w.clone()))
        return real(logits_w, z_w, z_s, labels, memory_prev, *a, **k)

    monkeypatch.setattr(trainer_mod, "compute_refresh", spy)
    tr = CTALTrainer(tiny_data(), tiny_cfg(epochs=6), device="cpu")
    tr.train()
    assert len(seen) == 3                                   # epochs 4, 5, 6 (= ramp 4-5, then 6)
    for t in range(1, len(seen)):
        m_prev, logits_prev = seen[t - 1]
        expected = 0.8 * m_prev + 0.2 * torch.softmax(logits_prev, 1)
        assert torch.allclose(seen[t][0], expected)
    assert torch.allclose(tr.memory, 0.8 * seen[-1][0] + 0.2 * torch.softmax(seen[-1][1], 1))


def test_trainer_refresh_epochs_and_weights():
    tr = CTALTrainer(tiny_data(), tiny_cfg(), device="cpu")
    out = tr.train()
    epochs = [r["epoch"] for r in out["history"]["refreshes"]]
    assert epochs == refresh_epochs(tr.cfg) == [4, 5, 6, 8]
    assert abs(tr.weights.double().mean().item() - 1) < 1e-6
    assert out["history"]["refreshes"][0]["first_active_refresh"]
    assert abs(out["history"]["refreshes"][0]["lambda_t"] - 1.0) < 1e-12
    assert len(out["history"]["epochs"]) == 8
    assert 0 <= out["final_acc"] <= 1
    assert "auroc" in out["history"]["refreshes"][0]


def test_paired_runs_share_warmup_and_ce_equals_plain_training():
    """Variants differ only after the first active refresh; 'ce' never deviates from ERM."""
    runs = {}
    for v in ("ce", "full"):
        tr = CTALTrainer(tiny_data(), tiny_cfg(variant=v, epochs=5), device="cpu")
        tr.train()
        runs[v] = tr
    h_ce = runs["ce"].history["epochs"]; h_full = runs["full"].history["epochs"]
    for e in range(3):                                     # warm-up epochs identical
        assert h_ce[e]["train_loss"] == h_full[e]["train_loss"]
    assert h_ce[4]["train_loss"] != h_full[4]["train_loss"]
    assert torch.equal(runs["ce"].weights, torch.ones_like(runs["ce"].weights))

    # 'ce' variant == plain unweighted training with the same RNG streams
    data = tiny_data()
    cfg = tiny_cfg(epochs=5)
    torch.manual_seed(cfg.seed)
    model = build_model("smallcnn", data.num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    from ctal.data import normalize
    for epoch in range(1, 6):
        model.train()
        order = torch.randperm(data.x_train.shape[0], generator=gen)
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            xb = normalize(random_crop_flip(data.x_train[idx], gen), data.mean, data.std)
            loss = F.cross_entropy(model(xb), data.y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    for p_ref, p_ce in zip(model.parameters(), runs["ce"].model.parameters()):
        assert torch.allclose(p_ref, p_ce, atol=1e-5)


def test_determinism():
    a = CTALTrainer(tiny_data(), tiny_cfg(epochs=5), device="cpu"); a.train()
    b = CTALTrainer(tiny_data(), tiny_cfg(epochs=5), device="cpu"); b.train()
    assert torch.equal(a.weights, b.weights)
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)


def test_permuted_control_runs():
    tr = CTALTrainer(tiny_data(), tiny_cfg(epochs=5, activation_control="permute"), device="cpu")
    tr.train()
    assert abs(tr.weights.double().mean().item() - 1) < 1e-6
