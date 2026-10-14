"""CTAL training loop (Algorithm 1)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .core import CTALConfig, VARIANTS, compute_refresh, update_memory, weighted_minibatch_loss
from .data import ImageDataset, normalize, random_crop_flip
from .ema import ModelEMA
from .losses import BASE_LOSSES, per_sample_loss
from .metrics import refresh_diagnostics
from .models import build_model

BATCH_SAMPLERS = ("random", "class_blocked")
DUMP_MODES = ("none", "first_last", "all")
DUMP_KEYS = ("h_L", "h_T", "h_R", "U_L", "U_T", "U_R", "x_L", "x_T", "x_R", "c", "tau",
             "s", "L", "H", "a", "a_used", "r", "w")


@dataclass
class TrainConfig:
    """Optimization / schedule settings (defaults = paper's CIFAR-N protocol)."""

    arch: str = "resnet34"
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: tuple = (60,)       # lr /10 after epoch 60
    lr_gamma: float = 0.1
    ema_delta: float = 0.999
    memory_init_after: int = 5         # E_m: memory initialized before epoch 6
    warmup: int = 10                   # E_w: weights = 1 during epochs 1..10
    ramp_epochs: int = 10              # lambda reaches full strength at epoch 20
    refresh_every: int = 5             # after the ramp: 21, 26, ..., 96
    variant: str = "full"              # base/ce | loss_tail | pointwise | collective | full
    activation_control: str = "none"   # none | permute (negative control)
    base_loss: str = "ce"              # student loss wrapped by CTAL: ce | gce | sce
    gce_q: float = 0.7
    sce_alpha: Optional[float] = None  # None -> SCE paper defaults per dataset
    sce_beta: Optional[float] = None
    batch_sampler: str = "random"      # random | class_blocked (composition stress test)
    dump_refreshes: str = "first_last" # per-sample refresh states saved for analysis
    seed: int = 42
    score_batch_size: int = 1000
    knn_block_size: int = 2048
    eval_ema: bool = True              # also report EMA test accuracy (diagnostic)
    log_every: int = 0                 # minibatch logging interval (0 = off)
    ctal: CTALConfig = field(default_factory=CTALConfig)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        if self.base_loss not in BASE_LOSSES:
            raise ValueError(f"base_loss must be one of {BASE_LOSSES}")
        if self.batch_sampler not in BATCH_SAMPLERS:
            raise ValueError(f"batch_sampler must be one of {BATCH_SAMPLERS}")
        if self.dump_refreshes not in DUMP_MODES:
            raise ValueError(f"dump_refreshes must be one of {DUMP_MODES}")
        if self.memory_init_after >= self.warmup:
            raise ValueError("memory must be initialized before the first refresh "
                             "(memory_init_after < warmup)")
        if self.ramp_epochs < 1 or self.refresh_every < 1:
            raise ValueError("ramp_epochs and refresh_every must be >= 1")


def refresh_epochs(cfg: TrainConfig) -> List[int]:
    """Refresh schedule E: every ramp epoch, then every ``refresh_every`` epochs."""
    ramp_end = cfg.warmup + cfg.ramp_epochs
    ramp = list(range(cfg.warmup + 1, min(ramp_end, cfg.epochs) + 1))
    after = list(range(ramp_end + 1, cfg.epochs + 1, cfg.refresh_every))
    return ramp + after


def lambda_at(epoch: int, cfg: TrainConfig) -> float:
    """Linear ramp: lambda_e = lambda * (e - E_w) / R for E_w < e <= E_w + R, then lambda."""
    if epoch <= cfg.warmup:
        return 0.0
    return cfg.ctal.lam * min(1.0, (epoch - cfg.warmup) / cfg.ramp_epochs)


def lr_at(epoch: int, cfg: TrainConfig) -> float:
    """Learning rate used during (1-based) ``epoch``."""
    n_decays = sum(1 for m in cfg.lr_milestones if epoch > m)
    return cfg.lr * (cfg.lr_gamma ** n_decays)


def epoch_order(labels: torch.Tensor, mode: str, batch_size: int,
                generator: torch.Generator) -> torch.Tensor:
    """Sample order for one epoch.

    random        : uniform random permutation.
    class_blocked : adversarial composition -- samples are grouped by observed
                    label (class order random, order within class random), cut
                    into minibatches, and the minibatch order is shuffled.  Most
                    minibatches then contain a single class.  CTAL's weights are
                    computed on the full training set, so they cannot depend on
                    this; the test checks the training dynamics end to end.
    """
    N = labels.numel()
    dev = labels.device
    perm = torch.randperm(N, device=dev, generator=generator)
    if mode == "random":
        return perm
    K = int(labels.max()) + 1
    class_rank = torch.randperm(K, device=dev, generator=generator)
    key = class_rank[labels[perm]]
    order = perm[torch.sort(key, stable=True).indices]
    batches = list(torch.split(order, batch_size))
    bperm = torch.randperm(len(batches), device=dev, generator=generator).tolist()
    return torch.cat([batches[i] for i in bperm])


class CTALTrainer:
    def __init__(self, data: ImageDataset, cfg: TrainConfig, device: Optional[str] = None,
                 log_path: Optional[str] = None, dump_dir: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.data = data.to(self.device)
        self.N = data.x_train.shape[0]
        self.K = data.num_classes
        self.log_path = log_path
        self.dump_dir = dump_dir

        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg.arch, self.K).to(self.device)
        self.ema = ModelEMA(self.model, cfg.ema_delta)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                   weight_decay=cfg.weight_decay)

        # Independent RNG streams so paired variants see identical training data.
        self.train_gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 1)
        self.score_gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 2)
        self.perm_gen = torch.Generator().manual_seed(cfg.seed + 3)
        self.ctrl_gen = torch.Generator().manual_seed(cfg.seed + 4)

        self.weights = torch.ones(self.N, dtype=torch.float32, device=self.device)
        self.memory: Optional[torch.Tensor] = None   # (N, K) float64 on CPU
        self.schedule = refresh_epochs(cfg)
        self.refresh_schedule = set(self.schedule)
        self.history: Dict[str, list] = {"epochs": [], "refreshes": []}
        self.time = {"train_s": 0.0, "refresh_s": 0.0, "memory_init_s": 0.0}

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _score(self, scorer: nn.Module, with_strong: bool):
        """Detached pass of a frozen scorer over the full training set."""
        logits_w, z_w, z_s = [], [], []
        x_all = self.data.x_train
        bs = self.cfg.score_batch_size
        for start in range(0, self.N, bs):
            raw = x_all[start:start + bs]
            xb = normalize(raw, self.data.mean, self.data.std)       # weak view a_w
            lw, fw = scorer(xb, return_features=True)
            logits_w.append(lw.double().cpu()); z_w.append(fw.float())
            if with_strong:                                           # strong view a_s
                xs = normalize(random_crop_flip(raw, self.score_gen), self.data.mean, self.data.std)
                _, fs = scorer(xs, return_features=True)
                z_s.append(fs.float())
        logits_w = torch.cat(logits_w)
        z_w = torch.cat(z_w)
        z_s = torch.cat(z_s) if with_strong else None
        return logits_w, z_w, z_s

    def _init_memory(self) -> None:
        t0 = time.time()
        scorer = self.ema.snapshot()
        logits_w, _, _ = self._score(scorer, with_strong=False)
        self.memory = torch.softmax(logits_w, dim=1)
        self.time["memory_init_s"] += time.time() - t0

    def _should_dump(self, t: int) -> bool:
        mode = self.cfg.dump_refreshes
        if mode == "none" or self.dump_dir is None:
            return False
        if mode == "all":
            return True
        n_active = sum(1 for e in self.schedule if e > self.cfg.warmup)
        return t == 1 or t == n_active

    def _dump(self, t: int, epoch: int, state: Dict[str, torch.Tensor]) -> None:
        os.makedirs(self.dump_dir, exist_ok=True)
        arrays = {k: state[k].float().numpy() for k in DUMP_KEYS}
        arrays["nbr"] = state["nbr"].numpy().astype(np.int32)
        arrays["y_obs"] = self.data.y_train.cpu().numpy()
        if self.data.y_train_clean is not None:
            arrays["y_clean"] = self.data.y_train_clean.cpu().numpy()
        if self.data.train_group is not None:
            arrays["group"] = self.data.train_group.cpu().numpy()
        arrays["epoch"] = np.array(epoch); arrays["refresh"] = np.array(t)
        np.savez_compressed(os.path.join(self.dump_dir, f"refresh_{t:03d}_epoch_{epoch:03d}.npz"), **arrays)

    def _refresh(self, epoch: int, t: int) -> Dict:
        cfg = self.cfg
        t0 = time.time()
        scorer = self.ema.snapshot()                       # Alg. 1 line 7
        logits_w, z_w, z_s = self._score(scorer, True)     # line 8
        lam_t = lambda_at(epoch, cfg)
        state = compute_refresh(                           # lines 9-16
            logits_w, z_w, z_s, self.data.y_train, self.memory, cfg.ctal, lam_t,
            variant=cfg.variant, perm_generator=self.perm_gen,
            activation_control=cfg.activation_control, control_generator=self.ctrl_gen,
            knn_block_size=cfg.knn_block_size,
        )
        self.weights = state["w"].float().to(self.device)
        # line 17: memory updated only after every refresh-t score is complete
        self.memory = update_memory(self.memory, torch.softmax(logits_w, dim=1), cfg.ctal.beta)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        self.time["refresh_s"] += elapsed
        diag = refresh_diagnostics(state, self.data.noisy_mask, cfg.ctal.q,
                                   group=self.data.train_group, group_names=self.data.group_names,
                                   labels=self.data.y_train)
        diag.update({"epoch": epoch, "refresh_index": t, "lambda_t": lam_t,
                     "first_active_refresh": t == 1, "refresh_time_s": elapsed})
        if self._should_dump(t):
            self._dump(t, epoch, state)
        return diag

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def predict(self, model: nn.Module) -> torch.Tensor:
        was_training = model.training
        model.eval()
        preds = []
        bs = self.cfg.score_batch_size
        for start in range(0, self.data.x_test.shape[0], bs):
            xb = normalize(self.data.x_test[start:start + bs], self.data.mean, self.data.std)
            preds.append(model(xb).argmax(dim=1))
        model.train(was_training)
        return torch.cat(preds)

    def evaluate(self, model: nn.Module) -> float:
        return (self.predict(model) == self.data.y_test).double().mean().item()

    def _detailed_eval(self, pred: torch.Tensor) -> Dict:
        correct = (pred == self.data.y_test).double()
        y = self.data.y_test
        per_class = [correct[y == c].mean().item() if (y == c).any() else float("nan")
                     for c in range(self.K)]
        out = {"per_class_acc": per_class}
        if self.data.test_group is not None:
            names = self.data.group_names or {}
            out["group_acc"] = {
                names.get(int(gid), str(int(gid))): correct[self.data.test_group == gid].mean().item()
                for gid in torch.unique(self.data.test_group).tolist()
            }
        return out

    # ------------------------------------------------------------------ train
    def train(self) -> Dict:
        cfg = self.cfg
        t = 0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            for g in self.opt.param_groups:
                g["lr"] = lr_at(epoch, cfg)

            if self.memory is None and epoch > cfg.memory_init_after:      # lines 3-5
                self._init_memory()

            if epoch > cfg.warmup and epoch in self.refresh_schedule:      # lines 6-18
                t += 1
                diag = self._refresh(epoch, t)
                self.history["refreshes"].append(diag)
                self._log({"type": "refresh", **diag})

            t_train = time.time()
            self.model.train()
            order = epoch_order(self.data.y_train, cfg.batch_sampler, cfg.batch_size, self.train_gen)
            tot_loss, tot_n = 0.0, 0
            for b, start in enumerate(range(0, self.N, cfg.batch_size)):       # lines 19-23
                idx = order[start:start + cfg.batch_size]
                xb = random_crop_flip(self.data.x_train[idx], self.train_gen)
                xb = normalize(xb, self.data.mean, self.data.std)
                yb = self.data.y_train[idx]
                per_sample = per_sample_loss(self.model(xb), yb, cfg.base_loss, cfg.gce_q,
                                             cfg.sce_alpha, cfg.sce_beta)
                loss = weighted_minibatch_loss(per_sample, self.weights[idx])
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                self.ema.update(self.model)
                tot_loss += per_sample.detach().sum().item(); tot_n += idx.numel()
                if cfg.log_every and b % cfg.log_every == 0:
                    print(f"  epoch {epoch} batch {b} loss {loss.item():.4f}", flush=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.time["train_s"] += time.time() - t_train

            pred = self.predict(self.model)
            rec = {"epoch": epoch, "lr": lr_at(epoch, cfg), "train_loss": tot_loss / tot_n,
                   "test_acc": (pred == self.data.y_test).double().mean().item(),
                   "time_s": time.time() - t0}
            detail = self._detailed_eval(pred)
            if "group_acc" in detail:
                rec["group_acc"] = detail["group_acc"]
            if epoch == cfg.epochs:
                rec["per_class_acc"] = detail["per_class_acc"]
            if cfg.eval_ema:
                rec["test_acc_ema"] = self.evaluate(self.ema.module)
            self.history["epochs"].append(rec)
            self._log({"type": "epoch", **rec})
            print(json.dumps({k: v for k, v in rec.items() if k != "per_class_acc"}), flush=True)
        return self.summary()

    def summary(self) -> Dict:
        eps = self.history["epochs"]
        accs = [e["test_acc"] for e in eps]
        best = max(range(len(accs)), key=lambda i: accs[i])
        last = eps[-10:]
        out = {
            "final_acc": accs[-1],                                   # primary metric
            "last10_mean_acc": sum(accs[-10:]) / len(accs[-10:]),    # stability
            "best_acc_diagnostic": accs[best],                       # diagnostic only
            "best_epoch_diagnostic": best + 1,
            "final_per_class_acc": eps[-1].get("per_class_acc"),
            "timing": dict(self.time),
            "config": _cfg_dict(self.cfg),
            "dataset": self.data.name,
            "history": self.history,
        }
        if "group_acc" in eps[-1]:
            out["final_group_acc"] = eps[-1]["group_acc"]
            out["last10_group_acc"] = {k: sum(e["group_acc"][k] for e in last) / len(last)
                                       for k in eps[-1]["group_acc"]}
        return out

    def _log(self, rec: Dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


def _cfg_dict(cfg: TrainConfig) -> Dict:
    d = asdict(cfg)
    d["lr_milestones"] = list(cfg.lr_milestones)
    return d
