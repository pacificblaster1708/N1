"""Data: CIFAR-N loading, a synthetic structured-contamination benchmark, and
deterministic batched augmentation.

All training images are held as one uint8 tensor (N, 3, 32, 32) on the
training device.  Augmentation is applied per batch on that device with an
explicit ``torch.Generator``, so

* scoring randomness never touches the training RNG streams, and
* paired runs (same seed, different CTAL variant) see identical minibatch
  orders and identical training augmentations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

CIFAR10N_TYPES = ("aggre_label", "random_label1", "random_label2", "random_label3", "worse_label")
CIFAR100N_TYPES = ("noisy_label",)


@dataclass
class ImageDataset:
    x_train: torch.Tensor            # uint8 (N, 3, H, W)
    y_train: torch.Tensor            # long (N,)  observed (possibly noisy) labels
    x_test: torch.Tensor             # uint8 (M, 3, H, W)
    y_test: torch.Tensor             # long (M,)  clean test labels
    num_classes: int
    mean: tuple
    std: tuple
    y_train_clean: Optional[torch.Tensor] = None  # analysis only, never used for weights
    name: str = ""
    # Optional analysis-only subgroup ids (never used for training or weights).
    train_group: Optional[torch.Tensor] = None
    test_group: Optional[torch.Tensor] = None
    group_names: Optional[dict] = None

    @property
    def noisy_mask(self) -> Optional[torch.Tensor]:
        if self.y_train_clean is None:
            return None
        return self.y_train != self.y_train_clean

    def to(self, device) -> "ImageDataset":
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)
        self.x_test = self.x_test.to(device)
        self.y_test = self.y_test.to(device)
        if self.y_train_clean is not None:
            self.y_train_clean = self.y_train_clean.to(device)
        if self.train_group is not None:
            self.train_group = self.train_group.to(device)
        if self.test_group is not None:
            self.test_group = self.test_group.to(device)
        return self

    def subset_train(self, keep: torch.Tensor) -> "ImageDataset":
        """Keep only the training indices in ``keep`` (sorted)."""
        keep = torch.sort(keep).values
        self.x_train = self.x_train[keep]
        self.y_train = self.y_train[keep]
        if self.y_train_clean is not None:
            self.y_train_clean = self.y_train_clean[keep]
        if self.train_group is not None:
            self.train_group = self.train_group[keep]
        return self


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def normalize(x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
    x = x_uint8.float().div_(255.0)
    m = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


def random_crop_flip(x: torch.Tensor, generator: torch.Generator, padding: int = 4) -> torch.Tensor:
    """Standard CIFAR augmentation, batched: zero-padded random crop + h-flip.

    Equivalent to torchvision RandomCrop(32, padding=4) + RandomHorizontalFlip
    applied independently to each image.  Apply it to the raw uint8 images
    *before* ``normalize`` so the padding is black pixels, exactly as in the
    standard torchvision pipeline.
    """
    B, C, H, W = x.shape
    dev = x.device
    xp = F.pad(x, (padding, padding, padding, padding))
    oy = torch.randint(0, 2 * padding + 1, (B,), device=dev, generator=generator)
    ox = torch.randint(0, 2 * padding + 1, (B,), device=dev, generator=generator)
    flip = torch.rand(B, device=dev, generator=generator) < 0.5
    ar_h = torch.arange(H, device=dev)
    ar_w = torch.arange(W, device=dev)
    rows = oy[:, None] + ar_h[None, :]                                   # (B, H)
    cols_fwd = ox[:, None] + ar_w[None, :]                               # (B, W)
    cols = torch.where(flip[:, None], ox[:, None] + (W - 1 - ar_w)[None, :], cols_fwd)
    b = torch.arange(B, device=dev)[:, None, None, None]
    ch = torch.arange(C, device=dev)[None, :, None, None]
    return xp[b, ch, rows[:, None, :, None], cols[:, None, None, :]]


# ---------------------------------------------------------------------------
# CIFAR-N
# ---------------------------------------------------------------------------

def _load_label_file(path: str) -> dict:
    try:
        obj = torch.load(path, weights_only=False)
    except TypeError:  # very old torch without weights_only
        obj = torch.load(path)
    return {k: np.asarray(v) for k, v in obj.items()}


def load_cifar_n(root: str, dataset: str, label_file: str, noise_type: str,
                 download: bool = False) -> ImageDataset:
    """Load CIFAR-10N / CIFAR-100N.

    ``label_file`` is CIFAR-10_human.pt or CIFAR-100_human.pt from
    https://github.com/UCSC-REAL/cifar-10-100n (folder ``data/``).
    The file's ``clean_label`` is checked against torchvision's training
    labels to make sure the sample order matches.
    """
    import torchvision

    dataset = dataset.lower()
    if dataset == "cifar10":
        cls, mean, std, K, allowed = torchvision.datasets.CIFAR10, CIFAR10_MEAN, CIFAR10_STD, 10, CIFAR10N_TYPES
    elif dataset == "cifar100":
        cls, mean, std, K, allowed = torchvision.datasets.CIFAR100, CIFAR100_MEAN, CIFAR100_STD, 100, CIFAR100N_TYPES
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    if noise_type not in allowed:
        raise ValueError(f"noise_type for {dataset} must be one of {allowed}")

    train = cls(root, train=True, download=download)
    test = cls(root, train=False, download=download)
    labels = _load_label_file(label_file)
    clean = labels["clean_label"].astype(np.int64)
    targets = np.asarray(train.targets, dtype=np.int64)
    if clean.shape != targets.shape or not np.array_equal(clean, targets):
        raise ValueError("clean_label in the CIFAR-N file does not match the torchvision "
                         "training labels; the label file / dataset do not correspond")
    noisy = labels[noise_type].astype(np.int64)

    def to_tensor(arr):
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))

    return ImageDataset(
        x_train=to_tensor(train.data),
        y_train=torch.from_numpy(noisy),
        x_test=to_tensor(test.data),
        y_test=torch.tensor(test.targets, dtype=torch.long),
        num_classes=K, mean=mean, std=std,
        y_train_clean=torch.from_numpy(clean),
        name=f"{dataset}n-{noise_type}",
    )


# ---------------------------------------------------------------------------
# Synthetic structured contamination
# ---------------------------------------------------------------------------

def make_synthetic_structured(
    n_per_class: int = 300, n_test_per_class: int = 100, num_classes: int = 10,
    contaminated_fraction: float = 0.3, noise_std: float = 0.9, seed: int = 0,
) -> ImageDataset:
    """Small 32x32 benchmark with *structured* label contamination.

    Each class has a smooth random prototype.  In every class a fixed fraction
    of training examples shares an acquisition artifact (a bright corner patch)
    and is relabelled, *as a group*, to the next class.  Affected examples are
    therefore similar to each other and carry the same wrong observed label --
    the regime of Assumption 1.  The test set is clean and artifact-free.
    """
    g = torch.Generator().manual_seed(seed)
    K = num_classes
    protos = F.interpolate(torch.randn(K, 3, 8, 8, generator=g), size=32, mode="bilinear",
                           align_corners=False)
    protos = protos / protos.flatten(1).std(dim=1).view(K, 1, 1, 1)

    def sample(n_per, with_artifact):
        xs, ys, yc = [], [], []
        for c in range(K):
            x = protos[c].expand(n_per, -1, -1, -1) + noise_std * torch.randn(n_per, 3, 32, 32, generator=g)
            y = torch.full((n_per,), c, dtype=torch.long)
            obs = y.clone()
            if with_artifact:
                n_aff = int(round(contaminated_fraction * n_per))
                x[:n_aff, :, :10, :10] += 3.0       # shared artifact
                obs[:n_aff] = (c + 1) % K           # shared, structured label error
            xs.append(x); ys.append(obs); yc.append(y)
        x = torch.cat(xs)
        x = ((x * 40.0) + 128.0).clamp(0, 255).to(torch.uint8)
        return x, torch.cat(ys), torch.cat(yc)

    xtr, ytr, ytr_clean = sample(n_per_class, True)
    xte, _, yte = sample(n_test_per_class, False)
    perm = torch.randperm(xtr.shape[0], generator=g)
    return ImageDataset(
        x_train=xtr[perm], y_train=ytr[perm], x_test=xte, y_test=yte,
        num_classes=K, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25),
        y_train_clean=ytr_clean[perm], name="synthetic-structured",
    )


# ---------------------------------------------------------------------------
# Controlled structured contamination on clean CIFAR-10 (reviewer experiments)
# ---------------------------------------------------------------------------

# Semantically plausible class pairs: airplane<->bird, automobile<->truck,
# cat<->dog, deer<->horse, frog<->ship.
CIFAR10_PAIRS = {0: 2, 2: 0, 1: 9, 9: 1, 3: 5, 5: 3, 4: 7, 7: 4, 6: 8, 8: 6}
CONTROLLED_NOISE = ("none", "cluster", "pair_random", "symmetric")
GROUP_OTHER, GROUP_CLUSTER, GROUP_RARE = 0, 1, 2


def kmeans(x: torch.Tensor, k: int, generator: torch.Generator, iters: int = 100):
    """k-means with k-means++ initialization; returns (centroids, assignment)."""
    n = x.shape[0]
    first = torch.randint(n, (1,), generator=generator)
    centers = [x[first].squeeze(0)]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=generator)
        centers.append(x[idx].squeeze(0))
        d2 = torch.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    C = torch.stack(centers)
    for _ in range(iters):
        assign = torch.cdist(x, C).argmin(1)
        newC = torch.stack([x[assign == j].mean(0) if (assign == j).any() else C[j] for j in range(k)])
        if torch.equal(newC, C):
            break
        C = newC
    return C, torch.cdist(x, C).argmin(1)


def pixel_descriptor(x_uint8: torch.Tensor) -> torch.Tensor:
    """Coarse 8x8 colour layout (192-d): an 'acquisition condition' proxy."""
    return F.avg_pool2d(x_uint8.float() / 255.0, 4).flatten(1).double()


def make_controlled_cifar10(root: str, noise: str, rate: float = 0.2, n_clusters: int = 5,
                            rare_keep: float = 0.05, data_seed: int = 0, download: bool = False,
                            train_features: Optional[torch.Tensor] = None,
                            test_features: Optional[torch.Tensor] = None) -> ImageDataset:
    """Clean CIFAR-10 with controlled, rate-matched label contamination.

    Within every true class, images are clustered (k-means on a coarse colour
    layout, or on supplied features).  One cluster whose size is closest to
    ``rate * n_c`` is the *contamination cluster*; the smallest remaining
    cluster is the *rare clean subgroup*, of which only ``rare_keep`` is kept
    for training (a rare, clean, typically hard subset; never relabelled).

    noise = 'cluster'     : the whole contamination cluster is flipped to the
                            paired class (structured / collective noise).
            'pair_random' : the SAME number of images per class, chosen at random
                            outside the rare subgroup, flipped to the SAME paired
                            class (identical transition matrix, no structure).
            'symmetric'   : the same per-class counts, random images, flipped to a
                            uniformly random other class.
            'none'        : clean labels.
    The training images, the rare subgroup and the test set are identical
    across noise types; only *where* the flips land differs.  Group ids are
    for analysis only.
    """
    import torchvision

    if noise not in CONTROLLED_NOISE:
        raise ValueError(f"noise must be one of {CONTROLLED_NOISE}")
    train = torchvision.datasets.CIFAR10(root, train=True, download=download)
    test = torchvision.datasets.CIFAR10(root, train=False, download=download)
    x_train = torch.from_numpy(np.ascontiguousarray(train.data.transpose(0, 3, 1, 2)))
    x_test = torch.from_numpy(np.ascontiguousarray(test.data.transpose(0, 3, 1, 2)))
    y_clean = torch.tensor(train.targets, dtype=torch.long)
    y_test = torch.tensor(test.targets, dtype=torch.long)
    return controlled_contamination(x_train, y_clean, x_test, y_test, noise, rate, n_clusters,
                                    rare_keep, data_seed, train_features, test_features,
                                    mean=CIFAR10_MEAN, std=CIFAR10_STD, pairs=CIFAR10_PAIRS)


def controlled_contamination(x_train, y_clean, x_test, y_test, noise, rate, n_clusters, rare_keep,
                             data_seed, train_features=None, test_features=None,
                             mean=CIFAR10_MEAN, std=CIFAR10_STD, pairs=None) -> ImageDataset:
    """Dataset-agnostic core of :func:`make_controlled_cifar10` (also used in tests)."""
    g = torch.Generator().manual_seed(data_seed)
    K = int(y_clean.max()) + 1
    pairs = pairs or {c: (c + 1) % K for c in range(K)}
    f_train = train_features.double() if train_features is not None else pixel_descriptor(x_train)
    f_test = test_features.double() if test_features is not None else pixel_descriptor(x_test)

    train_group = torch.zeros(len(y_clean), dtype=torch.long)
    test_group = torch.zeros(len(y_test), dtype=torch.long)
    for c in range(K):
        idx = (y_clean == c).nonzero().flatten()
        C, assign = kmeans(f_train[idx], n_clusters, g)
        sizes = torch.bincount(assign, minlength=n_clusters)
        j_cont = int(torch.argmin((sizes.double() - rate * len(idx)).abs()))
        rest = [j for j in range(n_clusters) if j != j_cont]
        j_rare = min(rest, key=lambda j: int(sizes[j]))
        train_group[idx[assign == j_cont]] = GROUP_CLUSTER
        train_group[idx[assign == j_rare]] = GROUP_RARE
        tidx = (y_test == c).nonzero().flatten()
        t_assign = torch.cdist(f_test[tidx], C).argmin(1)
        test_group[tidx[t_assign == j_cont]] = GROUP_CLUSTER
        test_group[tidx[t_assign == j_rare]] = GROUP_RARE

    # Make the rare subgroup rare: keep only a fraction of it.
    rare = (train_group == GROUP_RARE).nonzero().flatten()
    n_keep = max(1, int(round(rare_keep * len(rare))))
    drop = rare[torch.randperm(len(rare), generator=g)[n_keep:]]
    keep_mask = torch.ones(len(y_clean), dtype=torch.bool)
    keep_mask[drop] = False
    keep = keep_mask.nonzero().flatten()
    x_train, y_clean, train_group = x_train[keep], y_clean[keep], train_group[keep]

    y = y_clean.clone()
    for c in range(K):
        in_c = y_clean == c
        cluster = (in_c & (train_group == GROUP_CLUSTER)).nonzero().flatten()
        m_c = len(cluster)
        if noise == "cluster":
            chosen = cluster
        elif noise in ("pair_random", "symmetric"):
            pool = (in_c & (train_group != GROUP_RARE)).nonzero().flatten()
            chosen = pool[torch.randperm(len(pool), generator=g)[:m_c]]
        else:
            continue
        if noise == "symmetric":
            off = torch.randint(1, K, (len(chosen),), generator=g)
            y[chosen] = (c + off) % K
        else:
            y[chosen] = pairs[c]

    return ImageDataset(
        x_train=x_train, y_train=y, x_test=x_test, y_test=y_test, num_classes=K,
        mean=mean, std=std, y_train_clean=y_clean, name=f"c10ctrl-{noise}-r{rate}",
        train_group=train_group, test_group=test_group,
        group_names={GROUP_OTHER: "other", GROUP_CLUSTER: "contamination_cluster",
                     GROUP_RARE: "rare_clean_subgroup"},
    )


def make_long_tailed(data: ImageDataset, imbalance_ratio: float, data_seed: int = 0) -> ImageDataset:
    """Exponential long-tailed subset: n_c = n_max * ratio^(-c/(K-1)) by *clean* class.

    Standard long-tail construction (the benchmark is built with clean labels;
    training and weighting use only the observed labels).  Groups = clean class.
    """
    if data.y_train_clean is None:
        raise ValueError("long-tailed construction needs clean labels")
    g = torch.Generator().manual_seed(data_seed)
    K = data.num_classes
    yc = data.y_train_clean.cpu()
    n_max = int(torch.bincount(yc, minlength=K).max())
    keep = []
    for c in range(K):
        idx = (yc == c).nonzero().flatten()
        n_c = int(n_max * imbalance_ratio ** (-c / (K - 1)))
        keep.append(idx[torch.randperm(len(idx), generator=g)[:n_c]])
    data.subset_train(torch.cat(keep))
    data.train_group = data.y_train_clean.clone()
    data.test_group = data.y_test.clone()
    data.group_names = {c: f"class{c}" for c in range(K)}
    data.name = f"{data.name}-lt{imbalance_ratio:g}"
    return data
