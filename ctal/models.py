"""Classifiers that expose penultimate features.

Every model's ``forward(x, return_features=False)`` returns logits, or
``(logits, features)`` where ``features`` is the penultimate representation
(input of the final linear layer) used by CTAL's representation view and
neighbour graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CIFARResNet(nn.Module):
    """CIFAR-style ResNet (3x3 stem, no max-pool); penultimate dim = 512."""

    def __init__(self, num_blocks, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.linear = nn.Linear(512, num_classes)
        self.feature_dim = 512

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        feat = torch.flatten(F.adaptive_avg_pool2d(out, 1), 1)
        logits = self.linear(feat)
        return (logits, feat) if return_features else logits


def resnet34(num_classes: int = 10) -> CIFARResNet:
    return CIFARResNet([3, 4, 6, 3], num_classes)


def resnet18(num_classes: int = 10) -> CIFARResNet:
    return CIFARResNet([2, 2, 2, 2], num_classes)


class SmallCNN(nn.Module):
    """Small CNN for CPU smoke tests and the synthetic benchmark."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.BatchNorm2d(4 * width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = 4 * width
        self.linear = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        feat = torch.flatten(self.body(x), 1)
        logits = self.linear(feat)
        return (logits, feat) if return_features else logits


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet34":
        return resnet34(num_classes)
    if arch == "resnet18":
        return resnet18(num_classes)
    if arch == "smallcnn":
        return SmallCNN(num_classes)
    raise ValueError(f"unknown architecture {arch!r}")
