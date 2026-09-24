"""Tiny CNN used only to prove the template forward pass and train step."""

from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small grayscale classifier — placeholder until a real project model lands.

    Input:  (N, 1, H, W)
    Output: (N, num_classes) logits
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = torch.flatten(features, 1)
        return self.classifier(flattened)
