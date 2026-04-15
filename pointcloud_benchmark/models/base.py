"""Base model components for point cloud classification."""

from __future__ import annotations

import torch
from torch import nn


class PlaceholderClassifier(nn.Module):
    """Tiny placeholder backbone that keeps the pipeline executable."""

    def __init__(self, name: str, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.name = name
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, num_classes)

        # TODO: Replace this placeholder with the real architecture.

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.encoder(points)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)

