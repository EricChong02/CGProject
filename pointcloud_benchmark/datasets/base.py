"""Base dataset classes for point cloud classification."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class BasePointCloudDataset(Dataset):
    """Simple synthetic fallback dataset used until real loaders are implemented."""

    def __init__(
        self,
        split: str,
        num_points: int,
        num_classes: int,
        length: int = 64,
    ) -> None:
        self.split = split
        self.num_points = num_points
        self.num_classes = num_classes
        self.length = length

        # TODO: Replace synthetic samples with real data loading and transforms.
        generator = torch.Generator().manual_seed(1234 if split == "train" else 5678)
        self.points = torch.randn(length, num_points, 3, generator=generator)
        self.labels = torch.randint(0, num_classes, (length,), generator=generator)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "points": self.points[index].clone(),
            "label": self.labels[index].clone(),
        }

