"""PointNet++ SSG classification model."""

from __future__ import annotations

import torch
from torch import nn

from pointcloud_benchmark.models.pointnet2_utils import PointNetSetAbstraction


class PointNet2Classifier(nn.Module):
    """Standard PointNet++ single-scale grouping classifier for ModelNet40."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.name = "pointnet2"
        self.input_channels = model_cfg.get("input_channels", 3)
        self.num_classes = model_cfg["num_classes"]

        if self.input_channels != 3:
            raise ValueError(
                f"PointNet2Classifier currently expects xyz-only input with 3 channels, got {self.input_channels}."
            )

        # Hierarchical set abstraction layers:
        # SA1 samples 512 centroids and learns local features from 32-point neighborhoods.
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=0,
            mlp=[64, 64, 128],
            group_all=False,
        )
        # SA2 samples 128 higher-level regions and expands the receptive field.
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128,
            mlp=[128, 128, 256],
            group_all=False,
        )
        # SA3 groups all remaining points into one global feature vector.
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # Global feature aggregation feeds the final classification head.
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(f"Expected input of shape [B, N, 3], got tensor with shape {tuple(points.shape)}.")
        if points.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 for xyz input, got shape {tuple(points.shape)}.")

        xyz = points.contiguous()

        # Hierarchical set abstraction:
        # local neighborhoods -> higher-level local regions -> one global feature vector.
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        global_features = l3_points.squeeze(1)
        x = self.drop1(torch.relu(self.bn1(self.fc1(global_features))))
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)
        return logits
