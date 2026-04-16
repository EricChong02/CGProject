"""DGCNN classification model."""

from __future__ import annotations

import torch
from torch import nn


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-NN indices from point features."""
    batch_size, _, num_points = x.shape
    if num_points < 2:
        raise ValueError("DGCNN requires at least 2 points per sample.")
    k = min(k, num_points)

    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def _get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build edge features for EdgeConv: [x_j - x_i, x_i]."""
    batch_size, num_dims, num_points = x.shape
    idx = _knn(x, k=k)
    device = x.device

    idx_base = torch.arange(batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x_transposed = x.transpose(2, 1).contiguous()
    neighbors = x_transposed.view(batch_size * num_points, num_dims)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)
    central = x_transposed.view(batch_size, num_points, 1, num_dims).expand(-1, -1, neighbors.size(2), -1)

    features = torch.cat((neighbors - central, central), dim=3)
    return features.permute(0, 3, 1, 2).contiguous()


class DGCNNClassifier(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.name = "dgcnn"
        self.input_channels = int(model_cfg.get("input_channels", 3))
        self.num_classes = int(model_cfg["num_classes"])

        if self.input_channels < 3:
            raise ValueError(
                f"DGCNN expects at least xyz channels, got input_channels={self.input_channels}."
            )

        self.k = int(model_cfg.get("k", 20))
        self.emb_dims = int(model_cfg.get("emb_dims", 1024))
        dropout = float(model_cfg.get("dropout", 0.5))

        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.edge_conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.edge_conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(f"Expected input of shape [B, N, C], got {tuple(points.shape)}.")
        if points.size(-1) != self.input_channels:
            raise ValueError(
                f"Expected last dimension {self.input_channels}, got {points.size(-1)}."
            )

        x = points.transpose(2, 1).contiguous()

        x1 = self.edge_conv1(_get_graph_feature(x, k=self.k)).max(dim=-1)[0]
        x2 = self.edge_conv2(_get_graph_feature(x1, k=self.k)).max(dim=-1)[0]
        x3 = self.edge_conv3(_get_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        x4 = self.edge_conv4(_get_graph_feature(x3, k=self.k)).max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_global = self.fuse_conv(x_cat)

        x_max = torch.max(x_global, dim=2)[0]
        x_avg = torch.mean(x_global, dim=2)
        logits = self.classifier(torch.cat((x_max, x_avg), dim=1))
        return logits
