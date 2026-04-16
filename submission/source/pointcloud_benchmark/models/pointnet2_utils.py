"""Core PointNet++ sampling, grouping, and set abstraction modules."""

from __future__ import annotations

import torch
from torch import nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distance.

    Args:
        src: Source points of shape [B, N, C].
        dst: Target points of shape [B, M, C].

    Returns:
        Pairwise squared distances of shape [B, N, M].
    """

    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points using batch-aware indices."""

    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(batch_size, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling selects a spatially well-spread subset of points.

    PointNet++ uses FPS so each set-abstraction layer covers the geometry more uniformly
    than random sampling would. The algorithm grows the sampled set one point at a time,
    always picking the point farthest from the currently selected subset.
    """

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """Ball query groups local neighborhoods around sampled centroids.

    For each centroid in `new_xyz`, we gather nearby points from `xyz` that lie inside
    a radius-defined sphere. This gives PointNet++ a local region to encode.
    """

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, num_centroids, _ = new_xyz.shape

    group_idx = torch.arange(num_points, dtype=torch.long, device=device)
    group_idx = group_idx.view(1, 1, num_points).repeat(batch_size, num_centroids, 1)

    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    first_group = group_idx[:, :, 0].view(batch_size, num_centroids, 1).repeat(1, 1, nsample)
    mask = group_idx == num_points
    group_idx[mask] = first_group[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample centroids and group their local neighborhoods.

    This performs the standard PointNet++ local set abstraction step:
    1. Use FPS to choose centroids.
    2. Use ball query to gather neighbors around each centroid.
    3. Concatenate normalized local coordinates with any incoming point features.
    """

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, group_idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]

    if points is None:
        new_points = grouped_xyz_norm
    else:
        grouped_points = index_points(points, group_idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)

    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group all points into one global region for the final abstraction layer."""

    batch_size, num_points, _ = xyz.shape
    new_xyz = torch.zeros(batch_size, 1, 3, device=xyz.device, dtype=xyz.dtype)
    grouped_xyz = xyz.view(batch_size, 1, num_points, 3)

    if points is None:
        new_points = grouped_xyz
    else:
        grouped_points = points.view(batch_size, 1, num_points, -1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Single-scale set abstraction layer from PointNet++."""

    def __init__(
        self,
        npoint: int | None,
        radius: float | None,
        nsample: int | None,
        in_channel: int,
        mlp: list[int],
        group_all: bool,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel + 3
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply PointNet++ set abstraction.

        Args:
            xyz: Point coordinates of shape [B, N, 3].
            points: Optional point features of shape [B, N, D].

        Returns:
            new_xyz: Sampled centroids of shape [B, S, 3].
            new_points: Aggregated local features of shape [B, S, D_out].
        """

        # Set abstraction forms local neighborhoods, runs shared MLPs on each grouped
        # neighborhood, then max-pools across neighbors to produce one feature per centroid.
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            if self.npoint is None or self.radius is None or self.nsample is None:
                raise ValueError("npoint, radius, and nsample must be set when group_all=False.")
            new_xyz, new_points = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz,
                points,
            )

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = torch.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

