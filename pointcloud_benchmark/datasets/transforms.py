"""Reusable point-cloud sampling, normalization, and augmentation helpers."""

from __future__ import annotations

import math

import numpy as np


def sample_points(
    points: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample a fixed number of points from a point cloud."""

    total_points = points.shape[0]
    if total_points == 0:
        raise ValueError("Point cloud contains zero points and cannot be sampled.")

    if total_points >= num_points:
        if deterministic:
            indices = np.arange(num_points)
        else:
            indices = rng.choice(total_points, size=num_points, replace=False)
    else:
        base_indices = np.arange(total_points)
        if deterministic:
            extra_indices = np.resize(base_indices, num_points - total_points)
        else:
            extra_indices = rng.choice(total_points, size=num_points - total_points, replace=True)
        indices = np.concatenate([base_indices, extra_indices], axis=0)
        if not deterministic:
            rng.shuffle(indices)

    return points[indices]


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Normalize to zero mean and unit scale based on max point radius."""

    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-12:
        return centered
    return centered / scale


def random_point_dropout(
    points: np.ndarray,
    rng: np.random.Generator,
    max_dropout_ratio: float = 0.875,
) -> np.ndarray:
    """Randomly replace dropped points with the first point."""

    dropout_ratio = rng.uniform(0.0, max_dropout_ratio)
    dropout_mask = rng.random(points.shape[0]) <= dropout_ratio
    if np.any(dropout_mask):
        points = points.copy()
        points[dropout_mask] = points[0]
    return points


def random_scale_point_cloud(
    points: np.ndarray,
    rng: np.random.Generator,
    scale_low: float = 0.8,
    scale_high: float = 1.25,
) -> np.ndarray:
    scale = rng.uniform(scale_low, scale_high)
    return points * scale


def random_shift_point_cloud(
    points: np.ndarray,
    rng: np.random.Generator,
    shift_range: float = 0.1,
) -> np.ndarray:
    shift = rng.uniform(-shift_range, shift_range, size=(1, 3))
    return points + shift


def jitter_point_cloud(
    points: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.01,
    clip: float = 0.05,
) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=sigma, size=points.shape)
    noise = np.clip(noise, -clip, clip)
    return points + noise


def random_rotate_upright_axis(
    points: np.ndarray,
    rng: np.random.Generator,
    axis: str = "y",
) -> np.ndarray:
    """Rotate around the specified upright axis."""

    angle = rng.uniform(0.0, 2.0 * math.pi)
    cosval = math.cos(angle)
    sinval = math.sin(angle)

    if axis == "x":
        rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, cosval, -sinval], [0.0, sinval, cosval]],
            dtype=np.float32,
        )
    elif axis == "y":
        rotation = np.array(
            [[cosval, 0.0, sinval], [0.0, 1.0, 0.0], [-sinval, 0.0, cosval]],
            dtype=np.float32,
        )
    elif axis == "z":
        rotation = np.array(
            [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported upright axis: {axis!r}. Expected one of: x, y, z.")

    return points @ rotation.T
