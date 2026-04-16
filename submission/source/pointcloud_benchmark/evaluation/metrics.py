"""Evaluation metrics."""

from __future__ import annotations

import torch


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    # TODO: Add overall accuracy, mean class accuracy, confusion matrix, and per-class metrics.
    if labels.numel() == 0:
        return 0.0
    return (predictions == labels).float().mean().item()
