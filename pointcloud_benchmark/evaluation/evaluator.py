"""Evaluation loop placeholder."""

from __future__ import annotations

from pathlib import Path

import torch

from pointcloud_benchmark.evaluation.metrics import compute_accuracy
from pointcloud_benchmark.utils.io import save_json


class Evaluator:
    """Minimal evaluator for smoke-testing the benchmark pipeline."""

    def __init__(self, config: dict, model, dataloader, device: torch.device, logger) -> None:
        self.config = config
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.logger = logger

    @torch.no_grad()
    def run(self) -> dict:
        self.model.eval()
        predictions = []
        labels = []

        for batch in self.dataloader:
            if "points" not in batch or "label" not in batch:
                raise KeyError("Each evaluation batch must contain 'points' and 'label' entries.")
            points = batch["points"]
            batch_labels = batch["label"]
            if not isinstance(points, torch.Tensor) or not isinstance(batch_labels, torch.Tensor):
                raise TypeError("Expected both batch['points'] and batch['label'] to be torch.Tensor objects.")
            if points.dim() != 3 or points.size(-1) != 3:
                raise ValueError(
                    f"Expected evaluation point tensor shape [B, N, 3], got {tuple(points.shape)}."
                )
            if batch_labels.dim() != 1:
                batch_labels = batch_labels.view(-1)

            points = points.to(self.device, dtype=torch.float32)
            batch_labels = batch_labels.to(self.device, dtype=torch.long)
            logits = self.model(points)
            predictions.append(logits.argmax(dim=1))
            labels.append(batch_labels)

        accuracy = compute_accuracy(torch.cat(predictions), torch.cat(labels))
        metrics = {
            "accuracy": accuracy,
            "dataset": self.config["dataset"]["name"],
            "model": self.config["model"]["name"],
        }

        result_path = Path(self.config["output"]["result_dir"]) / "evaluation_metrics.json"
        save_json(metrics, result_path)
        self.logger.info("Saved evaluation metrics to %s", result_path)

        # TODO: Add richer evaluation outputs such as confusion matrices and class-wise scores.
        return metrics
