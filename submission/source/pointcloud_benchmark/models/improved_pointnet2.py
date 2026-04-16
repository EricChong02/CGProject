"""Improved PointNet++ placeholder model."""

from __future__ import annotations

from pointcloud_benchmark.models.base import PlaceholderClassifier


class ImprovedPointNet2Classifier(PlaceholderClassifier):
    def __init__(self, config: dict) -> None:
        model_cfg = config["model"]
        super().__init__(
            name="improved_pointnet2",
            input_channels=model_cfg.get("input_channels", 3),
            num_classes=model_cfg["num_classes"],
        )

        # TODO: Add your improved PointNet++ ideas here.
        # TODO: Candidates include attention, residual blocks, or adaptive sampling.

