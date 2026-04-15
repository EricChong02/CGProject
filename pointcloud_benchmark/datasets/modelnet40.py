"""ModelNet40 dataset placeholder."""

from __future__ import annotations

from pathlib import Path

from pointcloud_benchmark.datasets.base import BasePointCloudDataset


class ModelNet40Dataset(BasePointCloudDataset):
    """Synthetic fallback for ModelNet40 until the real parser is added."""

    def __init__(self, config: dict, split: str) -> None:
        dataset_cfg = config["dataset"]
        root = Path(dataset_cfg["root"])
        processed_root = Path(dataset_cfg["processed_root"])

        # TODO: Scan ModelNet40 files from root and processed_root.
        # TODO: Add normalization, sampling, caching, and augmentation.
        super().__init__(
            split=split,
            num_points=dataset_cfg["num_points"],
            num_classes=dataset_cfg["num_classes"],
            length=96 if split == "train" else 32,
        )

        self.root = root
        self.processed_root = processed_root

