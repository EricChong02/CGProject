"""Dataset and dataloader factory functions."""

from __future__ import annotations

from torch.utils.data import DataLoader

from pointcloud_benchmark.datasets.modelnet40 import ModelNet40Dataset
from pointcloud_benchmark.datasets.scanobjectnn import ScanObjectNNDataset


DATASET_REGISTRY = {
    "modelnet40": ModelNet40Dataset,
    "scanobjectnn": ScanObjectNNDataset,
}


def build_dataset(config: dict, split: str):
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name](config=config, split=split)


def build_dataloader(config: dict, split: str) -> DataLoader:
    dataset = build_dataset(config=config, split=split)
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    batch_size = (
        training_cfg.get("batch_size", 8)
        if split == "train"
        else evaluation_cfg.get("batch_size", training_cfg.get("batch_size", 8))
    )
    shuffle = split == "train"

    # TODO: Add a custom collate function for more advanced dataset outputs.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=training_cfg.get("num_workers", 0),
    )

