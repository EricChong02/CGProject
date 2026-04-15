"""Dataset and dataloader factory functions."""

from __future__ import annotations

from torch.utils.data import DataLoader

from pointcloud_benchmark.datasets.modelnet40 import ModelNet40Dataset
from pointcloud_benchmark.datasets.scanobjectnn import ScanObjectNNDataset


DATASET_REGISTRY = {
    "modelnet40": ModelNet40Dataset,
    "scanobjectnn": ScanObjectNNDataset,
}

SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "val": "test",
    "eval": "test",
}


def build_dataset(config: dict, split: str):
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    normalized_split = SPLIT_ALIASES.get(split.lower(), split.lower())
    return DATASET_REGISTRY[dataset_name](config=config, split=normalized_split)


def build_dataloader(config: dict, split: str) -> DataLoader:
    normalized_split = SPLIT_ALIASES.get(split.lower(), split.lower())
    dataset = build_dataset(config=config, split=normalized_split)
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    batch_size = (
        training_cfg.get("batch_size", 8)
        if normalized_split == "train"
        else evaluation_cfg.get("batch_size", training_cfg.get("batch_size", 8))
    )
    shuffle = normalized_split == "train"
    drop_last = training_cfg.get("drop_last", False) if normalized_split == "train" else False

    # TODO: Add a custom collate function for more advanced dataset outputs.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=training_cfg.get("num_workers", 0),
        drop_last=drop_last,
    )
