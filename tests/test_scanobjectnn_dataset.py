from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pointcloud_benchmark.datasets import build_dataset


def _write_scanobjectnn_split(split_root: Path) -> None:
    split_root.mkdir(parents=True, exist_ok=True)

    train_data = np.stack(
        [
            np.concatenate(
                [
                    np.full((6, 3), 5.0, dtype=np.float32),
                    np.full((6, 3), -5.0, dtype=np.float32),
                ],
                axis=0,
            ),
            np.linspace(0.0, 1.0, num=36, dtype=np.float32).reshape(12, 3),
        ],
        axis=0,
    )
    train_labels = np.array([[1], [2]], dtype=np.int64)
    train_masks = np.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    test_data = np.stack(
        [
            np.linspace(0.0, 0.9, num=30, dtype=np.float32).reshape(10, 3),
        ],
        axis=0,
    )
    test_labels = np.array([[3]], dtype=np.int64)
    test_masks = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float32)

    with h5py.File(split_root / "training_objectdataset_augmentedrot_scale75.h5", "w") as handle:
        handle.create_dataset("data", data=train_data)
        handle.create_dataset("label", data=train_labels)
        handle.create_dataset("mask", data=train_masks)

    with h5py.File(split_root / "test_objectdataset_augmentedrot_scale75.h5", "w") as handle:
        handle.create_dataset("data", data=test_data)
        handle.create_dataset("label", data=test_labels)
        handle.create_dataset("mask", data=test_masks)


def test_scanobjectnn_dataset_loads_official_h5_layout(tmp_path: Path) -> None:
    root = tmp_path / "scanobjectnn"
    split_root = root / "h5_files" / "main_split"
    _write_scanobjectnn_split(split_root)

    config = {
        "dataset": {
            "name": "scanobjectnn",
            "root": str(root),
            "processed_root": str(tmp_path / "processed"),
            "num_points": 8,
            "num_classes": 15,
            "normalize": True,
            "variant": "pb_t50_rs",
            "split_name": "main_split",
            "augmentations": {
                "random_point_dropout": {"enabled": False},
                "random_scaling": {"enabled": False},
                "random_shifting": {"enabled": False},
                "gaussian_jitter": {"enabled": False},
                "random_rotation_upright_axis": {"enabled": False},
            },
            "debug": {
                "max_train_samples": 1,
                "max_test_samples": 1,
            },
        },
        "training": {"batch_size": 2, "num_workers": 0},
        "evaluation": {"batch_size": 2},
    }

    train_dataset = build_dataset(config, split="train")
    test_dataset = build_dataset(config, split="test")

    assert len(train_dataset) == 1
    assert len(test_dataset) == 1

    train_sample = train_dataset[0]
    test_sample = test_dataset[0]

    assert tuple(train_sample["points"].shape) == (8, 3)
    assert int(train_sample["label"]) == 1
    assert tuple(test_sample["points"].shape) == (8, 3)
    assert int(test_sample["label"]) == 3


def test_scanobjectnn_can_filter_background_with_mask(tmp_path: Path) -> None:
    root = tmp_path / "scanobjectnn"
    split_root = root / "main_split"
    _write_scanobjectnn_split(split_root)

    config = {
        "dataset": {
            "name": "scanobjectnn",
            "root": str(root),
            "processed_root": str(tmp_path / "processed"),
            "num_points": 6,
            "num_classes": 15,
            "normalize": False,
            "variant": "hardest",
            "split_name": "main_split",
            "filter_background_with_mask": True,
            "augmentations": {
                "random_point_dropout": {"enabled": False},
                "random_scaling": {"enabled": False},
                "random_shifting": {"enabled": False},
                "gaussian_jitter": {"enabled": False},
                "random_rotation_upright_axis": {"enabled": False},
            },
        },
        "training": {"batch_size": 1, "num_workers": 0},
        "evaluation": {"batch_size": 1},
    }

    dataset = build_dataset(config, split="test")
    sample = dataset[0]

    assert tuple(sample["points"].shape) == (6, 3)
    assert float(sample["points"].min().item()) >= 0.0
