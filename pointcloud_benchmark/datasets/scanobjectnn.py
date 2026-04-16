"""ScanObjectNN h5 dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pointcloud_benchmark.datasets.transforms import (
    jitter_point_cloud,
    normalize_point_cloud,
    random_point_dropout,
    random_rotate_upright_axis,
    random_scale_point_cloud,
    random_shift_point_cloud,
    sample_points,
)


def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "ScanObjectNN h5 loading requires the optional dependency 'h5py'. "
            "Install it with `pip install h5py` or `pip install -r requirements.txt`."
        ) from error
    return h5py


class ScanObjectNNDataset(Dataset):
    """ScanObjectNN loader for the official h5 release layout."""

    _VARIANT_SUFFIXES = {
        "obj_bg": "objectdataset.h5",
        "obj_only": "objectdataset.h5",
        "pb_t25": "objectdataset_augmented25_norot.h5",
        "pb_t25_r": "objectdataset_augmented25rot.h5",
        "pb_t50_r": "objectdataset_augmentedrot.h5",
        "pb_t50_rs": "objectdataset_augmentedrot_scale75.h5",
    }

    _VARIANT_ALIASES = {
        "obj_bg": "obj_bg",
        "objbg": "obj_bg",
        "obj": "obj_bg",
        "obj_only": "obj_only",
        "objonly": "obj_only",
        "nobg": "obj_only",
        "pb_t25": "pb_t25",
        "pbt25": "pb_t25",
        "pb_t25_r": "pb_t25_r",
        "pbt25r": "pb_t25_r",
        "pb_t50_r": "pb_t50_r",
        "pbt50r": "pb_t50_r",
        "pb_t50_rs": "pb_t50_rs",
        "pbt50rs": "pb_t50_rs",
        "hardest": "pb_t50_rs",
    }

    def __init__(self, config: dict, split: str) -> None:
        dataset_cfg = config["dataset"]
        self.root = Path(dataset_cfg["root"])
        self.processed_root = Path(dataset_cfg.get("processed_root", "datasets_processed/scanobjectnn"))
        self.split = self._canonicalize_split(split)
        self.num_points = int(dataset_cfg["num_points"])
        self.num_classes = int(dataset_cfg.get("num_classes", 15))
        self.normalize = bool(dataset_cfg.get("normalize", True))
        self.variant = self._canonicalize_variant(str(dataset_cfg.get("variant", "pb_t50_rs")))
        self.split_name = str(dataset_cfg.get("split_name", "main_split"))
        self.filter_background_with_mask = bool(dataset_cfg.get("filter_background_with_mask", False))
        self.require_mask = bool(dataset_cfg.get("require_mask", False))

        augment_cfg = dataset_cfg.get("augmentations", {})
        self.rotation_axis = str(dataset_cfg.get("upright_axis", "y")).lower()
        self.augmentations = {
            "random_point_dropout": augment_cfg.get(
                "random_point_dropout",
                {"enabled": True, "max_dropout_ratio": 0.875},
            ),
            "random_scaling": augment_cfg.get(
                "random_scaling",
                {"enabled": True, "scale_low": 0.8, "scale_high": 1.25},
            ),
            "random_shifting": augment_cfg.get(
                "random_shifting",
                {"enabled": True, "shift_range": 0.1},
            ),
            "gaussian_jitter": augment_cfg.get(
                "gaussian_jitter",
                {"enabled": True, "sigma": 0.01, "clip": 0.05},
            ),
            "random_rotation_upright_axis": augment_cfg.get(
                "random_rotation_upright_axis",
                {"enabled": False},
            ),
        }

        self.dataset_root = self._resolve_split_root(self.root, self.split_name)
        self.h5_path = self._resolve_h5_path(self.dataset_root, self.split, self.variant)
        self.data, self.labels, self.masks = self._load_h5_arrays(self.h5_path)
        self.max_samples = self._resolve_max_samples(dataset_cfg, self.split)

        if self.data.ndim != 3:
            raise ValueError(
                "Expected ScanObjectNN data to have shape [num_samples, num_points, channels], "
                f"got {self.data.shape}."
            )
        if self.data.shape[-1] < 3:
            raise ValueError(f"Expected at least 3 channels per point, got shape {self.data.shape}.")
        if self.labels.ndim != 1:
            raise ValueError(f"Expected labels to be 1D after loading, got shape {self.labels.shape}.")
        if len(self.data) != len(self.labels):
            raise ValueError(
                f"Loaded mismatched data/label lengths: {len(self.data)} samples vs {len(self.labels)} labels."
            )

        if self.masks is not None:
            if self.masks.ndim != 2:
                raise ValueError(f"Expected masks to be 2D, got shape {self.masks.shape}.")
            if self.masks.shape[0] != self.data.shape[0]:
                raise ValueError(
                    "Loaded mismatched data/mask lengths: "
                    f"{self.data.shape[0]} samples vs {self.masks.shape[0]} masks."
                )
            if self.masks.shape[1] != self.data.shape[1]:
                raise ValueError(
                    "Loaded mismatched point counts between data and masks: "
                    f"{self.data.shape[1]} vs {self.masks.shape[1]}."
                )
        elif self.require_mask:
            raise KeyError(
                f"Configured require_mask=true, but the ScanObjectNN file does not contain a 'mask' dataset: "
                f"{self.h5_path}."
            )

        max_label = int(self.labels.max()) if len(self.labels) > 0 else -1
        if max_label >= self.num_classes:
            raise ValueError(
                f"Found class label {max_label}, which exceeds configured num_classes={self.num_classes}."
            )

        if self.max_samples is not None:
            if self.max_samples <= 0:
                raise ValueError(
                    f"Configured max_samples for split={self.split!r} must be positive, got {self.max_samples}."
                )
            self.data = self.data[: self.max_samples]
            self.labels = self.labels[: self.max_samples]
            if self.masks is not None:
                self.masks = self.masks[: self.max_samples]

    @staticmethod
    def _canonicalize_split(split: str) -> str:
        normalized = split.lower()
        split_aliases = {"train": "train", "test": "test", "val": "test", "eval": "test"}
        if normalized not in split_aliases:
            raise ValueError(
                f"Unsupported ScanObjectNN split: {split!r}. Expected one of: train, test, val, eval."
            )
        return split_aliases[normalized]

    @classmethod
    def _canonicalize_variant(cls, variant: str) -> str:
        normalized = variant.lower().replace("-", "_").replace(" ", "")
        if normalized in cls._VARIANT_ALIASES:
            return cls._VARIANT_ALIASES[normalized]
        raise ValueError(
            f"Unsupported ScanObjectNN variant: {variant!r}. "
            f"Expected one of: {', '.join(sorted(cls._VARIANT_SUFFIXES))}."
        )

    @staticmethod
    def _looks_like_split_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        return bool(list(path.glob("training_*.h5"))) and bool(list(path.glob("test_*.h5")))

    @classmethod
    def _resolve_split_root(cls, root: Path, split_name: str) -> Path:
        if not root.exists():
            raise FileNotFoundError(
                f"ScanObjectNN dataset root does not exist: {root}. "
                "Expected a directory containing `h5_files/<split_name>/` or the split h5 files directly."
            )

        candidates = [
            root,
            root / split_name,
            root / "h5_files",
            root / "h5_files" / split_name,
            root / "ScanObjectNN",
            root / "ScanObjectNN" / split_name,
            root / "ScanObjectNN" / "h5_files",
            root / "ScanObjectNN" / "h5_files" / split_name,
        ]

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if cls._looks_like_split_dir(candidate):
                return candidate

        raise FileNotFoundError(
            f"Could not find a valid ScanObjectNN h5 split under {root}. "
            f"Expected files like `training_*.h5` and `test_*.h5` inside `{split_name}/` or `h5_files/{split_name}/`."
        )

    @classmethod
    def _resolve_h5_path(cls, dataset_root: Path, split: str, variant: str) -> Path:
        prefix = "training" if split == "train" else "test"
        suffix = cls._VARIANT_SUFFIXES[variant]
        preferred = dataset_root / f"{prefix}_{suffix}"
        if preferred.exists():
            return preferred

        available = sorted(path.name for path in dataset_root.glob(f"{prefix}_*.h5"))
        if not available:
            raise FileNotFoundError(
                f"No ScanObjectNN h5 files found for split={split!r} in {dataset_root}. "
                f"Expected a file named `{prefix}_{suffix}`."
            )

        raise FileNotFoundError(
            f"Could not find the configured ScanObjectNN variant file `{prefix}_{suffix}` in {dataset_root}. "
            f"Available files for split={split!r}: {available}."
        )

    @staticmethod
    def _load_h5_arrays(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        h5py = _import_h5py()
        with h5py.File(h5_path, "r") as handle:
            if "data" not in handle or "label" not in handle:
                raise KeyError(
                    f"Malformed ScanObjectNN h5 file: {h5_path}. Expected datasets named 'data' and 'label'."
                )
            data = handle["data"][:].astype(np.float32)
            labels = handle["label"][:].astype(np.int64).reshape(-1)
            masks = handle["mask"][:].astype(np.float32) if "mask" in handle else None
        return data, labels, masks

    @staticmethod
    def _resolve_max_samples(dataset_cfg: dict, split: str) -> int | None:
        debug_cfg = dataset_cfg.get("debug", {})
        split_key = f"max_{split}_samples"

        if split_key in debug_cfg:
            return int(debug_cfg[split_key])
        if "max_samples" in debug_cfg:
            return int(debug_cfg["max_samples"])
        return None

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        rng = np.random.default_rng()
        points = self.data[index][:, :3].copy()
        label = int(self.labels[index])
        point_mask = None if self.masks is None else self.masks[index].reshape(-1)

        if self.filter_background_with_mask and point_mask is not None:
            foreground_mask = point_mask > 0.5
            if not np.any(foreground_mask):
                raise ValueError(
                    f"Foreground filtering removed all points for sample index={index} in {self.h5_path}."
                )
            points = points[foreground_mask]

        points = sample_points(
            points,
            num_points=self.num_points,
            rng=rng,
            deterministic=self.split != "train",
        )

        if self.normalize:
            points = normalize_point_cloud(points)

        if self.split == "train":
            points = self._apply_train_augmentations(points, rng)

        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "label": label,
        }

    def _apply_train_augmentations(
        self,
        points: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dropout_cfg = self.augmentations["random_point_dropout"]
        if dropout_cfg.get("enabled", False):
            points = random_point_dropout(
                points,
                rng=rng,
                max_dropout_ratio=float(dropout_cfg.get("max_dropout_ratio", 0.875)),
            )

        rotation_cfg = self.augmentations["random_rotation_upright_axis"]
        if rotation_cfg.get("enabled", False):
            points = random_rotate_upright_axis(points, rng=rng, axis=self.rotation_axis)

        scale_cfg = self.augmentations["random_scaling"]
        if scale_cfg.get("enabled", False):
            points = random_scale_point_cloud(
                points,
                rng=rng,
                scale_low=float(scale_cfg.get("scale_low", 0.8)),
                scale_high=float(scale_cfg.get("scale_high", 1.25)),
            )

        shift_cfg = self.augmentations["random_shifting"]
        if shift_cfg.get("enabled", False):
            points = random_shift_point_cloud(
                points,
                rng=rng,
                shift_range=float(shift_cfg.get("shift_range", 0.1)),
            )

        jitter_cfg = self.augmentations["gaussian_jitter"]
        if jitter_cfg.get("enabled", False):
            points = jitter_point_cloud(
                points,
                rng=rng,
                sigma=float(jitter_cfg.get("sigma", 0.01)),
                clip=float(jitter_cfg.get("clip", 0.05)),
            )

        return points.astype(np.float32, copy=False)
