"""ModelNet40 h5 dataset loader."""

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
            "ModelNet40 h5 loading requires the optional dependency 'h5py'. "
            "Install it with `pip install h5py` or `pip install -r requirements.txt`."
        ) from error
    return h5py


class ModelNet40Dataset(Dataset):
    """ModelNet40 loader for the common `modelnet40_ply_hdf5_2048` layout."""

    def __init__(self, config: dict, split: str) -> None:
        dataset_cfg = config["dataset"]
        self.root = Path(dataset_cfg["root"])
        self.processed_root = Path(dataset_cfg.get("processed_root", "datasets_processed/modelnet40"))
        self.split = self._canonicalize_split(split)
        self.num_points = int(dataset_cfg["num_points"])
        self.num_classes = int(dataset_cfg.get("num_classes", 40))
        self.normalize = bool(dataset_cfg.get("normalize", True))

        augment_cfg = dataset_cfg.get("augmentations", {})
        self.rotation_axis = str(dataset_cfg.get("upright_axis", "y")).lower()
        self.augmentations = {
            "random_point_dropout": augment_cfg.get("random_point_dropout", {"enabled": True, "max_dropout_ratio": 0.875}),
            "random_scaling": augment_cfg.get("random_scaling", {"enabled": True, "scale_low": 0.8, "scale_high": 1.25}),
            "random_shifting": augment_cfg.get("random_shifting", {"enabled": True, "shift_range": 0.1}),
            "gaussian_jitter": augment_cfg.get("gaussian_jitter", {"enabled": True, "sigma": 0.01, "clip": 0.05}),
            "random_rotation_upright_axis": augment_cfg.get(
                "random_rotation_upright_axis",
                {"enabled": False},
            ),
        }

        self.dataset_root = self._resolve_dataset_root(self.root)
        self.h5_files = self._resolve_split_files(self.dataset_root, self.split)
        self.class_names = self._load_class_names(self.dataset_root)
        self.data, self.labels = self._load_split_arrays(self.h5_files)
        self.max_samples = self._resolve_max_samples(dataset_cfg, self.split)

        if self.data.ndim != 3:
            raise ValueError(
                f"Expected ModelNet40 data to have shape [num_samples, num_points, channels], got {self.data.shape}."
            )
        if self.data.shape[-1] < 3:
            raise ValueError(
                f"Expected at least 3 channels per point, got shape {self.data.shape}."
            )
        if self.labels.ndim != 1:
            raise ValueError(f"Expected labels to be 1D after loading, got shape {self.labels.shape}.")
        if len(self.data) != len(self.labels):
            raise ValueError(
                f"Loaded mismatched data/label lengths: {len(self.data)} samples vs {len(self.labels)} labels."
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

    @staticmethod
    def _canonicalize_split(split: str) -> str:
        normalized = split.lower()
        split_aliases = {"train": "train", "test": "test", "val": "test", "eval": "test"}
        if normalized not in split_aliases:
            raise ValueError(
                f"Unsupported ModelNet40 split: {split!r}. Expected one of: train, test, val, eval."
            )
        return split_aliases[normalized]

    @staticmethod
    def _resolve_dataset_root(root: Path) -> Path:
        if not root.exists():
            raise FileNotFoundError(
                f"ModelNet40 dataset root does not exist: {root}. "
                "Expected a directory containing `modelnet40_ply_hdf5_2048/` or the h5 files directly."
            )

        candidate_roots = [root, root / "modelnet40_ply_hdf5_2048"]
        for candidate in candidate_roots:
            if not candidate.exists():
                continue
            if (candidate / "train_files.txt").exists() and (candidate / "test_files.txt").exists():
                return candidate
            if list(candidate.glob("ply_data_train*.h5")) and list(candidate.glob("ply_data_test*.h5")):
                return candidate

        raise FileNotFoundError(
            f"Could not find a valid ModelNet40 h5 layout under {root}. "
            "Expected `train_files.txt` and `test_files.txt`, or files matching "
            "`ply_data_train*.h5` and `ply_data_test*.h5` inside `modelnet40_ply_hdf5_2048/`."
        )

    @staticmethod
    def _load_class_names(dataset_root: Path) -> list[str]:
        shape_names_path = dataset_root / "shape_names.txt"
        if not shape_names_path.exists():
            return []

        with shape_names_path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _resolve_split_files(self, dataset_root: Path, split: str) -> list[Path]:
        split_file = dataset_root / f"{split}_files.txt"
        if split_file.exists():
            files = []
            with split_file.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    files.append(self._resolve_h5_reference(dataset_root, line))
        else:
            files = sorted(dataset_root.glob(f"ply_data_{split}*.h5"))

        if not files:
            raise FileNotFoundError(
                f"No h5 files found for split={split!r} in {dataset_root}. "
                f"Expected `{split}_files.txt` or files matching `ply_data_{split}*.h5`."
            )
        return files

    @staticmethod
    def _resolve_h5_reference(dataset_root: Path, reference: str) -> Path:
        ref_path = Path(reference)
        candidates = []

        if ref_path.is_absolute():
            candidates.append(ref_path)
        else:
            candidates.extend(
                [
                    dataset_root / ref_path,
                    dataset_root / ref_path.name,
                    dataset_root.parent / ref_path,
                    dataset_root.parent / ref_path.name,
                ]
            )

            if "modelnet40_ply_hdf5_2048" in ref_path.parts:
                marker_index = ref_path.parts.index("modelnet40_ply_hdf5_2048")
                suffix = Path(*ref_path.parts[marker_index + 1 :])
                candidates.append(dataset_root / suffix)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not resolve h5 file reference {reference!r} from dataset root {dataset_root}."
        )

    @staticmethod
    def _load_split_arrays(h5_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        h5py = _import_h5py()
        data_arrays = []
        label_arrays = []

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as handle:
                if "data" not in handle or "label" not in handle:
                    raise KeyError(
                        f"Malformed ModelNet40 h5 file: {h5_path}. Expected datasets named 'data' and 'label'."
                    )
                data_arrays.append(handle["data"][:].astype(np.float32))
                label_arrays.append(handle["label"][:].astype(np.int64).reshape(-1))

        if not data_arrays:
            raise ValueError("No ModelNet40 arrays were loaded from the discovered h5 files.")

        return np.concatenate(data_arrays, axis=0), np.concatenate(label_arrays, axis=0)

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
