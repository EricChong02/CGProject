"""Smoke test for the real ScanObjectNN data pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud_benchmark.datasets import build_dataloader
from pointcloud_benchmark.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the ScanObjectNN dataloader.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_name = config["dataset"]["name"].lower()
    if dataset_name != "scanobjectnn":
        raise ValueError(
            f"Smoke test expects a ScanObjectNN config, but got dataset={dataset_name!r}."
        )

    dataloader = build_dataloader(config=config, split=args.split)
    batch = next(iter(dataloader))
    points = batch["points"]
    labels = batch["label"]

    print(f"Batch tensor shape: {tuple(points.shape)}")
    print(f"Label shape: {tuple(labels.shape)}")
    print(f"Min/Max values: {points.min().item():.6f} / {points.max().item():.6f}")


if __name__ == "__main__":
    main()
