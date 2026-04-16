"""Smoke test for the DGCNN classification forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud_benchmark.models import build_model
from pointcloud_benchmark.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the DGCNN classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the random forward pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config)
    model.eval()

    num_points = int(config["dataset"]["num_points"])
    input_channels = int(config["model"].get("input_channels", 3))
    x = torch.randn(args.batch_size, num_points, input_channels)

    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
