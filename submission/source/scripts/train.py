"""CLI entry point for training."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud_benchmark.datasets import build_dataloader
from pointcloud_benchmark.models import build_model
from pointcloud_benchmark.training import Trainer
from pointcloud_benchmark.utils import create_logger, load_config, prepare_output_dirs, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a point cloud classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepare_output_dirs(config)
    set_seed(config["experiment"]["seed"])

    log_file = Path(config["output"]["log_dir"]) / "train.log"
    logger = create_logger(config["experiment"]["name"], log_file)
    logger.info("Starting training for %s", config["experiment"]["name"])

    config_snapshot = Path(config["output"]["experiment_dir"]) / "config_snapshot.yaml"
    shutil.copy2(args.config, config_snapshot)

    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested CUDA but no CUDA device is available. Set runtime.device to 'cpu' or enable CUDA."
        )
    train_loader = build_dataloader(config, split="train")
    val_loader = build_dataloader(config, split="test")
    dataset_name = config["dataset"]["name"]
    if len(train_loader.dataset) == 0:
        raise ValueError(
            f"Training dataset is empty for dataset={dataset_name!r}. "
            "Check the dataset path and debug sample limits."
        )
    if len(val_loader.dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty for dataset={dataset_name!r}. "
            "Check the dataset path and debug sample limits."
        )
    if len(train_loader) == 0:
        raise ValueError(
            "Training dataloader produced zero batches. "
            "This can happen when training.drop_last=true and the dataset subset is smaller than batch_size."
        )
    model = build_model(config)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
    )
    history = trainer.train()
    logger.info(
        "Finished training. Saved %d epochs of history. Best val_acc=%.4f at epoch=%s",
        len(history["train_loss"]),
        history["best_val_acc"],
        history["best_epoch"],
    )


if __name__ == "__main__":
    main()
