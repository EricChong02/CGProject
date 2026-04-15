"""CLI entry point for placeholder evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud_benchmark.datasets import build_dataloader
from pointcloud_benchmark.evaluation import Evaluator
from pointcloud_benchmark.models import build_model
from pointcloud_benchmark.utils import create_logger, load_config, prepare_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a point cloud classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def maybe_load_checkpoint(model, checkpoint_path: str | None, logger) -> None:
    if not checkpoint_path:
        logger.info("No checkpoint provided. Evaluating placeholder weights.")
        return

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        logger.warning("Checkpoint %s does not exist. Continuing without loading.", checkpoint)
        return

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    logger.info("Loaded checkpoint from %s", checkpoint)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepare_output_dirs(config)

    log_file = Path(config["output"]["log_dir"]) / "evaluate.log"
    logger = create_logger(f"{config['experiment']['name']}_eval", log_file)
    device = torch.device(config["runtime"].get("device", "cpu"))

    dataloader = build_dataloader(config, split="test")
    model = build_model(config)
    maybe_load_checkpoint(model, args.checkpoint, logger)

    evaluator = Evaluator(
        config=config,
        model=model,
        dataloader=dataloader,
        device=device,
        logger=logger,
    )
    metrics = evaluator.run()
    logger.info("Evaluation accuracy: %.4f", metrics["accuracy"])


if __name__ == "__main__":
    main()

