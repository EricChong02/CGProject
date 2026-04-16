"""CLI entry point for evaluation."""

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


def resolve_checkpoint_path(config: dict, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    preferred_paths = [checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"]
    for candidate in preferred_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No checkpoint was provided and none were found in {checkpoint_dir}. "
        "Expected `best.pt` or `latest.pt`."
    )


def load_checkpoint(model, checkpoint_path: Path, logger) -> None:
    checkpoint = checkpoint_path
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" not in state:
        raise KeyError(
            f"Malformed checkpoint at {checkpoint}. Expected a 'model_state_dict' entry."
        )
    model.load_state_dict(state["model_state_dict"], strict=True)
    logger.info("Loaded checkpoint from %s", checkpoint)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepare_output_dirs(config)

    log_file = Path(config["output"]["log_dir"]) / "evaluate.log"
    logger = create_logger(f"{config['experiment']['name']}_eval", log_file)
    device = torch.device(config["runtime"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested CUDA but no CUDA device is available. Set runtime.device to 'cpu' or enable CUDA."
        )

    dataloader = build_dataloader(config, split="test")
    if len(dataloader.dataset) == 0:
        raise ValueError(
            f"Evaluation dataset is empty for dataset={config['dataset']['name']!r}. "
            "Check the dataset path and debug sample limits."
        )
    model = build_model(config)
    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    load_checkpoint(model, checkpoint_path, logger)

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
