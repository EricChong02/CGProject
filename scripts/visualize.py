"""CLI entry point for placeholder visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud_benchmark.utils import create_logger, load_config, prepare_output_dirs
from pointcloud_benchmark.utils.io import load_json, save_json
from pointcloud_benchmark.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training outputs.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def load_or_create_history(history_path: Path, logger) -> dict:
    if history_path.exists():
        return load_json(history_path)

    logger.warning("History file %s not found. Creating a placeholder curve.", history_path)
    history = {
        "train_loss": [1.5, 1.2, 0.9],
        "train_acc": [0.25, 0.45, 0.60],
        "val_acc": [0.20, 0.35, 0.50],
    }
    save_json(history, history_path)
    return history


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepare_output_dirs(config)

    log_file = Path(config["output"]["log_dir"]) / "visualize.log"
    logger = create_logger(f"{config['experiment']['name']}_viz", log_file)

    history_path = Path(config["output"]["result_dir"]) / "train_history.json"
    figure_path = Path(config["output"]["figure_dir"]) / "training_curves.svg"
    history = load_or_create_history(history_path, logger)

    plot_training_history(
        history=history,
        output_path=figure_path,
        title=config["experiment"]["name"],
        dpi=config.get("visualization", {}).get("dpi", 150),
    )
    logger.info("Saved figure to %s", figure_path)


if __name__ == "__main__":
    main()
