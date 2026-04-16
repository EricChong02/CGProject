"""Summarize ModelNet40 evaluation metrics across experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation metrics from multiple experiments.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment names under results/, e.g. pointnet2_modelnet40_debug dgcnn_modelnet40_debug.",
    )
    return parser.parse_args()


def load_metrics(experiment_name: str) -> dict:
    metrics_path = PROJECT_ROOT / "results" / experiment_name / "evaluation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}. Run scripts/evaluate.py for this experiment first."
        )
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    rows = []

    for experiment_name in args.experiments:
        metrics = load_metrics(experiment_name)
        rows.append(
            {
                "experiment": experiment_name,
                "model": str(metrics.get("model", "unknown")),
                "dataset": str(metrics.get("dataset", "unknown")),
                "accuracy": float(metrics.get("accuracy", 0.0)),
            }
        )

    rows.sort(key=lambda row: row["accuracy"], reverse=True)

    print("| Experiment | Model | Dataset | Accuracy |")
    print("|---|---|---|---:|")
    for row in rows:
        print(
            f"| {row['experiment']} | {row['model']} | {row['dataset']} | {row['accuracy']:.4f} |"
        )


if __name__ == "__main__":
    main()
