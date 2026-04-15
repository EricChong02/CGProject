"""Model factory functions."""

from __future__ import annotations

from pointcloud_benchmark.models.dgcnn import DGCNNClassifier
from pointcloud_benchmark.models.improved_pointnet2 import ImprovedPointNet2Classifier
from pointcloud_benchmark.models.pointnet2 import PointNet2Classifier


MODEL_REGISTRY = {
    "pointnet2": PointNet2Classifier,
    "dgcnn": DGCNNClassifier,
    "improved_pointnet2": ImprovedPointNet2Classifier,
}


def build_model(config: dict):
    model_name = config["model"]["name"].lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[model_name](config=config)

