"""Basic import smoke tests."""

from pointcloud_benchmark.datasets import build_dataloader, build_dataset
from pointcloud_benchmark.evaluation import Evaluator
from pointcloud_benchmark.models import build_model
from pointcloud_benchmark.training import Trainer
from pointcloud_benchmark.visualization import plot_training_history


def test_imports_exist() -> None:
    assert build_dataset is not None
    assert build_dataloader is not None
    assert build_model is not None
    assert Trainer is not None
    assert Evaluator is not None
    assert plot_training_history is not None
