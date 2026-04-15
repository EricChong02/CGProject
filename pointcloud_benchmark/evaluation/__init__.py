"""Evaluation utilities."""

from __future__ import annotations

__all__ = ["Evaluator"]


def __getattr__(name: str):
    if name == "Evaluator":
        from pointcloud_benchmark.evaluation.evaluator import Evaluator

        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
