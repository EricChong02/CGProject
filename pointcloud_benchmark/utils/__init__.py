"""General utilities."""

from __future__ import annotations

__all__ = ["load_config", "prepare_output_dirs", "create_logger", "set_seed"]


def __getattr__(name: str):
    if name in {"load_config", "prepare_output_dirs"}:
        from pointcloud_benchmark.utils.config import load_config, prepare_output_dirs

        return {"load_config": load_config, "prepare_output_dirs": prepare_output_dirs}[name]
    if name == "create_logger":
        from pointcloud_benchmark.utils.logger import create_logger

        return create_logger
    if name == "set_seed":
        from pointcloud_benchmark.utils.seed import set_seed

        return set_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
