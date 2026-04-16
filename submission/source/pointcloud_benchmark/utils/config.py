"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

from pointcloud_benchmark.utils.io import ensure_dir

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(config_text: str) -> dict:
    """Parse the small mapping-only YAML files used in this scaffold."""

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw_line in config_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = raw_line.strip().partition(":")
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value == "":
            new_dict: dict = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)

    return root


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config_text = handle.read()

    if yaml is not None:
        config = yaml.safe_load(config_text)
    else:
        config = _load_simple_yaml(config_text)

    config["config_path"] = str(config_path)
    return config


def prepare_output_dirs(config: dict) -> None:
    for key in ("experiment_dir", "log_dir", "checkpoint_dir", "result_dir", "figure_dir"):
        ensure_dir(config["output"][key])

    # TODO: Save versioned config snapshots once experiment management is more mature.
