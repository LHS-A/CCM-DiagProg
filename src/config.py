from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default_path = ROOT / "configs" / "default.yaml"
    with default_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle) or {})
    config["_root"] = str(ROOT)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["project"][key])
    return path if path.is_absolute() else Path(config["_root"]) / path


def get_task(config: dict[str, Any], task: str) -> dict[str, Any]:
    if task not in config["tasks"]:
        raise KeyError(f"Unknown task {task!r}; expected one of {sorted(config['tasks'])}")
    result = deepcopy(config["tasks"][task])
    result["id"] = task
    return result

