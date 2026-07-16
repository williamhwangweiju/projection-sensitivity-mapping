"""Shared configuration helpers for the Phase 1-4 pipeline."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration {path} must contain a YAML mapping.")
    return payload


def save_yaml(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(payload), stream, sort_keys=False)


def nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"Missing configuration key: {dotted_key}")
        value = value[key]
    return value


def with_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    result = deepcopy(dict(config))
    experiment = result.setdefault("experiment", {})
    if not isinstance(experiment, MutableMapping):
        raise ValueError("experiment must be a mapping.")
    experiment["seed"] = int(seed)
    experiment["placement_seed"] = int(seed)
    return result


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path
