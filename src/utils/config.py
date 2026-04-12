from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extends = cfg.pop("extends", None)
    if not extends:
        return cfg
    parent_path = path.parent / extends
    if not parent_path.exists():
        parent_path = Path("configs") / extends
    parent = load_config(parent_path)
    return _deep_merge(parent, cfg)
