"""Config loading utilities for nested YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import re

import yaml


DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_numeric(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _coerce_numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric(v) for v in value]
    if isinstance(value, str) and _NUMERIC_RE.match(value.strip()):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = _coerce_numeric(value)
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_user_config(path: Path, seen: set[Path]) -> Dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in seen:
        cycle = " -> ".join(str(p) for p in [*seen, resolved_path])
        raise ValueError(f"Config extends cycle detected: {cycle}")

    user_cfg = _load_yaml(path)
    extends = user_cfg.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]

    cfg: Dict[str, Any] = {}
    next_seen = {*seen, resolved_path}
    for base_ref in extends:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        cfg = _deep_update(cfg, _load_user_config(base_path, next_seen))

    return _deep_update(cfg, user_cfg)


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    cfg: Dict[str, Any] = {}
    if DEFAULTS_PATH.exists():
        cfg = _deep_update(cfg, _load_yaml(DEFAULTS_PATH))

    user_cfg = _load_user_config(path, seen=set())
    cfg = _deep_update(cfg, user_cfg)
    return cfg
