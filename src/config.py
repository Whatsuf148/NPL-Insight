"""Single entry point for loading project configuration.

Everything that varies (teams, seasons, paths, thresholds) lives in
config/config.yaml. Code must read through this module instead of
hardcoding values, so the system scales to new seasons/teams without
code changes.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-declared path relative to the project root."""
    return PROJECT_ROOT / relative_path


def get(*keys: str, default: Any = None) -> Any:
    """Dotted-key lookup into config, e.g. get('paths', 'master_dataset')."""
    cfg: Any = load_config()
    for key in keys:
        if not isinstance(cfg, dict) or key not in cfg:
            return default
        cfg = cfg[key]
    return cfg
