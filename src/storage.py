"""Storage abstraction — backend (csv/sqlite) is a config switch, not a code fork.

Downstream code (analytics, dashboard) calls `save_table`/`load_table`
and never touches pandas.to_csv/to_sql directly, so switching storage
backend is a one-line config change.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path


def _db_path(config: dict) -> Path:
    return resolve_path(config["paths"]["db_path"])


def save_table(name: str, df: pd.DataFrame, config: dict | None = None) -> None:
    config = config or load_config()
    backend = config["storage"]["backend"]

    if backend == "sqlite":
        db_path = _db_path(config)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            df.to_sql(name, conn, if_exists="replace", index=False)
    elif backend == "csv":
        out_dir = resolve_path(config["paths"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{name}.csv", index=False)
    else:
        raise ValueError(f"Unknown storage backend '{backend}'")


def load_table(name: str, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    backend = config["storage"]["backend"]

    if backend == "sqlite":
        db_path = _db_path(config)
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {name}", conn)
    elif backend == "csv":
        out_dir = resolve_path(config["paths"]["processed_dir"])
        return pd.read_csv(out_dir / f"{name}.csv")
    else:
        raise ValueError(f"Unknown storage backend '{backend}'")
