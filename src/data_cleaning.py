"""Cleans and normalizes raw per-season data into one validated master dataset.

Rules:
- Required identifier columns must never be missing — rows failing
  that are dropped and counted, not silently kept.
- Numeric metric columns get missing values filled with 0 (a player
  who didn't bowl has 0 overs, not NaN) rather than dropped, so the
  dataset stays complete.
- Categorical text columns are stripped/title-cased for consistency
  across seasons (e.g. venue name spelling drift).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["match_id", "season", "team", "opponent", "player", "venue", "phase"]
NUMERIC_COLUMNS = [
    "runs", "balls", "strike_rate", "wickets", "overs", "economy",
    "catches_taken", "catches_dropped", "stumping_missed", "fielding_errors",
    # Real ball-by-ball-only columns (Cricsheet) — backfilled to 0 for
    # sources that don't produce them (e.g. the simulator fallback), so
    # downstream code can always rely on these columns existing.
    "fours", "sixes", "dismissals", "maidens",
]
CATEGORICAL_COLUMNS = ["team", "opponent", "player", "venue", "phase", "match_result"]


def clean(raw: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    df = raw.copy()

    missing_required = df[REQUIRED_COLUMNS].isna().any(axis=1)
    if missing_required.any():
        logger.warning("Dropping %d rows missing required identifiers", missing_required.sum())
        df = df[~missing_required]

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    valid_phases = set(config["phases"])
    invalid_phase = ~df["phase"].isin(valid_phases)
    if invalid_phase.any():
        logger.warning("Dropping %d rows with invalid phase values", invalid_phase.sum())
        df = df[~invalid_phase]

    df = df.drop_duplicates(subset=["match_id", "team", "player", "phase"])

    df["strike_rate"] = df.apply(
        lambda r: round(r["runs"] / r["balls"] * 100, 2) if r["balls"] > 0 else 0.0, axis=1
    )

    return df.reset_index(drop=True)


def merge_seasons(season_frames: dict[int, pd.DataFrame], config: dict | None = None) -> pd.DataFrame:
    """Clean each season independently, then merge into the master dataset."""
    config = config or load_config()
    cleaned = [clean(df, config) for df in season_frames.values()]
    master = pd.concat(cleaned, ignore_index=True)

    expected_seasons = {s["id"] for s in config["seasons"]}
    present_seasons = set(master["season"].unique())
    if present_seasons != expected_seasons:
        raise ValueError(
            f"Master dataset is incomplete: expected seasons {expected_seasons}, "
            f"found {present_seasons}."
        )
    return master
