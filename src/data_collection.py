"""Orchestrates data collection across all enabled sources/seasons.

Iterates config-declared seasons and config-declared enabled sources,
writes one raw CSV per (season, source) to data/raw — never silently
skips a failing source, since incomplete raw data leads to incomplete
datasets downstream.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config, resolve_path
from src.data_sources import SOURCE_REGISTRY

logger = logging.getLogger(__name__)


def collect_all(config: dict | None = None) -> dict[int, pd.DataFrame]:
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    enabled_sources = config["data_sources"]["enabled"]
    seasons = [s["id"] for s in config["seasons"]]

    collected: dict[int, pd.DataFrame] = {}
    for season_id in seasons:
        season_frames = []
        for source_name in enabled_sources:
            if source_name not in SOURCE_REGISTRY:
                raise ValueError(f"Unknown data source '{source_name}' in config.")
            source_config = {**config, **config.get(source_name, {})}
            source = SOURCE_REGISTRY[source_name](source_config)
            logger.info("Collecting season %s from source '%s'", season_id, source_name)
            df = source.fetch(season_id)
            if df.empty:
                raise RuntimeError(
                    f"Source '{source_name}' returned no data for season {season_id}; "
                    "refusing to proceed with an incomplete dataset."
                )
            out_path = raw_dir / f"season_{season_id}_{source_name}.csv"
            df.to_csv(out_path, index=False)
            season_frames.append(df)

        collected[season_id] = pd.concat(season_frames, ignore_index=True)

    return collected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all()
