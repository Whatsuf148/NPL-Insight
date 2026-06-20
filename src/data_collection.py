"""Orchestrates data collection across all enabled sources/seasons.

Two kinds of sources:
  - The per-phase player-match source (the simulator) that produces the
    master dataset the rest of the pipeline depends on. This must always
    succeed — never silently skips a failing source.
  - Real-data enrichment (wikipedia.py today; espncricinfo.py/cricbuzz.py
    once unblocked) that supplies real rosters, leaderboards, awards, and
    real match scorecards. This is best-effort: if the network/site is
    unavailable, the simulator falls back to synthetic player names rather
    than failing the whole pipeline, since the enrichment is additive, not
    the core dataset.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config, resolve_path
from src.data_sources import SOURCE_REGISTRY
from src.data_sources.wikipedia import WikipediaSource

logger = logging.getLogger(__name__)

MASTER_SOURCE_NAMES = {"simulator", "espncricinfo", "cricbuzz"}


def collect_real_enrichment(config: dict | None = None) -> dict:
    """Best-effort real-data layer. Returns combined rosters/leaders/awards/
    final-scorecards across all configured seasons, or empty structures if
    the source is disabled or unreachable."""
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    enrichment = {
        "rosters": {},
        "leaders_runs": [],
        "leaders_wickets": [],
        "awards": [],
        "final_scorecards": [],
        "standings": [],
        "head_to_head": [],
    }

    wikipedia_config = config["data_sources"].get("wikipedia", {})
    if not wikipedia_config.get("enabled", False):
        return enrichment

    source = WikipediaSource({**config, **wikipedia_config})
    for season in config["seasons"]:
        season_id = season["id"]
        try:
            squads = source.fetch_squads(season_id)
            for team, players in squads.items():
                enrichment["rosters"].setdefault(team, set()).update(players)

            leaders = source.fetch_leaders(season_id)
            if "most_runs" in leaders:
                enrichment["leaders_runs"].append(leaders["most_runs"])
            if "most_wickets" in leaders:
                enrichment["leaders_wickets"].append(leaders["most_wickets"])

            awards = source.fetch_awards(season_id)
            if not awards.empty:
                enrichment["awards"].append(awards)

            standings = source.fetch_standings(season_id)
            if not standings.empty:
                enrichment["standings"].append(standings)

            head_to_head = source.fetch_head_to_head(season_id)
            if not head_to_head.empty:
                enrichment["head_to_head"].append(head_to_head)

            final_scorecard = source.fetch(season_id)
            if not final_scorecard.empty:
                final_scorecard.to_csv(raw_dir / f"season_{season_id}_wikipedia_final.csv", index=False)
                enrichment["final_scorecards"].append(final_scorecard)

            logger.info("Wikipedia enrichment OK for season %s (%d real squads)", season_id, len(squads))
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment, never fatal
            logger.warning("Wikipedia enrichment unavailable for season %s (%s); "
                            "falling back to synthetic player names.", season_id, exc)

    enrichment["rosters"] = {team: sorted(players) for team, players in enrichment["rosters"].items()}
    return enrichment


def collect_all(config: dict | None = None) -> tuple[dict[int, pd.DataFrame], dict]:
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    enrichment = collect_real_enrichment(config)

    enabled_sources = [s for s in config["data_sources"]["enabled"] if s in MASTER_SOURCE_NAMES]
    seasons = [s["id"] for s in config["seasons"]]

    collected: dict[int, pd.DataFrame] = {}
    for season_id in seasons:
        season_frames = []
        for source_name in enabled_sources:
            if source_name not in SOURCE_REGISTRY:
                raise ValueError(f"Unknown data source '{source_name}' in config.")
            source_config = {**config, **config["data_sources"].get(source_name, {})}
            if source_name == "simulator" and enrichment["rosters"]:
                source_config = {**source_config, "_real_rosters": enrichment["rosters"]}

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

    return collected, enrichment


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all()
