"""End-to-end pipeline: collect -> clean -> engineer features -> store.

Run this whenever raw data or config changes; the dashboard only ever
reads the processed output this script produces.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config
from src.data_cleaning import merge_seasons
from src.data_collection import collect_all
from src.feature_engineering import build_feature_set
from src.real_data_anchor import anchor_real_totals
from src.storage import save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    logger.info("Step 1/5: Collecting raw data (+ real-data enrichment)...")
    raw_by_season, enrichment = collect_all(config)

    logger.info("Step 2/5: Cleaning and merging seasons...")
    master_df = merge_seasons(raw_by_season, config)

    leaders_runs = pd.concat(enrichment["leaders_runs"], ignore_index=True) if enrichment["leaders_runs"] else None
    leaders_wickets = pd.concat(enrichment["leaders_wickets"], ignore_index=True) if enrichment["leaders_wickets"] else None
    # Only needed for the simulator fallback path — cricsheet's ball-by-ball
    # totals are already exactly real, and its abbreviated player names
    # ("S Lamichhane") don't name-match Wikipedia's full names ("Sandeep
    # Lamichhane") used in leaders_runs/leaders_wickets anyway, so this is a
    # harmless no-op when cricsheet is the active master source.
    using_simulator = "cricsheet" not in config["data_sources"]["enabled"]
    if using_simulator and (leaders_runs is not None or leaders_wickets is not None):
        before_rows = len(master_df)
        master_df = anchor_real_totals(master_df, leaders_runs, leaders_wickets)
        assert len(master_df) == before_rows, "anchor_real_totals must not change row count"
        logger.info("  anchored real season totals for known leaderboard players onto the simulated dataset")

    save_table("master_dataset", master_df, config)
    logger.info("Master dataset: %d rows, %d columns", *master_df.shape)

    logger.info("Step 3/5: Engineering features...")
    feature_tables = build_feature_set(master_df, config)
    for name, table in feature_tables.items():
        save_table(name, table, config)
        logger.info("  saved feature table '%s' (%d rows)", name, len(table))

    logger.info("Step 4/5: Saving real-data enrichment tables...")
    if leaders_runs is not None:
        save_table("real_leaders_runs", leaders_runs, config)
    if leaders_wickets is not None:
        save_table("real_leaders_wickets", leaders_wickets, config)
    if enrichment["awards"]:
        save_table("real_awards", pd.concat(enrichment["awards"], ignore_index=True), config)
    if enrichment["final_scorecards"]:
        save_table("real_final_scorecards", pd.concat(enrichment["final_scorecards"], ignore_index=True), config)
    if enrichment["standings"]:
        save_table("real_standings", pd.concat(enrichment["standings"], ignore_index=True), config)
    if enrichment["head_to_head"]:
        save_table("real_head_to_head", pd.concat(enrichment["head_to_head"], ignore_index=True), config)
    if enrichment["match_results"]:
        save_table("real_match_results", pd.concat(enrichment["match_results"], ignore_index=True), config)
    if enrichment["final_tosses"]:
        tosses_df = pd.DataFrame(
            [{"season": s, "toss": t} for s, t in enrichment["final_tosses"].items()]
        )
        save_table("real_final_tosses", tosses_df, config)
    if enrichment["real_tosses"]:
        save_table("real_toss_results", pd.concat(enrichment["real_tosses"], ignore_index=True), config)
    logger.info("  real squads collected for %d teams", len(enrichment["rosters"]))

    logger.info("Step 5/5: Pipeline complete. Launch the dashboard with:")
    logger.info("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
