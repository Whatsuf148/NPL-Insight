"""End-to-end pipeline: collect -> clean -> engineer features -> store.

Run this whenever raw data or config changes; the dashboard only ever
reads the processed output this script produces.
"""
from __future__ import annotations

import logging

from src.config import load_config
from src.data_cleaning import merge_seasons
from src.data_collection import collect_all
from src.feature_engineering import build_feature_set
from src.storage import save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    logger.info("Step 1/4: Collecting raw data...")
    raw_by_season = collect_all(config)

    logger.info("Step 2/4: Cleaning and merging seasons...")
    master_df = merge_seasons(raw_by_season, config)
    save_table("master_dataset", master_df, config)
    logger.info("Master dataset: %d rows, %d columns", *master_df.shape)

    logger.info("Step 3/4: Engineering features...")
    feature_tables = build_feature_set(master_df, config)
    for name, table in feature_tables.items():
        save_table(name, table, config)
        logger.info("  saved feature table '%s' (%d rows)", name, len(table))

    logger.info("Step 4/4: Pipeline complete. Launch the dashboard with:")
    logger.info("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
