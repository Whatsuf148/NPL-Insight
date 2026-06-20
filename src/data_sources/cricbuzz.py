"""Cricbuzz scraper — pluggable, disabled by default. See espncricinfo.py for rationale."""
from __future__ import annotations

import pandas as pd

from .base import DataSource


class CricbuzzSource(DataSource):
    name = "cricbuzz"

    def fetch(self, season_id: int) -> pd.DataFrame:
        if not self.config.get("enabled", False):
            raise RuntimeError(
                "CricbuzzSource is disabled in config/config.yaml "
                "(data_sources.cricbuzz.enabled). Enable it and implement "
                "the actual scraping/request logic against "
                f"{self.config.get('base_url')} before use."
            )
        raise NotImplementedError("Cricbuzz scraping logic not yet implemented.")
