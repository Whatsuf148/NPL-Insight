"""ESPNcricinfo scraper — pluggable, disabled by default.

This is a stub defining the contract for a real scraper. Wire it up
by implementing `fetch` against actual scorecard/ball-by-ball pages
(or the official API if available) once real endpoints/credentials
are agreed on. Until then it raises clearly instead of silently
returning empty/fake data.
"""
from __future__ import annotations

import pandas as pd

from .base import DataSource


class ESPNCricinfoSource(DataSource):
    name = "espncricinfo"

    def fetch(self, season_id: int) -> pd.DataFrame:
        if not self.config.get("enabled", False):
            raise RuntimeError(
                "ESPNCricinfoSource is disabled in config/config.yaml "
                "(data_sources.espncricinfo.enabled). Enable it and implement "
                "the actual scraping/request logic against "
                f"{self.config.get('base_url')} before use."
            )
        raise NotImplementedError("ESPNcricinfo scraping logic not yet implemented.")
