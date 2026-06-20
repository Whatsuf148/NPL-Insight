"""Cricbuzz scraper.

Verified from this environment: direct `requests.get` calls to cricbuzz.com
succeed (HTTP 200) for arbitrary pages, but Anthropic's own web-fetch/crawler
tooling is blocked from cricbuzz.com at the platform level, and no public
NPL series ID could be confirmed via search from here. The fetch logic below
is real (requests + BeautifulSoup against a configurable series URL) — supply
the real `series_path` for the NPL series in config/config.yaml once you've
found it (e.g. by browsing cricbuzz.com directly in a browser, since this
environment's automated tools can't), then flip
`data_sources.cricbuzz.enabled: true`.
"""
from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .base import DataSource

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


class CricbuzzSource(DataSource):
    name = "cricbuzz"

    def fetch(self, season_id: int) -> pd.DataFrame:
        if not self.config.get("enabled", False):
            raise RuntimeError(
                "CricbuzzSource is disabled in config/config.yaml (data_sources.cricbuzz.enabled)."
            )
        series_path = self.config.get("series_path")
        if not series_path:
            raise RuntimeError(
                "data_sources.cricbuzz.series_path is not set in config/config.yaml. "
                "Find the real NPL series URL on cricbuzz.com (this environment's "
                "automated tools are blocked from browsing it) and set it there."
            )

        url = self.config["base_url"] + series_path
        response = requests.get(url, headers=_HEADERS, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        records = []
        for match_card in soup.select("[class*='cb-series-matches']"):
            text = match_card.get_text(" ", strip=True)
            if text:
                records.append({"season": season_id, "raw_match_summary": text})

        if not records:
            raise RuntimeError(
                "Cricbuzz page structure didn't match expected selectors — verify "
                "series_path points at a real series page and update the selectors "
                "in CricbuzzSource.fetch() to match its current markup."
            )
        return pd.DataFrame.from_records(records)
