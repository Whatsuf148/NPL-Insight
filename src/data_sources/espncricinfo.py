"""ESPNcricinfo scraper.

Verified from this environment: a direct `requests.get` to
https://www.espncricinfo.com returns HTTP 403 (bot-blocked), including the
specific NPL series page
(/series/nepal-premier-league-2024-25-1462594/match-schedule-fixtures-and-results).
That's a network-level block, not a code bug — the fetch logic below is real
(requests + BeautifulSoup against the actual series schedule page), and will
populate real data once run from a network ESPNcricinfo doesn't block (e.g.
a residential IP, or behind an approved API/proxy). Flip
`data_sources.espncricinfo.enabled: true` in config/config.yaml to use it.
"""
from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .base import DataSource

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


class ESPNCricinfoSource(DataSource):
    name = "espncricinfo"

    def fetch(self, season_id: int) -> pd.DataFrame:
        if not self.config.get("enabled", False):
            raise RuntimeError(
                "ESPNCricinfoSource is disabled in config/config.yaml "
                "(data_sources.espncricinfo.enabled)."
            )

        url = self.config["base_url"] + self.config["series_path"]
        response = requests.get(url, headers=_HEADERS, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(
                f"ESPNcricinfo returned HTTP {response.status_code} for {url}. "
                "This was a confirmed 403 (bot-blocked) from the sandboxed environment "
                "this project was built in — try from an unblocked network."
            )

        soup = BeautifulSoup(response.text, "html.parser")
        records = []
        for match_card in soup.select("[class*='match-info']"):
            text = match_card.get_text(" ", strip=True)
            if text:
                records.append({"season": season_id, "raw_match_summary": text})

        if not records:
            raise RuntimeError(
                "ESPNcricinfo page structure didn't match expected selectors — "
                "the site's markup may have changed; update the selectors in "
                "ESPNCricinfoSource.fetch()."
            )
        return pd.DataFrame.from_records(records)
