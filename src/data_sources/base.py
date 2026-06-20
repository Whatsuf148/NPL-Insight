"""Abstract interface every data source must implement.

Keeping collection behind a common interface means the pipeline never
cares whether data came from a scraper, an API, or the simulator —
new sources plug in by registering in data_sources/__init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    name: str = "base"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fetch(self, season_id: int) -> pd.DataFrame:
        """Return a raw ball/innings-level dataframe for one season.

        Must include at minimum: match_id, season, team, opponent,
        player, venue — downstream cleaning fills/validates the rest.
        """
        raise NotImplementedError
