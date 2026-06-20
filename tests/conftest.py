import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_master_df():
    """Minimal but schema-complete master dataset: 2 seasons, 2 matches,
    2 teams, a few players, all three phases — enough to exercise every
    feature/analytics function without depending on the simulator."""
    rows = []
    for season in (1, 2):
        for match_idx in (1, 2):
            match_id = f"S{season}M{match_idx:03d}"
            for team, opponent, result in [("Team A", "Team B", "Win"), ("Team B", "Team A", "Loss")]:
                for player, is_bowler in [(f"{team} Bat1", False), (f"{team} Bowl1", True)]:
                    for phase in ("powerplay", "middle", "death"):
                        rows.append({
                            "match_id": match_id,
                            "season": season,
                            "team": team,
                            "opponent": opponent,
                            "player": player,
                            "runs": 20 if not is_bowler else 5,
                            "balls": 15 if not is_bowler else 4,
                            "strike_rate": 0.0,  # recomputed by cleaning
                            "wickets": 1 if is_bowler else 0,
                            "overs": 1.0 if is_bowler else 0.0,
                            "economy": 7.0 if is_bowler else 0.0,
                            "catches_taken": 1,
                            "catches_dropped": 0,
                            "stumping_missed": 0,
                            "fielding_errors": 0,
                            "match_result": result,
                            "venue": "Test Ground",
                            "phase": phase,
                        })
    return pd.DataFrame.from_records(rows)
