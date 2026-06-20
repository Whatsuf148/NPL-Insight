import numpy as np

from src.data_cleaning import clean
from src.feature_engineering import (
    batting_metrics_by_phase,
    bowling_metrics_by_phase,
    build_feature_set,
    fielding_metrics,
)


def test_batting_metrics_strike_rate_matches_runs_over_balls(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = batting_metrics_by_phase(cleaned)
    nonzero = result[result["balls"] > 0]
    expected = round(nonzero["runs"] / nonzero["balls"] * 100, 2)
    assert np.allclose(nonzero["strike_rate_by_phase"].values, expected.values)


def test_bowling_metrics_economy_matches_runs_conceded_over_overs(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = bowling_metrics_by_phase(cleaned)
    # every bowler row in the fixture concedes at economy 7.0
    assert (result["economy_by_phase"] == 7.0).all()


def test_fielding_catch_efficiency_is_one_when_no_drops(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = fielding_metrics(cleaned)
    # fixture has catches_taken=1, catches_dropped=0 for every player -> perfect efficiency
    assert (result["catch_efficiency"] == 1.0).all()


def test_fielding_runs_lost_to_errors_zero_when_no_drops_or_misses(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = fielding_metrics(cleaned)
    assert (result["runs_lost_to_errors"] == 0).all()


def test_build_feature_set_returns_every_expected_table(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    expected_keys = {
        "batting_by_phase", "boundary_percentage", "consistency_index",
        "bowling_by_phase", "dot_ball_percentage", "fielding",
        "player_impact_score", "clutch_performance_index",
        "pressure_performance_score", "win_contribution_pct",
    }
    assert expected_keys.issubset(tables.keys())
    for name, df in tables.items():
        assert not df.empty, f"feature table '{name}' should not be empty"


def test_player_impact_score_bounded_zero_to_hundred(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    scores = tables["player_impact_score"]["player_impact_score"]
    assert scores.between(-100, 100).all()
