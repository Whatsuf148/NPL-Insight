import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean, merge_seasons


def test_clean_drops_rows_missing_required_identifiers(sample_master_df, config):
    df = sample_master_df.copy()
    df.loc[0, "player"] = np.nan
    cleaned = clean(df, config)
    assert len(cleaned) == len(df) - 1


def test_clean_drops_invalid_phase(sample_master_df, config):
    df = sample_master_df.copy()
    df.loc[0, "phase"] = "garbage_phase"
    cleaned = clean(df, config)
    assert "garbage_phase" not in cleaned["phase"].unique()
    assert len(cleaned) == len(df) - 1


def test_clean_fills_missing_numeric_with_zero_not_drop(sample_master_df, config):
    df = sample_master_df.copy()
    df.loc[0, "wickets"] = np.nan
    cleaned = clean(df, config)
    # row must still be present (filled, not dropped)
    assert len(cleaned) == len(df)
    assert cleaned["wickets"].isna().sum() == 0


def test_clean_recomputes_strike_rate_from_runs_and_balls(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    bat_rows = cleaned[cleaned["balls"] > 0]
    expected = round(bat_rows["runs"] / bat_rows["balls"] * 100, 2)
    assert np.allclose(bat_rows["strike_rate"].values, expected.values)


def test_merge_seasons_raises_if_a_configured_season_is_missing(sample_master_df, config):
    only_season_1 = sample_master_df[sample_master_df["season"] == 1]
    with pytest.raises(ValueError):
        merge_seasons({1: only_season_1}, config)


def test_merge_seasons_succeeds_with_all_configured_seasons(sample_master_df, config):
    by_season = {s: sample_master_df[sample_master_df["season"] == s] for s in (1, 2)}
    master = merge_seasons(by_season, config)
    assert set(master["season"].unique()) == {1, 2}
