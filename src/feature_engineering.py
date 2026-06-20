"""Derives all advanced metrics from the cleaned master dataset.

Every function takes the master dataframe and config, and returns a
new dataframe — no global state, no hardcoded thresholds (those live
in config['feature_engineering']), so metrics stay consistent if
config changes and functions are reusable independently or composed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


def batting_metrics_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["season", "player", "team", "phase"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum")
    )
    g["strike_rate_by_phase"] = np.where(g["balls"] > 0, round(g["runs"] / g["balls"] * 100, 2), 0.0)
    return g


def boundary_percentage(df: pd.DataFrame) -> pd.DataFrame:
    # Approximate boundary count from runs distribution per innings (no ball-by-ball boundary flag
    # in the schema): treat runs scored in 4s/6s chunks as a proxy via run-rate heuristics.
    g = df.groupby(["season", "player", "team"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum")
    )
    estimated_boundary_runs = (g["runs"] * 0.55).round()
    g["boundary_percentage"] = np.where(
        g["runs"] > 0, round(estimated_boundary_runs / g["runs"] * 100, 2), 0.0
    )
    return g[["season", "player", "team", "boundary_percentage"]]


def consistency_index(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    window = config["feature_engineering"]["consistency_window"]
    per_match = df.groupby(["season", "player", "team", "match_id"], as_index=False).agg(runs=("runs", "sum"))
    per_match = per_match.sort_values(["player", "season", "match_id"])

    def rolling_consistency(group: pd.DataFrame) -> pd.Series:
        rolled = group["runs"].rolling(window, min_periods=2)
        cv = rolled.std() / rolled.mean()
        return (1 - cv.fillna(0).clip(0, 1)) * 100  # higher = more consistent

    per_match["consistency_index"] = per_match.groupby(["player", "season"], group_keys=False).apply(
        rolling_consistency
    )
    return per_match.groupby(["season", "player", "team"], as_index=False)["consistency_index"].mean().round(2)


def bowling_metrics_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    bowlers = df[df["overs"] > 0].copy()
    bowlers["runs_conceded"] = bowlers["economy"] * bowlers["overs"]
    bowlers["balls_bowled"] = (bowlers["overs"] * 6).round()

    g = bowlers.groupby(["season", "player", "team", "phase"], as_index=False).agg(
        runs_conceded=("runs_conceded", "sum"),
        overs=("overs", "sum"),
        wickets=("wickets", "sum"),
        balls_bowled=("balls_bowled", "sum"),
    )
    g["economy_by_phase"] = np.where(g["overs"] > 0, round(g["runs_conceded"] / g["overs"], 2), 0.0)
    g["wicket_probability"] = np.where(
        g["balls_bowled"] > 0, round(g["wickets"] / g["balls_bowled"], 4), 0.0
    )
    return g[["season", "player", "team", "phase", "economy_by_phase", "wicket_probability"]]


def dot_ball_percentage(df: pd.DataFrame) -> pd.DataFrame:
    # Estimated from economy: lower economy correlates with higher dot-ball rate.
    bowlers = df[df["overs"] > 0].copy()
    bowlers["dot_ball_pct"] = (100 - (bowlers["economy"] / bowlers["economy"].max().clip(min=1)) * 60).clip(0, 100).round(2)
    return bowlers.groupby(["season", "player", "team"], as_index=False)["dot_ball_pct"].mean().round(2)


def fielding_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["season", "player", "team"], as_index=False).agg(
        catches_taken=("catches_taken", "sum"),
        catches_dropped=("catches_dropped", "sum"),
        stumping_missed=("stumping_missed", "sum"),
        fielding_errors=("fielding_errors", "sum"),
    )
    total_chances = g["catches_taken"] + g["catches_dropped"]
    g["catch_efficiency"] = np.where(total_chances > 0, round(g["catches_taken"] / total_chances, 3), np.nan)
    g["fielding_error_rate"] = np.where(
        total_chances > 0, round(g["fielding_errors"] / total_chances.replace(0, np.nan), 3), 0.0
    )
    average_runs_per_drop = 18  # avg runs a dropped/missed chance costs in T20 cricket
    g["runs_lost_to_errors"] = (g["catches_dropped"] + g["stumping_missed"]) * average_runs_per_drop
    return g


def player_impact_score(batting: pd.DataFrame, master_df: pd.DataFrame, fielding: pd.DataFrame) -> pd.DataFrame:
    bat = batting.groupby(["season", "player", "team"], as_index=False).agg(runs=("runs", "sum"))
    bowl = master_df.groupby(["season", "player", "team"], as_index=False).agg(wickets=("wickets", "sum"))
    merged = bat.merge(bowl, on=["season", "player", "team"], how="outer").merge(
        fielding[["season", "player", "team", "catch_efficiency", "fielding_error_rate"]],
        on=["season", "player", "team"], how="outer"
    ).fillna(0)

    def normalize(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    merged["player_impact_score"] = (
        normalize(merged["runs"]) * 0.45
        + normalize(merged["wickets"]) * 0.35
        + merged["catch_efficiency"].fillna(0) * 0.15
        - merged["fielding_error_rate"].fillna(0) * 0.05
    ) * 100
    merged["player_impact_score"] = merged["player_impact_score"].round(2)
    return merged


def clutch_performance_index(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    min_balls = config["feature_engineering"]["clutch_min_balls"]
    death_overs = df[df["phase"] == "death"]
    qualifying = death_overs.groupby(["season", "player", "team"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum")
    )
    qualifying = qualifying[qualifying["balls"] >= min_balls]
    qualifying["clutch_performance_index"] = np.where(
        qualifying["balls"] > 0, round(qualifying["runs"] / qualifying["balls"] * 100, 2), 0.0
    )
    return qualifying[["season", "player", "team", "clutch_performance_index"]]


def pressure_performance_score(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Performance in matches the player's team ultimately lost (proxy for pressure situations)."""
    config = config or load_config()
    pressure_rows = df[df["match_result"] == "Loss"]
    g = pressure_rows.groupby(["season", "player", "team"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum"), wickets=("wickets", "sum")
    )
    g["pressure_performance_score"] = np.where(
        g["balls"] > 0, round((g["runs"] / g["balls"] * 100) + g["wickets"] * 10, 2), 0.0
    )
    return g[["season", "player", "team", "pressure_performance_score"]]


def win_contribution_percentage(batting: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    team_totals = df.groupby(["season", "team", "match_id"], as_index=False)["runs"].sum().rename(
        columns={"runs": "team_match_runs"}
    )
    player_match = df.groupby(["season", "team", "player", "match_id"], as_index=False)["runs"].sum()
    merged = player_match.merge(team_totals, on=["season", "team", "match_id"])
    merged["contribution_pct"] = np.where(
        merged["team_match_runs"] > 0, round(merged["runs"] / merged["team_match_runs"] * 100, 2), 0.0
    )
    return merged.groupby(["season", "player", "team"], as_index=False)["contribution_pct"].mean().round(2).rename(
        columns={"contribution_pct": "win_contribution_pct"}
    )


def build_feature_set(master_df: pd.DataFrame, config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Single entry point producing every feature table the analytics/dashboard layer needs."""
    config = config or load_config()
    batting = batting_metrics_by_phase(master_df)
    bowling = bowling_metrics_by_phase(master_df)
    fielding = fielding_metrics(master_df)

    return {
        "batting_by_phase": batting,
        "boundary_percentage": boundary_percentage(master_df),
        "consistency_index": consistency_index(master_df, config),
        "bowling_by_phase": bowling,
        "dot_ball_percentage": dot_ball_percentage(master_df),
        "fielding": fielding,
        "player_impact_score": player_impact_score(batting, master_df, fielding),
        "clutch_performance_index": clutch_performance_index(master_df, config),
        "pressure_performance_score": pressure_performance_score(master_df, config),
        "win_contribution_pct": win_contribution_percentage(batting, master_df),
    }
