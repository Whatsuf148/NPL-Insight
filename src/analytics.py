"""Insight-generation layer — turns feature tables into ranked, comparative,
narrative-ready outputs. This is what makes the dashboard insight-driven
rather than a pile of charts: every function here answers a specific
question ("who improved most?", "which team wins more from the field?")
instead of just reshaping data for plotting.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config


def player_rankings(player_impact_score: pd.DataFrame, season: int | None = None, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    top_n = config["analytics"]["player_ranking_top_n"]
    df = player_impact_score if season is None else player_impact_score[player_impact_score["season"] == season]
    return df.sort_values("player_impact_score", ascending=False).head(top_n).reset_index(drop=True)


def catch_drop_leaderboard(fielding: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    df = fielding if season is None else fielding[fielding["season"] == season]
    return df.sort_values("catches_dropped", ascending=False).reset_index(drop=True)


def best_fielders_ranking(fielding: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    df = fielding if season is None else fielding[fielding["season"] == season]
    qualifying = df[(df["catches_taken"] + df["catches_dropped"]) >= 2]
    return qualifying.sort_values("catch_efficiency", ascending=False).reset_index(drop=True)


def season_comparison(master_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    season_names = {s["id"]: s["name"] for s in config["seasons"]}

    summary = master_df.groupby("season").agg(
        total_runs=("runs", "sum"),
        avg_strike_rate=("strike_rate", "mean"),
        total_wickets=("wickets", "sum"),
        avg_economy=("economy", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        catches_dropped=("catches_dropped", "sum"),
        fielding_errors=("fielding_errors", "sum"),
    ).reset_index()
    summary["season_name"] = summary["season"].map(season_names)
    summary = summary.round(2)
    return summary


def player_performance_change(player_impact_score: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Per-player impact score delta between consecutive seasons — the headline
    'who improved / declined' insight for season comparison."""
    config = config or load_config()
    pivoted = player_impact_score.pivot_table(
        index=["player", "team"], columns="season", values="player_impact_score", aggfunc="mean"
    ).reset_index()

    season_ids = sorted(s["id"] for s in config["seasons"])
    if len(season_ids) >= 2:
        first, last = season_ids[0], season_ids[-1]
        if first in pivoted.columns and last in pivoted.columns:
            pivoted["impact_score_change"] = (pivoted[last] - pivoted[first]).round(2)
            pivoted = pivoted.sort_values("impact_score_change", ascending=False)
    return pivoted


def generate_match_insights(master_df: pd.DataFrame, match_id: str) -> list[str]:
    """Plain-language insight bullets for a single match — drill-down narrative,
    not just a chart."""
    match_df = master_df[master_df["match_id"] == match_id]
    if match_df.empty:
        return [f"No data found for match {match_id}."]

    insights = []
    top_scorer = match_df.groupby("player")["runs"].sum().idxmax()
    top_runs = match_df.groupby("player")["runs"].sum().max()
    insights.append(f"Top scorer: {top_scorer} with {int(top_runs)} runs.")

    bowlers = match_df[match_df["overs"] > 0]
    if not bowlers.empty:
        best_bowler = bowlers.groupby("player")["wickets"].sum().idxmax()
        best_wkts = bowlers.groupby("player")["wickets"].sum().max()
        if best_wkts > 0:
            insights.append(f"Best bowling figures: {best_bowler} with {int(best_wkts)} wicket(s).")

    drops = match_df["catches_dropped"].sum()
    if drops > 0:
        insights.append(f"{int(drops)} catch(es) dropped — a potential turning point.")

    return insights


def win_probability_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Match-team level feature table feeding the win-probability model in models/."""
    g = master_df.groupby(["season", "match_id", "team"], as_index=False).agg(
        runs=("runs", "sum"),
        wickets=("wickets", "sum"),
        catches_dropped=("catches_dropped", "sum"),
        fielding_errors=("fielding_errors", "sum"),
        match_result=("match_result", "first"),
    )
    g["won"] = (g["match_result"] == "Win").astype(int)
    return g
