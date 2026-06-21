"""Anchors known real season totals onto the simulated master dataset.

The simulator generates statistically generic performances for every
player — it has no concept of "this specific player is an elite bowler."
That's fine for most of the ~150 real names in the dataset (no public
per-player target to match against), but it produces obviously wrong
results for the handful of players Wikipedia's season leaderboards do
publish real totals for: a real bowler with 17 wickets in Season 2
(Sandeep Lamichhane) showing up with 6 in the simulated data looks like
fabricated/fake data to anyone who knows the real numbers, because it is
materially wrong for that specific player.

This module rescales — not replaces — each such player's existing
per-match-phase runs/wickets so the season total matches the real
published figure exactly, while preserving the relative shape of their
existing performances (a big match stays relatively big, a quiet match
stays relatively quiet). Strike rate / economy are recomputed from the
adjusted runs/wickets against the player's existing (unchanged) balls/overs,
since real ball-by-ball-level detail isn't available to anchor those too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _scale_to_exact_total(values: pd.Series, real_total: int) -> pd.Series:
    """Scales a series of non-negative integers so they sum to `real_total`,
    preserving relative proportions and keeping every value an integer."""
    current_total = values.sum()
    if current_total <= 0 or real_total <= 0:
        # No existing signal to scale proportionally — spread evenly instead.
        n = len(values)
        base = real_total // n
        remainder = real_total - base * n
        result = pd.Series(base, index=values.index)
        if remainder > 0:
            result.iloc[:remainder] += 1
        return result

    scaled = values * (real_total / current_total)
    floored = np.floor(scaled).astype(int)
    shortfall = int(real_total - floored.sum())
    if shortfall > 0:
        # Give the leftover units to the rows with the largest fractional
        # remainder, so the result stays as close as possible to the
        # proportional scaling rather than dumping it all on one row.
        fractional = (scaled - floored).sort_values(ascending=False)
        bump_index = fractional.index[:shortfall]
        floored.loc[bump_index] += 1
    return floored


def anchor_real_totals(
    master_df: pd.DataFrame,
    leaders_runs: pd.DataFrame | None,
    leaders_wickets: pd.DataFrame | None,
) -> pd.DataFrame:
    df = master_df.copy()

    if leaders_runs is not None and not leaders_runs.empty:
        runs_table = leaders_runs.copy()
        runs_table["Runs"] = pd.to_numeric(runs_table["Runs"], errors="coerce")
        for _, row in runs_table.dropna(subset=["Runs"]).iterrows():
            season, player, real_total = row["season"], row["Player"], int(row["Runs"])
            mask = (df["season"] == season) & (df["player"] == player)
            if not mask.any():
                continue
            df.loc[mask, "runs"] = _scale_to_exact_total(df.loc[mask, "runs"], real_total)
            df.loc[mask, "strike_rate"] = (
                df.loc[mask, "runs"] / df.loc[mask, "balls"].replace(0, np.nan) * 100
            ).fillna(0.0).round(2)

    if leaders_wickets is not None and not leaders_wickets.empty:
        wickets_table = leaders_wickets.copy()
        wickets_table["Wickets"] = pd.to_numeric(wickets_table["Wickets"], errors="coerce")
        for _, row in wickets_table.dropna(subset=["Wickets"]).iterrows():
            season, player, real_total = row["season"], row["Player"], int(row["Wickets"])
            mask = (df["season"] == season) & (df["player"] == player) & (df["overs"] > 0)
            if not mask.any():
                continue
            df.loc[mask, "wickets"] = _scale_to_exact_total(df.loc[mask, "wickets"], real_total)

    return df
