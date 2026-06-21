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


def player_stats_table(master_df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Per-player, per-team career line across the selected scope — the
    detailed name+stats view, not just aggregate leaderboards."""
    df = master_df if season is None else master_df[master_df["season"] == season]
    bowled = df[df["overs"] > 0]
    runs_conceded = (bowled["economy"] * bowled["overs"]).groupby(
        [bowled["season"], bowled["player"], bowled["team"]]
    ).sum()

    table = df.groupby(["season", "player", "team"], as_index=False).agg(
        matches=("match_id", "nunique"),
        runs=("runs", "sum"),
        balls_faced=("balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("dismissals", "sum"),
        wickets=("wickets", "sum"),
        overs_bowled=("overs", "sum"),
        maidens=("maidens", "sum"),
        catches_taken=("catches_taken", "sum"),
        catches_dropped=("catches_dropped", "sum"),
        stumping_missed=("stumping_missed", "sum"),
    )
    import numpy as np

    table["strike_rate"] = (table["runs"] / table["balls_faced"].replace(0, np.nan) * 100).round(2).fillna(0.0)
    # Career-style batting average: runs per dismissal, computed once across
    # the whole season (every phase combined) — not per-phase, since an
    # "average" is meaningless split into 20-ball windows. A player who was
    # never dismissed (true not-out across the whole sample) has an
    # undefined/infinite average by cricket convention; we report their raw
    # runs total instead of dividing by zero.
    table["batting_average"] = np.where(
        table["dismissals"] > 0, (table["runs"] / table["dismissals"]).round(2), table["runs"].astype(float)
    )
    table["not_outs"] = (table["matches"] - table["dismissals"]).clip(lower=0)
    table["boundary_pct"] = np.where(
        table["balls_faced"] > 0,
        ((table["fours"] + table["sixes"]) / table["balls_faced"] * 100).round(2),
        0.0,
    )

    runs_conceded = runs_conceded.reset_index(name="runs_conceded")
    table = table.merge(runs_conceded, on=["season", "player", "team"], how="left")
    table["runs_conceded"] = table["runs_conceded"].fillna(0)
    table["economy"] = (table["runs_conceded"] / table["overs_bowled"].replace(0, np.nan)).round(2).fillna(0.0)
    table["bowling_average"] = np.where(
        table["wickets"] > 0, (table["runs_conceded"] / table["wickets"]).round(2), np.nan
    )

    team_matches = df.groupby(["season", "team"])["match_id"].nunique().rename("team_matches")
    table = table.merge(team_matches, on=["season", "team"], how="left")
    table["matches_played_pct"] = (table["matches"] / table["team_matches"] * 100).round(1)

    return table.drop(columns=["runs_conceded", "team_matches"]).sort_values(
        "runs", ascending=False
    ).reset_index(drop=True)


def best_individual_performances(master_df: pd.DataFrame, season: int | None = None) -> dict[str, pd.DataFrame]:
    """Best single-match performances per player: highest individual score,
    and best bowling figures (most wickets, tie-broken by fewest runs conceded)."""
    df = master_df if season is None else master_df[master_df["season"] == season]

    per_match_batting = df.groupby(["season", "player", "team", "match_id", "opponent"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum")
    )
    best_score_idx = per_match_batting.groupby(["season", "player"])["runs"].idxmax()
    best_scores = per_match_batting.loc[best_score_idx].rename(
        columns={"runs": "best_score", "match_id": "match", "opponent": "vs"}
    ).sort_values("best_score", ascending=False).reset_index(drop=True)

    bowled = df[df["overs"] > 0].copy()
    bowled["runs_conceded"] = bowled["economy"] * bowled["overs"]
    per_match_bowling = bowled.groupby(["season", "player", "team", "match_id", "opponent"], as_index=False).agg(
        wickets=("wickets", "sum"), runs_conceded=("runs_conceded", "sum")
    )
    per_match_bowling = per_match_bowling.sort_values(["wickets", "runs_conceded"], ascending=[False, True])
    best_bowling = per_match_bowling.groupby(["season", "player"], as_index=False).first().rename(
        columns={"match_id": "match", "opponent": "vs"}
    )
    best_bowling["runs_conceded"] = best_bowling["runs_conceded"].round(0).astype(int)
    best_bowling = best_bowling.sort_values(
        ["wickets", "runs_conceded"], ascending=[False, True]
    ).reset_index(drop=True)

    return {"best_batting": best_scores, "best_bowling": best_bowling}


def player_profile(master_df: pd.DataFrame, player_impact_score: pd.DataFrame, player: str) -> dict:
    """Single-player summary: every team/season they appear under, career
    totals, best performances, appearance rate, and where they rank by
    impact score — the one-stop view answering 'who is this player and how
    good are they', not just a row in a leaderboard."""
    rows = master_df[master_df["player"] == player]
    if rows.empty:
        return {}

    stats = player_stats_table(master_df)
    player_rows = stats[stats["player"] == player]
    best = best_individual_performances(master_df)
    best_batting = best["best_batting"][best["best_batting"]["player"] == player]
    best_bowling = best["best_bowling"][best["best_bowling"]["player"] == player]

    impact_rows = player_impact_score[player_impact_score["player"] == player]
    ranks = []
    for _, impact_row in impact_rows.iterrows():
        season_scores = player_impact_score[player_impact_score["season"] == impact_row["season"]]
        rank = (season_scores["player_impact_score"] > impact_row["player_impact_score"]).sum() + 1
        ranks.append({"season": impact_row["season"], "impact_rank": int(rank), "out_of": len(season_scores)})

    return {
        "teams": sorted(rows["team"].unique()),
        "seasons": sorted(rows["season"].unique()),
        "stats_by_season": player_rows,
        "best_batting": best_batting,
        "best_bowling": best_bowling,
        "impact_ranks": pd.DataFrame(ranks),
    }


def top_wicket_taker_per_opponent(master_df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Answers: who is the most dangerous bowler against each specific
    opponent team — a matchup-level insight, not a tournament-wide average."""
    df = master_df if season is None else master_df[master_df["season"] == season]
    bowled = df[df["overs"] > 0]
    by_matchup = bowled.groupby(["season", "opponent", "player", "team"], as_index=False)["wickets"].sum()
    top = by_matchup.sort_values("wickets", ascending=False).groupby(["season", "opponent"], as_index=False).first()
    return top.rename(columns={"opponent": "against_team"}).sort_values(
        ["season", "wickets"], ascending=[True, False]
    ).reset_index(drop=True)


def best_strike_rate_by_phase(master_df: pd.DataFrame, season: int | None = None, min_balls: int = 10) -> pd.DataFrame:
    """Answers: who scores fastest in each phase of the innings (powerplay,
    middle, death) — qualifying on a minimum-balls-faced threshold so a
    two-ball cameo can't top the list."""
    df = master_df if season is None else master_df[master_df["season"] == season]
    by_phase_player = df.groupby(["season", "phase", "player", "team"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum")
    )
    qualifying = by_phase_player[by_phase_player["balls"] >= min_balls].copy()
    qualifying["strike_rate"] = (qualifying["runs"] / qualifying["balls"] * 100).round(2)
    return qualifying.sort_values(["phase", "strike_rate"], ascending=[True, False]).groupby(
        ["season", "phase"], as_index=False
    ).head(5).reset_index(drop=True)


def fun_facts(config: dict | None = None) -> list[str]:
    """Real, verified facts about the NPL — sourced from the season's real
    leaderboards/awards tables, not generated/estimated numbers. Falls back
    gracefully if real-data enrichment tables aren't present (e.g.
    `data_sources.wikipedia.enabled: false`)."""
    from src.storage import load_table

    config = config or load_config()
    season_names = {s["id"]: s["name"] for s in config["seasons"]}
    facts: list[str] = []

    try:
        runs_leaders = load_table("real_leaders_runs", config)
        for season_id, group in runs_leaders.groupby("season"):
            top = group.iloc[0]
            facts.append(
                f"{season_names.get(season_id, f'Season {season_id}')}: "
                f"{top['Player']} ({top['Team']}) was the leading run-scorer with {top['Runs']} runs."
            )
    except Exception:
        pass

    try:
        wicket_leaders = load_table("real_leaders_wickets", config)
        for season_id, group in wicket_leaders.groupby("season"):
            top = group.iloc[0]
            facts.append(
                f"{season_names.get(season_id, f'Season {season_id}')}: "
                f"{top['Player']} ({top['Team']}) led the wickets column with {top['Wickets']} dismissals."
            )
    except Exception:
        pass

    try:
        awards = load_table("real_awards", config)
        potm_rows = awards[awards["Award"].str.contains("tournament", case=False, na=False)]
        for _, row in potm_rows.head(4).iterrows():
            season_label = season_names.get(row["season"], f"Season {row['season']}")
            facts.append(f"{season_label} — {row['Award']}: {row['Player']} ({row.get('Team', 'N/A')}).")
    except Exception:
        pass

    if not facts:
        facts.append(
            "No real-data facts available — enable data_sources.wikipedia in "
            "config/config.yaml and re-run the pipeline to populate this section."
        )
    return facts


def all_time_leaders(master_df: pd.DataFrame, top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Cross-season cumulative leaders from the dataset (real player names,
    summed across every season present) — answers 'all-time highest run
    scorer / wicket-taker' across the whole dataset, not just one season."""
    runs = master_df.groupby(["player", "team"], as_index=False)["runs"].sum()
    runs = runs.sort_values("runs", ascending=False).head(top_n).reset_index(drop=True)

    wickets = master_df.groupby(["player", "team"], as_index=False)["wickets"].sum()
    wickets = wickets.sort_values("wickets", ascending=False).head(top_n).reset_index(drop=True)

    catches = master_df.groupby(["player", "team"], as_index=False)["catches_taken"].sum()
    catches = catches.sort_values("catches_taken", ascending=False).head(top_n).reset_index(drop=True)

    return {"runs": runs, "wickets": wickets, "catches": catches}


def season_leaderboards(
    master_df: pd.DataFrame, season: int | None = None, top_n: int = 10,
    min_balls_for_average: int = 30, min_overs_for_bowling_average: float = 4.0,
) -> dict[str, pd.DataFrame]:
    """Season-scoped leaderboards beyond plain run/wicket totals: most
    sixes, most fours, best batting average, best bowling average — each
    qualified on a minimum sample so a single big over doesn't top the list.
    Built on top of `player_stats_table`, which already computes every one
    of these from real ball-by-ball figures when Cricsheet is the active source.
    """
    stats = player_stats_table(master_df, season=season)

    most_sixes = stats.sort_values("sixes", ascending=False).head(top_n)[
        ["season", "player", "team", "sixes", "fours", "matches"]
    ].reset_index(drop=True)
    most_fours = stats.sort_values("fours", ascending=False).head(top_n)[
        ["season", "player", "team", "fours", "sixes", "matches"]
    ].reset_index(drop=True)

    qualified_batters = stats[stats["balls_faced"] >= min_balls_for_average]
    best_batting_average = qualified_batters.sort_values("batting_average", ascending=False).head(top_n)[
        ["season", "player", "team", "batting_average", "runs", "dismissals", "not_outs", "balls_faced"]
    ].reset_index(drop=True)

    qualified_bowlers = stats[stats["overs_bowled"] >= min_overs_for_bowling_average]
    best_bowling_average = qualified_bowlers.dropna(subset=["bowling_average"]).sort_values(
        "bowling_average", ascending=True
    ).head(top_n)[["season", "player", "team", "bowling_average", "wickets", "overs_bowled", "economy"]].reset_index(drop=True)

    most_maidens = stats.sort_values("maidens", ascending=False).head(top_n)[
        ["season", "player", "team", "maidens", "overs_bowled", "economy"]
    ].reset_index(drop=True)

    return {
        "most_sixes": most_sixes,
        "most_fours": most_fours,
        "best_batting_average": best_batting_average,
        "best_bowling_average": best_bowling_average,
        "most_maidens": most_maidens,
    }


def real_all_time_leaders(leaders_runs: pd.DataFrame, leaders_wickets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Combines each season's REAL published top-5 leaderboard into one
    cross-season total. Caveat: this only sums what each season's article
    published as its top 5 — a player just outside a season's top 5 won't
    be counted for that season, so totals are a lower bound, not exhaustive
    career figures."""
    runs = leaders_runs.copy()
    runs["Runs"] = pd.to_numeric(runs["Runs"], errors="coerce")
    runs_total = runs.groupby("Player", as_index=False).agg(
        total_runs=("Runs", "sum"), seasons_in_top5=("season", "nunique"), team=("Team", "first")
    ).sort_values("total_runs", ascending=False).reset_index(drop=True)

    wickets = leaders_wickets.copy()
    wickets["Wickets"] = pd.to_numeric(wickets["Wickets"], errors="coerce")
    wickets_total = wickets.groupby("Player", as_index=False).agg(
        total_wickets=("Wickets", "sum"), seasons_in_top5=("season", "nunique"), team=("Team", "first")
    ).sort_values("total_wickets", ascending=False).reset_index(drop=True)

    return {"runs": runs_total, "wickets": wickets_total}


def player_vs_player_matchup(master_df: pd.DataFrame, player_a: str, player_b: str) -> pd.DataFrame:
    """Side-by-side stat line for two players in every match where both
    appeared (on either the same or opposing teams). The dataset is
    player-match-phase aggregates, not ball-by-ball, so this is an honest
    match-level comparison — not a literal 'who bowled to whom' duel, which
    would need ball-by-ball data this project doesn't have."""
    a_matches = set(master_df.loc[master_df["player"] == player_a, "match_id"])
    b_matches = set(master_df.loc[master_df["player"] == player_b, "match_id"])
    shared_matches = a_matches & b_matches
    if not shared_matches:
        return pd.DataFrame()

    scope = master_df[master_df["match_id"].isin(shared_matches) & master_df["player"].isin([player_a, player_b])]
    summary = scope.groupby(["match_id", "player", "team"], as_index=False).agg(
        runs=("runs", "sum"), balls=("balls", "sum"), wickets=("wickets", "sum"),
        overs=("overs", "sum"), match_result=("match_result", "first"),
    )
    import numpy as np

    summary["strike_rate"] = (summary["runs"] / summary["balls"].replace(0, np.nan) * 100).round(2).fillna(0.0)
    return summary.sort_values("match_id").reset_index(drop=True)


def head_to_head_record(head_to_head: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    """Real fixture-by-fixture results between two specific teams, across
    whichever seasons are present in the real_head_to_head table."""
    mask = (
        ((head_to_head["team_a"] == team_a) & (head_to_head["team_b"] == team_b))
        | ((head_to_head["team_a"] == team_b) & (head_to_head["team_b"] == team_a))
    )
    return head_to_head[mask].reset_index(drop=True)


def batting_order_win_rate(match_results: pd.DataFrame) -> dict[str, object]:
    """Real win rate by batting order (first vs second), derived from
    `real_match_results` (Wikipedia's own scorecard convention: the team
    listed first in "Team A 150 v Team B 151" always batted first).

    This is presented as "batting order", not "toss win probability" — real
    toss-decision data is only published for the two season finals (one
    sentence each, see analytics.fun_facts), nowhere near enough matches to
    fit a meaningful toss-outcome model. Batting order is the closest thing
    with full real coverage (32 matches/season): in T20, winning the toss
    and choosing to bat/bowl is what *decides* batting order in the
    overwhelming majority of matches, so this is a reasonable, honestly
    labeled proxy rather than a literal toss statistic.
    """
    df = match_results.dropna(subset=["winner"]).copy()
    total = len(df)
    if total == 0:
        return {"overall": {}, "by_team": pd.DataFrame()}

    first_wins = int((df["winner"] == df["team1"]).sum())
    second_wins = int((df["winner"] == df["team2"]).sum())
    overall = {
        "total_matches": total,
        "batted_first_wins": first_wins,
        "batted_first_win_pct": round(first_wins / total * 100, 1),
        "batted_second_wins": second_wins,
        "batted_second_win_pct": round(second_wins / total * 100, 1),
    }

    teams = sorted(set(df["team1"]) | set(df["team2"]))
    rows = []
    for team in teams:
        first = df[df["team1"] == team]
        second = df[df["team2"] == team]
        first_w = int((first["winner"] == team).sum())
        second_w = int((second["winner"] == team).sum())
        rows.append({
            "team": team,
            "matches_batting_first": len(first),
            "win_pct_batting_first": round(first_w / len(first) * 100, 1) if len(first) else None,
            "matches_batting_second": len(second),
            "win_pct_batting_second": round(second_w / len(second) * 100, 1) if len(second) else None,
        })

    return {"overall": overall, "by_team": pd.DataFrame(rows)}


def toss_win_probability(toss_results: pd.DataFrame) -> dict[str, object]:
    """Real toss-outcome win rate, from `real_toss_results`
    (`CricsheetSource.fetch_toss_results`) — genuine toss winner/decision
    and match outcome for every match, not the batting-order proxy
    `batting_order_win_rate` used before real per-match toss data was
    available. Also breaks out win rate by the toss winner's decision
    (bat vs field), since that's the more specific real question."""
    df = toss_results.dropna(subset=["toss_winner", "match_winner"]).copy()
    total = len(df)
    if total == 0:
        return {"overall": {}, "by_decision": pd.DataFrame()}

    toss_winner_won = int(df["toss_winner_won_match"].sum())
    overall = {
        "total_matches": total,
        "toss_winner_won_match": toss_winner_won,
        "toss_winner_win_pct": round(toss_winner_won / total * 100, 1),
    }

    rows = []
    for decision in sorted(df["toss_decision"].dropna().unique()):
        subset = df[df["toss_decision"] == decision]
        wins = int(subset["toss_winner_won_match"].sum())
        rows.append({
            "decision": decision,
            "matches": len(subset),
            "toss_winner_win_pct": round(wins / len(subset) * 100, 1) if len(subset) else None,
        })

    return {"overall": overall, "by_decision": pd.DataFrame(rows)}


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
