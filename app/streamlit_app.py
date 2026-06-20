"""NPL Insight — Streamlit dashboard.

Reads exclusively from the processed master dataset + feature tables
(no hardcoded numbers/teams/players anywhere) and is organized around
the six sections from the project spec. Every section pairs a chart
with a generated insight statement, so the page reads as analysis,
not just visuals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import analytics
from src.config import load_config
from src.storage import load_table

st.set_page_config(page_title="NPL Insight", layout="wide", page_icon="🏏")

config = load_config()


@st.cache_data
def get_data():
    master_df = load_table("master_dataset", config)
    features = {
        name: load_table(name, config)
        for name in [
            "batting_by_phase", "boundary_percentage", "consistency_index",
            "bowling_by_phase", "dot_ball_percentage", "fielding",
            "player_impact_score", "clutch_performance_index",
            "pressure_performance_score", "win_contribution_pct",
        ]
    }
    return master_df, features


try:
    master_df, features = get_data()
except Exception as e:
    st.error(
        "Processed dataset not found or failed to load. Run `python run_pipeline.py` "
        f"first to generate it.\n\nDetails: {e}"
    )
    st.stop()

season_lookup = {s["name"]: s["id"] for s in config["seasons"]}

st.sidebar.title("🏏 NPL Insight")
st.sidebar.caption("Nepal Premier League Advanced Analytics")

selected_season_names = st.sidebar.multiselect(
    "Season", options=list(season_lookup.keys()), default=list(season_lookup.keys())
)
selected_seasons = [season_lookup[n] for n in selected_season_names] or list(season_lookup.values())

teams_available = sorted(master_df[master_df["season"].isin(selected_seasons)]["team"].unique())
selected_teams = st.sidebar.multiselect("Team", options=teams_available, default=teams_available)

filtered_for_players = master_df[
    master_df["season"].isin(selected_seasons) & master_df["team"].isin(selected_teams)
]
players_available = sorted(filtered_for_players["player"].unique())
selected_players = st.sidebar.multiselect("Player", options=players_available, default=[])

matches_available = sorted(filtered_for_players["match_id"].unique())
selected_match = st.sidebar.selectbox("Match (for drill-down)", options=["All"] + matches_available)

selected_phases = st.sidebar.multiselect(
    "Phase", options=config["phases"], default=config["phases"]
)


def apply_filters(df):
    out = df[df["season"].isin(selected_seasons)] if "season" in df.columns else df
    if "team" in out.columns:
        out = out[out["team"].isin(selected_teams)]
    if selected_players and "player" in out.columns:
        out = out[out["player"].isin(selected_players)]
    if "phase" in out.columns:
        out = out[out["phase"].isin(selected_phases)]
    return out


filtered_master = apply_filters(master_df)

sections = st.tabs([
    "Match Overview", "Batting Analysis", "Bowling Analysis",
    "Fielding Insights", "Player Insights", "Season Comparison", "Fun Facts",
])

# ---------------------------------------------------------------- Match Overview
with sections[0]:
    st.header("Match Overview")
    if filtered_master.empty:
        st.warning("No data for the current filter selection.")
    else:
        match_scope = filtered_master if selected_match == "All" else filtered_master[filtered_master["match_id"] == selected_match]
        score_summary = match_scope.groupby(["match_id", "team"], as_index=False)["runs"].sum()
        fig = px.bar(score_summary, x="match_id", y="runs", color="team", barmode="group",
                     title="Score Summary by Match")
        st.plotly_chart(fig, width='stretch')

        if selected_match != "All":
            st.subheader("Drill-down Insights")
            for insight in analytics.generate_match_insights(master_df, selected_match):
                st.info(insight)

        progression = match_scope.groupby(["match_id", "phase"], as_index=False)["runs"].sum()
        fig2 = px.line(progression, x="phase", y="runs", color="match_id", markers=True,
                        title="Run Progression by Phase")
        st.plotly_chart(fig2, width='stretch')

# ---------------------------------------------------------------- Batting Analysis
with sections[1]:
    st.header("Batting Analysis")
    batting = apply_filters(features["batting_by_phase"])
    if batting.empty:
        st.warning("No batting data for the current filter selection.")
    else:
        fig = px.bar(batting, x="player", y="strike_rate_by_phase", color="phase", barmode="group",
                     title="Strike Rate by Phase")
        st.plotly_chart(fig, width='stretch')

        top_sr = batting.loc[batting["strike_rate_by_phase"].idxmax()]
        st.success(
            f"Insight: {top_sr['player']} ({top_sr['team']}) posts the highest phase strike rate "
            f"at {top_sr['strike_rate_by_phase']} during the {top_sr['phase']} phase."
        )

        consistency = apply_filters(features["consistency_index"])
        fig2 = px.bar(consistency.sort_values("consistency_index", ascending=False).head(15),
                      x="player", y="consistency_index", color="team", title="Consistency Index (Top 15)")
        st.plotly_chart(fig2, width='stretch')

        st.subheader("Best Strike Rate by Phase (qualified, min. balls faced)")
        sr_by_phase = analytics.best_strike_rate_by_phase(filtered_master, min_balls=10)
        if sr_by_phase.empty:
            st.info("No players meet the minimum-balls-faced threshold for the current filters.")
        else:
            powerplay_leaders = sr_by_phase[sr_by_phase["phase"] == "powerplay"]
            if not powerplay_leaders.empty:
                pp_top = powerplay_leaders.sort_values("strike_rate", ascending=False).iloc[0]
                st.success(
                    f"Insight: {pp_top['player']} ({pp_top['team']}) has the best powerplay strike rate "
                    f"at {pp_top['strike_rate']} (qualified on {int(pp_top['balls'])} balls faced)."
                )
            st.dataframe(
                sr_by_phase[["season", "phase", "player", "team", "runs", "balls", "strike_rate"]],
                width='stretch',
            )

# ---------------------------------------------------------------- Bowling Analysis
with sections[2]:
    st.header("Bowling Analysis")
    bowling = apply_filters(features["bowling_by_phase"])
    if bowling.empty:
        st.warning("No bowling data for the current filter selection.")
    else:
        fig = px.box(bowling, x="phase", y="economy_by_phase", color="phase", title="Economy vs Phase")
        st.plotly_chart(fig, width='stretch')

        wicket_dist = bowling.groupby("player", as_index=False)["wicket_probability"].mean()
        fig2 = px.bar(wicket_dist.sort_values("wicket_probability", ascending=False).head(15),
                      x="player", y="wicket_probability", title="Wicket Probability (Top 15)")
        st.plotly_chart(fig2, width='stretch')

        cheapest = bowling.loc[bowling["economy_by_phase"].idxmin()] if bowling["economy_by_phase"].gt(0).any() else None
        if cheapest is not None:
            st.success(
                f"Insight: {cheapest['player']} is the most economical bowler in the {cheapest['phase']} "
                f"phase, conceding at {cheapest['economy_by_phase']} runs/over."
            )

        st.subheader("Highest Wicket-Taker Against Each Team")
        matchups = analytics.top_wicket_taker_per_opponent(filtered_master)
        if matchups.empty:
            st.info("No bowling matchup data for the current filters.")
        else:
            fig3 = px.bar(matchups, x="against_team", y="wickets", color="player",
                          title="Top Bowler vs Each Opponent", text="player")
            st.plotly_chart(fig3, width='stretch')
            st.dataframe(
                matchups[["season", "against_team", "player", "team", "wickets"]],
                width='stretch',
            )

# ---------------------------------------------------------------- Fielding Insights
with sections[3]:
    st.header("Fielding Insights (Key Differentiator)")
    fielding = apply_filters(features["fielding"])
    if fielding.empty:
        st.warning("No fielding data for the current filter selection.")
    else:
        leaderboard = analytics.catch_drop_leaderboard(fielding)
        fig = px.bar(leaderboard.head(15), x="player", y="catches_dropped", color="team",
                     title="Catch Drop Leaderboard")
        st.plotly_chart(fig, width='stretch')

        best_fielders = analytics.best_fielders_ranking(fielding)
        fig2 = px.bar(best_fielders.head(15), x="player", y="catch_efficiency", color="team",
                      title="Best Fielders Ranking (Catch Efficiency)")
        st.plotly_chart(fig2, width='stretch')

        total_lost = int(fielding["runs_lost_to_errors"].sum())
        st.warning(f"Insight: Estimated {total_lost} runs lost across selected teams/players due to dropped catches and missed stumpings.")

# ---------------------------------------------------------------- Player Insights
with sections[4]:
    st.header("Player Insights")
    impact = apply_filters(features["player_impact_score"])
    if impact.empty:
        st.warning("No player data for the current filter selection.")
    else:
        rankings = analytics.player_rankings(impact, config=config)
        fig = px.bar(rankings, x="player", y="player_impact_score", color="team",
                     title="Player Impact Score Ranking")
        st.plotly_chart(fig, width='stretch')

        change = analytics.player_performance_change(features["player_impact_score"], config)
        if "impact_score_change" in change.columns:
            st.subheader("Performance Consistency / Season-over-Season Change")
            st.dataframe(change[["player", "team", "impact_score_change"]].dropna().head(15),
                         width='stretch')

        st.subheader("Player Stats Explorer")
        st.caption(
            "Every player's full batting, bowling, and fielding line for the current filter selection. "
            "`matches_played_pct` is how often this player appears relative to their team's matches that "
            "season — useful for spotting whether a name is a regular starter or a bench player in the "
            "simulated data."
        )
        stats_table = analytics.player_stats_table(filtered_master)
        search = st.text_input("Search player name", value="", key="player_search")
        if search:
            stats_table = stats_table[stats_table["player"].str.contains(search, case=False, na=False)]
        st.dataframe(
            stats_table[[
                "season", "player", "team", "matches", "matches_played_pct", "runs", "balls_faced",
                "strike_rate", "wickets", "overs_bowled", "economy",
                "catches_taken", "catches_dropped", "stumping_missed",
            ]],
            width='stretch', height=400,
        )

        st.subheader("Player Profile")
        st.caption("Full one-stop summary for a single player: career line, best performances, and impact rank.")
        profile_players = sorted(filtered_master["player"].unique())
        if profile_players:
            profile_player = st.selectbox("Select a player", options=profile_players, key="player_profile_select")
            profile = analytics.player_profile(filtered_master, features["player_impact_score"], profile_player)
            if profile:
                st.write(f"**Teams:** {', '.join(profile['teams'])}  |  **Seasons:** {', '.join(map(str, profile['seasons']))}")
                st.dataframe(
                    profile["stats_by_season"][[
                        "season", "team", "matches", "matches_played_pct", "runs", "strike_rate",
                        "wickets", "economy", "catches_taken",
                    ]],
                    width='stretch', hide_index=True,
                )
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.caption("Best Batting Performance(s)")
                    if not profile["best_batting"].empty:
                        st.dataframe(
                            profile["best_batting"][["season", "best_score", "balls", "vs", "match"]],
                            width='stretch', hide_index=True,
                        )
                with pcol2:
                    st.caption("Best Bowling Performance(s)")
                    if not profile["best_bowling"].empty:
                        st.dataframe(
                            profile["best_bowling"][["season", "wickets", "runs_conceded", "vs", "match"]],
                            width='stretch', hide_index=True,
                        )
                if not profile["impact_ranks"].empty:
                    for _, rank_row in profile["impact_ranks"].iterrows():
                        st.info(
                            f"Season {rank_row['season']}: ranked #{rank_row['impact_rank']} of "
                            f"{rank_row['out_of']} by Player Impact Score."
                        )

        st.subheader("All-Time Leaders (across all selected seasons)")
        all_time = analytics.all_time_leaders(filtered_master)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Highest Run Scorer")
            st.dataframe(all_time["runs"], width='stretch', hide_index=True)
        with col2:
            st.caption("Highest Wicket Taker")
            st.dataframe(all_time["wickets"], width='stretch', hide_index=True)
        with col3:
            st.caption("Most Catches Taken")
            st.dataframe(all_time["catches"], width='stretch', hide_index=True)

        st.subheader("Player vs Player")
        st.caption(
            "Compares two players' stat lines in every match where both appeared. This is a "
            "match-level comparison, not a literal ball-by-ball duel — the dataset doesn't track "
            "which specific ball a batter faced from which specific bowler."
        )
        all_players = sorted(filtered_master["player"].unique())
        if len(all_players) >= 2:
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                player_a = st.selectbox("Player A", options=all_players, key="pvp_a")
            with pcol2:
                player_b = st.selectbox(
                    "Player B", options=[p for p in all_players if p != player_a], key="pvp_b"
                )
            matchup = analytics.player_vs_player_matchup(filtered_master, player_a, player_b)
            if matchup.empty:
                st.info(f"{player_a} and {player_b} haven't appeared in the same match in the current filters.")
            else:
                st.dataframe(matchup, width='stretch', hide_index=True)

# ---------------------------------------------------------------- Season Comparison
with sections[5]:
    st.header("Season Comparison")
    comparison = analytics.season_comparison(master_df, config)
    if len(comparison) < 2:
        st.warning("Need data from at least two seasons to compare — check both seasons are selected.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(comparison, x="season_name", y="total_runs", title="Total Runs by Season")
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig2 = px.bar(comparison, x="season_name", y="avg_strike_rate", title="Average Strike Rate by Season")
            st.plotly_chart(fig2, width='stretch')

        st.dataframe(comparison, width='stretch')

        s1, s2 = comparison.iloc[0], comparison.iloc[-1]
        run_delta = s2["total_runs"] - s1["total_runs"]
        direction = "increased" if run_delta > 0 else "decreased"
        st.success(
            f"Insight: Total runs scored {direction} by {abs(int(run_delta))} between "
            f"{s1['season_name']} and {s2['season_name']}."
        )

# ---------------------------------------------------------------- Fun Facts
with sections[6]:
    st.header("🎉 Fun Facts")
    st.caption(
        "Real, verified NPL facts sourced from Wikipedia's season articles "
        "(src/data_sources/wikipedia.py) — not estimated or simulated numbers."
    )
    for fact in analytics.fun_facts(config):
        st.info(fact)

    try:
        real_runs = load_table("real_leaders_runs", config)
        real_wickets = load_table("real_leaders_wickets", config)
        real_at = analytics.real_all_time_leaders(real_runs, real_wickets)
        st.subheader("Real All-Time Leaders (combined from each season's published top 5)")
        st.caption(
            "A player not in a given season's published top-5 isn't counted for that season — "
            "this is a lower bound, not an exhaustive career total."
        )
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.caption("Most Runs")
            st.dataframe(real_at["runs"], width='stretch', hide_index=True)
        with rcol2:
            st.caption("Most Wickets")
            st.dataframe(real_at["wickets"], width='stretch', hide_index=True)
    except Exception:
        pass

    try:
        final_scorecards = load_table("real_final_scorecards", config)
        if not final_scorecards.empty:
            st.subheader("Real Final-Match Scorecards")
            for season_id, group in final_scorecards.groupby("season"):
                season_name = next((s["name"] for s in config["seasons"] if s["id"] == season_id), season_id)
                with st.expander(f"{season_name} Final"):
                    st.dataframe(
                        group[["team", "player", "runs", "balls", "wickets", "overs", "economy"]],
                        width='stretch',
                    )
    except Exception:
        st.caption("Real final-match scorecards not available — run `python run_pipeline.py` with "
                   "data_sources.wikipedia.enabled: true in config/config.yaml.")

    try:
        standings = load_table("real_standings", config)
        if not standings.empty:
            st.subheader("Real Final Standings")
            standings_season = st.selectbox(
                "Season", options=sorted(standings["season"].unique()),
                format_func=lambda sid: next((s["name"] for s in config["seasons"] if s["id"] == sid), sid),
                key="standings_season",
            )
            st.dataframe(
                standings[standings["season"] == standings_season].drop(columns=["season"]),
                width='stretch', hide_index=True,
            )
    except Exception:
        pass

    try:
        head_to_head = load_table("real_head_to_head", config)
        if not head_to_head.empty:
            st.subheader("Real Head-to-Head Lookup")
            all_teams = sorted(set(head_to_head["team_a"]) | set(head_to_head["team_b"]))
            col1, col2 = st.columns(2)
            with col1:
                team_a_choice = st.selectbox("Team A", options=all_teams, key="h2h_team_a")
            with col2:
                team_b_choice = st.selectbox(
                    "Team B", options=[t for t in all_teams if t != team_a_choice], key="h2h_team_b"
                )
            record = analytics.head_to_head_record(head_to_head, team_a_choice, team_b_choice)
            if record.empty:
                st.caption(f"No recorded fixtures found between {team_a_choice} and {team_b_choice}.")
            else:
                wins_a = (record["winner"] == team_a_choice).sum()
                wins_b = (record["winner"] == team_b_choice).sum()
                st.info(f"Head-to-head: {team_a_choice} {wins_a} — {wins_b} {team_b_choice}")
                st.dataframe(record.drop(columns=["team_a", "team_b"]), width='stretch', hide_index=True)
    except Exception:
        pass
