from src import analytics
from src.data_cleaning import clean
from src.feature_engineering import build_feature_set


def test_player_rankings_respects_top_n_from_config(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    top_n = config["analytics"]["player_ranking_top_n"]
    rankings = analytics.player_rankings(tables["player_impact_score"], config=config)
    assert len(rankings) <= top_n


def test_player_rankings_sorted_descending(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    rankings = analytics.player_rankings(tables["player_impact_score"], config=config)
    scores = rankings["player_impact_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_season_comparison_has_one_row_per_configured_season(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    comparison = analytics.season_comparison(cleaned, config)
    assert set(comparison["season"]) == {s["id"] for s in config["seasons"]}


def test_generate_match_insights_handles_unknown_match_gracefully(sample_master_df):
    insights = analytics.generate_match_insights(sample_master_df, "does_not_exist")
    assert len(insights) == 1
    assert "No data found" in insights[0]


def test_generate_match_insights_identifies_top_scorer(sample_master_df):
    match_id = sample_master_df["match_id"].iloc[0]
    insights = analytics.generate_match_insights(sample_master_df, match_id)
    assert any("Top scorer" in i for i in insights)


def test_best_fielders_ranking_requires_minimum_chances(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    ranking = analytics.best_fielders_ranking(tables["fielding"])
    assert (ranking["catches_taken"] + ranking["catches_dropped"] >= 2).all()


def test_player_stats_table_has_one_row_per_player_team_season(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    table = analytics.player_stats_table(cleaned)
    dupes = table.duplicated(subset=["season", "player", "team"]).sum()
    assert dupes == 0


def test_player_stats_table_strike_rate_matches_runs_over_balls(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    table = analytics.player_stats_table(cleaned)
    nonzero = table[table["balls_faced"] > 0]
    expected = (nonzero["runs"] / nonzero["balls_faced"] * 100).round(2)
    assert (nonzero["strike_rate"] == expected).all()


def test_player_stats_table_batting_average_is_runs_per_dismissal():
    import pandas as pd

    df = pd.DataFrame([
        {"season": 1, "player": "P", "team": "A", "match_id": "M1", "runs": 50, "balls": 30,
         "dismissals": 1, "fours": 4, "sixes": 1, "wickets": 0, "overs": 0, "economy": 0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
        {"season": 1, "player": "P", "team": "A", "match_id": "M2", "runs": 30, "balls": 20,
         "dismissals": 1, "fours": 2, "sixes": 0, "wickets": 0, "overs": 0, "economy": 0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
    ])
    table = analytics.player_stats_table(df)
    row = table.iloc[0]
    assert row["batting_average"] == 40.0  # (50+30) runs / 2 dismissals


def test_player_stats_table_batting_average_uses_raw_runs_when_never_dismissed():
    import pandas as pd

    df = pd.DataFrame([
        {"season": 1, "player": "P", "team": "A", "match_id": "M1", "runs": 50, "balls": 30,
         "dismissals": 0, "fours": 4, "sixes": 1, "wickets": 0, "overs": 0, "economy": 0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
    ])
    table = analytics.player_stats_table(df)
    assert table.iloc[0]["batting_average"] == 50.0
    assert table.iloc[0]["not_outs"] == 1


def test_player_stats_table_bowling_average_is_runs_conceded_per_wicket():
    import pandas as pd

    df = pd.DataFrame([
        {"season": 1, "player": "P", "team": "A", "match_id": "M1", "runs": 0, "balls": 0,
         "dismissals": 0, "fours": 0, "sixes": 0, "wickets": 2, "overs": 4.0, "economy": 6.0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
    ])
    table = analytics.player_stats_table(df)
    # economy 6.0 over 4 overs = 24 runs conceded; 24 / 2 wickets = 12.0
    assert table.iloc[0]["bowling_average"] == 12.0


def test_player_stats_table_matches_played_pct_is_100_when_player_in_every_team_match(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    table = analytics.player_stats_table(cleaned)
    # fixture has every player appearing in every one of their team's matches
    assert (table["matches_played_pct"] == 100.0).all()


def test_best_individual_performances_picks_highest_score_per_player(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    best = analytics.best_individual_performances(cleaned)
    per_match = cleaned.groupby(["season", "player", "match_id"], as_index=False)["runs"].sum()
    for _, row in best["best_batting"].iterrows():
        actual_max = per_match[
            (per_match["season"] == row["season"]) & (per_match["player"] == row["player"])
        ]["runs"].max()
        assert row["best_score"] == actual_max


def test_best_individual_performances_bowling_ties_broken_by_fewest_runs():
    import pandas as pd

    df = pd.DataFrame([
        {"season": 1, "player": "P", "team": "A", "opponent": "B", "match_id": "M1",
         "overs": 4.0, "economy": 5.0, "wickets": 3, "runs": 0, "balls": 0,
         "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0, "fielding_errors": 0,
         "match_result": "Win", "venue": "X", "phase": "death"},
        {"season": 1, "player": "P", "team": "A", "opponent": "C", "match_id": "M2",
         "overs": 4.0, "economy": 3.0, "wickets": 3, "runs": 0, "balls": 0,
         "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0, "fielding_errors": 0,
         "match_result": "Win", "venue": "X", "phase": "death"},
    ])
    best = analytics.best_individual_performances(df)
    row = best["best_bowling"].iloc[0]
    assert row["match"] == "M2"  # same wickets, fewer runs conceded


def test_player_profile_returns_empty_dict_for_unknown_player(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    profile = analytics.player_profile(cleaned, tables["player_impact_score"], "Nonexistent Player")
    assert profile == {}


def test_player_profile_impact_rank_is_consistent_with_score_ordering(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    tables = build_feature_set(cleaned, config)
    impact = tables["player_impact_score"]
    player = impact.iloc[0]["player"]
    profile = analytics.player_profile(cleaned, impact, player)
    for _, rank_row in profile["impact_ranks"].iterrows():
        season_scores = impact[impact["season"] == rank_row["season"]].sort_values(
            "player_impact_score", ascending=False
        ).reset_index(drop=True)
        expected_rank = season_scores[season_scores["player"] == player].index[0] + 1
        assert rank_row["impact_rank"] == expected_rank


def test_season_leaderboards_qualify_batting_average_on_minimum_balls():
    import pandas as pd

    df = pd.DataFrame([
        {"season": 1, "player": "TinySample", "team": "A", "match_id": "M1", "runs": 20, "balls": 5,
         "dismissals": 1, "fours": 0, "sixes": 0, "wickets": 0, "overs": 0, "economy": 0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
        {"season": 1, "player": "BigSample", "team": "A", "match_id": "M2", "runs": 50, "balls": 40,
         "dismissals": 1, "fours": 0, "sixes": 0, "wickets": 0, "overs": 0, "economy": 0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
    ])
    result = analytics.season_leaderboards(df, min_balls_for_average=30)
    assert "TinySample" not in result["best_batting_average"]["player"].values
    assert "BigSample" in result["best_batting_average"]["player"].values


def test_season_leaderboards_qualify_bowling_average_on_minimum_overs():
    import pandas as pd

    df = pd.DataFrame([
        {"season": 1, "player": "TinyBowler", "team": "A", "match_id": "M1", "runs": 0, "balls": 0,
         "dismissals": 0, "fours": 0, "sixes": 0, "wickets": 1, "overs": 1.0, "economy": 5.0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
        {"season": 1, "player": "BigBowler", "team": "A", "match_id": "M2", "runs": 0, "balls": 0,
         "dismissals": 0, "fours": 0, "sixes": 0, "wickets": 2, "overs": 8.0, "economy": 6.0,
         "maidens": 0, "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0},
    ])
    result = analytics.season_leaderboards(df, min_overs_for_bowling_average=4.0)
    assert "TinyBowler" not in result["best_bowling_average"]["player"].values
    assert "BigBowler" in result["best_bowling_average"]["player"].values


def test_top_wicket_taker_per_opponent_has_one_row_per_season_and_opponent(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = analytics.top_wicket_taker_per_opponent(cleaned)
    dupes = result.duplicated(subset=["season", "against_team"]).sum()
    assert dupes == 0


def test_best_strike_rate_by_phase_respects_min_balls_threshold(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = analytics.best_strike_rate_by_phase(cleaned, min_balls=10)
    assert (result["balls"] >= 10).all()


def test_best_strike_rate_by_phase_only_includes_configured_phases(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = analytics.best_strike_rate_by_phase(cleaned, min_balls=10)
    assert set(result["phase"]).issubset(set(config["phases"]))


def test_all_time_leaders_sums_runs_across_all_seasons_for_same_player(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    leaders = analytics.all_time_leaders(cleaned)
    # fixture has "Team A Bat1" scoring 20 runs per phase x 3 phases x 2 matches x 2 seasons
    expected_total = 20 * 3 * 2 * 2
    row = leaders["runs"][leaders["runs"]["player"] == "Team A Bat1"]
    assert row.iloc[0]["runs"] == expected_total


def test_all_time_leaders_sorted_descending(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    leaders = analytics.all_time_leaders(cleaned)
    runs = leaders["runs"]["runs"].tolist()
    assert runs == sorted(runs, reverse=True)


def test_real_all_time_leaders_combines_seasons_for_repeat_players():
    import pandas as pd

    leaders_runs = pd.DataFrame([
        {"Player": "X", "Team": "A", "Runs": "100", "season": 1},
        {"Player": "X", "Team": "A", "Runs": "150", "season": 2},
        {"Player": "Y", "Team": "B", "Runs": "300", "season": 1},
    ])
    leaders_wickets = pd.DataFrame([
        {"Player": "Z", "Team": "C", "Wickets": "10", "season": 1},
    ])
    result = analytics.real_all_time_leaders(leaders_runs, leaders_wickets)
    x_row = result["runs"][result["runs"]["Player"] == "X"].iloc[0]
    assert x_row["total_runs"] == 250
    assert x_row["seasons_in_top5"] == 2


def test_player_vs_player_matchup_only_includes_shared_matches(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    players = cleaned["player"].unique()
    result = analytics.player_vs_player_matchup(cleaned, players[0], players[0])
    # a player "vs themself" trivially shares every match they're in
    assert set(result["match_id"]) == set(cleaned.loc[cleaned["player"] == players[0], "match_id"])


def test_player_vs_player_matchup_empty_when_no_shared_matches(sample_master_df, config):
    cleaned = clean(sample_master_df, config)
    result = analytics.player_vs_player_matchup(cleaned, "Nonexistent Player A", "Nonexistent Player B")
    assert result.empty


def test_head_to_head_record_finds_fixtures_regardless_of_home_away_order():
    import pandas as pd

    h2h = pd.DataFrame([
        {"season": 1, "team_a": "Team X", "team_b": "Team Y", "winner": "Team X", "margin": "10 runs"},
        {"season": 1, "team_a": "Team Y", "team_b": "Team X", "winner": "Team Y", "margin": "3 wickets"},
        {"season": 1, "team_a": "Team Z", "team_b": "Team Y", "winner": "Team Z", "margin": "1 run"},
    ])
    result = analytics.head_to_head_record(h2h, "Team X", "Team Y")
    assert len(result) == 2


def test_fun_facts_returns_fallback_message_when_no_real_tables_exist(monkeypatch, config):
    import src.storage as storage

    def _raise(*args, **kwargs):
        raise FileNotFoundError("no such table")

    monkeypatch.setattr(storage, "load_table", _raise)
    facts = analytics.fun_facts(config)
    assert len(facts) == 1
    assert "enable data_sources.wikipedia" in facts[0]


def test_batting_order_win_rate_overall_percentages_sum_correctly():
    import pandas as pd

    match_results = pd.DataFrame([
        {"team1": "A", "team2": "B", "winner": "A"},
        {"team1": "A", "team2": "B", "winner": "B"},
        {"team1": "B", "team2": "A", "winner": "B"},
        {"team1": "B", "team2": "A", "winner": "B"},
    ])
    result = analytics.batting_order_win_rate(match_results)
    overall = result["overall"]
    assert overall["total_matches"] == 4
    assert overall["batted_first_wins"] + overall["batted_second_wins"] == 4
    assert overall["batted_first_win_pct"] + overall["batted_second_win_pct"] == 100.0


def test_batting_order_win_rate_per_team_breakdown():
    import pandas as pd

    match_results = pd.DataFrame([
        {"team1": "A", "team2": "B", "winner": "A"},  # A bats first, wins
        {"team1": "B", "team2": "A", "winner": "A"},  # A bats second, wins
    ])
    result = analytics.batting_order_win_rate(match_results)
    by_team = result["by_team"].set_index("team")
    assert by_team.loc["A", "win_pct_batting_first"] == 100.0
    assert by_team.loc["A", "win_pct_batting_second"] == 100.0


def test_batting_order_win_rate_handles_empty_input():
    import pandas as pd

    result = analytics.batting_order_win_rate(pd.DataFrame(columns=["team1", "team2", "winner"]))
    assert result["overall"] == {}
    assert result["by_team"].empty


def test_toss_win_probability_overall_pct_matches_raw_counts():
    import pandas as pd

    toss_results = pd.DataFrame([
        {"toss_winner": "A", "match_winner": "A", "toss_decision": "bat", "toss_winner_won_match": True},
        {"toss_winner": "A", "match_winner": "B", "toss_decision": "bat", "toss_winner_won_match": False},
        {"toss_winner": "B", "match_winner": "B", "toss_decision": "field", "toss_winner_won_match": True},
        {"toss_winner": "B", "match_winner": "B", "toss_decision": "field", "toss_winner_won_match": True},
    ])
    result = analytics.toss_win_probability(toss_results)
    assert result["overall"]["total_matches"] == 4
    assert result["overall"]["toss_winner_won_match"] == 3
    assert result["overall"]["toss_winner_win_pct"] == 75.0


def test_toss_win_probability_breaks_down_by_decision():
    import pandas as pd

    toss_results = pd.DataFrame([
        {"toss_winner": "A", "match_winner": "A", "toss_decision": "bat", "toss_winner_won_match": True},
        {"toss_winner": "A", "match_winner": "B", "toss_decision": "bat", "toss_winner_won_match": False},
        {"toss_winner": "B", "match_winner": "B", "toss_decision": "field", "toss_winner_won_match": True},
    ])
    result = analytics.toss_win_probability(toss_results)
    by_decision = result["by_decision"].set_index("decision")
    assert by_decision.loc["bat", "matches"] == 2
    assert by_decision.loc["bat", "toss_winner_win_pct"] == 50.0
    assert by_decision.loc["field", "matches"] == 1


def test_toss_win_probability_handles_empty_input():
    import pandas as pd

    result = analytics.toss_win_probability(
        pd.DataFrame(columns=["toss_winner", "match_winner", "toss_decision", "toss_winner_won_match"])
    )
    assert result["overall"] == {}
    assert result["by_decision"].empty
