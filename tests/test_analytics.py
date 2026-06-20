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


def test_fun_facts_returns_fallback_message_when_no_real_tables_exist(monkeypatch, config):
    import src.storage as storage

    def _raise(*args, **kwargs):
        raise FileNotFoundError("no such table")

    monkeypatch.setattr(storage, "load_table", _raise)
    facts = analytics.fun_facts(config)
    assert len(facts) == 1
    assert "enable data_sources.wikipedia" in facts[0]
