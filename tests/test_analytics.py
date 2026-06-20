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
