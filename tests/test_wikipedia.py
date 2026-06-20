"""Integration tests against the live Wikipedia source. These hit a real
network endpoint (en.wikipedia.org) — skipped automatically if unreachable
rather than failing the suite in offline environments."""
from __future__ import annotations

import pytest
import requests

from src.data_sources.wikipedia import WikipediaSource, _canonicalize_team


def test_canonicalize_team_matches_known_spelling_variant():
    known = ["Kathmandu Gorkhas", "Lumbini Lions"]
    assert _canonicalize_team("Kathmandu Gurkhas", known) == "Kathmandu Gorkhas"


def test_canonicalize_team_leaves_exact_match_untouched():
    known = ["Kathmandu Gorkhas", "Lumbini Lions"]
    assert _canonicalize_team("Lumbini Lions", known) == "Lumbini Lions"


def test_canonicalize_team_leaves_unrelated_name_untouched():
    known = ["Kathmandu Gorkhas", "Lumbini Lions"]
    assert _canonicalize_team("Completely Different Team", known) == "Completely Different Team"


@pytest.fixture
def source(config):
    try:
        requests.get("https://en.wikipedia.org", timeout=5)
    except requests.RequestException:
        pytest.skip("en.wikipedia.org unreachable from this environment")
    return WikipediaSource({**config, **config["data_sources"]["wikipedia"]})


def test_fetch_squads_returns_real_teams_with_plausible_roster_sizes(source):
    squads = source.fetch_squads(season_id=1)
    assert len(squads) == 8
    for team, players in squads.items():
        assert 10 <= len(players) <= 30, f"{team} roster size {len(players)} looks implausible"
        assert len(set(players)) == len(players), f"{team} roster has duplicate names"


def test_final_scorecard_bowlers_are_tagged_with_their_own_team_not_opponent(source):
    """Regression guard: a real bug had bowling-table rows tagged with the
    *batting* team's name instead of the bowling team's, because team/opponent
    were inferred from table order instead of each table's own caption."""
    df = source.fetch(season_id=1)
    bowling_rows = df[df["overs"] > 0]
    assert not bowling_rows.empty

    # Kishore Mahato and Mohammad Mohsin are real Janakpur Bolts squad members
    # (per fetch_squads) who bowled in the Season 1 final — their bowling rows
    # must be tagged team="Janakpur Bolts", not the opponent's name.
    squads = source.fetch_squads(season_id=1)
    for _, row in bowling_rows.iterrows():
        player_team = next((team for team, players in squads.items() if row["player"] in players), None)
        if player_team is not None:
            assert row["team"] == player_team, (
                f"{row['player']} bowled for {player_team} but was tagged team={row['team']!r}"
            )


def test_final_scorecard_team_and_opponent_are_never_equal(source):
    df = source.fetch(season_id=1)
    assert (df["team"] != df["opponent"]).all()


def test_fetch_standings_has_one_row_per_real_team_with_valid_played_count(source):
    standings = source.fetch_standings(season_id=1)
    assert len(standings) == 8
    assert (standings["Pld"].astype(int) > 0).all()


def test_fetch_head_to_head_has_28_unique_round_robin_fixtures(source):
    """8 teams, single round-robin league stage = 8*7/2 = 28 fixtures."""
    h2h = source.fetch_head_to_head(season_id=1)
    assert len(h2h) == 28
    assert h2h["winner"].notna().all()
    assert (h2h["team_a"] != h2h["team_b"]).all()


def test_fetch_head_to_head_winner_is_always_one_of_the_two_teams(source):
    h2h = source.fetch_head_to_head(season_id=1)
    assert ((h2h["winner"] == h2h["team_a"]) | (h2h["winner"] == h2h["team_b"])).all()


def test_fetch_leaders_includes_joint_wicket_leaders_merged_via_rowspan(source):
    """Regression guard: a real bug dropped tied-rank rows entirely, because
    Wikipedia merges the rank cell across ties with rowspan and a naive
    cell-per-row parse left those rows one column short. Season 2's most-wickets
    table has three players tied at 17 — all three must be present."""
    leaders = source.fetch_leaders(season_id=2)
    wickets = leaders["most_wickets"]
    tied_at_17 = wickets[wickets["Wickets"] == "17"]
    assert len(tied_at_17) == 3
    assert set(tied_at_17["Player"]) == {"Sandeep Lamichhane", "Abinash Bohara", "Sher Malla"}


def test_fetch_squads_resolves_every_team_to_configs_canonical_spelling(source, config):
    """Regression guard: config.yaml spells one franchise "Kathmandu Gorkhas",
    but Season 1's Wikipedia article spells it "Kathmandu Gurkhas" — a real
    spelling mismatch across Wikipedia's own articles. Before canonicalization,
    that team's roster lookup silently returned nothing and every player on
    it fell back to a placeholder name ("Kathmandu Player N") instead of a
    real one. Every team fetch_squads() returns must exactly match config's
    spelling, regardless of how the source article spells it."""
    squads = source.fetch_squads(season_id=1)
    assert set(squads.keys()) == set(config["teams"])


def test_fetch_captains_resolves_every_team_to_configs_canonical_spelling(source, config):
    captains = source.fetch_captains(season_id=1)
    assert set(captains.keys()).issubset(set(config["teams"]))
    assert "Kathmandu Gorkhas" in captains
    assert captains["Kathmandu Gorkhas"] == "Karan KC"
