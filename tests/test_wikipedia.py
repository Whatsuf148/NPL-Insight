"""Integration tests against the live Wikipedia source. These hit a real
network endpoint (en.wikipedia.org) — skipped automatically if unreachable
rather than failing the suite in offline environments."""
from __future__ import annotations

import pytest
import requests

from src.data_sources.wikipedia import WikipediaSource


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
