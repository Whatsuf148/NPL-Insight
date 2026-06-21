import pandas as pd

from src.data_collection import _build_season_rosters, _match_result_evidence


def test_match_result_evidence_extracts_players_per_team():
    match_results = pd.DataFrame([
        {"Team A_top_batter": "Alice", "Team A_top_bowler": "Bob", "Team B_top_batter": "Carl"},
    ])
    evidence = _match_result_evidence(match_results, ["Team A", "Team B"])
    assert evidence["Team A"] == {"Alice", "Bob"}
    assert evidence["Team B"] == {"Carl"}


def test_build_season_rosters_moves_a_transferred_player_to_their_real_season_team():
    """The core regression case: a player in Season 1's squad table for
    Team A who has direct Season 2 evidence of playing for Team B must end
    up in Team B's Season 2 roster, and NOT in Team A's Season 2 roster."""
    squads_by_season = {
        1: {"Team A": ["Transfer Player", "Stayer A"], "Team B": ["Stayer B"]},
    }
    evidence_by_season = {
        2: {"Team A": set(), "Team B": {"Transfer Player"}},
    }
    rosters = _build_season_rosters(squads_by_season, evidence_by_season, seasons=[1, 2])

    assert "Transfer Player" in rosters[1]["Team A"]
    assert "Transfer Player" in rosters[2]["Team B"]
    assert "Transfer Player" not in rosters[2]["Team A"]


def test_build_season_rosters_falls_back_to_any_available_squad_for_seasons_without_one():
    squads_by_season = {1: {"Team A": ["Player X"]}, 2: {}}
    rosters = _build_season_rosters(squads_by_season, {}, seasons=[1, 2])
    assert rosters[2]["Team A"] == ["Player X"]


def test_build_season_rosters_keeps_untouched_players_in_their_squad_team():
    squads_by_season = {1: {"Team A": ["Stayer"]}}
    rosters = _build_season_rosters(squads_by_season, {}, seasons=[1])
    assert rosters[1]["Team A"] == ["Stayer"]
