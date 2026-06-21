"""Integration tests against the real Cricsheet ball-by-ball source. These
hit a real network endpoint (cricsheet.org) — skipped automatically if
unreachable rather than failing the suite in offline environments."""
from __future__ import annotations

import pytest
import requests

from src.data_sources.cricsheet import CricsheetSource


@pytest.fixture
def source(config):
    try:
        requests.get("https://cricsheet.org", timeout=5)
    except requests.RequestException:
        pytest.skip("cricsheet.org unreachable from this environment")
    return CricsheetSource({**config, **config["data_sources"]["cricsheet"]})


def test_fetch_returns_required_schema_columns(source):
    df = source.fetch(season_id=1)
    required = {
        "match_id", "season", "team", "opponent", "player", "runs", "balls",
        "strike_rate", "wickets", "overs", "economy", "catches_taken",
        "catches_dropped", "stumping_missed", "fielding_errors",
        "match_result", "venue", "phase",
    }
    assert required.issubset(df.columns)


def test_fetch_covers_all_32_matches_per_season(source):
    df = source.fetch(season_id=1)
    assert df["match_id"].nunique() == 32
    df2 = source.fetch(season_id=2)
    assert df2["match_id"].nunique() == 32


def test_fetch_has_no_unknown_match_results(source):
    """Regression guard: tied matches decided by Super Over record the
    winner under outcome['eliminator'], not outcome['winner'] — missing
    that fallback left both teams as match_result='Unknown' for an
    otherwise fully-resolved match."""
    df = source.fetch(season_id=1)
    assert (df["match_result"] != "Unknown").all()


def test_fetch_matches_real_sandeep_lamichhane_season2_wickets(source):
    """Regression guard for the bug the user reported: a generic simulator
    gave Sandeep Lamichhane far fewer wickets than his real Season 2 total
    (17, per Wikipedia's published leaderboard). Real ball-by-ball data
    must reproduce that exact figure, since it's literally counting real
    dismissals, not estimating them."""
    df = source.fetch(season_id=2)
    wickets = df[df["player"] == "S Lamichhane"]["wickets"].sum()
    assert wickets == 17


def test_fielding_columns_not_tracked_in_real_data_are_always_zero(source):
    """Drops/missed stumpings/misfields aren't published anywhere real for
    NPL — these must stay an honest 0, never an estimate."""
    df = source.fetch(season_id=1)
    assert (df["catches_dropped"] == 0).all()
    assert (df["stumping_missed"] == 0).all()
    assert (df["fielding_errors"] == 0).all()


def test_catches_taken_are_nonzero_somewhere(source):
    """Real catches *are* derivable from wicket records (kind == 'caught'),
    so unlike drops, this should not be uniformly zero."""
    df = source.fetch(season_id=1)
    assert df["catches_taken"].sum() > 0


def test_fetch_toss_results_covers_every_match(source):
    toss = source.fetch_toss_results(season_id=1)
    assert len(toss) == 32
    assert toss["toss_winner"].notna().all()
    assert toss["toss_decision"].notna().all()
