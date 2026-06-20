from src.config import load_config
from src.data_sources.simulator import SimulatorSource


def test_simulator_output_has_required_schema_columns(config):
    source = SimulatorSource(config)
    df = source.fetch(season_id=1)
    required = {
        "match_id", "season", "team", "opponent", "player", "runs", "balls",
        "strike_rate", "wickets", "overs", "economy", "catches_taken",
        "catches_dropped", "stumping_missed", "fielding_errors",
        "match_result", "venue", "phase",
    }
    assert required.issubset(df.columns)


def test_simulator_is_deterministic_given_same_seed(config):
    df1 = SimulatorSource(config).fetch(season_id=1)
    df2 = SimulatorSource(config).fetch(season_id=1)
    assert df1.equals(df2)


def test_real_captain_plays_nearly_every_match_their_team_plays(config):
    """Regression guard: a real bug had every match's playing XI sampled
    independently from the full squad, so a real captain like Kushal Bhurtel
    (21-player Pokhara Avengers squad) could land in only 2 of 15 matches by
    chance — unrealistic once real names were wired in. Captains now anchor
    a fixed core XI reused (with minor rotation) across the season."""
    real_rosters = {"Test Team": [f"Player {i}" for i in range(20)]}
    real_captains = {"Test Team": "Player 0"}
    config = {
        **config,
        "teams": ["Test Team", "Other Team"],
        "_real_rosters": real_rosters,
        "_real_captains": real_captains,
        "simulator": {**config["simulator"], "matches_per_season": 10, "use_real_rosters": True},
    }
    df = SimulatorSource(config).fetch(season_id=1)
    team_matches = df[df["team"] == "Test Team"]["match_id"].nunique()
    captain_matches = df[(df["team"] == "Test Team") & (df["player"] == "Player 0")]["match_id"].nunique()
    assert captain_matches / team_matches >= 0.9


def test_simulator_uses_real_names_for_every_configured_team_when_rosters_cover_them(config):
    """Regression guard: if a real-roster dict's team-name spelling doesn't
    exactly match config['teams'] (e.g. "Kathmandu Gurkhas" vs config's
    "Kathmandu Gorkhas"), that team's roster lookup silently returns nothing
    and every player on it falls back to a placeholder name. When rosters
    are provided for every configured team (as they should be after
    canonicalization in wikipedia.py), zero placeholder names should appear."""
    real_rosters = {team: [f"{team.split()[0]}Star{i}" for i in range(15)] for team in config["teams"]}
    config = {**config, "_real_rosters": real_rosters}
    df = SimulatorSource(config).fetch(season_id=1)
    # The synthetic-fallback naming pattern is "<TeamFirstWord> Player <N>" exactly —
    # distinct from the real-roster names above, which contain no literal " Player ".
    placeholder_rows = df[df["player"].str.contains(r"^\w+ Player \d+$", regex=True)]
    assert placeholder_rows.empty


def test_simulator_winner_correlates_with_runs_and_wickets(config):
    """Regression guard: match outcomes must be tied to performance, not chosen
    independently of it — otherwise no win-probability model can learn from the data
    (this was a real bug: accuracy was 0.25 before the fix, 0.83 after)."""
    df = SimulatorSource(config).fetch(season_id=1)
    team_match = df.groupby(["match_id", "team"], as_index=False).agg(
        runs=("runs", "sum"), wickets=("wickets", "sum"), match_result=("match_result", "first")
    )
    team_match["strength"] = team_match["runs"] + team_match["wickets"] * 6
    correlation = team_match["strength"].corr((team_match["match_result"] == "Win").astype(int))
    assert correlation > 0.3
