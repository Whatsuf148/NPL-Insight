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
