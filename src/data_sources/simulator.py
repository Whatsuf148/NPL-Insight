"""Realistic synthetic data generator.

Produces player-match-phase level records matching the master schema
defined in config (teams, venues, phases all sourced from config —
nothing hardcoded here). This lets the full pipeline + dashboard run
end-to-end today, and is swapped out for real scrapers later without
touching any downstream code, since the output schema is identical.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .base import DataSource


class SimulatorSource(DataSource):
    name = "simulator"

    def fetch(self, season_id: int) -> pd.DataFrame:
        sim_cfg = self.config.get("simulator", {})
        teams: list[str] = self.config["teams"]
        venues: list[str] = self.config["venues"]
        phases: list[str] = self.config["phases"]
        matches_per_season = sim_cfg.get("matches_per_season", 15)
        players_per_team = sim_cfg.get("players_per_team", 14)
        seed = sim_cfg.get("random_seed", 42)

        rng = np.random.default_rng(seed + season_id)
        rosters = {
            team: [f"{team.split()[0]} Player {i+1}" for i in range(players_per_team)]
            for team in teams
        }

        records = []
        match_id_counter = 1
        for home, away in itertools.combinations(teams, 2):
            for _ in range(max(1, matches_per_season // len(list(itertools.combinations(teams, 2))) + 1)):
                if match_id_counter > matches_per_season:
                    break
                match_id = f"S{season_id}M{match_id_counter:03d}"
                venue = rng.choice(venues)
                match_id_counter += 1

                team_rows = {home: [], away: []}
                team_score = {}
                for team, opponent in [(home, away), (away, home)]:
                    playing_xi = rng.choice(rosters[team], size=11, replace=False)
                    bowlers = rng.choice(playing_xi, size=6, replace=False)

                    total_runs = total_wickets = total_errors = 0
                    for player in playing_xi:
                        is_bowler = player in bowlers
                        for phase in phases:
                            balls_faced = int(rng.poisson(6 if phase == "middle" else 4))
                            runs = int(max(0, rng.normal(balls_faced * 1.3, 4)))
                            strike_rate = round((runs / balls_faced * 100), 2) if balls_faced else 0.0

                            overs_bowled = round(rng.uniform(0, 1.5), 1) if is_bowler else 0.0
                            runs_conceded = int(max(0, rng.normal(overs_bowled * 8, 3))) if is_bowler else 0
                            economy = round(runs_conceded / overs_bowled, 2) if overs_bowled else 0.0
                            wickets = int(rng.poisson(0.3)) if is_bowler else 0

                            catches_taken = int(rng.poisson(0.15))
                            catches_dropped = int(rng.binomial(1, 0.05))
                            stumping_missed = int(rng.binomial(1, 0.02))
                            fielding_errors = int(rng.binomial(1, 0.07))

                            total_runs += runs
                            total_wickets += wickets
                            total_errors += catches_dropped + stumping_missed + fielding_errors

                            team_rows[team].append({
                                "match_id": match_id,
                                "season": season_id,
                                "team": team,
                                "opponent": opponent,
                                "player": player,
                                "runs": runs,
                                "balls": balls_faced,
                                "strike_rate": strike_rate,
                                "wickets": wickets,
                                "overs": overs_bowled,
                                "economy": economy,
                                "catches_taken": catches_taken,
                                "catches_dropped": catches_dropped,
                                "stumping_missed": stumping_missed,
                                "fielding_errors": fielding_errors,
                                "venue": venue,
                                "phase": phase,
                            })

                    # Match strength score: runs scored, wickets taken with the ball, and
                    # fielding errors conceded all genuinely drive the outcome — this keeps
                    # match_result causally linked to performance instead of independent of it,
                    # so downstream win-probability modeling has real signal to learn from.
                    upset_noise = rng.normal(0, 15)
                    team_score[team] = total_runs + total_wickets * 6 - total_errors * 4 + upset_noise

                winner = home if team_score[home] >= team_score[away] else away
                for team in (home, away):
                    match_result = "Win" if team == winner else "Loss"
                    for row in team_rows[team]:
                        row["match_result"] = match_result
                        records.append(row)

                if match_id_counter > matches_per_season:
                    break

        return pd.DataFrame.from_records(records)
