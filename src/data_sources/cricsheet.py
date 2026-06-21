"""Real ball-by-ball data source: Cricsheet (https://cricsheet.org).

Cricsheet publishes genuine ball-by-ball JSON for NPL — 64 matches across
both seasons (32 each), including real toss winner/decision *per match*
(not just the two finals, unlike the Wikipedia source), real wickets with
dismissal type and fielders, real venues, and real outcomes. This is the
primary, most-accurate source in this project: every batting/bowling/
fielding number in the master dataset comes directly from a real delivery
when this source is enabled, not a statistical simulation.

What real ball-by-ball data does *not* include: dropped catches, missed
stumpings, or misfields — no public source records fielding *mistakes* for
NPL, only completed dismissals. `catches_dropped`, `stumping_missed`, and
`fielding_errors` are therefore always 0 from this source — a real "we
don't have this data" 0, not an estimate. Fielding Insights will show fewer
interesting numbers as a result; that's the honest tradeoff for everything
else being verifiably real.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .base import DataSource
from .wikipedia import _canonicalize_team

_DOWNLOAD_URL = "https://cricsheet.org/downloads/npl_json.zip"
_HEADERS = {"User-Agent": "Mozilla/5.0 (NPL-Insight research bot; non-commercial analytics)"}

# Dismissal kinds not credited to the bowler (run outs/retirements are not a
# bowler's wicket in cricket's own scoring rules).
_NON_BOWLER_WICKET_KINDS = {"run out", "retired hurt", "retired out", "obstructing the field"}


class CricsheetSource(DataSource):
    name = "cricsheet"

    def _cache_dir(self) -> Path:
        external_dir = Path(self.config["paths"]["external_dir"])
        cache_dir = external_dir / "cricsheet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _ensure_downloaded(self) -> Path:
        cache_dir = self._cache_dir()
        extracted_dir = cache_dir / "extracted"
        if extracted_dir.exists() and list(extracted_dir.glob("*.json")):
            return extracted_dir

        response = requests.get(_DOWNLOAD_URL, headers=_HEADERS, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(extracted_dir)
        return extracted_dir

    def _season_year(self, season_id: int) -> int:
        for season in self.config["seasons"]:
            if season["id"] == season_id:
                return season["year"]
        raise ValueError(f"No year configured for season {season_id}")

    def fetch(self, season_id: int) -> pd.DataFrame:
        known_teams = self.config["teams"]
        powerplay_overs = self.config["powerplay_overs"]
        death_overs_start = self.config["death_overs_start"]
        target_year = self._season_year(season_id)

        extracted_dir = self._ensure_downloaded()
        records = []

        for match_file in sorted(extracted_dir.glob("*.json")):
            with open(match_file, encoding="utf-8") as f:
                match = json.load(f)
            info = match["info"]
            match_year = int(info["dates"][0][:4])
            if match_year != target_year:
                continue

            teams = [_canonicalize_team(t, known_teams) for t in info["teams"]]
            if len(teams) != 2:
                continue
            venue = info.get("venue", "")
            outcome = info.get("outcome", {})
            # Tied matches decided by Super Over record the winner under
            # "eliminator", not "winner" — without this, the match falls
            # back to "Unknown" for both teams even though there's a real,
            # known result.
            winner_raw = outcome.get("winner") or outcome.get("eliminator")
            winner = _canonicalize_team(winner_raw, known_teams) if winner_raw else None
            match_id = f"S{season_id}_{match_file.stem}"

            team_canon = {raw: _canonicalize_team(raw, known_teams) for raw in info["teams"]}

            for innings in match["innings"]:
                batting_team = team_canon.get(innings["team"], innings["team"])
                bowling_team = next((t for t in teams if t != batting_team), batting_team)

                for over in innings["overs"]:
                    over_num = over["over"]
                    if over_num < powerplay_overs:
                        phase = "powerplay"
                    elif over_num >= death_overs_start:
                        phase = "death"
                    else:
                        phase = "middle"

                    over_runs_off_bowler = 0
                    over_is_all_legal = True
                    over_bowler = None

                    for delivery in over["deliveries"]:
                        extras = delivery.get("extras", {})
                        runs = delivery["runs"]
                        is_wide = "wides" in extras
                        bowler_chargeable_extras = extras.get("wides", 0) + extras.get("noballs", 0)
                        is_legal_delivery = "wides" not in extras and "noballs" not in extras

                        batter = delivery["batter"]
                        bowler = delivery["bowler"]
                        over_bowler = bowler
                        runs_off_bat = runs["batter"]

                        records.append({
                            "_kind": "bat", "match_id": match_id, "team": batting_team,
                            "opponent": bowling_team, "player": batter, "phase": phase,
                            "runs": runs_off_bat, "faced": 0 if is_wide else 1,
                            "fours": 1 if runs_off_bat == 4 else 0,
                            "sixes": 1 if runs_off_bat == 6 else 0,
                        })
                        records.append({
                            "_kind": "bowl", "match_id": match_id, "team": bowling_team,
                            "opponent": batting_team, "player": bowler, "phase": phase,
                            "runs_conceded": runs_off_bat + bowler_chargeable_extras,
                            "legal_ball": 1 if is_legal_delivery else 0,
                        })
                        over_runs_off_bowler += runs["total"]
                        over_is_all_legal = over_is_all_legal and is_legal_delivery

                        for wicket in delivery.get("wickets", []):
                            kind = wicket["kind"]
                            dismissed_player = wicket.get("player_out", batter)
                            if kind not in _NON_BOWLER_WICKET_KINDS:
                                records.append({
                                    "_kind": "wicket", "match_id": match_id, "team": bowling_team,
                                    "opponent": batting_team, "player": bowler, "phase": phase,
                                })
                            # Credited to whoever was actually dismissed, not necessarily
                            # the batter on strike for this ball (a run-out can dismiss
                            # the non-striker).
                            records.append({
                                "_kind": "dismissal", "match_id": match_id, "team": batting_team,
                                "opponent": bowling_team, "player": dismissed_player, "phase": phase,
                            })
                            if kind == "caught and bowled":
                                records.append({
                                    "_kind": "catch", "match_id": match_id, "team": bowling_team,
                                    "opponent": batting_team, "player": bowler, "phase": phase,
                                })
                            elif kind == "caught":
                                for fielder in wicket.get("fielders", []):
                                    if "name" in fielder:
                                        records.append({
                                            "_kind": "catch", "match_id": match_id, "team": bowling_team,
                                            "opponent": batting_team, "player": fielder["name"], "phase": phase,
                                        })

                    if over_is_all_legal and over_runs_off_bowler == 0 and over_bowler is not None:
                        records.append({
                            "_kind": "maiden", "match_id": match_id, "team": bowling_team,
                            "opponent": batting_team, "player": over_bowler, "phase": phase,
                        })

            for team in teams:
                records.append({
                    "_kind": "match_meta", "match_id": match_id, "team": team,
                    "opponent": next(t for t in teams if t != team),
                    "match_result": "Win" if team == winner else ("Loss" if winner else "Unknown"),
                    "venue": venue,
                })

        if not records:
            return pd.DataFrame()

        raw = pd.DataFrame.from_records(records)
        return self._aggregate(raw, season_id)

    def fetch_toss_results(self, season_id: int) -> pd.DataFrame:
        """Real toss winner + decision for every match (not just the two
        finals, unlike the Wikipedia source) — enables a real toss-outcome
        win-rate analysis rather than the batting-order proxy."""
        known_teams = self.config["teams"]
        target_year = self._season_year(season_id)
        extracted_dir = self._ensure_downloaded()

        rows = []
        for match_file in sorted(extracted_dir.glob("*.json")):
            with open(match_file, encoding="utf-8") as f:
                match = json.load(f)
            info = match["info"]
            if int(info["dates"][0][:4]) != target_year:
                continue
            toss = info.get("toss", {})
            teams = [_canonicalize_team(t, known_teams) for t in info["teams"]]
            toss_winner = _canonicalize_team(toss.get("winner"), known_teams) if toss.get("winner") else None
            outcome = info.get("outcome", {})
            match_winner_raw = outcome.get("winner") or outcome.get("eliminator")
            match_winner = _canonicalize_team(match_winner_raw, known_teams) if match_winner_raw else None
            rows.append({
                "season": season_id,
                "match_id": f"S{season_id}_{match_file.stem}",
                "team1": teams[0] if teams else None,
                "team2": teams[1] if len(teams) > 1 else None,
                "toss_winner": toss_winner,
                "toss_decision": toss.get("decision"),
                "match_winner": match_winner,
                "toss_winner_won_match": (toss_winner == match_winner) if toss_winner and match_winner else None,
            })
        return pd.DataFrame.from_records(rows)

    @staticmethod
    def _aggregate(raw: pd.DataFrame, season_id: int) -> pd.DataFrame:
        group_cols = ["match_id", "team", "opponent", "player", "phase"]

        bat = raw[raw["_kind"] == "bat"].groupby(group_cols, as_index=False).agg(
            runs=("runs", "sum"), balls=("faced", "sum"),
            fours=("fours", "sum"), sixes=("sixes", "sum"),
        )
        bowl = raw[raw["_kind"] == "bowl"].groupby(group_cols, as_index=False).agg(
            runs_conceded=("runs_conceded", "sum"), legal_balls=("legal_ball", "sum")
        )
        wickets = raw[raw["_kind"] == "wicket"].groupby(group_cols, as_index=False).size().rename(
            columns={"size": "wickets"}
        )
        catches = raw[raw["_kind"] == "catch"].groupby(group_cols, as_index=False).size().rename(
            columns={"size": "catches_taken"}
        )
        dismissals = raw[raw["_kind"] == "dismissal"].groupby(group_cols, as_index=False).size().rename(
            columns={"size": "dismissals"}
        )
        maidens = raw[raw["_kind"] == "maiden"].groupby(group_cols, as_index=False).size().rename(
            columns={"size": "maidens"}
        )

        merged = bat.merge(bowl, on=group_cols, how="outer") \
            .merge(wickets, on=group_cols, how="outer") \
            .merge(catches, on=group_cols, how="outer") \
            .merge(dismissals, on=group_cols, how="outer") \
            .merge(maidens, on=group_cols, how="outer")
        for col in ("runs", "balls", "fours", "sixes", "runs_conceded", "legal_balls",
                    "wickets", "catches_taken", "dismissals", "maidens"):
            merged[col] = merged[col].fillna(0)

        merged["strike_rate"] = (merged["runs"] / merged["balls"].replace(0, pd.NA) * 100).fillna(0.0).round(2)
        merged["overs"] = (merged["legal_balls"] / 6).round(4)
        merged["economy"] = (merged["runs_conceded"] / merged["overs"].replace(0, pd.NA)).fillna(0.0).round(2)
        merged["catches_dropped"] = 0
        merged["stumping_missed"] = 0
        merged["fielding_errors"] = 0
        merged["season"] = season_id

        meta = raw[raw["_kind"] == "match_meta"][["match_id", "team", "opponent", "match_result", "venue"]].drop_duplicates()
        merged = merged.merge(meta, on=["match_id", "team", "opponent"], how="left")

        column_order = [
            "match_id", "season", "team", "opponent", "player", "runs", "balls", "strike_rate",
            "fours", "sixes", "dismissals", "wickets", "overs", "economy", "maidens",
            "catches_taken", "catches_dropped", "stumping_missed",
            "fielding_errors", "match_result", "venue", "phase",
        ]
        return merged[column_order]
