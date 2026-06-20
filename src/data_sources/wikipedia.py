"""Real-data source: scrapes verified NPL facts from Wikipedia's season articles.

This is genuinely live, tested HTTP + parsing (requests + BeautifulSoup) against
en.wikipedia.org, not a stub. It is the source of truth for:
  - real team rosters (used to give simulated player-match rows real player names)
  - real season leaderboards (most runs / most wickets)
  - real award winners
  - the one full real match scorecard per season that Wikipedia publishes in
    detail (the final)

ESPNcricinfo and Cricbuzz would be the richer/primary real sources, but are
blocked from this environment (see espncricinfo.py / cricbuzz.py for the
verified reasons) — Wikipedia fills that gap with real, citable data today.
"""
from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .base import DataSource

_HEADERS = {"User-Agent": "Mozilla/5.0 (NPL-Insight research bot; non-commercial analytics)"}
_NAME_ANNOTATION_RE = re.compile(r"\([^)]*\)")


def _get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=_HEADERS, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _table_rows(table) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    return rows


class WikipediaSource(DataSource):
    name = "wikipedia"

    def _season_url(self, season_id: int) -> str:
        for season in self.config["seasons"]:
            if season["id"] == season_id:
                return season["wikipedia_url"]
        raise ValueError(f"No wikipedia_url configured for season {season_id}")

    def fetch_squads(self, season_id: int) -> dict[str, list[str]]:
        """Real team -> real player name list. Only Season 1's article publishes
        full squads in a parseable table; season 2 reuses the same core rosters
        (same league, same 8 franchises) when its own squad table isn't available."""
        soup = _get_soup(self._season_url(season_id))
        squads: dict[str, list[str]] = {}
        for table in soup.find_all("table", class_="wikitable"):
            header_cells = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]
            if header_cells[:1] != ["Team"] and header_cells[:1] != ["Teams"]:
                continue
            if "Playing squads" not in header_cells and "Squad" not in " ".join(header_cells):
                continue
            for row in table.find_all("tr")[1:]:
                cells = row.find_all(["th", "td"])
                if len(cells) < 3:
                    continue
                team = cells[0].get_text(strip=True)
                squad_raw = cells[-1].get_text(",", strip=True)
                squad_raw = _NAME_ANNOTATION_RE.sub("", squad_raw)
                players = [n.strip() for n in squad_raw.split(",") if n.strip()]
                if team and players:
                    squads[team] = players
            if squads:
                break
        return squads

    def fetch_leaders(self, season_id: int) -> dict[str, pd.DataFrame]:
        """Real season leaderboards: most runs, most wickets."""
        soup = _get_soup(self._season_url(season_id))
        leaders: dict[str, pd.DataFrame] = {}
        for table in soup.find_all("table", class_="wikitable"):
            caption = table.find("caption")
            label = caption.get_text(strip=True) if caption else ""
            if label not in ("Most runs", "Most wickets"):
                continue
            rows = _table_rows(table)
            header, data_rows = rows[0], rows[1:]
            data_rows = [r for r in data_rows if len(r) == len(header)]
            df = pd.DataFrame(data_rows, columns=header)
            df["season"] = season_id
            key = "most_runs" if label == "Most runs" else "most_wickets"
            leaders[key] = df
        return leaders

    def fetch_awards(self, season_id: int) -> pd.DataFrame:
        soup = _get_soup(self._season_url(season_id))
        for table in soup.find_all("table", class_="wikitable"):
            header = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]
            if "Award" in header and "Player" in header:
                rows = _table_rows(table)
                data_rows = [r for r in rows[1:] if len(r) == len(header)]
                df = pd.DataFrame(data_rows, columns=header)
                df["season"] = season_id
                return df
        return pd.DataFrame(columns=["Award", "Prize", "Player", "Team", "season"])

    def fetch_standings(self, season_id: int) -> pd.DataFrame:
        """Real points table: position, played/won/lost, points, NRR."""
        soup = _get_soup(self._season_url(season_id))
        for table in soup.find_all("table", class_="wikitable"):
            header = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]
            if header[:2] == ["Pos", "Team"]:
                rows = _table_rows(table)
                data_rows = []
                for r in rows[1:]:
                    if len(r) < len(header):
                        r = r + [""] * (len(header) - len(r))
                    data_rows.append(r[: len(header)])
                df = pd.DataFrame(data_rows, columns=header)
                df["season"] = season_id
                df["Team"] = df["Team"].str.replace(r"\s*\([^)]*\)", "", regex=True)
                return df
        return pd.DataFrame()

    def fetch_head_to_head(self, season_id: int) -> pd.DataFrame:
        """Real winner/margin for every league-stage fixture, parsed from
        Wikipedia's home-vs-visitor results grid and resolved to full team
        names (the raw grid only gives abbreviations/city prefixes)."""
        soup = _get_soup(self._season_url(season_id))
        for table in soup.find_all("table", class_="wikitable"):
            first_row_text = table.find("tr").get_text(" ", strip=True)
            if "Visitor team" not in first_row_text:
                continue
            rows = table.find_all("tr")
            header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            data_rows = rows[2:]  # row 0 = abbreviations header, row 1 = "Home team" label

            home_teams_in_order = [
                r.find_all(["th", "td"])[0].get_text(strip=True)
                for r in data_rows if r.find_all(["th", "td"])
            ]
            # Header columns (after the first "Visitor team →" cell) are in the
            # same team order as the row order, since it's a symmetric NxN grid.
            abbrev_to_team = dict(zip(header_cells[1:], home_teams_in_order))

            records = []
            for row in data_rows:
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue
                home_team = cells[0].get_text(strip=True)
                for col_idx, cell in enumerate(cells[1:], start=1):
                    text = cell.get_text(" ", strip=True)
                    if not text:
                        continue
                    if text.endswith("Super Over"):
                        winner_prefix, margin = text[: -len("Super Over")].strip(), "Super Over"
                    else:
                        parts = text.rsplit(" ", 1)
                        if len(parts) != 2:
                            continue
                        winner_prefix, margin = parts
                    visitor_abbrev = header_cells[col_idx] if col_idx < len(header_cells) else ""
                    visitor_team = abbrev_to_team.get(visitor_abbrev, visitor_abbrev)
                    winner_team = next(
                        (t for t in (home_team, visitor_team) if t.split()[0] == winner_prefix),
                        winner_prefix,
                    )
                    records.append({
                        "season": season_id,
                        "team_a": home_team,
                        "team_b": visitor_team,
                        "winner": winner_team,
                        "margin": margin,
                    })
            return pd.DataFrame.from_records(records)
        return pd.DataFrame()

    def fetch(self, season_id: int) -> pd.DataFrame:
        """Satisfies the DataSource interface: returns the one real, full match
        scorecard Wikipedia publishes (the final), reshaped to the master schema.
        Phase is intentionally left as `match_total` rather than split into
        powerplay/middle/death — Wikipedia's scorecard doesn't report over-by-over
        timing, and fabricating a phase split for genuinely real figures would
        defeat the point of having a real-data source."""
        soup = _get_soup(self._season_url(season_id))
        tables = soup.find_all("table", class_="wikitable")

        innings_tables = [t for t in tables if t.find("tr") and "innings" in t.find("tr").get_text(strip=True).lower()]
        bowling_tables = [t for t in tables if t.find("tr") and "bowling" in t.find("tr").get_text(strip=True).lower()]
        if len(innings_tables) < 2 or len(bowling_tables) < 2:
            return pd.DataFrame()

        records = []
        team_names = [t.find("tr").get_text(strip=True).replace(" innings", "") for t in innings_tables[:2]]
        winner = None  # Wikipedia summary text states the winner; left unset here since
        # the final's result is already captured authoritatively in fetch_awards()/season
        # standings — avoid guessing it from scorecard order alone.

        for team, innings_table, opponent in zip(team_names, innings_tables[:2], reversed(team_names)):
            for row in _table_rows(innings_table)[1:]:
                if len(row) < 7 or row[0].lower().startswith("extras"):
                    continue
                player, _status, runs, balls = row[0], row[1], row[2], row[3]
                if not runs.replace("*", "").isdigit():
                    continue
                records.append({
                    "match_id": f"S{season_id}_FINAL", "season": season_id, "team": team,
                    "opponent": opponent, "player": player,
                    "runs": int(runs.replace("*", "")), "balls": int(balls) if balls.isdigit() else 0,
                    "strike_rate": 0.0, "wickets": 0, "overs": 0.0, "economy": 0.0,
                    "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0, "fielding_errors": 0,
                    "match_result": "Unknown", "venue": "Tribhuvan University International Cricket Ground",
                    "phase": "match_total",
                })

        for bowling_table in bowling_tables[:2]:
            # Identify the bowling side from its own caption ("<Team> bowling")
            # rather than assuming table order — that assumption was the source
            # of a real bug where bowlers were tagged with their opponent's name.
            bowling_team = bowling_table.find("tr").get_text(strip=True).replace(" bowling", "")
            opponent = next((t for t in team_names if t != bowling_team), bowling_team)

            for row in _table_rows(bowling_table)[1:]:
                if len(row) < 6:
                    continue
                player, overs, _maidens, runs, wickets = row[0], row[1], row[2], row[3], row[4]
                if not overs.replace(".", "").isdigit():
                    continue
                overs_f = float(overs)
                runs_i = int(runs) if runs.isdigit() else 0
                records.append({
                    "match_id": f"S{season_id}_FINAL", "season": season_id, "team": bowling_team,
                    "opponent": opponent, "player": player,
                    "runs": 0, "balls": 0, "strike_rate": 0.0,
                    "wickets": int(wickets) if wickets.isdigit() else 0, "overs": overs_f,
                    "economy": round(runs_i / overs_f, 2) if overs_f else 0.0,
                    "catches_taken": 0, "catches_dropped": 0, "stumping_missed": 0, "fielding_errors": 0,
                    "match_result": "Unknown", "venue": "Tribhuvan University International Cricket Ground",
                    "phase": "match_total",
                })

        return pd.DataFrame.from_records(records)
