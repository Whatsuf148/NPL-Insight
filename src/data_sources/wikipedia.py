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

import difflib
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


def _canonicalize_team(name: str, known_teams: list[str]) -> str:
    """Maps a team name as spelled in one Wikipedia article/table to config's
    canonical spelling.

    Real bug this fixes: config.yaml's "Kathmandu Gorkhas" didn't match
    Season 1's article, which spells the same franchise "Kathmandu Gurkhas".
    Every roster lookup for that team silently returned nothing, so every
    "Kathmandu Gorkhas" player in the simulated dataset was a fallback
    placeholder name ("Kathmandu Player N") instead of a real one — not a
    crash, just quietly wrong data. Wikipedia's spelling is inconsistent
    even within the same article family, so this matches fuzzily against
    the authoritative team list instead of hardcoding one alias.
    """
    if name in known_teams:
        return name
    matches = difflib.get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else name


def _table_rows(table) -> list[list[str]]:
    """Parses <tr> rows into aligned cell lists, honoring rowspan.

    Wikipedia's ranking tables (e.g. "Most wickets") merge the rank/value
    cell across tied rows via rowspan instead of repeating it — a naive
    cell-per-row parse leaves tied rows short by one column and silently
    drops them downstream (a real bug: joint wicket-leaders were missing
    entirely). This carries a rowspan cell's text down into the rows it
    visually spans, so every row ends up with the same column count.
    """
    raw_trs = table.find_all("tr")
    # pending[col_index] = (text, rows_remaining)
    pending: dict[int, tuple[str, int]] = {}
    rows: list[list[str]] = []

    for tr in raw_trs:
        cells = tr.find_all(["th", "td"])
        row: list[str] = []
        col = 0
        cell_iter = iter(cells)
        current_cell = next(cell_iter, None)

        while current_cell is not None or col in pending:
            if col in pending:
                text, remaining = pending[col]
                row.append(text)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col] = (text, remaining - 1)
                col += 1
                continue

            text = current_cell.get_text(" ", strip=True)
            rowspan = int(current_cell.get("rowspan", 1) or 1)
            row.append(text)
            if rowspan > 1:
                pending[col] = (text, rowspan - 1)
            col += 1
            current_cell = next(cell_iter, None)

        if row:
            rows.append(row)
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
        known_teams = self.config["teams"]
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
                team = _canonicalize_team(cells[0].get_text(strip=True), known_teams)
                squad_raw = cells[-1].get_text(",", strip=True)
                squad_raw = _NAME_ANNOTATION_RE.sub("", squad_raw)
                players = [n.strip() for n in squad_raw.split(",") if n.strip()]
                if team and players:
                    squads[team] = players
            if squads:
                break
        return squads

    def fetch_captains(self, season_id: int) -> dict[str, str]:
        """Real team -> real captain name, parsed from the "(c)" annotation
        in the same squad table fetch_squads() reads (which strips that
        annotation off). Captains and other near-every-match starters are
        what make a real per-team core XI realistic in the simulator."""
        known_teams = self.config["teams"]
        soup = _get_soup(self._season_url(season_id))
        captains: dict[str, str] = {}
        for table in soup.find_all("table", class_="wikitable"):
            header_cells = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]
            if header_cells[:1] not in (["Team"], ["Teams"]):
                continue
            if "Playing squads" not in header_cells and "Squad" not in " ".join(header_cells):
                continue
            for row in table.find_all("tr")[1:]:
                cells = row.find_all(["th", "td"])
                if len(cells) < 3:
                    continue
                team = _canonicalize_team(cells[0].get_text(strip=True), known_teams)
                # Empty separator (not ",") — get_text(",") inserts a comma at every
                # tag boundary, which splits "Name(c)" into separate "Name" / "(c)"
                # chunks (the (c) annotation is its own <a> tag) and breaks the
                # name-to-annotation association this method depends on.
                squad_raw = cells[-1].get_text("", strip=True)
                match = re.search(r"([A-Za-z][\w .'\-]*?)\(c[,)]", squad_raw)
                if match:
                    captains[team] = match.group(1).strip()
            if captains:
                break
        return captains

    def fetch_leaders(self, season_id: int) -> dict[str, pd.DataFrame]:
        """Real season leaderboards: most runs, most wickets."""
        known_teams = self.config["teams"]
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
            if "Team" in df.columns:
                df["Team"] = df["Team"].apply(lambda t: _canonicalize_team(t, known_teams))
            key = "most_runs" if label == "Most runs" else "most_wickets"
            leaders[key] = df
        return leaders

    def fetch_awards(self, season_id: int) -> pd.DataFrame:
        known_teams = self.config["teams"]
        soup = _get_soup(self._season_url(season_id))
        for table in soup.find_all("table", class_="wikitable"):
            header = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]
            if "Award" in header and "Player" in header:
                rows = _table_rows(table)
                data_rows = [r for r in rows[1:] if len(r) == len(header)]
                df = pd.DataFrame(data_rows, columns=header)
                df["season"] = season_id
                if "Team" in df.columns:
                    df["Team"] = df["Team"].apply(lambda t: _canonicalize_team(t, known_teams))
                return df
        return pd.DataFrame(columns=["Award", "Prize", "Player", "Team", "season"])

    def fetch_final_toss(self, season_id: int) -> str | None:
        """The one real toss sentence Wikipedia publishes for the season —
        only the final's scorecard includes a "Toss:" line; no other match
        in either season's article has toss data at all. Not enough real
        coverage to build a toss-outcome model from (see
        analytics.batting_order_win_rate for the real-data proxy used
        instead), but worth surfacing as a verified fact."""
        soup = _get_soup(self._season_url(season_id))
        text = soup.get_text()
        match = re.search(r"Toss:\s*(.+?\.)", text)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else None

    def fetch_standings(self, season_id: int) -> pd.DataFrame:
        """Real points table: position, played/won/lost, points, NRR."""
        known_teams = self.config["teams"]
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
                df["Team"] = df["Team"].apply(lambda t: _canonicalize_team(t, known_teams))
                return df
        return pd.DataFrame()

    def fetch_head_to_head(self, season_id: int) -> pd.DataFrame:
        """Real winner/margin for every league-stage fixture, parsed from
        Wikipedia's home-vs-visitor results grid and resolved to full team
        names (the raw grid only gives abbreviations/city prefixes)."""
        known_teams = self.config["teams"]
        soup = _get_soup(self._season_url(season_id))
        for table in soup.find_all("table", class_="wikitable"):
            first_row_text = table.find("tr").get_text(" ", strip=True)
            if "Visitor team" not in first_row_text:
                continue
            rows = table.find_all("tr")
            header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            data_rows = rows[2:]  # row 0 = abbreviations header, row 1 = "Home team" label

            home_teams_in_order = [
                _canonicalize_team(r.find_all(["th", "td"])[0].get_text(strip=True), known_teams)
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
                home_team = _canonicalize_team(cells[0].get_text(strip=True), known_teams)
                for col_idx, cell in enumerate(cells[1:], start=1):
                    text = cell.get_text(" ", strip=True)
                    if not text:
                        continue
                    if text.endswith("Super Over"):
                        winner_prefix, margin = text[: -len("Super Over")].strip(), "Super Over"
                    else:
                        # Format is "<city prefix> <N> <runs|wickets|run|wicket>" —
                        # the prefix is always the first word, margin is everything else.
                        parts = text.split(" ", 1)
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

    def fetch_match_results(self, season_id: int) -> pd.DataFrame:
        """Real per-match results for every match of the season (league stage
        + playoffs + final) — not just the one match `fetch()` covers.

        Wikipedia's season articles embed a per-match info box for every
        single match (team scores, overs, top batter/bowler per innings,
        winner, margin, venue, Player of the Match), distinct from the
        head-to-head grid (`fetch_head_to_head`, winner/margin only, one row
        per pairing) and the one fully-detailed final scorecard (`fetch()`).
        This is the richest real source on the page: ~4 real player-team
        associations per match (32 matches/season), which is what makes it
        possible to catch a real mid-season transfer (e.g. Marchant de Lange
        moving from Chitwan Rhinos in Season 1 to Biratnagar Kings in Season
        2) that a single static squad table can't reflect.
        """
        known_teams = self.config["teams"]
        soup = _get_soup(self._season_url(season_id))
        all_tables = soup.find_all("table")
        none_class_tables = [t for t in all_tables if t.get("class") is None]

        label_re = re.compile(r"^(Match \d+|Final|Qualifier \d+|Eliminator)")
        score_re = re.compile(r"^(.*?)\s+([\d/]+)\s*\((\d+(?:\.\d+)?)\s*overs?\)$")
        batter_re = re.compile(r"^(.*?)\s+(\d+)\s*\*?\s*\((\d+)\)\s*(.*)$")
        bowler_re = re.compile(r"^(.*?)\s+(\d+/\d+)\s*\(([\d.]+)\s*overs?\)$")
        winner_re = re.compile(r"^(.*?)\s+won by\s+(.*?)\s+Tribhuvan", re.IGNORECASE)
        super_over_re = re.compile(r"Match tied \((.*?) won the Super Over", re.IGNORECASE)
        potm_re = re.compile(r"Player of the match:\s*(.*?)\s*\((.*?)\)")

        groups = []
        i = 0
        while i < len(none_class_tables):
            header_text = none_class_tables[i].get_text("|", strip=True)
            if label_re.match(header_text) and i + 2 < len(none_class_tables):
                groups.append((header_text, none_class_tables[i + 1], none_class_tables[i + 2]))
                i += 3
            else:
                i += 1

        records = []
        for label_text, score_table, result_table in groups:
            label_parts = label_text.split("|")
            match_label = label_parts[0]
            date = label_parts[1] if len(label_parts) > 1 else ""

            score_rows = score_table.find_all("tr")
            if len(score_rows) < 1:
                continue
            score_cells = [c.get_text(" ", strip=True) for c in score_rows[0].find_all(["th", "td"])]
            if len(score_cells) < 3:
                continue
            m1, m2 = score_re.match(score_cells[0]), score_re.match(score_cells[2])
            if not (m1 and m2):
                continue
            team1 = _canonicalize_team(m1.group(1).strip(), known_teams)
            team2 = _canonicalize_team(m2.group(1).strip(), known_teams)

            performers = {}
            if len(score_rows) > 1:
                perf_cells = [c.get_text(" ", strip=True) for c in score_rows[1].find_all(["th", "td"])]
                perf_cells = [c for c in perf_cells if c]
                for cell, batting_team, bowling_team in zip(perf_cells, (team1, team2), (team2, team1)):
                    bm = batter_re.match(cell)
                    if not bm:
                        continue
                    batter, runs, _balls, rest = bm.groups()
                    performers[f"{batting_team}_top_batter"] = batter.strip()
                    performers[f"{batting_team}_top_batter_runs"] = int(runs)
                    bowl_m = bowler_re.match(rest.strip())
                    if bowl_m:
                        bowler, figures, _overs = bowl_m.groups()
                        performers[f"{bowling_team}_top_bowler"] = bowler.strip()
                        performers[f"{bowling_team}_top_bowler_figures"] = figures

            result_text = result_table.get_text(" ", strip=True)
            wm = winner_re.search(result_text)
            winner_raw, margin = (wm.groups() if wm else (None, None))
            if winner_raw is None:
                som = super_over_re.search(result_text)
                if som:
                    winner_raw, margin = som.group(1), "Super Over"
            # winner_re captures the team's full name as spelled in this specific
            # match's result sentence, which can use a different spelling variant
            # than team1/team2 above (e.g. "Kathmandu Gurkhas" vs "Kathmandu
            # Gorkhas") — canonicalize directly rather than assuming it matches
            # team1/team2 by substring.
            winner = _canonicalize_team(winner_raw, known_teams) if winner_raw else None
            potm_match = potm_re.search(result_text)
            player_of_match = potm_match.group(1) if potm_match else None
            potm_team = _canonicalize_team(potm_match.group(2), known_teams) if potm_match else None

            records.append({
                "season": season_id,
                "match_label": match_label,
                "date": date,
                "team1": team1, "team1_score": m1.group(2), "team1_overs": m1.group(3),
                "team2": team2, "team2_score": m2.group(2), "team2_overs": m2.group(3),
                "winner": winner,
                "margin": margin,
                "player_of_match": player_of_match,
                "player_of_match_team": potm_team,
                **performers,
            })

        return pd.DataFrame.from_records(records)

    def fetch(self, season_id: int) -> pd.DataFrame:
        """Satisfies the DataSource interface: returns the one real, full match
        scorecard Wikipedia publishes (the final), reshaped to the master schema.
        Phase is intentionally left as `match_total` rather than split into
        powerplay/middle/death — Wikipedia's scorecard doesn't report over-by-over
        timing, and fabricating a phase split for genuinely real figures would
        defeat the point of having a real-data source."""
        known_teams = self.config["teams"]
        soup = _get_soup(self._season_url(season_id))
        tables = soup.find_all("table", class_="wikitable")

        innings_tables = [t for t in tables if t.find("tr") and "innings" in t.find("tr").get_text(strip=True).lower()]
        bowling_tables = [t for t in tables if t.find("tr") and "bowling" in t.find("tr").get_text(strip=True).lower()]
        if len(innings_tables) < 2 or len(bowling_tables) < 2:
            return pd.DataFrame()

        records = []
        team_names = [
            _canonicalize_team(t.find("tr").get_text(strip=True).replace(" innings", ""), known_teams)
            for t in innings_tables[:2]
        ]
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
            bowling_team = _canonicalize_team(
                bowling_table.find("tr").get_text(strip=True).replace(" bowling", ""), known_teams
            )
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
