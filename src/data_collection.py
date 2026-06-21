"""Orchestrates data collection across all enabled sources/seasons.

Two kinds of sources:
  - The per-phase player-match source that produces the master dataset the
    rest of the pipeline depends on. This must always succeed — never
    silently skips a failing source. `cricsheet` (real ball-by-ball data)
    is the default; `simulator` (real names, simulated performances) is a
    statistical fallback if cricsheet is disabled or unreachable.
  - Real-data enrichment (wikipedia.py) that supplies real rosters,
    leaderboards, awards, and real match scorecards. This is best-effort:
    if the network/site is unavailable, the simulator falls back to
    synthetic player names rather than failing the whole pipeline, since
    the enrichment is additive, not the core dataset (cricsheet doesn't
    depend on it at all — it has real names built in).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config, resolve_path
from src.data_sources import SOURCE_REGISTRY
from src.data_sources.wikipedia import WikipediaSource

logger = logging.getLogger(__name__)

MASTER_SOURCE_NAMES = {"cricsheet", "simulator", "espncricinfo", "cricbuzz"}


def _match_result_evidence(match_results: pd.DataFrame, teams: list[str]) -> dict[str, set[str]]:
    """Extracts {team: {player, ...}} from fetch_match_results()'s per-match
    top-batter/top-bowler columns — direct real evidence of who played for
    whom in this specific season, as opposed to a season-agnostic squad list."""
    evidence: dict[str, set[str]] = {team: set() for team in teams}
    for team in teams:
        for col in (f"{team}_top_batter", f"{team}_top_bowler"):
            if col in match_results.columns:
                evidence[team].update(match_results[col].dropna().unique())
    return evidence


def _build_season_rosters(
    squads_by_season: dict[int, dict[str, list[str]]],
    evidence_by_season: dict[int, dict[str, set[str]]],
    seasons: list[int],
) -> dict[int, dict[str, list[str]]]:
    """Combines static squad-table rosters with direct match-evidence per
    season into one roster per (season, team), and — this is the part that
    actually matters — removes a player from a team's roster for a season
    where direct evidence says they played elsewhere.

    Real motivating case: Marchant de Lange is in Season 1's Chitwan Rhinos
    squad table (the only season with a parseable full squad list), but
    really played for Biratnagar Kings in Season 2 (confirmed via
    fetch_match_results, where he appears as a bowler in Biratnagar Kings'
    matches). Season 1's squad table, naively reused for both seasons, would
    keep assigning him to Chitwan Rhinos in the Season 2 simulated data too.
    """
    # Fallback base: any season's squad list, used for seasons with no squad
    # table of their own (today, that's every season except Season 1).
    fallback_squads = next((sq for sq in squads_by_season.values() if sq), {})

    rosters_by_season: dict[int, dict[str, list[str]]] = {}
    for season_id in seasons:
        base_squads = squads_by_season.get(season_id) or fallback_squads
        evidence = evidence_by_season.get(season_id, {})

        player_to_evidenced_team: dict[str, str] = {}
        for team, players in evidence.items():
            for player in players:
                player_to_evidenced_team[player] = team

        season_rosters: dict[str, list[str]] = {}
        for team, players in base_squads.items():
            # Drop anyone this season's evidence says actually played for a
            # *different* team — don't let a stale squad list override
            # direct, season-specific proof of a transfer.
            season_rosters[team] = [
                p for p in players if player_to_evidenced_team.get(p, team) == team
            ]

        for player, team in player_to_evidenced_team.items():
            season_rosters.setdefault(team, [])
            if player not in season_rosters[team]:
                season_rosters[team].append(player)

        rosters_by_season[season_id] = season_rosters
    return rosters_by_season


def collect_real_enrichment(config: dict | None = None) -> dict:
    """Best-effort real-data layer. Returns combined rosters/leaders/awards/
    final-scorecards across all configured seasons, or empty structures if
    the source is disabled or unreachable."""
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    teams = config["teams"]

    enrichment = {
        "rosters": {},
        "rosters_by_season": {},
        "captains": {},
        "leaders_runs": [],
        "leaders_wickets": [],
        "awards": [],
        "final_scorecards": [],
        "standings": [],
        "head_to_head": [],
        "match_results": [],
        "final_tosses": {},
        "real_tosses": [],
    }

    cricsheet_config = config["data_sources"].get("cricsheet", {})
    if cricsheet_config.get("enabled", False):
        from src.data_sources.cricsheet import CricsheetSource

        cricsheet_source = CricsheetSource({**config, **cricsheet_config})
        for season in config["seasons"]:
            try:
                toss_df = cricsheet_source.fetch_toss_results(season["id"])
                if not toss_df.empty:
                    enrichment["real_tosses"].append(toss_df)
            except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
                logger.warning("Cricsheet toss enrichment unavailable for season %s (%s)", season["id"], exc)

    wikipedia_config = config["data_sources"].get("wikipedia", {})
    if not wikipedia_config.get("enabled", False):
        return enrichment

    source = WikipediaSource({**config, **wikipedia_config})
    squads_by_season: dict[int, dict[str, list[str]]] = {}
    evidence_by_season: dict[int, dict[str, set[str]]] = {}

    for season in config["seasons"]:
        season_id = season["id"]
        try:
            squads = source.fetch_squads(season_id)
            squads_by_season[season_id] = squads
            for team, players in squads.items():
                enrichment["rosters"].setdefault(team, set()).update(players)

            captains = source.fetch_captains(season_id)
            enrichment["captains"].update(captains)

            match_results = source.fetch_match_results(season_id)
            if not match_results.empty:
                match_results.to_csv(raw_dir / f"season_{season_id}_wikipedia_matches.csv", index=False)
                enrichment["match_results"].append(match_results)
                evidence_by_season[season_id] = _match_result_evidence(match_results, teams)

            leaders = source.fetch_leaders(season_id)
            if "most_runs" in leaders:
                enrichment["leaders_runs"].append(leaders["most_runs"])
            if "most_wickets" in leaders:
                enrichment["leaders_wickets"].append(leaders["most_wickets"])

            awards = source.fetch_awards(season_id)
            if not awards.empty:
                enrichment["awards"].append(awards)

            standings = source.fetch_standings(season_id)
            if not standings.empty:
                enrichment["standings"].append(standings)

            head_to_head = source.fetch_head_to_head(season_id)
            if not head_to_head.empty:
                enrichment["head_to_head"].append(head_to_head)

            final_scorecard = source.fetch(season_id)
            if not final_scorecard.empty:
                final_scorecard.to_csv(raw_dir / f"season_{season_id}_wikipedia_final.csv", index=False)
                enrichment["final_scorecards"].append(final_scorecard)

            toss = source.fetch_final_toss(season_id)
            if toss:
                enrichment["final_tosses"][season_id] = toss

            logger.info("Wikipedia enrichment OK for season %s (%d real squads)", season_id, len(squads))
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment, never fatal
            logger.warning("Wikipedia enrichment unavailable for season %s (%s); "
                            "falling back to synthetic player names.", season_id, exc)

    enrichment["rosters"] = {team: sorted(players) for team, players in enrichment["rosters"].items()}
    seasons = [s["id"] for s in config["seasons"]]
    enrichment["rosters_by_season"] = _build_season_rosters(squads_by_season, evidence_by_season, seasons)
    return enrichment


def collect_all(config: dict | None = None) -> tuple[dict[int, pd.DataFrame], dict]:
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    enrichment = collect_real_enrichment(config)

    enabled_sources = [s for s in config["data_sources"]["enabled"] if s in MASTER_SOURCE_NAMES]
    seasons = [s["id"] for s in config["seasons"]]

    collected: dict[int, pd.DataFrame] = {}
    for season_id in seasons:
        season_frames = []
        for source_name in enabled_sources:
            if source_name not in SOURCE_REGISTRY:
                raise ValueError(f"Unknown data source '{source_name}' in config.")
            source_config = {**config, **config["data_sources"].get(source_name, {})}
            if source_name == "simulator" and enrichment["rosters"]:
                source_config = {
                    **source_config,
                    "_real_rosters": enrichment["rosters"],
                    "_real_rosters_by_season": enrichment["rosters_by_season"],
                    "_real_captains": enrichment["captains"],
                }

            source = SOURCE_REGISTRY[source_name](source_config)
            logger.info("Collecting season %s from source '%s'", season_id, source_name)
            df = source.fetch(season_id)
            if df.empty:
                raise RuntimeError(
                    f"Source '{source_name}' returned no data for season {season_id}; "
                    "refusing to proceed with an incomplete dataset."
                )
            out_path = raw_dir / f"season_{season_id}_{source_name}.csv"
            df.to_csv(out_path, index=False)
            season_frames.append(df)

        collected[season_id] = pd.concat(season_frames, ignore_index=True)

    return collected, enrichment


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all()
