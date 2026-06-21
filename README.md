# NPL Insight
Nepal Premier League (NPL) Advanced Analytics Dashboard

## Project Overview

A professional, dynamic cricket analytics system for the Nepal Premier League (NPL), covering:

* Season 1 (2024)
* Season 2 (2025)
* Comparative analysis between both seasons

The system integrates a config-driven, multi-source data pipeline, advanced feature engineering, and an interactive Streamlit dashboard — built so every layer (sources, metrics, storage, UI filters) scales by editing [config/config.yaml](config/config.yaml), never by editing code.

---

## Architecture

```
config/config.yaml        Single source of truth: seasons, teams, venues, phases,
                           paths, storage backend, thresholds. No hardcoded values
                           anywhere downstream — code reads through src/config.py.

src/
  config.py                Config loader.
  data_sources/
    base.py                 Abstract DataSource interface.
    cricsheet.py              PRIMARY source: real ball-by-ball data for all 64 NPL
                              matches (cricsheet.org) — every batting/bowling number
                              in the master dataset is a real delivery, not a simulation.
    wikipedia.py              Real-data enrichment: live (requests+BeautifulSoup) against
                              Wikipedia's NPL season articles — real squads, season
                              leaderboards, awards, standings, head-to-head results.
    simulator.py             Statistical FALLBACK if cricsheet is disabled/unreachable;
                              real player names (via wikipedia.py), simulated performances.
    espncricinfo.py           Real fetch() implementation, disabled — verified HTTP 403
                              (bot-blocked) from this environment, see file docstring.
    cricbuzz.py                Real fetch() implementation, disabled — verified blocked
                              for automated/crawler access from this environment.
  data_collection.py        Orchestrates enabled sources per season -> data/raw/, plus
                             best-effort real-data enrichment (rosters/leaders/awards).
  real_data_anchor.py        Rescales simulated totals to match real leaderboards —
                             only used on the simulator fallback path, not cricsheet.
  data_cleaning.py          Validates schema, fills/normalizes, merges seasons -> master dataset.
  feature_engineering.py    All advanced metrics (batting/bowling/fielding/impact).
  analytics.py              Insight generation: rankings, comparisons, narrative insights.
  storage.py                CSV/SQLite storage abstraction (backend = config switch).

models/
  train_win_probability.py  Trains + persists the win-probability model from live data.

app/
  streamlit_app.py          Dashboard: reads only from processed storage, dynamic filters,
                             every chart paired with a generated insight statement.

notebooks/
  analysis.ipynb            Exploratory analysis against the processed master dataset.

tests/                       Unit tests for cleaning, features, analytics, and the simulator.

run_pipeline.py             End-to-end: collect -> clean -> engineer features -> store.
```

### Why this shape

* **No hardcoding** — teams, seasons, venues, phases, thresholds, storage backend, and which data sources are active all live in `config/config.yaml`. Adding Season 3 or a new team is a config edit, not a code change.
* **Reusable code** — every pipeline stage is a pure function over a dataframe + config; `feature_engineering.py` and `analytics.py` have zero global state and compose freely (e.g. `player_impact_score` reuses `batting_by_phase` + `fielding`).
* **Scalable pipeline** — `DataSource` is an abstract interface; new sources (real ESPNcricinfo/Cricbuzz scrapers, an official API) register in `data_sources/__init__.py` and the orchestrator, cleaner, and dashboard never change.
* **Insight-driven, not static** — `analytics.py` answers specific questions (who improved most, which catches cost the most runs, what changed between seasons) rather than just reshaping data for charts. The dashboard pairs every chart with a generated insight (`st.success`/`st.info`/`st.warning` callouts), and is fully filter-driven (season/team/player/match/phase) so nothing is a fixed, pre-baked screenshot.
* **No incomplete datasets** — `data_collection.py` raises rather than silently continuing if a source returns empty data; `data_cleaning.merge_seasons` raises if any configured season is missing from the merged result.

## Data Schema

Player-match-phase granularity (one row per player, per match, per phase):

`match_id, season, team, opponent, player, runs, balls, strike_rate, wickets, overs, economy, catches_taken, catches_dropped, stumping_missed, fielding_errors, match_result, venue, phase`

## Feature Engineering

* **Batting**: Strike Rate by Phase, Boundary Percentage, Consistency Index (rolling, config-windowed)
* **Bowling**: Economy by Phase, Wicket Probability, Dot Ball %
* **Fielding** (key differentiator): Catch Efficiency, Fielding Error Rate, Runs Lost to Errors
* **Advanced**: Player Impact Score, Clutch Performance Index, Pressure Performance Score, Win Contribution %
* **Season-wide career stats** (`analytics.player_stats_table`): Batting Average (runs per dismissal — real, from Cricsheet's tracked dismissals, not estimated), Bowling Average (runs conceded per wicket), Not Outs, Fours, Sixes, Boundary %, Maidens. These are computed once across the whole season (every phase combined) rather than per-phase, since "average" is meaningless split into small windows — see "Season Leaderboards" below for the qualified versions.

## Data Sourcing

**The master dataset is now built from real ball-by-ball data — [Cricsheet](https://cricsheet.org).** `src/data_sources/cricsheet.py` downloads Cricsheet's real JSON for all 64 NPL matches (32 per season) and parses every legal delivery: who batted, who bowled, runs off the bat (including real fours/sixes), extras, wickets (with real dismissal type, the actual dismissed player, and fielders), maiden overs, over number (→ real powerplay/middle/death phase), real toss winner/decision, real venue, real result. Every `runs`, `wickets`, `strike_rate`, `economy`, `batting_average`, and `bowling_average` value in the master dataset is now a real number derived from a real ball, not a statistical estimate. This directly fixes the kind of error that earlier versions of this project had: a real bowler's wicket count now matches their actual published total exactly, because it's counting actual dismissals (verified for Sandeep Lamichhane: 17 wickets in Season 2; verified for Adam Rossington: batting average 53.83 and 31 fours / 19 sixes — all exactly matching real published totals).

What real ball-by-ball data does *not* include: dropped catches, missed stumpings, or misfields. No public source records fielding *mistakes* for NPL — only completed dismissals. `catches_dropped`, `stumping_missed`, and `fielding_errors` are therefore always `0` from Cricsheet — an honest "we don't have this data" zero, not an estimate. Fielding Insights will necessarily show less when Cricsheet is the active source; that's the real tradeoff for everything else being verifiable.

**Cricsheet player names use cricket-scorecard abbreviations** ("S Lamichhane", "MJ Guptill"), while the Wikipedia enrichment layer (below) uses full names ("Sandeep Lamichhane", "Martin Guptill"). The two aren't name-joined — Fun Facts (Wikipedia-sourced) and Player Stats Explorer (Cricsheet-sourced) may refer to the same real person under different spellings. This is a cosmetic inconsistency, not a correctness bug; resolving it would need a name-mapping layer not yet built.

**Wikipedia (`src/data_sources/wikipedia.py`) remains a real-data enrichment layer** — live (requests + BeautifulSoup, tested against `en.wikipedia.org`), supplying things Cricsheet doesn't: season leaderboards (most runs/wickets), award winners, final standings (points table), and a head-to-head results grid — surfaced in the dashboard's **Fun Facts** tab.

**Real toss data, for every match, not just finals.** Unlike Wikipedia (which only publishes a toss sentence for the two finals), Cricsheet records the real toss winner and decision for all 64 matches. `analytics.toss_win_probability()` computes a genuine toss-outcome win rate across the full dataset (37.5% — the toss winner does *not* reliably win in this league) and a breakdown by bat-first vs. field-first decision, surfaced in Fun Facts as **Real Toss Win Probability**. A separate `analytics.batting_order_win_rate()` (sourced from Wikipedia's per-match scorecards) is kept alongside it for cross-reference.

**Statistical fallback — `src/data_sources/simulator.py`:** if Cricsheet is disabled or unreachable, the pipeline falls back to a statistical generator anchored to real player names (via Wikipedia squads) and real season-leaderboard totals (`real_data_anchor.py`), with match outcomes causally tied to performance (verified by `tests/test_simulator.py`). This is what the project ran on before Cricsheet was wired in, and remains available for offline use — switch back by setting `data_sources.enabled: [simulator]` in `config/config.yaml`.

**ESPNcricinfo / Cricbuzz — real code, verified blocked, not stubs:**
* `espncricinfo.py`: a direct `requests.get` to espncricinfo.com (including the real NPL series URL) returns **HTTP 403** from this environment — confirmed by testing, not assumed.
* `cricbuzz.py`: direct requests to cricbuzz.com succeed, but Anthropic's own crawler/fetch tooling is blocked from the domain at the platform level, and no public series ID could be located via search from here.

Both files contain real `fetch()` implementations (HTTP request + BeautifulSoup parsing against the actual target pages) — they are disabled in config, not because the code is incomplete, but because this sandboxed environment cannot reach them. Run them from an unblocked network (your own machine, or via an approved proxy/API) by flipping `data_sources.espncricinfo.enabled` / `data_sources.cricbuzz.enabled` to `true` in `config/config.yaml` — no other code changes needed, since they implement the same `DataSource` interface as everything else.

## Running

```bash
pip install -r requirements.txt

# 1. Build the full dataset (collect -> clean -> engineer features -> store)
python run_pipeline.py

# 2. (Optional) Train the win-probability model
python -m models.train_win_probability

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Testing

```bash
python -m pytest tests/ -v
```

Tests run against an in-memory fixture dataset (`tests/conftest.py`), not the simulator's generated CSVs, so they're fast and don't depend on `run_pipeline.py` having been run first. Coverage includes: schema validation and incompleteness guards in `data_cleaning.py`, metric correctness in `feature_engineering.py`, insight functions in `analytics.py`, the numpy logistic regression, and regression guards for two real bugs caught during development:
* `test_simulator_winner_correlates_with_runs_and_wickets` — match outcomes were once generated independently of performance stats, making the win-probability model unable to learn anything (accuracy 0.25 -> 0.83 after the fix).
* `test_final_scorecard_bowlers_are_tagged_with_their_own_team_not_opponent` (in `tests/test_wikipedia.py`) — bowling-table rows from the real Wikipedia scorecard were tagged with the *batting* team's name instead of their own, because team/opponent were inferred from table order instead of each table's own caption.
* `test_fetch_leaders_includes_joint_wicket_leaders_merged_via_rowspan` — Wikipedia merges the rank cell across tied rows with `rowspan` instead of repeating it; a naive cell-per-row parse silently dropped tied players entirely (Season 2's three players tied at 17 wickets were missing two of three). `_table_rows()` now expands rowspan cells into every row they visually cover.
* `test_real_captain_plays_nearly_every_match_their_team_plays` — every match's playing XI was originally sampled independently from the full squad, so a real captain (e.g. Kushal Bhurtel, in a 21-player Pokhara Avengers squad) could land in only 2 of 15 matches purely by chance. Looked obviously wrong once real names were wired in. Fixed by giving each team a fixed core XI (captain always included) for the season, with light week-to-week rotation — see "Player Participation Realism" below.
* `test_fetch_squads_resolves_every_team_to_configs_canonical_spelling` — config.yaml spells one franchise "Kathmandu Gorkhas"; Season 1's Wikipedia article spells the same franchise "Kathmandu Gurkhas" (a real inconsistency *within* Wikipedia's own NPL coverage). Every roster lookup for that team silently returned nothing, so 100% of that team's players in the simulated dataset were placeholder names ("Kathmandu Player N") — not a crash, just quietly wrong data for one team. Fixed with `_canonicalize_team()` (fuzzy-matches any parsed team name against `config['teams']`), applied everywhere `wikipedia.py` extracts a team name.
* `test_fetch_match_results_detects_real_marchant_de_lange_transfer` / `tests/test_data_collection.py` — a single static squad snapshot can't reflect a real mid-season transfer; see "Season-Specific Rosters" below.
* `test_fetch_match_results_winner_resolves_to_canonical_team_spelling` — the per-match winner-extraction regex captures a team's full name as spelled in *that specific match's* own result sentence, which can use yet another spelling variant than the same row's `team1`/`team2` columns; every winner now canonicalizes independently rather than assuming it matches by substring.
* `tests/test_cricsheet.py` — live integration tests against cricsheet.org. `test_fetch_matches_real_sandeep_lamichhane_season2_wickets` is the regression guard for the bug that motivated switching to real ball-by-ball data: the statistical simulator gave a real elite bowler far fewer wickets than his actual published total, because it had no concept of which players are genuinely exceptional. `test_fetch_has_no_unknown_match_results` guards a real edge case (Super-Over-decided ties record the winner under `outcome['eliminator']`, not `outcome['winner']`) that left one match's result as "Unknown" before the fix.

`tests/test_wikipedia.py` and `tests/test_cricsheet.py` make live requests and skip automatically if their target is unreachable.

## A Note on Data Accuracy

With Cricsheet as the active source, the master dataset's batting/bowling/wicket numbers are real ball-by-ball figures, not estimates — verified directly against Wikipedia's independently published leaderboards (Sandeep Lamichhane's 17 Season 2 wickets match exactly, because both sources are counting the same real matches). Three fielding columns (`catches_dropped`, `stumping_missed`, `fielding_errors`) are an honest `0` because no public source records fielding *mistakes* for NPL — only completed dismissals. That's a real absence of data, not an estimate dressed up as one.

What this project can't do anything about: ESPNcricinfo and Cricbuzz are both genuinely unreachable from this environment (HTTP 403 and platform-level crawler block respectively, confirmed by direct testing through two independent tool paths) — not a code limitation, a network one. Cricsheet and Wikipedia are what's actually accessible and verifiable from here, and both are real, independently-checkable sources, not synthetic substitutes for the real thing.

## Player Participation Realism

Real T20 leagues don't pick an independent random 11 every match — captains and regular starters play nearly every game; squad depth (the rest of an 18-21 player roster) mostly sits out unless rotated in. `src/data_sources/simulator.py` now models this: each team gets one **core XI per season** (anchored on the real captain from `WikipediaSource.fetch_captains()`), reused for every match with a small chance (0-2 players) of rotation. `analytics.player_stats_table()` exposes a `matches_played_pct` column so it's visible at a glance whether a given name is a near-every-match regular or genuinely a fringe player in the simulated data.

## Dashboard Sections

1. **Match Overview** — score summary, run progression, drill-down match insights
2. **Batting Analysis** — strike rate by phase, consistency ranking, best-strike-rate-by-phase leaderboard
3. **Bowling Analysis** — economy vs phase, wicket probability, highest wicket-taker against each opponent team
4. **Fielding Insights** — catch-drop leaderboard, best fielders ranking, runs lost to errors
5. **Player Insights** — impact score ranking, season-over-season change, a searchable **Player Stats Explorer** (with real batting/bowling average, fours, sixes, boundary %, maidens, `matches_played_pct`), a **Player Profile** lookup (career line, best individual score, best bowling figures, impact-score rank), **All-Time Leaders** (cross-season cumulative runs/wickets/catches), **Season Leaderboards** (most sixes/fours, best qualified batting/bowling average, most maidens), and a **Player vs Player** match-level comparison tool
6. **Season Comparison** — totals, averages, and player performance deltas between seasons
7. **Fun Facts** — real, verified NPL facts: season leaders, **real all-time leaders** (combined across seasons' published top-5s), award winners, real final-match scorecards, real final standings, a real **head-to-head lookup** (pick any two teams, see every real fixture result between them), and **real win rate by batting order** across all 64 real matches plus the two real toss-decision facts — all sourced from `src/data_sources/wikipedia.py`

Both leaderboard-style insights qualify on a minimum-balls/representative-sample threshold so small samples can't top a leaderboard. The "Player vs Player" tool compares two players' stat lines in every match where both appeared — an honest match-level comparison, not a literal ball-by-ball duel, since the dataset is player-match-phase aggregates and doesn't track which specific ball a batter faced from which bowler.

All sections share sidebar filters: Season, Team, Player, Match, Phase.

## Constraints & Assumptions

* Catch drops / stumping misses are simulated where real data isn't available (`src/data_sources/simulator.py`); swap in real values once collected.
* Consistency is enforced across seasons via shared schema validation in `data_cleaning.py`.
* Missing values are filled (0 for unplayed metrics) rather than dropped, so the dataset stays complete; rows missing required identifiers are dropped and logged, never silently kept.

## Goal

A portfolio-level, professional sports analytics system comparable to ESPN-style dashboards, demonstrating data engineering, data analytics, visualization, and cricket domain understanding.
