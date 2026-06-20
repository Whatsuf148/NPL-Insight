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
    simulator.py             Per-phase player-match data generator (active by default);
                              uses real player names/rosters when wikipedia.py succeeds.
    wikipedia.py              Real-data source: live (requests+BeautifulSoup) against
                              Wikipedia's NPL season articles — real squads, season
                              leaderboards, awards, and the one full real match
                              scorecard Wikipedia publishes per season.
    espncricinfo.py           Real fetch() implementation, disabled — verified HTTP 403
                              (bot-blocked) from this environment, see file docstring.
    cricbuzz.py                Real fetch() implementation, disabled — verified blocked
                              for automated/crawler access from this environment.
  data_collection.py        Orchestrates enabled sources per season -> data/raw/, plus
                             best-effort real-data enrichment (rosters/leaders/awards).
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

## Data Sourcing

**What's genuinely real today:** `src/data_sources/wikipedia.py` is a live, working scraper (requests + BeautifulSoup, tested against `en.wikipedia.org`) that pulls, for both NPL seasons:
* The real 8 franchise names, real season dates, and real venue.
* Real player rosters (~150 real player names across 8 teams), which feed directly into the simulator so every simulated player-match row carries a **real player name**, not a placeholder.
* Real season leaderboards (most runs, most wickets) and real award winners — surfaced verbatim in the dashboard's **Fun Facts** tab.
* The one full real match scorecard Wikipedia publishes per season (the final) — saved as `real_final_scorecards`, also viewable in Fun Facts.

**What's simulated and why:** per-phase (powerplay/middle/death), per-ball granularity for every league match isn't published anywhere scrapeable — it would require ball-by-ball provider access. `src/data_sources/simulator.py` fills that gap with statistically realistic data, anchored to real player names and team strength so that match outcomes stay causally linked to performance (verified by `tests/test_simulator.py`).

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

`tests/test_wikipedia.py` makes live requests to `en.wikipedia.org` and skips automatically if that's unreachable.

## A Note on "100% Accuracy"

Every team now resolves to a real roster with zero placeholder names (`master_dataset` has 0 rows matching the synthetic naming pattern, verified by `test_simulator_uses_real_names_for_every_configured_team_when_rosters_cover_them`), and every real-data table (`real_standings`, `real_head_to_head`, `real_leaders_runs/wickets`, `real_awards`) cross-checks consistent against the same 8 canonical team names. That's real-data *coverage*, and it's now complete.

What 100% accuracy *can't* mean here: the per-phase, per-match stats themselves (runs, wickets, economy by powerplay/middle/death) are still simulated, because no source available to this project publishes ball-by-ball data for NPL — that was an explicit, agreed tradeoff from the start (see "Data Sourcing" above), not an oversight. The simulator is anchored to real names, real captains, and real match outcomes' team strength, but the actual numbers in `batting_by_phase` / `bowling_by_phase` / etc. are statistically generated, not "the real strike rate". Anywhere this matters, the dashboard separates real (Fun Facts tab, `real_*` tables) from simulated (everything else) rather than blending them silently.

## Player Participation Realism

Real T20 leagues don't pick an independent random 11 every match — captains and regular starters play nearly every game; squad depth (the rest of an 18-21 player roster) mostly sits out unless rotated in. `src/data_sources/simulator.py` now models this: each team gets one **core XI per season** (anchored on the real captain from `WikipediaSource.fetch_captains()`), reused for every match with a small chance (0-2 players) of rotation. `analytics.player_stats_table()` exposes a `matches_played_pct` column so it's visible at a glance whether a given name is a near-every-match regular or genuinely a fringe player in the simulated data.

## Dashboard Sections

1. **Match Overview** — score summary, run progression, drill-down match insights
2. **Batting Analysis** — strike rate by phase, consistency ranking, best-strike-rate-by-phase leaderboard
3. **Bowling Analysis** — economy vs phase, wicket probability, highest wicket-taker against each opponent team
4. **Fielding Insights** — catch-drop leaderboard, best fielders ranking, runs lost to errors
5. **Player Insights** — impact score ranking, season-over-season change, a searchable **Player Stats Explorer** (with `matches_played_pct`), a **Player Profile** lookup (career line, best individual score, best bowling figures, impact-score rank), **All-Time Leaders** (cross-season cumulative runs/wickets/catches), and a **Player vs Player** match-level comparison tool
6. **Season Comparison** — totals, averages, and player performance deltas between seasons
7. **Fun Facts** — real, verified NPL facts: season leaders, **real all-time leaders** (combined across seasons' published top-5s), award winners, real final-match scorecards, real final standings, and a real **head-to-head lookup** (pick any two teams, see every real fixture result between them) — all sourced from `src/data_sources/wikipedia.py`

Both leaderboard-style insights qualify on a minimum-balls/representative-sample threshold so small samples can't top a leaderboard. The "Player vs Player" tool compares two players' stat lines in every match where both appeared — an honest match-level comparison, not a literal ball-by-ball duel, since the dataset is player-match-phase aggregates and doesn't track which specific ball a batter faced from which bowler.

All sections share sidebar filters: Season, Team, Player, Match, Phase.

## Constraints & Assumptions

* Catch drops / stumping misses are simulated where real data isn't available (`src/data_sources/simulator.py`); swap in real values once collected.
* Consistency is enforced across seasons via shared schema validation in `data_cleaning.py`.
* Missing values are filled (0 for unplayed metrics) rather than dropped, so the dataset stays complete; rows missing required identifiers are dropped and logged, never silently kept.

## Goal

A portfolio-level, professional sports analytics system comparable to ESPN-style dashboards, demonstrating data engineering, data analytics, visualization, and cricket domain understanding.
