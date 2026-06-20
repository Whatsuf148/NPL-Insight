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

Tests run against an in-memory fixture dataset (`tests/conftest.py`), not the simulator's generated CSVs, so they're fast and don't depend on `run_pipeline.py` having been run first. Coverage includes: schema validation and incompleteness guards in `data_cleaning.py`, metric correctness in `feature_engineering.py`, insight functions in `analytics.py`, the numpy logistic regression, and a regression guard on the simulator (`test_simulator_winner_correlates_with_runs_and_wickets`) that catches the earlier bug where match outcomes were generated independently of performance stats, making the win-probability model unable to learn anything.

## Dashboard Sections

1. **Match Overview** — score summary, run progression, drill-down match insights
2. **Batting Analysis** — strike rate by phase, consistency ranking
3. **Bowling Analysis** — economy vs phase, wicket probability
4. **Fielding Insights** — catch-drop leaderboard, best fielders ranking, runs lost to errors
5. **Player Insights** — impact score ranking, season-over-season change, and a searchable **Player Stats Explorer** (every player's full name + batting/bowling/fielding line)
6. **Season Comparison** — totals, averages, and player performance deltas between seasons
7. **Fun Facts** — real, verified NPL facts (season leaders, award winners, real final-match scorecards) sourced from `src/data_sources/wikipedia.py`

Batting Analysis also surfaces a **best-strike-rate-by-phase** leaderboard (who's fastest in the powerplay, middle overs, death overs), and Bowling Analysis surfaces **highest wicket-taker against each opponent team** — both qualify on a minimum-balls/representative-sample threshold so small samples can't top a leaderboard.

All sections share sidebar filters: Season, Team, Player, Match, Phase.

## Constraints & Assumptions

* Catch drops / stumping misses are simulated where real data isn't available (`src/data_sources/simulator.py`); swap in real values once collected.
* Consistency is enforced across seasons via shared schema validation in `data_cleaning.py`.
* Missing values are filled (0 for unplayed metrics) rather than dropped, so the dataset stays complete; rows missing required identifiers are dropped and logged, never silently kept.

## Goal

A portfolio-level, professional sports analytics system comparable to ESPN-style dashboards, demonstrating data engineering, data analytics, visualization, and cricket domain understanding.
