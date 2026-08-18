# Data Source Inventory

**Status:** Phase 1 complete. Every row below was verified against the live
endpoint on **2026-08-17**, not recalled from documentation. Where a claim is
inference rather than measurement it is labelled.

Verification entry points: `scripts/walk_one_qb_week.py`, `src/qbpair/data/loaders.py`.

---

## 1. Summary table

| # | Source | Covers | Access | Cost | Historical depth | Status |
|---|--------|--------|--------|------|------------------|--------|
| 1 | nflverse `load_schedules` | schedule, **closing betting lines**, venue, rest | `nflreadpy` | free | 1999-2026 | verified |
| 2 | nflverse `load_player_stats` | weekly box score + fantasy points | `nflreadpy` | free | 1999-2025 | verified |
| 3 | nflverse `load_ff_opportunity` | expected fantasy points (xFP) | `nflreadpy` | free | 2006-2025 | verified |
| 4 | nflverse `load_ff_rankings` | FantasyPros ECR consensus | `nflreadpy` | free | current scrape | verified |
| 5 | Fantasy Football Calculator | **2026 ADP from real drafts** | public REST | free | 2010s-2026 | verified |
| 6 | CBS Sports projections | season QB projections | public HTML | free | current | verified |
| 7 | Kalshi | game + QB prop markets | public REST | free | live only | verified, limited |
| 8 | Polymarket | NFL markets | public REST | free | live only | verified, weak |
| 9 | The Odds API | sportsbook props | REST + key | **$99/mo** | props from 2023-05 | **adopted**, key required |

**Headline result: the project can be built and backtested end to end at zero
data cost.** The one materially missing input is historical QB player props,
which the user elected to purchase (see §6.1).

---

## 2. The critical find: free historical betting lines

`load_schedules()` returns 7,548 games (1999-2026, 46 columns) and, decisively,
carries closing market data on every game:

`spread_line`, `total_line`, `away_moneyline`, `home_moneyline`,
`away_spread_odds`, `home_spread_odds`, `over_odds`, `under_odds`

Measured non-null coverage:

| Seasons | spread / total | moneyline / odds |
|---------|----------------|------------------|
| 2006 | 100% | 82% |
| **2007-2025** | **100%** | **100%** |
| 2026 (forward) | 41% | 41% |

This removes the assumed dependency on a paid odds provider for backtesting and
sets the backtest window at **2007-2025 (19 seasons)**.

**Sign convention, verified empirically rather than assumed:** `spread_line` is
positive when the **home** team is favoured (correlation with home margin =
**+0.443** over 2015-2025; mean home margin +5.61 when positive, -4.50 when
negative). Implied team total = `total_line / 2 + team_spread / 2`. This is
encoded once in `src/qbpair/features/pregame.py::team_game_context` so the
convention cannot drift.

### 2026 forward coverage is front-loaded, and this shapes the design

| Weeks | Games with lines |
|-------|------------------|
| 1-6 | 93 / 93 (100%) |
| 7 | 7 / 14 |
| 8-15 | 8 / 115 |
| **14-16 (fantasy playoffs)** | **4 / 47 (9%)** |

**Consequence (evidence-supported):** market data *cannot* drive the preseason
Weeks 14-16 evaluation the brief asks for, because the lines do not exist in
August. The playoff component must be built from schedule structure and
opponent priors, with market data entering only as an in-season refinement.

This splits the system into two operating modes, which the architecture should
reflect:

- **Draft mode (now):** schedule structure, opponent priors, ADP. Sparse markets.
- **Weekly mode (in-season):** full market data for start/sit decisions.

The backtest must respect the same split, or it will overstate what a manager
could actually have known on draft day.

---

## 3. Scoring: exact match confirmed

The league uses standard scoring with 4-point passing TDs. nflverse
`fantasy_points` was reconstructed from component stats across **1,389 QB-weeks
(2024-2025)**:

```
0.04/pass yd | 4/pass TD | -2/INT | 0.1/rush yd | 6/rush TD | -2/fumble lost | 2/2PC
```

**Maximum absolute difference: 0.0 across all 1,389 rows.** The league's scoring
and the dataset's built-in column are the same function, so no custom scorer is
required, though `Scoring` in `src/qbpair/config.py` keeps it explicit and
swappable for other formats.

Sanity check, 2025 standard-scoring QB leaders: Josh Allen 364.6 (16 games),
Drake Maye 351.0, Matthew Stafford 350.4, Trevor Lawrence 338.2.

---

## 4. Draft cost: 2026 data is already live

Fantasy Football Calculator's free API returns ADP from **real drafts** in the
exact league shape (12-team, Non-PPR):

- **1,415 drafts**, window **2026-08-10 to 2026-08-17**
- Per player: `adp`, `times_drafted`, `high`, `low`, `stdev`, `bye`

Current 2026 QB market (verified live):

| QB | ADP | Round | stdev |
|----|-----|-------|-------|
| Josh Allen | 21.5 | 2.10 | 10.9 |
| Drake Maye | 48.1 | 4.12 | 10.5 |
| Lamar Jackson | 52.2 | 5.04 | 10.0 |
| Joe Burrow | 54.5 | 5.06 | 9.6 |

The `stdev` field (roughly 10 picks) is itself useful: it quantifies how
reliably a target QB can actually be waited on, which the pairing engine needs
in order to recommend a *draftable* pair rather than a theoretical one.

Historical ADP is available by year (2025 verified), enabling backtests that
charge each strategy its true historical draft cost.

---

## 5. Benchmarks

**CBS Sports** (explicitly requested): 2026 season QB projections are public,
server-rendered HTML, **60 QBs** with projected fantasy points, passing yards
and passing TDs. Top of board: Allen 419.1, Jackson 378.7, Prescott 375.2,
Lawrence 371.3.

**FantasyPros ECR** via `load_ff_rankings()`: 5,849 rows scraped 2026-08-14,
including a `redraft-qb` page (98 QBs) with `ecr`, `sd`, `best`, `worst`.
Usefully it carries a `cbs_id` column, giving a ready-made crosswalk to CBS.

*Caution:* FantasyPros team codes differ from nflverse (`NEP` vs `NE`). A team
crosswalk is required before joining, or the join silently drops rows.
[measured]

---

## 6. Prediction markets: real, but narrower than hoped

**Kalshi** (`api.elections.kalshi.com`, public, no key required for reads). NFL
series confirmed live: `KXNFLGAME`, `KXNFLSPREAD`, `KXNFLTOTAL`, plus genuinely
QB-level series: `KXNFLPASSTDS`, `KXNFLPASSATT`, `KXNFLSEASONPASSYDS`,
`KXLEADERNFLPYDS`, `KXNFLMVP`.

That `KXNFLPASSTDS` exists and is free is notable: passing-TD props are exactly
what The Odds API gates behind $99/month.

**However**, queried 2026-08-17, `KXNFLPASSTDS` returned **zero open markets**.
These series activate during the regular season. Game-level series currently
list only preseason games.

**Polymarket** (`gamma-api.polymarket.com`, public). The `nfl` tag returns
mostly novelty and season-long markets in August ("Tush Push banned for 2026?",
"Will Aaron Rodgers retire?", "Chiefs Week 1 starting QB?").

**Assessment [reasonable inference]:** for *game-level* information both venues
are largely redundant with the sportsbook lines already free in source #1. Their
distinct value is QB-level markets on Kalshi during the season. Neither offers a
free historical price archive, so **neither can be backtested**; they can only
be collected prospectively. Per the brief's instruction to include market data
only where it genuinely contributes, prediction markets should be treated as an
in-season enrichment to be validated forward, not a modelling input the thesis
depends on.

---

### 6.1 Decisions taken 2026-08-17

**Backtest window: 2018-2025 (modern rushing-QB era).** Line coverage would
permit 2007+, but the user elected the narrower, more representative window.
Recorded risk: 8 seasons is thin for fitting QB-characteristic x opponent
interaction terms, and only ~5 are 17-game seasons. Mitigations are binding on
Phase 4 — prefer archetype-level effects over per-QB coefficients, regularise,
and report out-of-sample performance rather than in-sample fit.
`BACKTEST_SEASONS_EXTENDED` (2007-2025) remains available as a robustness check.

**Sportsbook props: adopted.** The Odds API Business tier ($99/mo) is the only
source of historical NFL player props, available from 2023-05-03. Note this
covers **2023, 2024, 2025 — three of the eight backtest seasons**, so prop
features must be built as an *additive* layer that the model can run without,
not as a core input, or five seasons of backtest become unusable.

Setup required from the user (not performed automatically — this is a paid
subscription):

1. Subscribe at <https://the-odds-api.com/> (Business tier for NFL props).
2. Put the key in the environment as `ODDS_API_KEY`, or a local `.env`
   (already gitignored). Never commit it.
3. Verify with `loaders.odds_api_status()`, which reports quota headers and
   never echoes the key.

Adapter: `src/qbpair/data/loaders.py` — `list_events`, `event_props`, with
`date_iso` snapshot support so a backtest can request only prices published
before kickoff.

**Prediction markets: forward-only.** Kalshi collection is retained to build a
price history prospectively, but nothing in the thesis will depend on it.

---

## 7. Gaps and limitations

| Gap | Impact | Handling |
|-----|--------|----------|
| **No free historical QB props** | Cannot backtest prop-based features; the "QB passing props" thread is forward-only | Backtest on game-level markets; collect Kalshi props prospectively from Week 1 |
| **2026 lines stop at Week 7** | Playoff-week market view unavailable at draft time | Playoff component from schedule + priors; markets refine in-season |
| No Kalshi/Polymarket price history | Prediction-market value unprovable historically | Log prices weekly starting Week 1 |
| `temp`/`wind` 38-75% non-null | Weather features unreliable, especially 2022+ | Use `roof`/`indoors` (100% populated) as the robust proxy |
| FantasyPros team codes differ | Silent join failures | Explicit crosswalk before joining |
| ADP window is a snapshot | ADP moves through August | Timestamp every pull (`adp_window_start`/`end` retained) |
| Paid props ($99/mo) | Only if free sources prove inadequate | Revisit after Phase 2 measures game-level market value |

---

## 8. Additional nflverse datasets available (not yet verified in depth)

`load_injuries`, `load_depth_charts`, `load_snap_counts`, `load_nextgen_stats`,
`load_pfr_advstats`, `load_pbp`, `load_participation`, `load_rosters_weekly`,
`load_players`, `load_teams`, `load_ftn_charting`.

`load_depth_charts` and `load_injuries` are the likely mechanism for the brief's
"QB injury or uncertain starter" edge case, to be verified in Phase 2.

---

## 9. Reproducibility

All sources are wrapped in `src/qbpair/data/loaders.py`; each pull is cached to
`data/raw/*.parquet` so a backtest reruns identically even if an upstream feed
changes. Environment: Python 3.12.10, `nflreadpy` 0.1.5, `polars` 1.43.2.

Note: `nfl_data_py` is **deprecated** and should not be used; `nflreadpy` is its
maintained successor (Polars-based).
