# Research Log

Chronological record of what was done, what was found, and what was decided.
Entries are append-only.

---

## 2026-08-17 — Phase 1: Prove the data foundation

**Goal:** verify real data sources against live endpoints (not from memory) and
walk one historical QB-week end to end using pregame-only information.

### Environment

Python 3.12.10, venv at `.venv`. Installed `nflreadpy` 0.1.5, `polars` 1.43.2,
`pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`,
`matplotlib`. `gh` CLI authenticated. No R.

### Findings

1. **`nfl_data_py` is deprecated.** nflverse has moved Python support to
   `nflreadpy` (Polars-based port of `nflreadr`). Adopted `nflreadpy`.

2. **Historical closing betting lines are free.** `load_schedules()` carries
   `spread_line`, `total_line`, moneylines and odds at **100% coverage for
   2007-2025**. This was the largest open risk in the brief (assumed dependency
   on a paid odds provider) and it is now closed. Backtest window set to
   **2007-2025, 19 seasons**.

3. **Spread sign convention verified empirically**, not assumed: positive
   `spread_line` = home favoured (corr +0.443 with home margin, 2015-2025).
   Wired into one function so it cannot drift.

4. **League scoring matches nflverse `fantasy_points` exactly** — 0.0 max
   absolute difference over 1,389 QB-weeks (2024-2025). No custom scorer needed.

5. **2026 forward lines stop after Week 7.** Weeks 14-16 have lines for only
   4 of 47 games. This is a structural constraint, not a data-fetching
   shortfall, and it forces a two-mode architecture (draft mode vs weekly mode).

6. **2026 ADP is live** at Fantasy Football Calculator: 1,415 real 12-team
   Non-PPR drafts, 2026-08-10 to 2026-08-17. Matches the league format exactly.

7. **CBS projections are publicly accessible** (60 QBs, 2026), satisfying the
   explicitly requested benchmark. FantasyPros ECR available and carries a
   `cbs_id` crosswalk column.

8. **Prediction markets are thinner than hoped.** Kalshi does have free
   QB-level series (`KXNFLPASSTDS`, `KXNFLPASSATT`) — the same data The Odds
   API charges $99/mo for — but they had **zero open markets** in August;
   they activate in-season. Polymarket's August NFL inventory is novelty
   markets. Neither has a free price history, so neither can be backtested.

### Built

- `src/qbpair/config.py` — league rules (12 team, 1QB/2RB/3WR/1TE/K/DST,
  7 bench, no flex), scoring, playoff weeks 14-16, backtest window.
- `src/qbpair/data/loaders.py` — source adapters with parquet caching, one
  function per provider so any source can be swapped.
- `src/qbpair/features/pregame.py` — pregame features behind a single
  `as_of_expr` primitive that encodes the no-look-ahead rule; game context
  (decomposed market inputs), QB rolling form, opponent QB defence shrunk
  toward league mean.
- `scripts/walk_one_qb_week.py` — the Phase 1 gate.

### Gate result

`scripts/walk_one_qb_week.py 2024 11 "Josh Allen"` reconstructs the QB-week
from pregame data only:

- context: BUF vs KC, spread +2.5, total 46.5, implied team total 24.50
- form (prior 8 games): 19.80 pts/game, 21.5% of points from rushing
- opponent prior: KC allowing 15.13 QB pts (shrunk), matchup adj -0.60
- naive projection **19.20** vs actual **23.98** (error -4.78)
- **leakage audit PASS** — latest game date used 2024-11-11 < kickoff 2024-11-17

Incidental validation: the first attempt (2024 Week 12) correctly returned no
row because Buffalo was on bye, confirming bye handling falls out of the
data model rather than needing a special case.

### Decisions taken

- `nflreadpy` over deprecated `nfl_data_py`.
- Backtest window 2007-2025 (line coverage, not arbitrary).
- Opponent priors shrunk toward league mean with k=6 games, so Week-2
  opponents cannot look elite on one game of evidence.
- Market inputs decomposed into implied team total + game closeness rather than
  used raw, per the brief's warning that a favourite is not automatically the
  better fantasy QB.

### Decisions surfaced to the human, and answered

Three choices were put to the user before being locked in.

1. **Backtest window → 2018-2025 (modern era only).** Chosen over the
   recommended 2011-2025. Accepted risk: 8 seasons is thin for interaction
   terms. Binding mitigation on Phase 4: archetype-level effects over per-QB
   coefficients, regularisation, out-of-sample reporting.
   `BACKTEST_SEASONS_EXTENDED` retained for robustness runs.

2. **Sportsbook props → adopted (The Odds API, $99/mo).** Adapter built and
   env-var gated; awaiting the user's subscription and `ODDS_API_KEY`.
   Constraint to respect: prop history starts 2023-05, covering only 3 of the
   8 backtest seasons, so the prop layer must be additive and optional.

3. **GitHub repo → private.** Created and pushed:
   <https://github.com/malhadeff28-gif/qb-pairing-lab>

### Still open

- Whether the QB2-on-bye-week case should be modelled with waiver replacement,
  or penalised flatly as the brief's default assumes.
- Regular-season vs playoff weighting — to be resolved by sensitivity analysis
  in Phase 6, per the brief, not chosen arbitrarily.

### Next

Phase 2: research what actually predicts QB fantasy outcomes. First test is
whether schedule-difficulty features carry real predictive signal *before*
they are given weight in the pairing model — the brief's own stated risk.

---

## 2026-08-17 — Phase 2: what actually predicts QB fantasy outcomes

**Headline: the schedule-complementarity premise does not survive contact with
the data. The second quarterback still has value, but for a different reason
than the thesis proposed.**

Built `features/panel.py` (vectorised as-of panel, 4,373 QB-weeks 2018-2025,
validated to 0.0 difference against the explicit-filter builder) and ran three
studies. All results walk-forward, train on seasons < S, test on S, S=2020-2025.

### Finding 1 — schedule difficulty adds almost nothing [evidence]

| model | MAE | pair accuracy |
|-------|-----|---------------|
| form only | 6.048 | 62.53% |
| form + schedule | 6.028 | 62.59% |
| form + market | 5.926 | **63.99%** |
| form + market + schedule | 5.916 | 63.98% |
| + QB x matchup interactions | 5.918 | 64.06% |

Opponent difficulty on top of recent form: **+0.06 percentage points** of
ranking accuracy. On top of the betting market: **−0.01 pp**, i.e. nothing.
The market fully subsumes it.

Replacing the naive "points allowed" rating with a QB-quality-controlled one
(mean actual minus form-expected) did not rescue it: MAE 6.023 vs 6.023, and
correlation with the form residual only 0.085.

### Finding 2 — the draft-day version is weaker still [evidence]

This is the test that matters, because in August there is no market and no
current-season defensive data. Only last season's defence is available.

- Team defence rating, season N-1 to season N: **r = 0.206**, so a defence
  explains 4.2% of its own next-season rating.
- Prior-season defence rating vs QB form residual: **r = 0.063**, explaining
  **0.39%** of the variance a drafter is trying to exploit.
- Easiest vs hardest third of prior-season matchups: **1.00 fantasy point per
  game**.

A whole tercile of schedule difficulty is worth one point per game, and the
ratings it is built on barely persist year to year.

### Finding 3 — QB-specific matchup interactions find no support [evidence]

Interactions of `rush_fp_share` with opponent difficulty, spread and implied
total made the model slightly *worse* (MAE +0.005, accuracy −0.01 pp).
`rush_fp_share` correlates −0.006 with the form residual. The brief flagged
this as a hypothesis to test rather than assume; it tested negative at this
level of granularity.

### Finding 4 — the market is the one real weekly signal [evidence]

Correlation with the form-only residual: implied team total **0.150**, game
total 0.114, home 0.071, indoors 0.070, opponent difficulty 0.068,
spread magnitude 0.017.

### Finding 5 — carrying QB2 is worth about +12 points per season [evidence]

Over 1,106 pairs, 2020-2025, against a single-QB manager who streams the
weekly median starter on his bye:

| strategy | pts/season | vs 1 QB |
|----------|-----------|---------|
| single QB1 + streamed bye | 314.8 | — |
| two QBs, model-selected | 326.9 | **+12.0** |
| two QBs, perfect foresight | 362.3 | +47.4 |

Only **25.4%** of the available prize is captured. Just **57.1%** of pairs beat
the one-QB baseline; median gain +2.8, quartiles −6.5 / +21.9. Season gains
ranged from −0.2 (2021) to +24.0 (2024).

### Finding 6 — the value is insurance, not complementarity [inference]

The highest-gain pairs are dominated by cases where QB1 collapsed or was hurt:
Tua/Stafford +155, Brady/Hurts +150, Minshew/Prescott +127. The benefit of a
second quarterback shows up mainly as protection against QB1 failing, not as
weekly matchup exploitation. That is a real and bankable effect — but it is a
different thesis, and notably it does *not* require schedule complementarity.

### What survives

The pairs above were selected with hindsight. Phase 2c measures the **average**
pair; the thesis needs a **selected** pair to do better. That remains open and
is exactly what Phase 5 should test. But the mechanism the selection was meant
to exploit — schedule complementarity — is the part the data does not support,
so any selection edge will have to come from somewhere else (QB1 fragility,
bye structure, or price).

### Built

- `features/panel.py`, `scripts/phase2_signal.py`,
  `scripts/phase2b_schedule_test.py`, `scripts/phase2c_pair_value.py`
- `research/METHODOLOGY.md`
- `outputs/phase2_model_comparison.csv`, `phase2_residuals.csv`,
  `phase2b_pair_structural_value.csv`, `phase2c_pair_value.csv`

### Environment note

OpenBLAS fails to allocate per-thread buffers on this machine at the default
thread count; every analysis script pins BLAS threads to 1 before importing
numpy.
