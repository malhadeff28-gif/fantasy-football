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

### Open questions raised for the human

Recorded in the session; see "Open modelling decisions" in
`QB_PAIRING_RESEARCH.md` when written.

1. Backtest depth: 19 seasons of line coverage vs. NFL rule/era drift.
2. Whether the QB2-on-bye-week case should be modelled with waiver replacement.
3. Whether prediction markets stay in scope given they cannot be backtested.
