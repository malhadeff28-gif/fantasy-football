# QB-018 — Do advanced QB variables beat box-score form?

**Question.** Do pressure, blitz, sack rate, time to throw, aggressiveness,
scrambling and red-zone rushing add predictive value for weekly QB fantasy
points *beyond* ordinary box-score form?

**Verdict: REJECT.** Twenty-two as-of charting and tracking features, joined to
96–99% of the panel, move out-of-sample start/sit accuracy by **−0.13 pp**
(95% CI [−0.52, +0.23]) and MAE by **+0.0045** points. No block helps. No
single variable helps. A gradient-boosted model finds nothing linear ridge
missed. Several of these variables are genuinely stable quarterback traits —
they are simply already priced into the eight-game form average the model
already has.

Run: `./.venv/Scripts/python.exe qb-pairing/scripts/wave1_advanced_qb_vars.py`
Date: 2026-08-18. Outputs: `qb-pairing/outputs/qb018_*.csv`.

---

## 1. What data actually exists

Both sources were probed, not assumed. Neither goes back as far as box scores,
and both are keyed differently from the panel.

### 1.1 PFR advanced passing — `load_pfr_advstats(seasons, stat_type="pass")`

**Coverage: 2018–2025 only.** The loader raises
`ValueError: Season must be between 2018 and 2025` for 2017 and for 2026.
The box-score panel could run from 2007 (line coverage) or 1999 (stats); PFR
cannot. **2007–2017 cannot be covered by any pressure/blitz feature.** Since
the project's backtest window is already 2018–2025 by user decision, this
costs nothing *here*, but it hard-caps any future extension of the window.

5,424 rows (2018–2025, all game types), 24 columns, ~640–706 rows per season.
One row per **player**-game, not per QB-game: receivers appear with drop stats
and zeros elsewhere. Keyed by `pfr_player_id`, **not** `gsis_id`.

| Column | Type | Notes |
|---|---|---|
| `game_id`, `pfr_game_id`, `season`, `week`, `game_type`, `team`, `opponent` | str/int | `game_type` ∈ REG, WC, DIV, CON, SB |
| `pfr_player_name`, `pfr_player_id` | str | PFR id, e.g. `FoleNi00` |
| `times_pressured`, `times_pressured_pct` | float | 0% null |
| `times_blitzed`, `times_hurried`, `times_hit`, `times_sacked` | float | 0% null |
| `passing_drops`, `passing_drop_pct` | float | 0% null |
| `passing_bad_throws`, `passing_bad_throw_pct` | float | 0% null |
| `receiving_drop`, `receiving_drop_pct` | float | **100% null** — unusable |
| `def_times_blitzed`, `def_times_hurried`, `def_times_hitqb` | float | **100% null** — unusable |

There are **no scramble columns** in this table, contrary to the loader's
docstring ("pressure, blitz, hurries, scrambles"). Scrambles had to come from
play-by-play instead (§1.3).

The dropback denominator was recovered rather than guessed: 2018 wk 1 Nick
Foles has 9 pressures at `times_pressured_pct` 0.250 → 36 dropbacks = 34
attempts + 2 sacks. So **dropbacks = attempts + sacks**, and every rate below
uses that denominator from the box score.

### 1.2 Next Gen Stats — `load_nextgen_stats(seasons, stat_type="passing")`

**Coverage: 2016–2025**, so two seasons deeper than PFR but still not to 2007.
5,933 rows over 2016–2025, 29 columns. Keyed by `player_gsis_id` — joins to
the panel directly.

Two traps, both real:

1. **`week == 0` rows are season aggregates, not week zero.** 38–45 such rows
   per season (407 total). Including them would silently mix season totals
   into a weekly panel. Filtered out.
2. **NGS publishes only qualified passers: the minimum `attempts` on any
   weekly row is 15.** The panel's inclusion rule is ≥ 10 attempts, so
   10–14-attempt starts have no NGS row at all and never will.

Weekly REG rows: 5,285 over 2016–2025 (512–549 per season). POST rows exist
(20–26/season) and are excluded.

Usable columns (all ≤ 0.1% null): `avg_time_to_throw`, `aggressiveness`,
`avg_intended_air_yards`, `avg_completed_air_yards`,
`avg_air_yards_differential`, `avg_air_yards_to_sticks`,
`expected_completion_percentage`, `completion_percentage_above_expectation`,
`avg_air_distance`, `max_air_distance`, `max_completed_air_distance`,
plus box-score repeats (`attempts`, `pass_yards`, `passer_rating`, …).

### 1.3 Play-by-play — scrambles and red-zone rushing

Neither source carries scrambles or field-position rushing, so both were
derived from nflverse play-by-play, 2017–2025.

**Memory.** PBP is 372 columns and ~47k rows per season; loading a season whole
is not affordable at ~1.8 GB free. Each season's parquet is fetched once and
read with **column pruning — 11 of 372 columns, ~4.5 MB per season** —
aggregated to one row per QB-game immediately, then released. The aggregate is
cached to `shared/nfl-data/cache/processed/qb018_pbp_rush_2017_2025.parquet`,
so a rerun costs nothing. Peak footprint stays under ~50 MB.

Scramble encoding was verified rather than assumed: of 874 `qb_scramble == 1`
plays in 2018, 822 are `play_type == 'run'` with `rush_attempt == 1` and a
populated `rusher_player_id`; 49 are `no_play` (penalties) and 3 are `pass`.
Only the 822-type rows are counted.

### 1.4 Coverage summary

| Source | Seasons available | Seasons NOT coverable | Key | Weekly granularity |
|---|---|---|---|---|
| Box score (`load_player_stats`) | 1999–2025 | — | `gsis_id` | yes |
| PFR advanced passing | **2018–2025** | 1999–2017, 2026 | `pfr_player_id` | yes |
| Next Gen Stats passing | **2016–2025** | 1999–2015 | `gsis_id` | yes, ≥ 15 att only |
| Play-by-play | 1999–2025 | — | `gsis_id` | yes |

---

## 2. Features built

All 22 features follow `features/panel.py` exactly: every rolling statistic is
computed on `.shift(1)` over `player_id`, so **a game can never enter its own
feature.** Window = 8 games (`FORM_WINDOW`) for QB-side features, 17
(`DEF_WINDOW`) for the opponent-defence features, matching the baseline panel.
History is restricted to ≥ 10-attempt games, as the baseline form panel does,
so a two-throw relief cameo cannot swing a rate.

Rates are **ratios of prior sums**, not means of per-game ratios: prior-8-game
Σnumerator / Σdenominator. A mean of per-game ratios is dominated by
small-denominator games.

| Block | Features |
|---|---|
| **Pressure** (PFR, 7) | `adv_pressure_rate`, `adv_blitz_rate`, `adv_hurry_rate`, `adv_qbhit_rate`, `adv_sack_rate`, `adv_press_to_sack` (sacks ÷ pressures), `adv_bad_throw_rate` |
| **NGS tracking** (7) | `adv_time_to_throw`, `adv_aggressiveness`, `adv_intended_air_yards`, `adv_air_yards_diff`, `adv_air_yards_to_sticks`, `adv_xcomp`, `adv_cpoe_ngs` — all attempt-weighted |
| **Scramble / red-zone rush** (5) | `adv_scramble_rate` (÷ dropbacks incl. scrambles), `adv_rz_rush_pg`, `adv_rz_rush_share` (QB ÷ team rushes inside the 20), `adv_gl_rush_pg` (inside the 5), `adv_designed_rush_pg` |
| **Opponent defence** (3) | `opp_def_pressure_rate`, `opp_def_blitz_rate`, `opp_def_sack_rate` — the defence's own prior-17-game pressure profile |

### Leakage audit

`adv_pressure_rate` was recomputed for 40 random panel rows the slow, obvious
way — take that quarterback's previous 8 games strictly earlier than this one,
divide summed pressures by summed dropbacks — and compared to the vectorised
shift-then-roll output.

**max |vectorised − explicit| = 0.00e+00.** The assertion is in the script and
fails the run if it is ever violated. This mirrors
`panel.validate_against_pregame()`.

---

## 3. Join rates — honestly

The `pfr_player_id` → `gsis_id` crosswalk comes from `nflreadpy.load_players()`
and is **100% complete**: all 5,189 PFR regular-season rows resolve to a gsis
id, and every one of the 4,373 panel QB-weeks has a matching PFR game row.

The join rates below are lower than 100% not because of matching failure but
because a **rolling feature needs prior games**: a quarterback's first start in
the window has no eight-game history to roll.

| Feature block | Rows matched | Panel rows | Join rate |
|---|---|---|---|
| Pressure (7 features) | 4,223 | 4,373 | **96.57%** |
| NGS tracking (7) | 4,253 | 4,373 | **97.26%** |
| Scramble / RZ rush (4) | 4,268 | 4,373 | **97.60%** |
| `adv_rz_rush_share` | 4,252 | 4,373 | 97.23% |
| Opponent defence (3) | 4,340 | 4,373 | **99.25%** |
| **All 22 present + baseline present** | **4,195** | **4,373** | **95.93%** |

Per-season coverage (panel rows / with pressure / with NGS / with rushing):

| Season | Panel | Pressure | NGS | Rushing |
|---|---|---|---|---|
| 2018 | 511 | 455 | 492 | 493 |
| 2019 | 507 | 488 | 492 | 492 |
| 2020 | 534 | 520 | 518 | 521 |
| 2021 | 563 | 548 | 547 | 550 |
| 2022 | 557 | 544 | 538 | 544 |
| 2023 | 572 | 560 | 558 | 560 |
| 2024 | 568 | 559 | 559 | 559 |
| 2025 | 561 | 549 | 549 | 549 |

2018 pressure coverage (455/511, 89%) is the worst cell and the reason is
structural: **PFR starts in 2018, so 2018 has no prior season of charting to
roll from.** NGS reaches back to 2016 and does not suffer this. 178 rows
(4.1%) are lost overall — first starts, plus 10–14-attempt games NGS never
publishes.

**Sensitivity to the row set.** The baseline scores 64.04% pairwise accuracy on
the full 4,268-row panel and 63.84% on the restricted 4,195-row set. The
0.20 pp gap is the row set, not the features; every model below is scored on
the identical restricted set, so the comparison is apples to apples.
(64.04% reproduces the 64.07% quoted in `QB_PAIRING_RESEARCH.md` §8.)

---

## 4. The increment: walk-forward, train seasons < S, test S (S = 2020–2025)

RidgeCV over 24 alphas, features standardised on the training fold only.
48,162 within-week pairwise comparisons per model.

| Model | k | MAE | RMSE | R² | Pair acc | Δ MAE | Δ acc |
|---|---|---|---|---|---|---|---|
| **baseline (9 box+market)** | 9 | **5.9216** | 7.4024 | 0.1713 | **63.84%** | — | — |
| baseline + pressure | 16 | 5.9160 | 7.4030 | 0.1712 | 63.88% | −0.0056 | **+0.04 pp** |
| baseline + NGS tracking | 16 | 5.9214 | 7.4077 | 0.1702 | 63.73% | −0.0002 | −0.11 pp |
| baseline + scramble/RZ rush | 14 | 5.9300 | 7.4248 | 0.1663 | 63.74% | +0.0084 | −0.10 pp |
| baseline + opp def pressure | 12 | 5.9199 | 7.4071 | 0.1703 | 63.71% | −0.0017 | −0.13 pp |
| **baseline + ALL advanced** | 31 | 5.9261 | 7.4281 | 0.1656 | **63.71%** | +0.0045 | **−0.13 pp** |
| advanced only (no box score) | 22 | 6.2487 | 7.7963 | 0.0808 | 60.39% | +0.3272 | −3.45 pp |

### Per-season spread — pairwise accuracy (%)

| Model | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| baseline | 64.09 | 63.19 | 66.63 | 64.36 | 64.09 | 60.77 | 63.85 | 1.90 |
| + pressure | 64.16 | 63.53 | 66.64 | 64.32 | 64.35 | 60.34 | 63.89 | 2.04 |
| + NGS | 64.35 | 63.02 | 65.69 | 64.19 | 64.29 | 60.92 | 63.74 | 1.62 |
| + scramble/RZ | 63.61 | 63.14 | 66.41 | 64.52 | 64.00 | 60.78 | 63.74 | 1.84 |
| + opp def | 63.61 | 62.88 | 66.37 | 64.42 | 64.12 | 60.91 | 63.72 | 1.81 |
| + ALL advanced | 64.01 | 63.21 | 65.70 | 64.41 | 64.50 | 60.43 | 63.71 | 1.80 |
| advanced only | 61.14 | 59.86 | 61.36 | 59.64 | 61.85 | 58.52 | 60.40 | 1.26 |

**The season-to-season spread (sd ≈ 1.9 pp, range 60.8%–66.6%) is fourteen
times the largest effect any advanced block produces.** Anyone quoting a
single-season improvement from these variables is quoting noise.

### Per-season spread — MAE

| Model | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| baseline | 6.368 | 6.007 | 5.203 | 5.738 | 6.033 | 6.193 | 5.924 | 0.410 |
| + pressure | 6.375 | 5.990 | 5.192 | 5.736 | 6.021 | 6.197 | 5.918 | 0.416 |
| + ALL advanced | 6.348 | 6.017 | 5.277 | 5.734 | 5.996 | 6.200 | 5.929 | 0.381 |

### Paired block bootstrap over weeks (500 resamples)

Weeks are resampled with replacement; both models are scored on an identical
comparison set, so the deltas are paired.

| Model | Δ pair acc | 95% CI | P(better) | Δ MAE | 95% CI |
|---|---|---|---|---|---|
| baseline + pressure | +0.04 pp | [−0.16, +0.23] | 0.61 | −0.0056 | [−0.0166, +0.0067] |
| baseline + NGS tracking | −0.11 pp | [−0.47, +0.22] | 0.28 | −0.0002 | [−0.0215, +0.0201] |
| baseline + scramble/RZ rush | −0.10 pp | [−0.31, +0.10] | 0.18 | +0.0084 | [−0.0070, +0.0242] |
| baseline + opp def pressure | −0.13 pp | [−0.28, +0.01] | 0.03 | −0.0017 | [−0.0104, +0.0062] |
| baseline + ALL advanced | −0.13 pp | [−0.52, +0.23] | 0.23 | +0.0045 | [−0.0214, +0.0317] |
| advanced only | −3.45 pp | [−4.46, −2.49] | 0.00 | +0.3272 | [+0.2402, +0.4203] |

Every interval straddles zero except "advanced only", which is decisively
**worse**. The best block, pressure, is a +0.04 pp point estimate with a
0.39 pp-wide interval and P(better) = 0.61 — a coin flip.

### 4b. Robustness — is the null a linear-model artefact?

Ridge can only miss a nonlinear signal. Same walk-forward, LightGBM
(300 trees, 7 leaves, lr 0.03, min_child 60, λ 5 — deliberately small for 4k
training rows):

| Model (LightGBM) | MAE | RMSE | Pair acc |
|---|---|---|---|
| baseline (9) | 5.9964 | 7.5120 | 63.04% |
| baseline + ALL advanced (31) | 6.0293 | 7.5655 | 62.54% |

Advanced increment under GBM: **−0.50 pp** [−1.09, +0.10], P(better) = 0.05,
Δ MAE +0.0329. The nonlinear learner finds nothing the linear one missed, and
is itself 0.80 pp worse than ridge — consistent with the project's existing
finding that complexity does not earn its place on 8 seasons of QB-weeks.

---

## 5. Which individual advanced variables carry signal

Two independent readings. **`r resid`** = correlation with the baseline model's
out-of-sample residual (what the model still gets wrong). **`+1 Δacc`** =
change in walk-forward pairwise accuracy when that one variable is added to
the deployed nine-feature model. Residual rows: 3,255 (test seasons only).

| Variable | r resid | 95% CI | var % | +1 Δacc | +1 ΔMAE |
|---|---|---|---|---|---|
| `adv_air_yards_diff` | **−0.0423** | [−0.077, −0.008] | 0.18 | −0.04 pp | −0.0101 |
| `adv_aggressiveness` | −0.0338 | [−0.068, +0.001] | 0.11 | +0.07 pp | +0.0026 |
| `adv_scramble_rate` | +0.0326 | [−0.002, +0.067] | 0.11 | +0.05 pp | −0.0030 |
| `adv_time_to_throw` | +0.0258 | [−0.009, +0.060] | 0.07 | −0.03 pp | −0.0022 |
| `adv_gl_rush_pg` | −0.0223 | [−0.057, +0.012] | 0.05 | −0.15 pp | +0.0129 |
| `adv_air_yards_to_sticks` | +0.0219 | [−0.012, +0.056] | 0.05 | −0.02 pp | −0.0013 |
| `adv_hurry_rate` | +0.0197 | [−0.015, +0.054] | 0.04 | −0.02 pp | −0.0038 |
| `adv_intended_air_yards` | +0.0174 | [−0.017, +0.052] | 0.03 | +0.03 pp | −0.0009 |
| `adv_sack_rate` | +0.0147 | [−0.020, +0.049] | 0.02 | +0.02 pp | −0.0019 |
| `adv_qbhit_rate` | −0.0140 | [−0.048, +0.020] | 0.02 | +0.04 pp | −0.0005 |
| `adv_pressure_rate` | +0.0112 | [−0.023, +0.046] | 0.01 | −0.02 pp | −0.0022 |
| `adv_blitz_rate` | +0.0111 | [−0.023, +0.045] | 0.01 | +0.02 pp | −0.0007 |
| `opp_def_blitz_rate` | +0.0058 | [−0.029, +0.040] | 0.00 | −0.08 pp | +0.0017 |
| `adv_bad_throw_rate` | −0.0056 | [−0.040, +0.029] | 0.00 | +0.06 pp | −0.0009 |
| `opp_def_pressure_rate` | −0.0055 | [−0.040, +0.029] | 0.00 | +0.01 pp | −0.0012 |
| `adv_rz_rush_pg` | +0.0049 | [−0.029, +0.039] | 0.00 | +0.07 pp | +0.0001 |
| `adv_press_to_sack` | +0.0041 | [−0.030, +0.038] | 0.00 | +0.02 pp | +0.0017 |
| `adv_designed_rush_pg` | −0.0039 | [−0.038, +0.030] | 0.00 | +0.03 pp | +0.0010 |
| `opp_def_sack_rate` | +0.0034 | [−0.031, +0.038] | 0.00 | −0.02 pp | +0.0000 |
| `adv_rz_rush_share` | −0.0031 | [−0.037, +0.031] | 0.00 | +0.00 pp | +0.0013 |
| `adv_xcomp` | −0.0025 | [−0.037, +0.032] | 0.00 | −0.02 pp | −0.0011 |
| `adv_cpoe_ngs` | −0.0004 | [−0.035, +0.034] | 0.00 | +0.03 pp | +0.0004 |

**Reading.** Exactly one variable, `adv_air_yards_diff` (intended minus
completed air yards), has a residual CI excluding zero — and with 22 variables
scanned, roughly one such result is expected at the 5% level by chance. It
explains **0.18% of residual variance**, and when added alone it *lowers*
pairwise accuracy by 0.04 pp. Best single addition by accuracy is
`adv_aggressiveness` at **+0.07 pp [−0.05, +0.20], P(better) = 0.87** — the
maximum of 22 draws, which is exactly what a null looks like.

Compare with the market variable the model already uses: `implied_team_total`
correlates with the form-only residual at **r = 0.150** (Phase 2). The best
advanced variable here is a quarter of that, and unlike the market it does not
survive into the model.

Note the pattern in the top rows: `air_yards_diff`, `aggressiveness`,
`time_to_throw`, `intended_air_yards` and `air_yards_to_sticks` are all
measuring the same underlying thing — *how far downfield and how riskily he
throws*. The panel already carries `form_adot` from box-score air yards. The
tracking data is re-describing a dimension the box score had.

---

## 6. Stable trait vs noisy outcome

Season N value against the same unit's season N+1 value; ratios of season sums
(not means of per-game ratios); ≥ 8 games in both seasons; Fisher-z 95%
intervals. 170 QB-season pairs, 224 team-season pairs.

### Quarterback level

| Variable | r(N, N+1) | 95% CI | Verdict |
|---|---|---|---|
| `designed_rush_per_game` | **0.892** | [+0.86, +0.92] | **STABLE TRAIT** |
| `scramble_rate` | **0.788** | [+0.72, +0.84] | **STABLE TRAIT** |
| `rz_rush_per_game` | **0.785** | [+0.72, +0.84] | **STABLE TRAIT** |
| `rz_rush_share` | **0.726** | [+0.65, +0.79] | **STABLE TRAIT** |
| `time_to_throw` | **0.660** | [+0.57, +0.74] | **STABLE TRAIT** |
| `qbhit_rate_faced` | **0.628** | [+0.53, +0.71] | **STABLE TRAIT** |
| `gl_rush_per_game` | 0.592 | [+0.48, +0.68] | STABLE TRAIT |
| *`fantasy_points_per_game`* | *0.555* | *[+0.44, +0.65]* | *(reference)* |
| `pressure_rate_faced` | 0.526 | [+0.41, +0.63] | semi-stable |
| `bad_throw_rate` | 0.494 | [+0.37, +0.60] | semi-stable |
| `xcomp` | 0.480 | [+0.36, +0.59] | semi-stable |
| `sack_rate_taken` | 0.480 | [+0.35, +0.59] | semi-stable |
| `intended_air_yards` | 0.476 | [+0.35, +0.58] | semi-stable |
| `cpoe_ngs` | 0.454 | [+0.33, +0.57] | semi-stable |
| `air_yards_diff` | 0.442 | [+0.31, +0.56] | semi-stable |
| `aggressiveness` | 0.424 | [+0.29, +0.54] | semi-stable |
| `blitz_rate_faced` | 0.406 | [+0.27, +0.52] | semi-stable |
| `hurry_rate_faced` | 0.403 | [+0.27, +0.52] | semi-stable |
| `air_yards_to_sticks` | 0.369 | [+0.23, +0.49] | semi-stable |
| `press_to_sack` | **0.277** | [+0.13, +0.41] | **NOISY OUTCOME** |

### Team (offence) level — the line, or the quarterback?

| Variable | r(N, N+1) | 95% CI |
|---|---|---|
| `designed_rush_per_game` | 0.719 | [+0.65, +0.78] |
| `rz_rush_per_game` | 0.625 | [+0.54, +0.70] |
| `scramble_rate` | 0.569 | [+0.47, +0.65] |
| `rz_rush_share` | 0.552 | [+0.45, +0.64] |
| `gl_rush_per_game` | 0.481 | [+0.37, +0.58] |
| `time_to_throw` | 0.462 | [+0.35, +0.56] |
| *`fantasy_points_per_game`* | *0.451* | *[+0.34, +0.55]* |
| `aggressiveness` | 0.408 | [+0.29, +0.51] |
| `sack_rate_taken` | 0.375 | [+0.26, +0.48] |
| `blitz_rate_faced` | 0.366 | [+0.25, +0.47] |
| `intended_air_yards` | 0.363 | [+0.24, +0.47] |
| `air_yards_to_sticks` | 0.356 | [+0.24, +0.47] |
| `cpoe_ngs` | 0.347 | [+0.23, +0.46] |
| `qbhit_rate_faced` | 0.333 | [+0.21, +0.44] |
| `air_yards_diff` | 0.333 | [+0.21, +0.44] |
| `xcomp` | 0.320 | [+0.20, +0.43] |
| `pressure_rate_faced` | 0.320 | [+0.20, +0.43] |
| `bad_throw_rate` | 0.308 | [+0.18, +0.42] |
| `hurry_rate_faced` | 0.297 | [+0.17, +0.41] |
| `press_to_sack` | 0.290 | [+0.17, +0.41] |

### What this says

1. **The stated hypothesis is backwards.** The brief guessed that pressure rate
   faced would be a stable offensive-line property while sacks taken would be
   noisier. The data says the opposite ordering at team level: *pressure rate
   faced* carries r = 0.32 team-to-team while *sack rate taken* carries 0.375;
   and at quarterback level pressure faced (0.526) and sacks taken (0.480) are
   statistically indistinguishable. **Pressure faced travels with the
   quarterback more than with the team** (0.526 vs 0.320) — it is largely a
   property of how long he holds the ball, not of the five men in front of him.
   `time_to_throw` is itself far more stable at QB level (0.660) than at team
   level (0.462), which is the same story told directly.

2. **`press_to_sack` is the one outright noisy outcome** (QB 0.277, team 0.290)
   — the most obviously "clutch-sounding" variable in the set is the least
   reliable, as usual.

3. **The rushing block is the most stable thing measured, on either unit** —
   `designed_rush_per_game` r = 0.892 at QB level, higher than fantasy points
   per game itself (0.555). Whether a quarterback is a designed runner is close
   to a fixed attribute. It still adds nothing, because the box-score panel
   already carries `form_rush_att` and `rush_fp_share`, which encode the same
   fact from cheaper data.

4. **Stability is necessary, not sufficient.** Six variables clear r = 0.59 and
   none of them helps. High season-to-season stability means a variable is
   *measurable*; it says nothing about whether it is *incremental*.

---

## 7. Verdict

**REJECT.**

Advanced charting and tracking variables do not add predictive value for weekly
QB fantasy points beyond box-score form:

- best block (+pressure): **+0.04 pp** accuracy, 95% CI [−0.16, +0.23],
  P(better) = 0.61;
- all 22 together: **−0.13 pp**, 95% CI [−0.52, +0.23], Δ MAE +0.0045;
- best single variable: **+0.07 pp**, 95% CI [−0.05, +0.20], and it is the
  maximum of 22 scanned candidates;
- nonlinear check (LightGBM): **−0.50 pp**, so it is not a linear-model
  artefact;
- advanced-only model is **3.45 pp worse** than the box-score baseline — the
  charting data is not a substitute either.

Against a season-to-season accuracy sd of 1.90 pp, none of these is
distinguishable from zero.

**The mechanism behind the null.** These variables are not noise — most are
strongly stable traits. They fail because they are *redundant*. Pressure,
blitz, time to throw and aggressiveness describe how a quarterback plays; that
already shows up in the eight-game rolling mean of what he actually scored,
which the model already has. Downfield-aggression tracking duplicates
`form_adot`; designed-rush and red-zone-rush usage duplicate `form_rush_att`
and `rush_fp_share`. The box score is a lossy but sufficient summary of the
charting data *for this target*.

This is the same shape as QB-002 through QB-005 (opponent difficulty) and
QB-007 (per-QB matchup effects): a plausible mechanism that is real as a
description and empty as a prediction. Weekly QB scoring remains dominated by
irreducible variance.

**Do not add these features to `models/projection.py`.**

### What would change the verdict

- A **target other than total fantasy points** — sack rate and pressure plainly
  belong in a *floor* or downside-risk model even if they do not move the mean.
  Untested here.
- **Within-season injury/line-change events** (a left tackle lost in week 4)
  that a rolling 8-game average smooths away. Would need a different feature
  construction, not different data.
- **Opponent-specific pressure matchups** at a granularity below team-season.
  QB-002/QB-007 already argue against this working.

### Limitations

- PFR advanced stats do not exist before 2018, so no pressure/blitz feature can
  ever be built for 2007–2017. Any future extension of the backtest window
  carries this hole.
- NGS never publishes 10–14-attempt games, so 152 panel rows (3.5%) are
  structurally unreachable by any tracking feature.
- 2018 has no prior season of PFR history, so its pressure features are built
  from a shorter window than later seasons (455/511 rows covered).
- Rolling windows are fixed at 8 games to match the baseline; no window search
  was performed, and a search over windows would need a holdout to be honest.
- 170 QB-season pairs is thin for the stability estimates; the Fisher intervals
  are given for exactly that reason.

---

## Artefacts

| File | Contents |
|---|---|
| `outputs/qb018_source_schemas.csv` | Full probed schemas, dtypes, null rates |
| `outputs/qb018_join_rates.csv` | Per-feature join rate against the panel |
| `outputs/qb018_coverage_by_season.csv` | Coverage by season and source |
| `outputs/qb018_model_comparison.csv` | All models, ridge and GBM, with bootstrap CIs |
| `outputs/qb018_per_season.csv` | Per-season MAE / RMSE / accuracy for every model |
| `outputs/qb018_residual_correlations.csv` | Residual correlation + solo-addition scan |
| `outputs/qb018_stability.csv` | Season N→N+1 correlations, QB and team level |
| `outputs/qb018_advanced_panel.csv` | The 4,195-row joined feature panel |
| `shared/nfl-data/cache/processed/qb018_pbp_*.parquet` | Cached PBP aggregates |
