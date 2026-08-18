# Methodology

How the modelling is done, and why each choice was made. Conclusions are
labelled **[evidence]**, **[inference]**, or **[hypothesis]** per the brief.

---

## 1. No look-ahead: the one rule everything routes through

Every feature is computed only from rows observable before the target game
kicked off. This is enforced in two independent implementations that are
cross-checked against each other:

- `features/pregame.py` — filters history explicitly per week via `as_of_expr`.
  Slow, but the rule is visible in one place and easy to audit.
- `features/panel.py` — vectorised shift-then-roll. Every rolling statistic is
  computed on `.shift(1)`, so a game can never enter its own feature.

`panel.validate_against_pregame()` confirms the two agree (0.0 difference on
the checked columns). `scripts/walk_one_qb_week.py` additionally asserts that
the latest game date feeding any feature is strictly earlier than kickoff.

**Why two implementations:** look-ahead bias is the single failure mode that
would invalidate every result in this repository, and it is silent when it
happens. A second construction that arrives at the same numbers is cheap
insurance.

---

## 2. The panel

`build_panel(seasons)` produces one row per QB-week: **4,373 rows, 2018–2025**,
roughly 510–575 per season.

**Inclusion rule:** at least 10 pass attempts. This makes the panel *"given he
started, how did he score"*. It deliberately sets aside the separate problem of
identifying the starter pregame, which needs depth charts and injury reports.
[Documented limitation — a deployed system needs that layer.]

**Debut quarterbacks** (105 rows, 2.4%) have no prior form and are dropped for
model comparison so every model sees an identical row set.

Target: standard-scoring `fantasy_points`, mean 16.1, sd 8.2.

### Feature groups

| Group | Features |
|-------|----------|
| Form | 8-game rolling mean/sd, career mean, pass & rush attempts |
| Style | `rush_fp_share` — share of production from rushing |
| Market | implied team total, `spread_abs`, game total |
| Schedule | opponent QB points allowed, shrunk toward an as-of league mean |
| Context | home, indoors, rest days, division game |

Market inputs are **decomposed rather than used raw**: implied team total
carries the scoring environment, spread magnitude carries game script. This
follows the brief's warning that a favourite is not automatically the better
fantasy quarterback.

Opponent priors are shrunk toward the league mean with k=6 games, so a Week-2
defence cannot look elite on one game of evidence. The shrinkage target is
itself an expanding as-of mean, so it leaks nothing.

---

## 3. Evaluation

**Walk-forward only.** Train on all seasons before S, test on S, for
S = 2020…2025. Never in-sample, never a random split — a random split over
time-series panel data would leak the future into the past.

Model: `RidgeCV` over 24 alphas, features standardised on the training fold.
Ridge rather than a gradient-boosted model on purpose: with 8 seasons the brief's
own risk of over-fitting interaction terms is live, and a regularised linear
model makes marginal contributions readable.

### Metrics

- **MAE / RMSE** — projection accuracy.
- **R²** against a naive "predict the training mean" forecast.
- **Pairwise ranking accuracy** — within each week, over all QB pairs, how often
  the model ranks two quarterbacks correctly. *This is the metric the project
  actually needs:* a start/sit call is a pairwise comparison, not an absolute
  projection. 49,026 comparisons evaluated. Coin flip = 50%.

---

## 4. Strategy simulation

To value a second quarterback, three strategies are run over every eligible
pair (both QBs starting ≥12 games; QB1 = higher **prior-season** points per
game, so the designation uses no in-season information):

1. **Single QB** — start QB1 every week; on his bye, stream the *weekly median
   starter*. Deliberately generous to the strategy the thesis must beat. An
   earlier version scored the bye as zero and overstated the pairing benefit by
   roughly a full replacement week; that was corrected.
2. **Two QBs, model-selected** — start whichever the walk-forward model projects
   higher. No hindsight.
3. **Two QBs, perfect foresight** — the ceiling, not achievable.

The gap between (1) and (3) is the entire prize available. The gap between (1)
and (2) is what a real manager captures.

---

## 5. Known limitations

| Limitation | Effect |
|-----------|--------|
| Starter identification excluded | Real deployment needs depth-chart/injury logic |
| 8-season window (user decision) | Thin for interactions; mitigated with ridge + archetype-level effects |
| Pairs evaluated are all eligible pairs | Measures the *average* pair, not a *selected* pair — the thesis needs selection to add value, which Phase 5 tests |
| Injuries not modelled explicitly | Shows up implicitly as QB1 failure, which is where most pairing value appears |
| No historical player props | Prop features are forward-only (see DATA_SOURCES.md §6.1) |
| Replacement level = weekly median | A sharper streamer would raise the single-QB baseline and shrink the measured gain |

---

## 6. Reproducing

```bash
python scripts/walk_one_qb_week.py 2024 11 "Josh Allen"   # leakage audit
python scripts/phase2_signal.py                           # feature signal
python scripts/phase2b_schedule_test.py                   # schedule + ceiling
python scripts/phase2c_pair_value.py                      # realistic value
```

Outputs land in `outputs/`. BLAS thread count is pinned to 1 inside each
script; OpenBLAS fails to allocate its per-thread buffers on this machine at
the default setting.
