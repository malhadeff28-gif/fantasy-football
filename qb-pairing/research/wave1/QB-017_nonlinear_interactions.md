# QB-017 - Does nonlinear structure or interaction effects exist that a regularised linear model missed?

**Agent:** Nonlinear Modeling Agent
**Date:** 2026-08-18
**Verdict:** REJECT
**Script:** `scripts/wave1_nonlinear_search.py` (runs in ~8 min, deterministic)

## Why this experiment exists

Wave 0 fitted ridge regression, reached out-of-sample R² ≈ 0.177 with 64%
pairwise start/sit accuracy, and stopped. Hand-picked interactions - rushing
share × opponent difficulty and similar - made the model slightly worse.

That is a claim about *hand-picked* interactions, not about interactions in
general. A model class can be rejected for the wrong reason: four guesses
failing says almost nothing about whether the 171 possible pairs contain
something real. The open question in the Wave 0 log was therefore whether the
linear ceiling is a real ceiling, or a failure of search.

This experiment searches properly, twice, with independent methods, and guards
both searches with permutation nulls.

## Design

Three models isolate the two things "is it nonlinear?" usually conflates:

| Model | What it can represent |
| --- | --- |
| ridge | linear, additive - the incumbent |
| GBM-additive (depth-1 stumps) | arbitrary univariate curvature, **no interactions possible** |
| GBM-full (depth unconstrained) | curvature **and** interactions |

`GBM-additive − ridge` isolates univariate curvature. `GBM-full − GBM-additive`
isolates interaction structure. Two further models sharpen the test: xgboost as
an independent implementation, and a **residual booster** where trees are fitted
to the ridge residual, so the linear model keeps everything it already explains
and the trees are asked only for what it missed.

**Data:** 4,167 complete-case QB-weeks, 2018-2025, from `build_panel`. 206 rows
(debut and short-history quarterbacks) are dropped so every model sees an
identical row set. 19 pregame features, the ridge baseline's 9 plus form
volatility, aDOT, CPOE, sack rate, rushing share, rest, division game and both
opponent-defence variables. Test rows: 3,213 over 2020-2025, 46,922 within-week
pairwise comparisons.

**Validation:** walk-forward only - train seasons < S, test season S, for
S = 2020…2025. Every hyperparameter (including the loss function) is selected on
inner folds carved out of the *training* seasons; the test season is invisible
during selection. Uncertainty comes from a paired block bootstrap that resamples
**weeks**, because the week is the correlated unit, plus per-season SEs over the
six test seasons.

## Result 1 - trees lose

| Model | MAE | RMSE | R²(oos) | pair acc | pred sd |
| --- | --- | --- | --- | --- | --- |
| train mean | 6.5373 | 8.1370 | 0.000 | 50.00% | 0.24 |
| **ridge (9 feat)** | **5.9233** | **7.4014** | **0.1726** | **63.78%** | 3.46 |
| ridge (19 feat) | 5.9256 | 7.4190 | 0.1687 | 63.67% | 3.62 |
| GBM additive | 5.9520 | 7.4688 | 0.1575 | 63.29% | 3.45 |
| GBM full (lightgbm) | 5.9525 | 7.4625 | 0.1589 | 63.22% | 3.49 |
| GBM full (xgboost) | 5.9449 | 7.4487 | 0.1620 | 63.51% | 3.31 |
| ridge + GBM on residual | 5.9419 | 7.4575 | 0.1600 | 63.39% | 3.68 |
| ridge + GBM blend (50/50) | 5.9171 | 7.4085 | 0.1710 | 63.87% | 3.43 |

Actual sd = 8.12. The ridge baseline reproduces Wave 0 (R² 0.176 on its native
4,268-row set, 0.1726 here on the common row set).

Paired block bootstrap over weeks, 2,000 draws, against ridge(9):

| Model | ΔMAE | 95% CI | P(beats ridge) | ΔAcc pp | P(beats ridge) |
| --- | --- | --- | --- | --- | --- |
| ridge (19 feat) | +0.0023 | [−0.026, +0.030] | 0.43 | −0.11 | 0.23 |
| GBM additive | +0.0286 | [−0.008, +0.069] | 0.07 | −0.49 | 0.01 |
| GBM full (lightgbm) | +0.0292 | [−0.012, +0.073] | 0.07 | −0.56 | 0.01 |
| GBM full (xgboost) | +0.0216 | [−0.019, +0.065] | 0.14 | −0.27 | 0.10 |
| ridge + GBM residual | +0.0186 | [−0.021, +0.058] | 0.17 | −0.39 | 0.03 |
| ridge + GBM blend | −0.0062 | [−0.026, +0.016] | 0.71 | +0.08 | 0.70 |

Positive ΔMAE = worse. **Every tree model is worse than ridge**, and on ranking
accuracy - the metric the start/sit decision actually needs - the gap is outside
the bootstrap CI (P(GBM beats ridge on accuracy) = 0.01).

Three things are worth separating:

- **Univariate curvature costs 0.029 MAE.** GBM-additive, which is free to bend
  every feature arbitrarily but cannot interact anything, already loses. The
  penalty is not about interactions at all.
- **Interactions add 0.0005 MAE on top of that** - literally nothing. Removing
  the tree model's ability to interact features changes its accuracy by 0.07 pp.
- **Trees find nothing in the ridge residual.** Handed a residual with all the
  linear structure removed, and free to use all 19 features, the booster makes
  the forecast 0.019 MAE *worse*.

The tuner's own choices say the same thing. Given `num_leaves ∈ {3, 7, 15}` and
selecting on honest inner folds, it picked **3 leaves in 5 of 6 folds**. Told to
choose its complexity, the model chooses near-stumps.

## Result 2 - stability

| Season | ridge | GBM additive | GBM full | + residual | blend |
| --- | --- | --- | --- | --- | --- |
| 2020 | 6.3491 | 6.3356 | 6.3384 | 6.3562 | 6.3336 |
| 2021 | 5.9690 | 5.9339 | 5.9575 | 5.9857 | 5.9380 |
| 2022 | 5.2638 | 5.3331 | 5.3410 | 5.2548 | 5.2661 |
| 2023 | 5.7399 | 5.8476 | 5.8246 | 5.8694 | 5.7602 |
| 2024 | 6.0332 | 6.0433 | 6.0228 | 5.9952 | 6.0142 |
| 2025 | 6.1990 | 6.2295 | 6.2432 | 6.2025 | 6.2041 |

GBM-full beats ridge in 3/6 seasons on MAE but only 2/6 on accuracy; mean ΔMAE
+0.0289 ± 0.0186 (SE). The season-to-season spread of the *baseline itself*
(5.26 to 6.35) is thirty times the model-class difference. Six test seasons
cannot resolve a 0.03-point effect, and the honest statement is that the
difference is small and negative, not that it is precisely estimated.

The 50/50 blend is 0.006 MAE better than ridge with P = 0.71. That is a coin
flip dressed as an improvement and is not a finding.

## Result 3 - calibration

| Model | slope | intercept | bias | mean abs bin gap | pred sd |
| --- | --- | --- | --- | --- | --- |
| ridge (9 feat) | 0.970 | 0.260 | +0.237 | 0.388 | 3.46 |
| GBM additive | 0.928 | 1.251 | −0.098 | 0.362 | 3.45 |
| GBM full | 0.922 | 1.281 | −0.025 | 0.436 | 3.49 |

Slope from regressing actual on predicted; 1.0 / 0.0 is perfect. Ridge is the
best-calibrated of the three (slope closest to 1, smallest intercept); the trees
are marginally over-spread and need slightly more shrinkage. Neither is badly
miscalibrated - `outputs/qb017_calibration.png` shows all three decile curves
sitting on the identity line.

The more important number in that table is **pred sd ≈ 3.5 against an actual sd
of 8.12**. Every model's decile bins span only 10 to 23 points while real
outcomes run −7 to +52. Roughly 83% of the weekly variance is not forecastable
from any pregame information in this panel, by any of these model classes. That
is the ceiling, and it is a property of the data.

## Result 4 - the interaction search and its permutation null

**Search A: SHAP interaction values** on a lightgbm fitted to the whole panel
(177 trees, the modal tuned configuration), 800 sampled rows, all 171 feature
pairs ranked by mean |interaction| in fantasy points.

**Null A:** shuffle the target *within (season, week)*, refit with identical
hyperparameters and tree count, re-run the entire SHAP search. 200 draws. The
tree count is held at the value the real signal justified, so the null model is
free to overfit noise - this makes the null harder to beat, deliberately.

| rank | interaction | observed | null p95 | p (pair) | p (family-wise) |
| --- | --- | --- | --- | --- | --- |
| 1 | career_fp_mean × implied_team_total | 0.1206 | 0.0185 | 0.000 | **0.000** |
| 2 | form_fp_mean × implied_team_total | 0.0929 | 0.0138 | 0.000 | **0.015** |
| 3 | form_rush_att × implied_team_total | 0.0508 | 0.0151 | 0.005 | 0.260 |
| 4 | career_fp_mean × total_line | 0.0412 | 0.0203 | 0.005 | 0.460 |
| 5 | form_fp_mean × career_fp_mean | 0.0400 | 0.0292 | 0.010 | 0.500 |
| 6 | rush_fp_share × implied_team_total | 0.0378 | 0.0134 | 0.005 | 0.540 |
| 7 | form_pass_att × form_rush_att | 0.0342 | 0.0132 | 0.000 | 0.700 |
| 8 | form_rush_att × form_cpoe | 0.0261 | 0.0090 | 0.005 | 0.915 |
| 9 | career_fp_mean × def_fp_allowed | 0.0231 | 0.0106 | 0.000 | 0.980 |
| 10 | form_fp_sd × form_rush_att | 0.0157 | 0.0154 | 0.040 | 1.000 |

Family-wise null max-statistic: mean 0.0434, 95th pct 0.0748. Taken at face
value, the top two pairs clear it: **quality × scoring environment**, exactly the
multiplicative structure one would hypothesise a priori. A good quarterback in a
28-point game environment beats the additive prediction.

### The magnitude null is misleading, and the scale-free version says the opposite

A model fitted to real signal has far more total SHAP mass to distribute than a
model fitted to shuffled noise. Comparing *absolute* interaction magnitudes
therefore partly compares how much signal each model found, not how much of it
is interaction. The scale-free statistic is the **interaction share**: interaction
magnitude ÷ (main-effect + interaction magnitude).

| | interaction share |
| --- | --- |
| observed model | **14.6%** |
| permutation null, mean | 29.2% |
| permutation null, 95th pct | 34.0% |
| p-value | **1.000** |

The real model attributes *half* as much of its explanation to interactions as a
pure-noise model does. Under the null the trees have nothing additive to find, so
their splits scatter into apparent interaction structure; the real model is
markedly more additive than that. On the statistic that is not confounded by
total signal, the interaction finding disappears entirely.

## Result 5 - Null B: do the interactions pay off out of sample?

The decisive test is not whether a fitted tree attributes SHAP mass to a pair,
but whether the interaction is worth anything on unseen seasons.

**Search B is exhaustive and independent of the tree model.** Each of the 171
feature pairs is added to ridge as an explicit product term and scored
walk-forward. Because it is exhaustive, the family-wise null covers the whole
search rather than a shortlist. The ridge penalty is pinned per fold at the
value chosen for the base model, so a candidate cannot win by quietly buying
itself different regularisation, and observed and permuted terms are scored by
exactly the same procedure.

**Null B:** permute the product column within (season, week) - identical
marginal distribution, identical week composition, row-level link destroyed -
and take the maximum gain over all 171 candidates in each draw. 200 draws.

Base ridge(19): MAE 5.9256, accuracy 63.67%.

| interaction | ΔMAE gain | null p95 | p (FW) | ΔAcc pp | p (FW) |
| --- | --- | --- | --- | --- | --- |
| form_cpoe × rest | +0.0081 | 0.0030 | **0.360** | +0.09 | 0.965 |
| form_pass_att × is_home | +0.0052 | 0.0022 | 0.780 | +0.05 | 1.000 |
| career_fp_mean × is_home | +0.0042 | 0.0018 | 0.915 | +0.00 | 1.000 |
| career_fp_mean × def_adj | +0.0033 | 0.0014 | 0.980 | +0.06 | 1.000 |
| career_fp_mean × def_fp_allowed | +0.0033 | 0.0017 | 0.980 | +0.07 | 1.000 |
| **career_fp_mean × implied_team_total** (SHAP #1) | **+0.0013** | 0.0021 | **1.000** | +0.03 | 1.000 |
| **form_fp_mean × implied_team_total** (SHAP #2) | **+0.0001** | 0.0016 | **1.000** | −0.03 | 1.000 |
| form_pass_att × form_rush_att (SHAP #7) | −0.0007 | 0.0030 | 1.000 | +0.12 | 0.805 |
| form_rush_att × implied_team_total (SHAP #3) | −0.0031 | 0.0018 | 1.000 | +0.00 | 1.000 |
| rush_fp_share × implied_team_total (SHAP #6) | −0.0039 | 0.0027 | 1.000 | −0.00 | 1.000 |
| rush_fp_share × def_adj (Wave-0 pick) | −0.0002 | 0.0026 | 1.000 | −0.00 | 1.000 |
| rush_fp_share × spread_abs (Wave-0 pick) | −0.0012 | 0.0021 | 1.000 | −0.00 | 1.000 |
| form_pass_att × def_fp_allowed (Wave-0 pick) | −0.0004 | 0.0028 | 1.000 | +0.09 | 0.980 |

**The two interactions that cleared Null A are worth nothing out of sample.**
career_fp_mean × implied_team_total buys +0.0013 MAE points - 0.02% of the
baseline error - with a family-wise p-value of 1.000. Ridge with both main
effects already captures essentially all of it.

**The best of all 171 pairs does not clear its null either.** form_cpoe × rest
gains +0.0081 MAE (0.14% of baseline error) against a family-wise null whose max
averages +0.0073 and whose 95th percentile is +0.0119: **p = 0.360**. It
replicates in 5/6 seasons (mean +0.0080 ± 0.0030 SE), which is why it is
reported rather than buried - but a 0.008-point effect is not a modelling
finding, and the search that produced it looked at 171 candidates.

Two summary statistics settle it:

- **37 of 171 pairs improved MAE at all**, against **34.1 expected under the
  null**. The observed interaction landscape is indistinguishable from noise.
- Mean gain over all 171 pairs: **−0.0019 observed vs −0.0023 null**. A randomly
  chosen interaction is a cost, not a benefit, and the real ones are barely
  cheaper than the fake ones.

Wave 0's four hand-picked interactions all sit in the negative range, so its
original conclusion was right - it just was not entitled to it on the evidence
it had. It is now.

## Result 6 - honest-split robustness

Sections 4-5 search and score on overlapping rows, which favours the search.
Repeating the discovery on **2018-2021 only** and grading on **2022-2025**:

| rank | interaction discovered pre-2022 | ΔMAE on 2022-2025 |
| --- | --- | --- |
| 1 | career_fp_mean × def_fp_allowed | −0.0010 |
| 2 | career_fp_mean × implied_team_total | −0.0027 |
| 3 | career_fp_mean × def_adj | −0.0034 |
| 4 | career_fp_mean × total_line | −0.0006 |
| 5 | form_rush_att × implied_team_total | −0.0001 |

**All five hurt.** An interaction selected on early seasons and applied to later
ones has never once helped. This is the cleanest statement in the experiment,
because nothing the search saw was part of what graded it.

## Interpretation

The relationship between pregame information and QB weekly fantasy points is,
for practical purposes, **linear and additive**.

`outputs/qb017_shape_functions.png` shows what curvature does exist. The additive
GBM does learn a real shape on `form_fp_mean`: steep through the middle, flat in
both tails - shrinking extreme form back toward the mean, which is exactly right
as a description of the training data. It does not survive contact with a new
season, because with ~2,500-3,500 training rows the step locations are fitted to
noise. `implied_team_total`, the strongest market signal, is close to perfectly
straight over its whole range - there is nothing there to bend.

The interaction result is best read as a lesson in what a permutation null is
for. A tree model *does* allocate SHAP interaction mass to
quality × scoring-environment, and on the naive magnitude comparison that
allocation looks overwhelmingly significant (p = 0.000). It is an artefact of
comparing a signal-rich model against a signal-free one. The scale-free share
statistic (p = 1.000), the exhaustive out-of-sample scan (best p = 0.360), and
the honest temporal split (all five candidates negative) agree with each other
and disagree with the naive test. Had this experiment stopped after the first
permutation null, it would have shipped a false positive with an impressive
p-value attached.

**The ceiling is data, not model class.** Residual sd is 7.4 points against an
outcome sd of 8.1. Even the SHAP-strongest interaction is worth 0.12 points of
mean absolute attribution against that 7.4-point residual - a real structure two
orders of magnitude too small to matter. Adding model flexibility spends degrees
of freedom against noise, which is why every tree variant lost. Progress on this
problem requires **new information** - starter identification, injury and
personnel status, weather at kickoff, player props - and not a richer functional
form over the information already in hand.

## Verdict

**REJECT.** Gradient-boosted trees do not beat ridge on any metric: MAE +0.029
(P(better) = 0.07), ranking accuracy −0.56 pp (P(better) = 0.01), R² 0.159 vs
0.173. Interactions specifically add +0.0005 MAE over a strictly additive tree
model, i.e. nothing. No interaction, from a SHAP search or an exhaustive
walk-forward scan, survives its permutation null once the null is stated in a
form that is not confounded by total model signal; the strongest out-of-sample
candidate has family-wise p = 0.360 and is worth 0.14% of baseline error.

The Wave 0 linear ceiling is a real ceiling. Ridge stays as the projection layer.

### What would change this verdict

- More rows. 4,167 QB-weeks is thin for 171 candidate interactions; extending to
  the 2007+ window (available, deliberately excluded as a different QB era)
  roughly doubles the sample and is the cheapest available test of whether these
  effects are real-but-underpowered rather than absent.
- New feature families with genuinely multiplicative structure - player props
  and injury/personnel status are the obvious candidates - which would be tested
  for interactions on their own terms, not retro-fitted to this feature set.
- A different target. Curvature that does not help mean prediction may still help
  *tail* prediction; nothing here tests quantile or ceiling-week forecasting, and
  a start/sit decision arguably cares about the upper tail.

## Artefacts

| File | Contents |
| --- | --- |
| `outputs/qb017_model_comparison.csv` | the model ladder, pooled metrics |
| `outputs/qb017_bootstrap_deltas.csv` | paired block bootstrap, 2,000 draws |
| `outputs/qb017_by_season.csv` | per-season MAE and accuracy, all models |
| `outputs/qb017_calibration.csv`, `_curves.csv` | slopes, bias, decile reliability |
| `outputs/qb017_shap_interactions.csv` | top 20 pairs with Null A p-values |
| `outputs/qb017_interaction_oos_test.csv` | exhaustive scan with Null B p-values |
| `outputs/qb017_honest_split.csv` | pre-2022 discovery graded on 2022-2025 |
| `outputs/qb017_tuned_params.csv` | hyperparameters chosen per fold, per model |
| `outputs/qb017_calibration.png` | reliability curves and prediction spread |
| `outputs/qb017_season_stability.png` | per-season ΔMAE and ΔAccuracy vs ridge |
| `outputs/qb017_shap_interactions.png` | top interactions against the Null A band |
| `outputs/qb017_permutation_nulls.png` | both null distributions with observed values |
| `outputs/qb017_shape_functions.png` | learned univariate shapes, additive GBM vs ridge |

Reproduce: `./.venv/Scripts/python.exe qb-pairing/scripts/wave1_nonlinear_search.py`
(`--quick` runs the same pipeline with few resampling draws for a smoke test;
its nulls are too coarse to quote).
