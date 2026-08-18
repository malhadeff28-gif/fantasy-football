# QB-019 - Do quarterbacks genuinely differ in matchup sensitivity?

**Agent:** Research Director (main context)
**Date:** 2026-08-18
**Verdict:** REJECT (strong)

## Why this experiment exists

Wave 0 rejected per-QB matchup effects two ways: fitting a separate slope per
quarterback made the model worse out of sample (MAE 5.9455 vs 5.9183), and a
split-half reliability test gave r = 0.065. Both are suggestive, neither is the
standard test.

The standard test is a **random-slopes mixed model**. Let every quarterback
have his own slope on opponent difficulty, with partial pooling, then inspect
the variance component. This is strictly better than fitting 41 separate
slopes, because partial pooling shrinks noisy individual estimates toward the
league mean instead of manufacturing spurious spread. `lme4` is the reference
implementation, and this is the reason R was installed for this repository.

## Method

- Panel: 3,842 QB-weeks, 69 quarterbacks with >= 16 games, 2018-2025
- Fixed effects: form, career mean, pass/rush attempts, implied team total,
  game total, spread magnitude, home, indoors, opponent difficulty
- `m0`: random intercept only - quarterbacks differ in *level*
- `m1`: `+ (1 + def_adj | player_id)` - quarterbacks differ in *matchup response*
- `m2`: `+ (1 + implied_team_total | player_id)` - quarterbacks differ in
  *market response* (never previously tested)
- ML fit, likelihood-ratio tests, bobyqa optimizer

## Results

| Model | npar | AIC | BIC | Chisq | df | p |
| --- | --- | --- | --- | --- | --- | --- |
| m0 random intercept | 13 | 26354 | 26436 | | | |
| m1 + slope on opponent difficulty | 15 | 26354 | **26448** | 4.07 | 2 | **0.130** |
| m2 + slope on implied team total | 15 | 26357 | **26451** | 1.41 | 2 | **0.495** |

Variance components:

| Term | SD |
| --- | --- |
| Intercept (m1) | 0.586 |
| Slope on opponent difficulty | 0.194 |
| Slope on implied team total (m2) | 0.062 |
| **Residual** | **7.413** |

**Both random-slope models are singular fits** (`isSingular = TRUE`), with
intercept-slope correlations pinned at exactly +1.000 and -1.000. A boundary
fit means the data cannot identify separate slope variance at all. That is not
a nuisance to be optimised away - it is the answer.

## Interpretation

Neither random slope earns its place. AIC does not improve, BIC gets worse for
both, and neither likelihood-ratio test approaches significance.

For scale: the fitted random-slope SD of 0.194 against an observed opponent-
difficulty SD of 2.10 implies a 1-SD harder matchup moves the typical
quarterback about **0.41 fantasy points** relative to the league-average
response - and that estimate comes from a singular fit, so it is not
distinguishable from zero. Residual SD is 7.41 points, roughly eighteen times
larger.

## What is new here

The market-sensitivity test (m2) had never been run. Quarterbacks do not differ
in how much their scoring responds to the betting market either (p = 0.495) -
which means the implied-team-total effect can be applied league-wide with a
single coefficient, and no per-quarterback tuning is warranted anywhere in the
model.

## Bearing on the thesis

This is the strongest available statement of a Wave 0 finding. The brief asked
whether an opponent affects quarterbacks differently by their characteristics.
Under the standard hierarchical test, with proper shrinkage, on eight seasons:
**no, and the data cannot even estimate such a difference.**

## Reproduce

```
.venv/Scripts/python qb-pairing/scripts/wave1_hierarchical_matchup.py
```

Requires R 4.6.1 with lme4 (user library at C:/Users/Max/R/win-library/4.6).
