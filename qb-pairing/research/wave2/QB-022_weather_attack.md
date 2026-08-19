# QB-022 — Adversarial attack on the weather finding (QB-008)

**Verdict: WEAKENED.**

The wind ≥ 15 mph effect is real and survives every attack aimed at it — but the
published number overstates both its size and its certainty, the wind 8–14 mph
effect does not survive at all, and the effect is worth roughly **0.7 fantasy
points per season** to a manager even under assumptions generous to it.

Script: `qb-pairing/scripts/wave2_weather_and_replication.py`
Evidence: `qb-pairing/outputs/qb022_*.csv`

---

## The claim under attack

> "Wind ≥ 15 mph costs a quarterback 1.61 fantasy points (t = −3.45, n = 304);
> wind 8–14 mph costs 0.57 (t = −2.39). This is the largest scheduling effect
> measured."

Reproduced exactly from `scripts/phase3_scheduling.py`:

| bucket | n | mean form residual | se (iid) | t (iid) |
|---|---|---|---|---|
| wind < 8 mph | 1,317 | −0.129 | 0.214 | −0.60 |
| wind 8–14 mph | 1,033 | −0.572 | 0.240 | **−2.39** |
| wind ≥ 15 mph | 304 | −1.608 | 0.466 | **−3.45** |

Panel: 4,268 QB-weeks, 2018–2025. Wind is populated on 2,654 (62.2%).

---

## Two errors in the estimator, before any data question

**1. The published figure is a bucket mean, not a contrast.** "Costs 1.61
points" implicitly compares against a zero counterfactual. The correct
comparison is against the low-wind bucket, which is itself −0.129. The
differential is **−1.478**, not −1.608.

**2. The standard errors treat two quarterbacks in the same game as independent
observations.** They are not — they share one wind reading. n = 304 QB-weeks is
**153 games**. Clustering the standard error by `game_id`:

| bucket | n | games | mean | se (iid) | t (iid) | se (clustered) | t (clustered) |
|---|---|---|---|---|---|---|---|
| wind < 8 | 1,317 | 655 | −0.129 | 0.214 | −0.60 | 0.233 | −0.55 |
| wind 8–14 | 1,033 | 514 | −0.572 | 0.240 | −2.39 | 0.258 | −2.21 |
| wind ≥ 15 | 304 | 153 | −1.608 | 0.466 | −3.45 | 0.539 | −2.98 |

Correcting both together — the contrast against low wind, with game-clustered
standard errors — gives the honest headline:

| effect | coefficient | se | t | 95% CI |
|---|---|---|---|---|
| wind ≥ 15 mph | **−1.484** | 0.586 | **−2.53** | **[−2.63, −0.33]** |
| wind 8–14 mph | **−0.448** | 0.348 | **−1.29** | **[−1.13, +0.23]** |

**The 8–14 mph claim is broken.** As a properly specified contrast with honest
standard errors it is indistinguishable from zero. Its published t = −2.39 came
entirely from (a) measuring against zero rather than against calm conditions and
(b) counting each game twice.

The ≥ 15 mph claim survives, but the interval is wide: the data are consistent
with anything from a trivial −0.33 to a large −2.63.

---

## Attack 1 — is the missingness informative? (Mostly no.)

### Coverage is structural, not random

| roof | QB-weeks | wind coverage | mean form residual |
|---|---|---|---|
| outdoors | 2,899 | 91.5% | −0.512 |
| dome | 699 | 0.0% | +0.407 |
| closed | 587 | 0.3% | +0.941 |
| **open** | 83 | **0.0%** | +1.114 |

A retractable roof left **open** is an outdoor game and gets no weather row at
all. "Indoors" and "no reading" are therefore not the same set — a point the
original analysis never had to confront, because of what comes next.

### The published estimate was never contaminated by domes

Of the 2,654 rows carrying a wind reading, **2,652 are outdoor games**. The
weather-present sample is already an outdoor sample; the low-wind reference
bucket is outdoor too. Restricting to outdoor games changes nothing:

| sample | < 8 | 8–14 | ≥ 15 |
|---|---|---|---|
| all wind-present rows | −0.129 | −0.572 | −1.608 |
| outdoor only | −0.124 | −0.572 | −1.608 |
| outdoor, drop 2022 (51% coverage) | −0.103 | −0.611 | −1.786 |
| outdoor, ≥ 94% coverage seasons only | −0.164 | −0.440 | −1.621 |
| outdoor, IPW-reweighted to full outdoor population | −0.107 | −0.572 | **−1.517** |

### Missing-at-random test, outdoor games only

Comparing outdoor rows *with* a reading against outdoor rows *without* one:

| variable | present | absent | diff | t |
|---|---|---|---|---|
| week | 9.75 | 5.52 | **+4.23** | **17.26** |
| season | 2021.42 | 2022.09 | −0.67 | −8.13 |
| spread_abs | 5.53 | 4.63 | +0.90 | 4.65 |
| rest | 7.49 | 7.12 | +0.37 | 3.74 |
| def_adj | −0.03 | −0.39 | +0.36 | 2.91 |
| implied_team_total | 22.43 | 22.39 | +0.04 | 0.19 |
| total_line | 44.77 | 44.77 | +0.01 | 0.03 |
| form_fp_mean | 16.36 | 16.10 | +0.27 | 0.97 |
| fantasy_points | 15.89 | 15.65 | +0.25 | 0.55 |
| **form residual** | **−0.468** | **−0.449** | **−0.020** | **−0.05** |

Missingness is **strongly non-random on timing** — missing outdoor readings are
concentrated in early weeks (week 5.5 vs 9.7) and in two bad vendor seasons
(2022 at 51% coverage, 2023 at 79%). A logit predicting missingness among
outdoor games reaches pseudo-R² = 0.106 from pregame observables, of which week
does nearly all the work; season dummies alone reach 0.287. This is a scrape
gap, not a game-characteristic gap.

But it is **not informative about the outcome**: the form residual is identical
in the present and absent groups (t = −0.05), and neither market variable
differs at all. Inverse-probability reweighting to the full outdoor population
moves the estimate from −1.608 to −1.517, a 6% attenuation well inside the
standard error.

**The missingness attack fails.** It was the right thing to suspect and it is
not what is wrong with this finding.

---

## Attack 2 — is the wind effect really the market, priced twice? (No.)

The market *does* price wind:

| target | wind ≥ 15 | se | t | wind 8–14 | se | t |
|---|---|---|---|---|---|---|
| total_line | **−1.203** | 0.406 | −2.96 | −0.758 | 0.272 | −2.79 |
| implied_team_total | **−0.573** | 0.216 | −2.66 | −0.407 | 0.140 | −2.91 |
| form_pass_att (prior form) | −0.004 | 0.353 | −0.01 | −0.273 | 0.223 | −1.22 |

Game totals are 1.2 points lower in high-wind games. That is direct evidence
that the forecast was public before the line closed. But controlling for it
changes nothing:

| specification | wind ≥ 15 | se | t | 95% CI |
|---|---|---|---|---|
| 1. no controls | −1.484 | 0.586 | −2.53 | [−2.63, −0.33] |
| 2. + implied team total, total line | −1.490 | 0.591 | −2.52 | [−2.65, −0.33] |
| 3. + market + form level | −1.132 | 0.570 | −1.99 | [−2.25, −0.02] |
| 4. + market + temperature | −1.448 | 0.588 | −2.46 | [−2.60, −0.29] |
| 5. + market + temp + week + home | −1.470 | 0.589 | −2.50 | [−2.62, −0.32] |
| continuous, per 10 mph, no controls | −1.035 | 0.352 | −2.94 | [−1.73, −0.35] |
| continuous, per 10 mph, + market + temp | −1.004 | 0.356 | −2.82 | [−1.70, −0.31] |

**Why there is no double counting:** implied team total earns only
+0.095 fantasy points per point (t = 2.24) against the form residual. The
market's 0.573-point downgrade of a windy offence therefore accounts for
0.573 × 0.095 ≈ **0.05 fantasy points** — about 4% of the 1.48-point effect.
The market registers the wind and prices roughly none of its fantasy
consequence. The residual estimate is not absorbing information already in the
line.

Adjacent finding from the same regression, worth recording: **temperature
matters on its own** (+0.321 per 10 °F, t = 2.53) and **home field is +1.32
(t = 4.38)** on the same outdoor sample.

---

## Attack 3 — is it wind, or is it windy stadiums / bad offences?

| specification | wind ≥ 15 | se | t |
|---|---|---|---|
| market controls only | −1.490 | 0.591 | −2.52 |
| + stadium fixed effects | −1.414 | 0.606 | −2.33 |
| + offence-team fixed effects | −1.576 | 0.596 | −2.65 |
| + season fixed effects | −1.517 | 0.592 | −2.56 |

Survives all three. It is not Buffalo-and-Chicago in disguise.

## Attack 4 — was the 15 mph threshold cherry-picked?

| threshold | n | mean | se (clustered) | t | contrast vs below |
|---|---|---|---|---|---|
| 10 | 902 | −1.239 | 0.285 | −4.35 | −1.168 |
| 12 | 620 | −1.322 | 0.346 | −3.82 | −1.114 |
| 14 | 367 | −1.415 | 0.474 | −2.99 | −1.099 |
| **15** | **304** | **−1.608** | **0.539** | **−2.98** | **−1.287** |
| 16 | 212 | −1.724 | 0.645 | −2.67 | −1.365 |
| 18 | 110 | −0.727 | 0.877 | −0.83 | −0.269 |
| 20 | 60 | −1.163 | 1.103 | −1.05 | −0.711 |
| 22 | 33 | −2.079 | 1.155 | −1.80 | −1.631 |

Monotone up to 16 mph, then noisy — which is what a real effect measured on
thinning samples looks like, not what a cherry-pick looks like. Permutation
null, shuffling wind within season-week and taking the maximum |t| across the
whole threshold grid (1,000 draws):

- observed max |t| = 3.65 (at 10 mph)
- **permutation p = 0.0040**, corrected for the threshold search

The effect survives multiplicity correction.

## Attack 5 — does panel selection manufacture it?

The panel requires ≥ 10 pass attempts. In high wind, teams throw less, so
suppressed starts could be filtered out.

| wind bucket | QB appearances | share kept at ≥ 10 att | mean attempts |
|---|---|---|---|
| < 8 | 1,489 | 90.9% | 29.5 |
| 8–14 | 1,154 | 91.3% | 29.3 |
| ≥ 15 | 352 | **88.1%** | 27.8 |

High-wind starts are dropped slightly *more* often. The surviving high-wind rows
are the ones that still threw the ball, so this selection biases the measured
penalty **toward zero**. Conservative, not inflationary.

---

## The real weakness: stability, and whether a manager could act

### It is not stable across the sample

| season | n (≥15) | mean | se | t | contrast vs < 8 |
|---|---|---|---|---|---|
| 2018 | 32 | −0.484 | 1.258 | −0.38 | +0.307 |
| 2019 | 42 | −1.211 | 1.691 | −0.72 | −0.691 |
| 2020 | 51 | −0.094 | 1.363 | −0.07 | −0.179 |
| 2021 | 57 | −2.600 | 1.215 | −2.14 | −2.478 |
| 2022 | 27 | +0.225 | 1.888 | +0.12 | +0.658 |
| 2023 | 25 | −3.447 | 1.544 | −2.23 | −3.776 |
| 2024 | 30 | −1.765 | 1.625 | −1.09 | −2.586 |
| 2025 | 40 | −3.407 | 1.606 | −2.12 | −3.087 |

Negative point estimate in 7 of 8 seasons; individually significant in 3.
Chronological split, no look-ahead:

- **2018–2021: −0.835 (se 0.766)** — not significant
- **2022–2025: −2.374 (se 0.908)** — significant

A manager running this study after the 2021 season would have found nothing.
The finding is carried by the second half of the sample. That is either a real
regime change or ordinary sampling variation on 153 games; eight seasons cannot
distinguish them.

### Is wind knowable pregame? Partly. Is it actionable? Barely.

**nflverse `wind` is the observed game-time reading, not the forecast.** It is
sourced from the game record. Nothing in this dataset tells us what a manager
saw on Sunday morning. Three things follow.

**1. The forecast existed and was public.** The 1.2-point drop in `total_line`
for high-wind games is proof: sportsbooks moved the number before kickoff, so
the wind was forecast and priced. A manager could have seen the same forecast.

**2. Forecast error attenuates the effect but does not destroy it.** Simulating
a manager who flags games on a noisy forecast `wind + N(0, σ)` and is scored on
actual outcomes:

| forecast σ (mph) | games flagged | mean form residual | t |
|---|---|---|---|
| 0 (perfect hindsight) | 304 | −1.608 | −2.98 |
| 1 | 261 | −1.605 | −2.76 |
| 2 | 282 | −1.426 | −2.55 |
| 3 | 319 | −1.363 | −2.59 |
| 4 | 368 | −1.250 | −2.55 |
| 5 | 428 | −1.108 | −2.44 |
| 7 | 556 | −0.907 | −2.27 |

At a realistic 24–48 hour wind-forecast error of 2–4 mph, roughly 80% of the
effect remains capturable. This is the one part of the finding that holds up
better than expected.

**3. The decision is almost never available.** This is what kills it:

- wind ≥ 15 mph occurs in **7.1%** of QB-weeks
- the measured differential is **1.48 points**
- only **18.5%** of within-week QB pairs are within 1.48 points of each other on
  recent form — the rest of the time the flag cannot change the call
- upper bound on annual value: 2 × 0.071 × 0.185 × 1.48 × 17 ≈ **0.66 points per
  season**

That upper bound assumes a perfect forecast and that the flag is always the
correct tiebreak. For comparison, QB-013 valued the *entire* weekly start/sit
decision at +4.9 points per season. Wind is a small fraction of a small number.

### "Largest scheduling effect measured" does not survive either

On the same outdoor sample and the same regression, **home field is +1.32
(t = 4.38)** — comparable in size, better identified, and applicable to every
game rather than 7% of them. Descriptively, the indoor/outdoor gap in the form
residual is larger still (outdoors −0.51 vs indoors +0.68, a 1.19-point spread
across 32% of games), though that comparison is confounded with which teams
play indoors. Wind has the largest *coefficient* among rare conditions; it does
not have the largest expected impact on a season.

---

## Verdict

**WEAKENED.**

| component of the claim | status |
|---|---|
| wind ≥ 15 mph is associated with lower QB scoring | **survives** — every attack (missingness, market, stadium/team FE, threshold multiplicity, panel selection) leaves it intact |
| the effect is −1.61 points | **corrected to −1.48** (contrast vs calm, not a raw bucket mean) |
| t = −3.45, n = 304 | **corrected to t = −2.53 on 153 games**, 95% CI [−2.63, −0.33] |
| wind 8–14 mph costs 0.57, t = −2.39 | **BROKEN** — −0.448, se 0.348, t = −1.29, CI [−1.13, +0.23] |
| effect is stable | **fails** — −0.84 (2018–21) vs −2.37 (2022–25); significant in 3/8 seasons |
| the largest scheduling effect measured | **fails** — home field is comparable and universal |
| actionable pregame | **partly** — the forecast was public and priced by the market, and ~80% of the effect survives realistic forecast error, but the flag is worth ~0.7 points per season at most |

Recommended log change: QB-008 downgraded from **KEEP** to **UNCERTAIN**. The
wind ≥ 15 mph direction is real; the magnitude, the precision, the 8–14 mph
tier, and the "largest scheduling effect" framing are not supportable, and the
practical value does not justify presenting it as a decision rule.
