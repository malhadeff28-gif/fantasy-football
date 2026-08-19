# DST-006 - Full outcome distributions per defense-week

**Verdict: KEEP.** The component simulator beats every rival out of sample on a
paired test, and its published percentiles and event probabilities are
calibrated. One known defect is documented below rather than hidden.

Script: `scripts/dst006_distributions.py`
Reusable module: `src/dst_distributions.py`
Outputs: `outputs/dst006_calibration.png`, `dst006_example_week.png`,
`dst006_oos_distributions.csv`, `dst006_variant_comparison.csv`,
`dst006_coverage.csv`, `dst006_reliability.csv`

---

## Why the mean is the wrong object

D/ST scoring is violently asymmetric. Over 4,222 defense-weeks the mean is 6.28
points, but the 10th percentile is 0, the 90th is 14, and the skew is +0.87. A
projection of "6.3 points" describes almost no actual defense-week. The two
tails also come from different places: sacks carry the middle of the
distribution, while the points-allowed tier (range -4 to +10) and the 6-point
defensive touchdown produce both ends.

## Design: model the components, not the total

Each scoring component gets its own shape, and the shape is chosen from the
data, not by habit:

| Component | mean | var/mean | pts per game | shape used |
| --- | --- | --- | --- | --- |
| `def_sacks` | 2.400 | **1.280** | 2.400 | negative binomial (overdispersed) |
| `def_interceptions` | 0.771 | 1.075 | 1.542 | Poisson |
| `fumble_recovery_opp` | 0.522 | 0.973 | 1.045 | Poisson |
| `td_any` (def + ST TDs) | 0.103 | 1.013 | 0.615 | Poisson |
| `points_allowed` | 22.886 | - | 0.484 | linear on market + empirical residuals, then the tier map |
| misc (safety, blocked kicks) | - | - | 0.197 | empirical draw |

Three design decisions, each with its reason:

1. **Only sacks are overdispersed** (var/mean = 1.28); everything else is within
   7% of Poisson. So one negative binomial and three Poissons, not four
   negative binomials.
2. **Points allowed is modelled, then mapped.** The tier is a known deterministic
   step function of the opponent's score, and the opponent's score is exactly
   what the betting market prices. Modelling the score and applying the shared
   `scoring.DSTScoring.points_allowed` map is strictly better than fitting a
   distribution to the tier value itself. The map is built by *calling* shared
   scoring, never by re-listing thresholds, so the league definition cannot
   drift between modules. Reconstruction check, out of sample: simulated tier
   frequencies match actual to within 1.8 pp at every one of the seven tiers
   (largest miss: the +1 tier, 23.9% simulated against 25.7% actual), and the
   simulated tier mean is 0.428 against an actual 0.482.
3. **One predictor per component.** DST-002 established that a single market
   number carries the whole signal, so every component is regressed on
   `implied_opp_total` and nothing else, with an intercept-only alternative
   selected against it by training-set AIC.

### Dependence is not optional

Components are correlated, and in the direction that matters:

|  | sacks | INT | fumble | def TD | pts allowed |
| --- | --- | --- | --- | --- | --- |
| **sacks** | 1.000 | 0.092 | 0.158 | 0.057 | **-0.355** |
| **INT** | 0.092 | 1.000 | 0.038 | **0.284** | **-0.261** |
| **fumble** | 0.158 | 0.038 | 1.000 | 0.034 | -0.167 |
| **def TD** | 0.057 | 0.284 | 0.034 | 1.000 | -0.053 |
| **pts allowed** | -0.355 | -0.261 | -0.167 | -0.053 | 1.000 |

A shutout arrives with sacks and interceptions attached. Sampling components
independently would delete exactly the joint upper tail a streamer is buying.

Dependence is restored by **resampling whole rows of the training PIT matrix**:
each simulated game reuses the relative position a single real game occupied
inside every one of its own marginals at once. That is an empirical copula. It
needs less code than a Gaussian copula (no correlation matrix, no Cholesky) and
it keeps joint-tail behaviour a Gaussian copula structurally cannot represent.
All three options are kept in the module and scored below.

## Thresholds, derived rather than picked

- **Bust = 2 points or fewer.** 2.0 is the median weekly 24th-best D/ST score,
  so a bust week is literally "the waiver wire would have beaten you". It is
  also the 26th percentile of all defense-weeks (25.7%).
- **Spike = 14 points or more.** The 89th percentile (10.8%), and clearly past
  the median weekly 6th-best score of 11 - a start that wins the position for
  the week.

Both are anchored to weekly replacement level, not to round numbers.

## Validation: walk-forward, six rivals

Fit on seasons < S, simulate season S, for S = 2020..2025. **3,230 out-of-sample
defense-weeks**, 3,000 simulations each.

| Variant | CRPS | skill vs climatology | KS(PIT) | p | Brier(top-12) | BSS | AUC(top-12) | Brier(bust) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A components + empirical copula** | **2.894** | **0.048** | **0.0297** | 0.007 | 0.2265 | 0.052 | 0.632 | **0.1787** |
| B components + Gaussian copula | 2.900 | 0.046 | 0.0417 | 0.000 | 0.2265 | 0.052 | 0.632 | 0.1793 |
| C components, independent | 2.924 | 0.038 | 0.0704 | 0.000 | 0.2272 | 0.049 | 0.632 | 0.1823 |
| D total only (regression + residual bootstrap) | 2.905 | 0.044 | 0.0310 | 0.004 | 0.2263 | 0.053 | 0.633 | 0.1805 |
| E climatology (no information) | 3.039 | 0.000 | 0.0188 | 0.204 | 0.2389 | 0.000 | 0.510 | 0.1908 |
| F variant A, 3-season recency weight | 2.896 | 0.047 | 0.0275 | 0.015 | 0.2265 | 0.052 | 0.632 | 0.1789 |

CRPS differences that small are meaningless without uncertainty, so they were
tested paired, bootstrapped over the 107 out-of-sample weeks:

| Variant | CRPS minus CRPS(A) | 95% CI | P(worse than A) |
| --- | --- | --- | --- |
| B Gaussian copula | +0.0059 | [+0.0019, +0.0097] | 0.996 |
| C independent | **+0.0302** | [+0.0216, +0.0389] | 1.000 |
| D total only | **+0.0109** | [+0.0064, +0.0156] | 1.000 |
| E climatology | **+0.1452** | [+0.1055, +0.1813] | 1.000 |
| F recency weighted | +0.0018 | [+0.0002, +0.0034] | 0.984 |

Every rival is worse, and every interval clears zero. Three conclusions, each
now earned rather than asserted:

- **Components beat "just fit the total"** (D, +0.011 [0.006, 0.016]). The
  margin is small, which is the DST-002 lesson repeating, but it is consistent
  across CRPS, Brier(bust) and calibration - and the component model is the only
  one that can answer questions like "what is P(shutout tier)".
- **Dependence matters most** (C, +0.030). Deleting it is three times as
  damaging as using the wrong dependence model, and it wrecks the percentiles:
  variant C's p10 covers 15.8% of outcomes against a nominal 10%.
- **The empirical copula beats the Gaussian one** (B, +0.006), as expected from
  the missing tail dependence.
- **Recency weighting does not help** (F is marginally worse). It was tested
  because the level bias below looked like drift; it is not. `half_life`
  therefore defaults to `None` in the module, with the knob retained and
  documented rather than tuned.

## Calibration

### Published percentiles, out of sample

Binomial SE at n = 3,230 is about 0.53 pp.

| Variant | p10 | p25 | p50 | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| **A empirical copula** | **10.2%** | **25.8%** | **51.9%** | **77.0%** | **91.0%** |
| B Gaussian copula | 11.2% | 28.0% | 53.5% | 76.3% | 89.9% |
| C independent | 15.8% | 31.8% | 54.7% | 74.9% | 87.7% |
| D total only | 10.9% | 26.5% | 52.3% | 76.5% | 91.5% |
| nominal | 10% | 25% | 50% | 75% | 90% |

Variant A is within 0.2 to 2.0 pp of nominal at every published percentile. The
residual pattern (all five slightly high) is the level bias discussed below.

The randomised PIT gives KS = 0.0297, p = 0.007 - formally distinguishable from
uniform. That is a statement about power at n = 3,230, not about magnitude: the
largest CDF departure anywhere in the distribution is 3 percentage points.

### Event probabilities - does P(top-12) = 0.30 happen 30% of the time?

Yes.

**P(top-12)** - every bin's 95% interval contains the predicted value:

| predicted bin | n | predicted | observed | 95% CI |
| --- | --- | --- | --- | --- |
| 0.15-0.20 | 65 | 0.180 | 0.215 | [0.129, 0.326] |
| 0.20-0.30 | 604 | 0.262 | 0.268 | [0.234, 0.305] |
| 0.30-0.40 | 1056 | 0.351 | 0.327 | [0.299, 0.355] |
| 0.40-0.50 | 875 | 0.447 | 0.469 | [0.436, 0.502] |
| 0.50-0.70 | 601 | 0.564 | 0.547 | [0.507, 0.587] |

**P(bust <= 2)** - every bin's interval contains the prediction:

| predicted bin | n | predicted | observed | 95% CI |
| --- | --- | --- | --- | --- |
| 0.05-0.10 | 120 | 0.083 | 0.058 | [0.026, 0.111] |
| 0.10-0.15 | 430 | 0.126 | 0.121 | [0.093, 0.154] |
| 0.15-0.20 | 575 | 0.175 | 0.174 | [0.145, 0.206] |
| 0.20-0.30 | 1079 | 0.249 | 0.248 | [0.223, 0.275] |
| 0.30-0.40 | 760 | 0.340 | 0.370 | [0.336, 0.404] |
| 0.40-0.50 | 226 | 0.439 | 0.429 | [0.366, 0.494] |
| 0.50-0.70 | 35 | 0.532 | 0.600 | [0.435, 0.749] |

**P(top-6)** is calibrated in six of seven bins; the 0.40-0.50 bin (n=55,
predicted 0.436, observed 0.309) is the one marginal miss.
**P(spike >= 14)** is calibrated in four of five bins; the 0.05-0.10 bin
(n=1274, predicted 0.077, observed 0.057, CI [0.046, 0.071]) over-states spike
probability by about 2 pp.

**Across all 23 reliability bins, 21 contain the prediction.** At a 95% interval
that is what chance alone would produce, so there is no systematic reliability
failure to report - only the level bias below.

Discrimination, not just reliability: AUC for top-12 is **0.632** against
climatology's 0.510, and the Brier skill score is +0.052. The probabilities are
informative as well as honest, though the information content is modest - which
is exactly what DST-002 predicted.

## The one known defect: +0.25 points of optimism

| Variant | mean simulated - mean actual | MAE |
| --- | --- | --- |
| A empirical copula | **+0.250** | 4.123 |
| D total only | +0.335 | 4.141 |
| E climatology | **+0.186** | 4.322 |

Every walk-forward variant is optimistic by about a quarter of a point, and
**climatology carries three quarters of it**. Climatology cannot be wrong about
anything except the level - it is just the pooled training distribution - so the
bulk of the bias is the league simply scoring fewer D/ST points in the test
seasons than in the training ones. Only about 0.06 points is attributable to the
model. Recency weighting (variant F) was tested as a fix and made things
slightly worse, confirming the drift is not monotone and therefore not
predictable from season order.

Consequence: published percentiles run 1-2 pp optimistic, which nudges P(bust)
down and P(spike) up by a similar amount. It is small enough to state and live
with, and any correction would be fitted to the test seasons, which is exactly
the mistake this lab's methodology rules out.

## Reusable API

```python
import dst_distributions as dd

sim = dd.fit(train_df)                      # train_df must be strictly earlier
out = dd.distributions(sim, future_weeks)   # one row per defense-week
```

`out` carries `exp_pts, sd, p10, p25, p50, p75, p90, p_top12, p_top6, p_bust,
p_spike`. Each `(season, week)` is simulated jointly so that P(top-12) and
P(top-6) are ranks against that week's real field; both are null when the week
has fewer than 12 or 6 defenses. `keep_sims=True` additionally returns the raw
ensemble, which the calibration script needs and the pairing engine does not.

`python src/dst_distributions.py` runs a self-check: it fits on 2018-2021,
simulates 2022, and asserts monotone percentiles, probabilities in [0,1],
P(top-12) summing to exactly 12 per week, a mean within 1 point of reality, and
that the tier map is the shared one rather than a local copy.

## Limitations

- **Defenses within a week are simulated independently.** Two defenses in the
  same game are mechanically linked (my points allowed is your points scored),
  and that link is ignored. It biases P(top-12) only to the extent that both
  members of one game are contenders in the same week. Marked with a
  `ponytail:` comment for DST-007 to revisit if pair selection ever puts two
  defenses from the same game in a lineup.
- **No defense-specific quality term.** Every defense facing the same implied
  opponent total gets the same distribution. That is defensible given DST-002
  (defensive form alone scores 53.6% pairwise) and DST-005 (in-season own-defense
  ratings sit at r <= 0.09), but it means all spread within a market bucket is
  treated as noise. Pooled out of sample the overall shape survives it well -
  simulated sd 5.66 against an actual 5.53, simulated skew 0.87 against 0.85 -
  so the model is if anything marginally too wide rather than too narrow. (The
  Gaussian-copula variant was the one that came out too narrow, at sd 5.34 and
  skew 0.74; replacing it with the empirical copula is what closed the gap.)
- **Misc points** (safeties, blocked kicks; 0.197 pts per game, non-zero in 9%
  of games) are drawn independently of the copula.
- 3,000 simulations per week gives a Monte Carlo standard error of about 0.9 pp
  on a probability near 0.5 - well below the calibration bin widths used, but
  worth raising if a downstream decision turns on a difference of under 2 pp.
