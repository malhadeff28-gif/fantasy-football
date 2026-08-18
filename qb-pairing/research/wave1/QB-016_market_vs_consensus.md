# QB-016 — Does the betting market add information *after* a public projection?

**Wave 1 · Market Agent · 2026-08-18**
Script: `qb-pairing/scripts/wave1_market_vs_consensus.py`
Outputs: `qb-pairing/outputs/qb016_*.csv`

**Verdict: KEEP.** The betting market carries information a preseason public
projection does not have, and cannot have. Against a consensus-proxy baseline
the market is worth **+1.85 pp** of within-week start/sit ranking accuracy
(SE 0.35 across six test seasons, 95% CI [0.95, 2.75], positive in 6 of 6
seasons). Against the harder, more realistic baseline of projection **plus**
in-season form it is still worth **+0.99 pp** (SE 0.30, CI [0.22, 1.76]).

The surprise runs the other way: the **public preseason projection is the
redundant input**, not the market. Once you have recent form and the market,
adding the projection is worth **−0.16 pp** (SE 0.19) — nothing.

---

## 1. The obstacle, stated plainly, and verified at run time

CBS publishes only a **current** board. The script probes this on every run
rather than asserting it:

```
CBS archive probe: page lengths {2022: 788195, 2025: 788195, 2026: 788195},
all identical = True
```

The URLs for 2022, 2025 and 2026 return the same 788,195-character page and
parse to the same 77-row table, to the value. The season path segment is
ignored. **There is no accessible historical CBS archive, so "CBS + market"
cannot be backtested.** Anything claiming otherwise is fabricating data.

## 2. What was built instead — the consensus proxy

A stand-in for a published preseason projection, using only what provably
existed before week 1 of each season:

| input | source | availability |
|---|---|---|
| `prev_ppg`, `prev_games` | season S−1 fantasy PPG and starts (≥10 att) | finished season |
| `prev2_ppg`, `prev2_games` | season S−2 | finished season |
| `log_adp`, `adp_qb_rank`, `is_drafted` | FFC preseason ADP for season S | draft window closes before week 1 |
| `has_prev` | missingness flag | — |

ADP windows, printed by the script, all close before week 1 of their season
(e.g. 2018-08-28→09-04, 2020-08-25→09-01, 2025-08-30→09-01). Nothing in the
proxy could have been unknown in August.

These are fitted walk-forward (train seasons < S, predict S) to **weekly**
fantasy points, so the proxy *is* a per-week baseline expectation and is
constant within a player-season — structurally identical to a published
preseason projection. Panel: 4,268 QB-weeks, 2018–2025 (4,373 from
`features/panel.py`, unmodified, less 105 rows missing a form or market
feature). Test seasons 2020–2025: 3,283 rows, 49,026 decided within-week pairs.
61.1% of rows carry an ADP, 83.1% a prior season — the remainder are backups
pressed into starting, and the missingness flags carry that information.

**It is a proxy. It is not CBS.** Three validations:

**(a) Against realised outcomes**, out of sample, player-season level (≥4 starts):

| predictor | mean Pearson r | mean Spearman r |
|---|---|---|
| consensus proxy | **0.612** | **0.557** |
| ADP QB rank alone | 0.614 | 0.533 |
| prior-season PPG alone | 0.418 | 0.455 |

Honest reading: the proxy is essentially *ADP with a prior-season tilt*. ADP
alone matches it on Pearson and loses on rank correlation. This is a real
limitation — see §7.

**(b) Against a genuine public projection.** 2026 is the one season where both
the proxy and a real CBS board exist. Correlation of the two, over the whole
77-QB board: **Pearson r = 0.867, Spearman 0.762**; over the 24 drafted QBs,
**Pearson 0.858, Spearman 0.812**. The proxy is a credible stand-in for a real
published projection. This is the only direct check that can ever be run.

**(c) Optimism check.** The primary spec applies the proxy in-sample to
training rows. Proxy MAE in-sample (train) vs out-of-sample (test) by season:
6.45/6.48, 6.45/6.11, 6.34/5.68, 6.19/6.01, 6.15/6.44, 6.19/6.43 — no
systematic gap. A fully nested robustness spec (every proxy value produced by a
model fitted on strictly earlier seasons) reproduces every headline (§5).

## 3. Results — error metrics

Walk-forward, train seasons < S, test S, S = 2020–2025. No random splits.

| model | MAE | RMSE | pair acc, all | pair acc, close calls | pair acc, drafted pairs |
|---|---|---|---|---|---|
| consensus | 6.193 | 7.673 | 61.12 | 53.38 | 58.72 |
| **consensus + market** | **6.020** | **7.503** | **62.97** | **58.01** | **60.83** |
| consensus + form | 6.041 | 7.510 | 62.84 | 57.51 | 60.16 |
| **consensus + form + market** | **5.928** | **7.394** | **63.83** | **59.15** | **61.48** |
| form (Wave 0 baseline) | 6.048 | 7.542 | 62.53 | 57.20 | 59.91 |
| form + market (Wave 0) | 5.926 | 7.396 | 63.99 | 59.44 | 61.77 |

n = 49,026 decided pairs (all), 18,607 (close calls: |consensus gap| < 2.0 pts),
17,083 (both QBs drafted, i.e. a plausible real roster pair). Ties in the
outcome are excluded; a tie in the prediction scores 0.5.

**Wave 0 replicates exactly.** form → form+market on these rows is **+1.46 pp**,
matching the prior +1.46 pp figure to two decimals on an independently
constructed pair set. The pipeline is not drifting.

## 4. Results — the contrast that answers the question

Deltas with the standard error **across the six test seasons** (n = 6, so
t has 5 df and the intervals are wide — this is the binding limitation), plus
a cluster bootstrap over (season, week) blocks.

| contrast | metric | Δ | SE (seasons) | t | boot SE | 95% CI |
|---|---|---|---|---|---|---|
| **market on top of consensus** | pair acc, all | **+1.85 pp** | 0.35 | 5.28 | 0.38 | [0.95, 2.75] |
| | pair acc, close calls | **+4.63 pp** | 0.78 | 5.93 | 0.81 | [2.62, 6.64] |
| | pair acc, drafted pairs | **+2.11 pp** | 0.50 | 4.25 | 0.68 | [0.83, 3.38] |
| | MAE | −0.172 | 0.032 | −5.46 | | [−0.253, −0.091] |
| | RMSE | −0.170 | 0.022 | −7.77 | | |
| **market on top of consensus + form** | pair acc, all | **+0.99 pp** | 0.30 | 3.31 | 0.30 | [0.22, 1.76] |
| | pair acc, close calls | +1.64 pp | 0.62 | 2.65 | 0.58 | [0.05, 3.24] |
| | pair acc, drafted pairs | +1.32 pp | 0.38 | 3.50 | 0.50 | [0.35, 2.29] |
| | MAE | −0.114 | 0.028 | −4.02 | | [−0.186, −0.041] |
| market on top of form *(Wave 0 replication)* | pair acc, all | +1.46 pp | 0.30 | 4.84 | 0.33 | |
| form on top of consensus | pair acc, all | +1.72 pp | 0.41 | 4.23 | 0.32 | |
| **consensus on top of form** | pair acc, all | +0.31 pp | 0.15 | 2.02 | 0.26 | |
| **consensus on top of form + market** | pair acc, all | **−0.16 pp** | 0.19 | −0.87 | 0.18 | [−0.65, 0.32] |

Per-season market gain over consensus (all pairs), pp:
2020 +1.60, 2021 +0.55, 2022 +2.77, 2023 +1.74, 2024 +1.60, 2025 +2.86 —
**positive in 6 of 6**. Over consensus+form: +0.36, +0.53, +2.42, +0.84,
+1.00, +0.81 — also **6 of 6**.

**Rough points translation.** QB-013 measured the whole weekly start/sit
decision at +4.9 pts/season for +3.05 pp of accuracy, i.e. ~1.6 pts/season per
pp. On that ratio the market's increment over a projection on realistic roster
pairs (+2.11 pp) is worth roughly **+3.4 pts/season**, about 70% of the entire
measured value of the weekly decision. This is a linear extrapolation across a
different pair-selection rule, not a re-run of the Phase 7 pipeline; treat it
as an order of magnitude.

### Why the market cannot be subsumed — the model-free version

Correlation of each market variable with the **residual** of the baseline
(no fitting, no model choice involved):

| market variable | vs consensus residual | vs consensus+form residual |
|---|---|---|
| implied_team_total | **0.141** | **0.109** |
| total_line | 0.108 | 0.082 |
| team_spread | 0.097 | 0.076 |
| implied_opp_total | −0.021 | −0.018 |
| spread_abs | 0.003 | 0.007 |

Wave 0 measured implied_team_total at **0.150** against the *form* residual.
Against the *consensus* residual it is **0.141**. Swapping the baseline from
recent form to a preseason projection barely moves it, and adding form on top
only takes it to 0.109. This is the structural point: a preseason projection
is written in August, and the implied team total for week 11 encodes injuries,
depth-chart changes, weather and opponent state that did not exist in August.
No improvement to a *preseason* projection can absorb it.

### Effect size in natural units

Standardised coefficients in `consensus + market`, by season:

| season | consensus | implied_team_total | total_line | spread_abs |
|---|---|---|---|---|
| 2020 | 1.24 | 1.68 | 0.73 | −0.30 |
| 2021 | 1.68 | 1.49 | 0.72 | −0.20 |
| 2022 | 1.80 | 1.45 | 0.61 | −0.16 |
| 2023 | 1.81 | 1.51 | 0.63 | −0.17 |
| 2024 | 1.74 | 1.49 | 0.67 | −0.08 |
| 2025 | 1.65 | 1.59 | 0.56 | −0.03 |

Stable across six seasons. SD(implied_team_total) = 3.93 points, so **+1 SD of
implied team total ≈ +1.54 fantasy points**, or ~0.39 fantasy points per point
of implied team total. Across the realistic range of implied totals (roughly 17
to 30) the market moves a QB's weekly expectation by about **5 fantasy points**.
The market's weight is comparable to the projection's own (1.5 vs 1.7).

### When it helps (exploratory — no SEs, post-hoc slice)

| weeks | n pairs | consensus | +market | gain | gain over consensus+form |
|---|---|---|---|---|---|
| 1–4 | 11,770 | 60.93 | 61.67 | +0.74 | +0.88 |
| 5–9 | 12,597 | 62.76 | 62.89 | +0.13 | −0.43 |
| 10–18 | 24,659 | 60.38 | 63.63 | **+3.26** | +1.77 |

The market's edge over a static projection concentrates late, which is what a
decaying preseason projection predicts: consensus-only accuracy falls to 60.4%
in weeks 10–18 while consensus+market holds at 63.6%. Flagged as **exploratory**
— it is one of several slices examined and carries no multiplicity correction.

## 5. Robustness — fully nested proxy

Every proxy value produced by a model fitted only on strictly earlier seasons
(no in-sample proxy anywhere; costs the 2018 rows and thins early training):

| contrast | metric | Δ primary | Δ nested |
|---|---|---|---|
| market on consensus | pair acc, all | +1.85 (t 5.28) | **+1.79 (t 5.13)** |
| market on consensus | pair acc, close | +4.63 | +4.42 |
| market on consensus | MAE | −0.172 | −0.175 |
| market on consensus+form | pair acc, all | +0.99 (t 3.31) | **+0.78 (t 2.67)** |
| market on form | pair acc, all | +1.46 | +1.30 |

Conclusion unchanged. Full table in `outputs/qb016_contrasts.csv`.

## 6. Market redundancy — how much is one opinion counted twice?

`markets.market_redundancy` over all five market variables, n = 4,268:

| a | b | r |
|---|---|---|
| implied_team_total | team_spread | **0.817** |
| team_spread | implied_opp_total | **−0.818** |
| total_line | implied_opp_total | 0.578 |
| implied_team_total | total_line | **0.574** |
| implied_team_total | implied_opp_total | −0.336 |
| spread_abs | (all others) | \|r\| < 0.007 |

Not three independent signals. `implied_team_total` is `(total + spread)/2` by
construction and correlates 0.57 with `total_line` and 0.82 with `team_spread`.
Of the three variables used, **implied_team_total and total_line span roughly
two effective dimensions with 33% shared variance, and spread_abs is orthogonal
to both but carries a near-zero coefficient (−0.03 to −0.30, shrinking over
time) and a residual correlation of 0.003.**

Practical statement: **the market's contribution here is essentially one
number — implied team total.** Nothing in this experiment awards a quarterback
three separate market bonuses; the second and third variables are, at best, a
partial second dimension and a rounding error.

## 7. Limitations — read these before quoting the headline

1. **The consensus proxy is not CBS.** It is validated at r = 0.867 against the
   real 2026 CBS board, but that is one season and a correlation, not a
   backtest. The historical result is "market beats *a defensible preseason
   projection proxy*", not "market beats CBS".
2. **The proxy is ADP-anchored.** Because ADP is its strongest input, it cannot
   disagree with the draft market the way a real projection can — visible in
   §8, where CBS is bearish on Burrow/Maye/Stafford and the proxy is not. A
   real projection may therefore be somewhat better than the proxy, which would
   shrink the +1.85 pp figure. It would **not** shrink it to zero: the market
   variables are orthogonal to *any* August information (§4, residual
   correlations), and the increment survives against the strongest baseline
   available in the data (+0.99 pp over consensus+form).
3. **The baseline's strength sets the effect size.** Market over consensus
   (61.1% base) = +1.85 pp; market over consensus+form (62.8% base) = +0.99 pp.
   Report the number with its baseline attached.
4. **n = 6 test seasons.** Across-season standard errors carry 5 df. The CIs
   above are wide and honest; the block bootstrap agrees with them, which is
   reassuring but is not independent evidence about season-to-season variation.
5. **Closing lines.** `spread_line`/`total_line` from nflverse are closing
   lines. They are pregame — no look-ahead — but they are the *most informed*
   pregame price. A manager locking a lineup Sunday morning has a slightly
   earlier line. The effect is likely small but is untested here.
6. **De facto starters only** (≥10 attempts), inherited from `panel.py`. This
   is "given he started", and sets aside identifying the starter pregame.
7. **The week-bucket split (§4) is exploratory** and uncorrected for multiple
   comparisons.

## 8. Forward check — 2026 CBS board vs ADP (DESCRIPTIVE ONLY)

**This section cannot be validated. The 2026 season has not been played.** It
describes where two live public opinions disagree, nothing more. Both are
standardised within the 24-QB drafted pool; positive = CBS more bullish than
the draft market. Full table: `outputs/qb016_cbs_vs_adp_2026.csv`.

**CBS bullish relative to draft price**

| QB | team | ADP | ADP QB rank | CBS ppg | proxy ppg | z-gap |
|---|---|---|---|---|---|---|
| Jaxson Dart | NYG | 122.0 | 16 | 18.11 | 17.21 | +1.05 |
| Trevor Lawrence | JAX | 93.3 | 11 | 18.28 | 18.14 | +0.62 |
| Patrick Mahomes | KC | 100.0 | 12 | 17.95 | 18.04 | +0.51 |
| Baker Mayfield | TB | 137.5 | 17 | 17.05 | 16.93 | +0.48 |
| Jalen Hurts | PHI | 77.5 | 8 | 18.59 | 18.68 | +0.47 |
| Daniel Jones | IND | 166.8 | 24 | 16.50 | 16.30 | +0.46 |
| Malik Willis | MIA | 150.4 | 20 | 16.74 | 17.24 | +0.43 |
| Brock Purdy | SF | 88.1 | 9 | 18.13 | 18.44 | +0.38 |

**CBS bearish relative to draft price**

| QB | team | ADP | ADP QB rank | CBS ppg | proxy ppg | z-gap |
|---|---|---|---|---|---|---|
| Matthew Stafford | LA | 75.0 | 7 | 16.69 | 18.69 | −1.06 |
| Joe Burrow | CIN | 54.7 | 4 | 17.59 | 19.27 | −1.02 |
| Drake Maye | NE | 47.4 | 2 | 18.07 | 19.82 | −0.96 |
| Jordan Love | GB | 161.4 | 23 | 15.20 | 16.11 | −0.61 |
| Bo Nix | DEN | 119.7 | 15 | 16.17 | 17.36 | −0.48 |
| Jared Goff | DET | 106.0 | 13 | 16.62 | 17.64 | −0.39 |
| Lamar Jackson | BAL | 52.7 | 3 | 18.52 | 19.43 | −0.39 |
| Sam Darnold | SEA | 153.2 | 21 | 15.86 | 16.19 | −0.21 |

The sharpest disagreements sit at the top of the board: CBS is materially
below the draft market on Maye (QB2 by ADP), Burrow (QB4) and Stafford (QB7),
and materially above it on Dart (QB16) and Mayfield (QB17). Note the proxy
sides with ADP in every one of those cases — the ADP-anchoring limitation of
§7.2, visible in the open.

## 9. Verdict

**KEEP.** Betting-market variables carry incremental QB fantasy information
after a public preseason projection is known.

- Headline: **+1.85 pp** start/sit ranking accuracy over a consensus proxy
  (SE 0.35, CI [0.95, 2.75], 6/6 seasons positive); **+4.63 pp** on the close
  calls that are actually live decisions; MAE −0.172.
- Against the realistic baseline of projection **plus** in-season form:
  **+0.99 pp** (SE 0.30, CI [0.22, 1.76]), MAE −0.114. Smaller, still real.
- Mechanism is structural, not statistical: the market prices game-week
  information that no August projection can contain. Residual correlation of
  implied team total is 0.141 against the consensus residual vs 0.150 against
  the form residual (Wave 0) — essentially unchanged by swapping the baseline.
- Use **one** market variable. Implied team total dominates; total_line adds a
  little; spread_abs adds nothing.

**Secondary finding, worth its own line: the public preseason projection is the
redundant input.** Given form and the market, adding the consensus projection
is worth −0.16 pp (SE 0.19); given form alone it is worth +0.31 pp (SE 0.15).
`form + market` (63.99%) is as good as `consensus + form + market` (63.83%).
For the *weekly start/sit decision*, a manager who knows a QB's recent form and
this week's implied team total gains nothing further from the published
preseason projection. That is the opposite of the direction the question was
asked from, and it is the more useful half of the answer.
