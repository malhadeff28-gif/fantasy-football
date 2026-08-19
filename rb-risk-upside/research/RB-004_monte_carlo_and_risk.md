# RB-004 — Season Monte Carlo, calibration, and risk-adjusted upside

**Verdict: KEEP with a stated ceiling.** The distributions are calibrated
out of sample and the ranking is robust across six risk metrics and three role
specifications. But the backtest shows the simulator's *point* prediction is no
sharper than a naive ADP curve (MAE 57.4 vs 58.7 over 139 player-seasons), so
the product here is the **distribution**, not the central estimate. Differences
between Swift, Pollard and Judkins are inside that error bar. Montgomery's gap
is not.

Reproduce: `./.venv/Scripts/python.exe rb-risk-upside/scripts/rb004_simulate.py`

---

## 1. The generative model, and why it is shaped this way

RB-002 dictated the architecture. Volume persists (r ≈ 0.77); efficiency and
touchdown rate barely do (split-half r = 0.17 for TD rate). So:

1. **Role** is drawn from RB-003's comparables — pick a comparable weighted by
   inverse distance, take its realised role change and its games played.
2. **Production** is drawn from the pool of *historical RB-games at the implied
   role level*, not from a per-player efficiency estimate. This mechanically
   regresses touchdown rate and yards per carry to the league, which is exactly
   what the stability evidence says happens, rather than assuming it in a prior.
3. **The finish bar is itself random.** Each draw samples its own RB12/RB24
   threshold from the eight observed seasons, so year-to-year movement of the
   bar is inside the answer rather than hidden by an average.

Thresholds, 2018–2025, with 2018–2020 scaled by 17/16 for a like-for-like season:

| | mean | min | max | sd |
|---|---|---|---|---|
| RB3 | 279.2 | 224.4 | 312.2 | 29.0 |
| RB5 | 245.9 | 218.7 | 269.2 | 19.7 |
| RB12 | 191.9 | 177.6 | 201.1 | 9.0 |
| RB24 | 146.6 | 138.3 | 158.0 | 5.7 |
| RB36 | 108.8 | 97.3 | 118.4 | 7.2 |

---

## 2. Convergence — the measurement, not a guess

Five independent seeds at each size; the columns are the spread of the estimate
*across* seeds, which is the actual Monte Carlo error.

| draws | sd of median | sd of P(top-12) | sd of p90 |
|---|---|---|---|
| 2,000 | 1.18 | 0.0107 | 1.65 |
| 5,000 | 0.96 | 0.0063 | 2.66 |
| 10,000 | 1.03 | 0.0056 | 1.46 |
| **25,000** | **0.71** | **0.0038** | 1.22 |
| 50,000 | 0.32 | 0.0030 | 0.80 |
| 100,000 | 0.29 | 0.0030 | 0.41 |

25,000 draws already meets ±1.0 point on the median and ±0.005 on P(top-12).
**50,000 is used for headroom. 500,000 would buy nothing measurable** — the
error is already flat between 50k and 100k, because it is dominated by the
30-comparable sample, not by the draw count. Section 5b quantifies that.

---

## 3. Season outcome distributions (50,000 draws each)

| | Team | ADP | mean | MC se | p10 | p25 | **median** | p75 | p90 | sd | games | touches/gm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Quinshon Judkins** | CLE | 49.1 | 184.1 | 0.31 | **88.8** | **133.1** | **185.1** | 233.8 | **274.5** | 69.9 | 14.6 | 18.4 |
| **D'Andre Swift** | CHI | 41.9 | 158.9 | 0.34 | 59.5 | 110.8 | 159.6 | 205.7 | 257.1 | 75.0 | 14.0 | 16.3 |
| **Tony Pollard** | TEN | 63.7 | 142.2 | 0.31 | 46.7 | 92.2 | 147.5 | 187.8 | 228.6 | 69.4 | 13.7 | 15.0 |
| **David Montgomery** | HOU | 54.6 | 88.0 | 0.23 | 27.1 | 46.0 | 78.8 | 126.6 | 161.5 | 51.2 | 13.5 | 10.0 |

Finish probabilities:

| | P(top-3) | P(top-5) | P(top-12) | P(top-24) | **P(bust)** | P(disaster) |
|---|---|---|---|---|---|---|
| Judkins | 0.110 | **0.204** | **0.463** | **0.691** | **0.309** | 0.160 |
| Swift | 0.073 | 0.129 | 0.315 | 0.580 | 0.420 | 0.243 |
| Pollard | 0.034 | 0.070 | 0.234 | 0.505 | 0.495 | 0.307 |
| Montgomery | 0.001 | 0.002 | 0.031 | 0.156 | **0.844** | 0.662 |

### Definitions, and why these and not others

**Bust = finishes outside the top 24 RBs.** This league starts 12 teams × 2 RBs
with no flex, so exactly 24 RBs start and RB24 *is* the replacement line —
`scoring.LEAGUE.replacement_rank["RB"]` returns 24 for precisely this reason. A
pick spent in rounds 4–6 that does not return a startable RB returned nothing;
the same roster spot could have held a waiver-wire RB24. This is a stricter and
more honest bar than "finished below ADP", which lets a pick count as a success
for beating a low expectation.

**Disaster = outside the top 36**, i.e. not even a useful bench asset in a
12-team league where three RBs per team is a normal allocation.

**League-winning = a top-5 RB finish.** Not top-12: a top-12 RB drafted in round
4 is a good pick, but a good pick does not win a league. Top-5 is the outcome
that converts a round-4/5 selection into a round-1 asset and produces a weekly
positional edge over eleven opponents. Top-3 is reported alongside for readers
who want the stricter bar; it does not change any ordering.

---

## 4. Where this disagrees with the market

| | ADP | CBS (rescored) | sim mean | difference | CBS's percentile in our distribution |
|---|---|---|---|---|---|
| Judkins | 49.1 / RB22 | 178.3 | 184.1 | **+5.8** | 0.46 |
| Pollard | 63.7 / RB26 | 150.7 | 142.2 | −8.5 | 0.52 |
| Swift | 41.9 / RB20 | 202.2 | 158.9 | **−43.3** | 0.74 |
| Montgomery | 54.6 / RB24 | 172.3 | 88.0 | **−84.3** | **0.93** |

Two disagreements are large enough to act on.

**Montgomery (0.93).** CBS's projection sits at the 93rd percentile of our
distribution: it is close to our *best case*, treated as the base case. The
mechanism is visible in RB-001 and RB-003 — CBS projects 263 carries for a back
who took 158 last year, moved teams, and joins a room where Woody Marks handled
196 carries and a 43.6% share in 2025. The market has priced the March depth
chart; the comparable evidence says the median outcome for this exact profile is
around RB40.

**Swift (0.74).** CBS projects 9 rushing TDs. Swift scored 9 rushing TDs on 223
carries in 2025 (a 4.0% rate against a 3.2% league mean) while Kyle Monangai took
12 of Chicago's goal-line carries to Swift's 14. RB-002 says a TD rate above the
mean regresses about 85% of the way back. Consensus is projecting the good
version of a coin flip.

### The availability gap, isolated

CBS projects all four for 17 healthy games. Ours does not. Stripping that out:

| | CBS | sim mean if 17 games | sim mean all-in | cost of missed games |
|---|---|---|---|---|
| Swift | 202.2 | 182.6 | 158.9 | **23.7** |
| Pollard | 150.7 | 167.1 | 142.2 | **24.9** |
| Judkins | 178.3 | 210.7 | 184.1 | **26.6** |
| Montgomery | 172.3 | 106.7 | 88.0 | **18.7** |

**Roughly 19–27 points per back — a seventh of the expected total — is quietly
assumed away by any projection that pencils in 17 games.** On a like-for-like
17-game basis we are *above* CBS on Judkins (+32) and Pollard (+16), and the
Swift gap shrinks from −43 to −20. Most of our apparent pessimism is just
counting injuries; the genuine disagreement is Montgomery.

---

## 5. Risk-adjusted upside — six specifications

Bars: RB24 = 147, RB12 = 192, RB5 = 246. The ADP baseline is fitted on 772 RB
seasons, 2018–2025, points against log ADP positional rank (R² = 0.465).

| | ADP bar | sharpe | sortino | **omega** | P(top-5) | CVaR-up | **omega@ADP** |
|---|---|---|---|---|---|---|---|
| Judkins | 106.8 | **0.537** | **1.267** | **0.763** | **0.198** | **305.4** | **17.94** |
| Swift | 113.0 | 0.164 | 0.265 | 0.328 | 0.124 | 294.4 | 4.63 |
| Pollard | 96.1 | −0.064 | −0.085 | 0.156 | 0.063 | 262.3 | 4.98 |
| Montgomery | 101.2 | −1.146 | −0.765 | 0.005 | 0.001 | 184.4 | 0.55 |

**What each one is for, and why it exists:**

- **sharpe** = (mean − RB24) / sd. The textbook baseline. It punishes upside
  variance exactly as hard as downside variance, which is the wrong objective
  for a late-round pick where the whole point is buying a tail. Included as the
  thing to beat, not as the answer.
- **sortino** = (mean − RB24) / downside deviation. Same idea, but only shortfall
  below replacement counts as risk. Closer to what a drafter feels; still a
  mean-based summary and still blind to how big the upside is.
- **omega** = E[points above RB12] / E[shortfall below RB12]. **The headline.**
  Justification: in redraft, a bust is *droppable*. A back who finishes RB40 is
  cut in week 6 and replaced from waivers, so the realised loss is capped near
  the replacement line while the gain is not capped at all. Omega is the only
  specification here whose asymmetry matches that structural asymmetry. RB12 is
  the reference because that is the bar for "this pick gave me a genuine weekly
  starter advantage".
- **P(top-5)** — the raw league-winning probability with no risk adjustment at
  all. Included precisely so the reader can see how much work the adjustment is
  doing. Answer: none of the ordering depends on it.
- **CVaR-up** = mean of the best decile. The size of the prize when it lands, as
  opposed to the chance of it landing.
- **omega@ADP** — the same gain/shortfall ratio, but each player is measured
  against what a back at *his own* ADP has historically returned. The only
  specification that charges each player his own price; the other five compare
  all four against a common bar even though they cost 22 picks apart.

### Ranking under each specification (1 = best)

| | sharpe | sortino | omega | P(top-5) | CVaR-up | omega@ADP | spread |
|---|---|---|---|---|---|---|---|
| Judkins | 1 | 1 | 1 | 1 | 1 | 1 | **0** |
| Swift | 2 | 2 | 2 | 2 | 2 | 3 | 1 |
| Pollard | 3 | 3 | 3 | 3 | 3 | 2 | 1 |
| Montgomery | 4 | 4 | 4 | 4 | 4 | 4 | **0** |

**The answer is not an artefact of the metric.** Judkins is first and Montgomery
last under all six, including the two that penalise variance hardest and the one
that ignores risk entirely. The single flip is Swift ↔ Pollard on omega@ADP:
Pollard is 22 picks cheaper, so once you charge each back his own price the
cheaper, worse player is marginally better value. That flip is small (4.98 vs
4.63) and sits well inside the uncertainty in section 5b.

### Robustness to the biggest modelling choice

How a comparable's role transfers is the single largest judgement call. Three
specifications, 50,000 draws each:

| median | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| ratio (comp's role *change* applied to his own prior role) | 159.6 | 147.5 | 185.1 | 78.8 |
| absolute (comp's realised role level used directly) | 170.8 | 151.9 | 173.3 | 93.7 |
| blend (geometric mean) | 164.9 | 149.1 | 177.5 | 86.8 |

| P(top-12) | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| ratio | 0.315 | 0.234 | 0.463 | 0.031 |
| absolute | 0.354 | 0.257 | 0.376 | 0.082 |
| blend | 0.325 | 0.229 | 0.410 | 0.034 |

Judkins first and Montgomery last under all three. The **absolute**
specification is the most generous available reading of Montgomery — it says the
situation determines the workload regardless of the man — and it still leaves him
at P(top-12) = 0.08 and P(bust) = 0.77. His conclusion does not depend on the
specification. Swift closes to within 0.02 of Judkins on P(top-12) under
*absolute*, so the Swift/Judkins gap is the one that is specification-sensitive.

---

## 5b. How much of that ranking is actually real

The Monte Carlo error is negligible. The real uncertainty is that each
distribution rests on 30 comparable seasons. Resampling those 30 with
replacement, 200 reps:

| | median | 95% interval | P(top-12) | 95% interval |
|---|---|---|---|---|
| Judkins | 185.1 | **[154.7, 211.4]** | 0.463 | [0.318, 0.609] |
| Swift | 159.6 | [130.0, 185.2] | 0.315 | [0.198, 0.467] |
| Pollard | 147.5 | [114.1, 170.7] | 0.234 | [0.134, 0.365] |
| Montgomery | 78.8 | [57.8, 104.7] | 0.031 | [0.006, 0.061] |

P(row's median exceeds column's median) over the same resamples:

| | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| **Swift** | – | 0.79 | 0.11 | 1.00 |
| **Pollard** | 0.21 | – | 0.03 | 1.00 |
| **Judkins** | **0.89** | **0.97** | – | **1.00** |
| **Montgomery** | 0.00 | 0.00 | 0.00 | – |

This is the honest statement of confidence:

- Judkins > Montgomery, Swift > Montgomery, Pollard > Montgomery: **certain**
  (1.00 in every resample).
- Judkins > Pollard: **97%**.
- Judkins > Swift: **89%** — likely, not settled.
- Swift > Pollard: **79%** — a lean, nothing more.

---

## 6. Is the simulator calibrated? Walk-forward backtest

Every RB with a top-40 positional ADP in 2022–2025, simulated using only seasons
earlier than his own. 139 player-seasons.

| season | n | p10–p90 coverage | p25–p75 coverage | frac below median | MAE sim | MAE ADP curve |
|---|---|---|---|---|---|---|
| 2022 | 34 | 0.853 | 0.559 | 0.559 | 48.0 | 48.7 |
| 2023 | 36 | 0.750 | 0.500 | 0.528 | 54.6 | 58.9 |
| 2024 | 37 | 0.730 | 0.405 | 0.459 | 67.9 | 66.6 |
| 2025 | 32 | 0.688 | 0.531 | 0.406 | 58.4 | 60.0 |
| **all** | **139** | **0.755** | **0.496** | **0.489** | **57.4** | **58.7** |

Read this carefully, because it is the most important table in the project.

**Good:** the median is unbiased (48.9% of outcomes fall below it, nominal 50%)
and the interquartile range is almost exactly right (49.6% vs 50%). The
distributions are honest where it matters most.

**Mildly bad:** p10–p90 coverage is 75.5% against a nominal 80%. The tails are
about 5 points too thin — real seasons produce slightly more extreme outcomes
than 30 comparables can show. Every tail probability in section 3 (P(top-5),
P(disaster)) should be read as a modest **under**statement.

**The honest ceiling:** MAE 57.4 for the simulator against 58.7 for a curve that
knows nothing but ADP rank. **A 2% improvement.** The simulator does not know
who will score more points than the market does. This is entirely consistent with
RB-002, where nineteen workload metrics could not beat one lagged column either.

What the simulator *does* add over the ADP curve is a calibrated shape: an ADP
curve gives one number, this gives a floor, a ceiling, and a bust probability
that survive a coverage test. That is the deliverable.

The practical consequence: **a 25-point median gap between two of these backs is
inside a 57-point error bar.** Judkins over Swift is a lean supported at 89%, not
a fact. The 106-point gap from Judkins to Montgomery is nearly two MAEs, agrees
across all six metrics and all three role specifications, and is the one
separation this project can genuinely defend.

---

## Outputs

`rb-risk-upside/outputs/`: `rb004_distributions.png`,
`rb004_floor_ceiling.png`, `rb004_adp_vs_value.png`, plus
`rb004_{distributions,probabilities,risk_metrics,convergence,role_mode_robustness,comparable_uncertainty,backtest}.csv`.

## Limitations

- Tails are ~5 percentage points too thin (see coverage above).
- P(top-N) treats the rest of the RB population as fixed at its historical
  distribution. It does not simulate the other 60 backs jointly, so a season in
  which the whole position is unusually healthy or unusually injured is
  represented only through the sampled threshold.
- The comparable pool is 2019–2025. Seven years of committee-usage norms.
- No injury-history feature beyond prior games played. A specific known injury
  would not be visible to this model.
- The backtest covers 139 player-seasons across 4 years. Coverage estimates carry
  roughly ±0.07 at that sample size, so 0.755 is not distinguishable from 0.80
  with high confidence — the tail-thinness conclusion is a lean, not a
  certainty.
