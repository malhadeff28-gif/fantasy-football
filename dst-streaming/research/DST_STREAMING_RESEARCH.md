# D/ST Streaming Research

**Two defenses, rotated across weeks 1–6, for a 12-team league**

Scoring: CBS/Yahoo points-allowed tiers · Backtest 2018–2025, walk-forward ·
2026 board built 2026-08-18 · Panel: 4,222 defense-weeks

---

## 1. Executive summary

**The strategy works, but not for the reason it was proposed, and the edge is
about a point and a half.**

Four independent lines of attack land in the same place: **the opponent's
implied team total is the entire signal.** Defensive form, opponent-offence
quality, opposing-QB identity, and prior-season defensive ratings each add
essentially nothing on top of it.

| Finding | Result |
|---|---|
| Best model vs. the trivial "lowest opponent implied total" rule | **+0.00 pp** |
| Opposing-QB identity, on top of team-level | +0.10 pp |
| Defensive form alone | 53.61% (coin flip = 50%) |
| Defensive touchdowns, season to season | **r = −0.075** (random) |
| Preseason defensive knowledge, half-life | ~6.5 weeks, CI [1.8, >50] |
| Drafting a second defense vs. streaming one | **+1.47 pts over weeks 1–6** |

**Recommendation:** draft one defense late if you want one, and stream the other
slot. The pair table below is real and usable, but the second *drafted* defense
buys +1.47 points for a pick worth roughly 4–7 points elsewhere.

---

## 2. Why weeks 1–6 — and the honest version of that argument

The window was chosen because forecasting gets harder further out. That is true,
but **not for the reason originally assumed.**

**What is true:** 2026 betting lines exist for **93 of 93 games in weeks 1–6**
and for 8 of 158 in weeks 8–18. The one predictor that works physically stops
existing after week 7. That alone justifies the window.

**What is not true:** that preseason knowledge of team quality survives into
week 6 and can be drafted on. Preseason defensive ratings decay with a half-life
of about **6.5 weeks (95% CI [1.8, >50])** — but they are weak *everywhere*, at
r ≈ 0.14 and 55% pairwise accuracy even in weeks 1–6. And in every window
including week 1, **preseason at its best (55.1%) loses to the market at its
worst (58.6%)**.

The sanity check that the decay method works: **the market shows no decay at
all**, best fit by a constant. A forward-looking price should not decay the way
a backward-looking rating does.

> **Do not draft defenses on last season's rankings.** [evidence]

---

## 3. What actually predicts a D/ST week

Walk-forward, train on seasons before S, test on S, S = 2020–2025:

| Model | MAE | Pair accuracy |
|---|---|---|
| League mean | 4.314 | — |
| **Opponent implied total alone** | **4.134** | **61.01%** |
| Defensive form only | 4.287 | 53.61% |
| Form + opponent offence | 4.190 | 58.12% |
| Everything | 4.116 | 60.95% |

A single variable beats every richer model, and adding features makes ranking
slightly *worse*.

**The benchmark was pre-registered before any model was fitted:** each week,
start whichever rostered defense faces the lowest opponent implied total. That
rule scores **61.01%**. The best model scores **61.01%**. The gap is
**+0.00 pp**.

### The opposing quarterback does not help either

The brief emphasised opposing-QB vulnerability most of all. Adding the opposing
quarterback's prior interception rate, sack rate, fumble rate, form and
experience is worth **+0.10 pp** — and the market-only model still beats the
model that knows who the quarterback is. A QB-only model reaches 57.92%.

Every QB trait correlates with the team-level residual at |r| < 0.031. The
brief's proposed combination — strong pass rush plus poor pressure-to-sack plus
elevated interception expectation — **did not win** an exhaustive pairwise
interaction search. The best product that did win is worth **0.17% of baseline
error**: detectable and useless.

### What is trait and what is noise

| Metric | r(season N, N+1) |
|---|---|
| Points allowed | 0.279 |
| Total D/ST points | 0.270 |
| Sacks | 0.195 |
| Interceptions | 0.171 |
| Fumble recoveries | 0.051 |
| **Defensive touchdowns** | **−0.075** |

Answering the brief directly: **defensive touchdowns are random**, with a
slightly negative point estimate. Even sacks — the largest scoring component at
2.40 points per game — persist at only 0.195.

---

## 4. Where D/ST points come from

Per game, on average:

| Component | Points/game |
|---|---|
| Sacks | **2.40** |
| Interceptions | 1.54 |
| Fumble recoveries | 1.04 |
| Points-allowed tier | 0.48 |
| Defensive TD | 0.41 |
| Special-teams TD | 0.21 |

Sacks dominate the **mean**; the points-allowed tier (range −4 to +10) and the
6-point defensive touchdown drive the **tails**. Floor and ceiling have
different sources, which is why the model simulates components separately rather
than fitting one distribution to the total.

---

## 5. The distributions are calibrated

3,230 walk-forward out-of-sample defense-weeks:

| Nominal percentile | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| **Observed coverage** | 10.2 | 25.8 | 51.9 | 77.0 | 91.0 |

Within 0.2–2.0 pp everywhere against a binomial standard error of 0.53 pp.
**21 of 23 reliability bins contain the prediction**, which is what chance
produces at 95%. Discrimination is real as well as calibration: AUC for top-12
is 0.632 against 0.510 for climatology.

Reported rather than hidden: a PIT Kolmogorov–Smirnov test formally rejects
uniformity (p = 0.007), but that is power at n = 3,230 — the largest CDF
departure anywhere is 3 pp. And every variant runs **+0.25 points optimistic**,
of which climatology carries +0.19, so the bulk is league scoring drift between
train and test seasons rather than model error.

Thresholds are derived, not chosen: **bust ≤ 2 points** (the median weekly
DST-24 line — the waiver wire beat you), **spike ≥ 14** (89th percentile).

---

## 6. The 2026 pair table

All 496 two-defense combinations scored across weeks 1–6, with the weekly
starter chosen **pregame** on expected points and the outcome drawn from the
calibrated simulator.

### Best overall, weeks 1–6

| Pair | ADP | W1–4 | W1–6 | p10 | p90 | Top-12 wks | Bad wks |
|---|---|---|---|---|---|---|---|
| **KC + NE** | 143/125 | 30.4 | **47.8** | **31.0** | **66.0** | 4.05 | 0 |
| DET + NE | 126/125 | 29.3 | 46.7 | 30.0 | 64.0 | 3.97 | 0 |
| NE + SEA | 125/84 | 29.0 | 46.4 | 29.0 | 65.0 | **4.14** | 0 |
| NE + SF | 125/undrafted | 28.7 | 46.2 | 29.0 | 64.0 | 4.07 | 0 |
| NE + PIT | 125/140 | 28.6 | 46.0 | 29.0 | 64.0 | 4.05 | 0 |

### Best first four weeks

| Pair | ADP | W1–4 | W1–6 |
|---|---|---|---|
| **KC + LAC** | 143/147 | **32.6** | 45.9 |
| JAX + SF | 151/undrafted | 32.1 | 44.8 |
| PIT + SF | 140/undrafted | 32.1 | 44.3 |
| LAC + SF | 147/undrafted | 31.9 | 44.8 |
| DET + LAC | 126/147 | 31.9 | 44.5 |

### Best inexpensive (both late or undrafted)

**NE + SF** (125/undrafted, 46.2) and **CHI + NE** (undrafted/125, 45.7) land
within 1.6 points of the best pair on the board at a fraction of the cost.

### Floor and ceiling

**KC + NE tops both lists** — highest 10th percentile (31.0) *and* highest 90th
percentile (66.0). There is no floor-versus-ceiling trade-off to exploit, which
is itself a finding: pairs that are good are good at both ends, because both
ends are driven by the same thing — facing bad offences.

### The honest caveat about this table

**New England appears in nine of the top ten pairs.** That is not a claim about
the New England defense. It is a schedule artifact: they face the Jets in week 6
(16.5 implied) and the Raiders in week 5 (18.0). The table ranks *schedules*,
exactly as intended — but do not read it as a scouting report.

---

## 7. The market signals that matter

Best single matchups in weeks 1–6, and what they have in common:

| Wk | D/ST | vs | Opp implied | Exp | p50 | p90 | P(top-12) | P(bust) |
|---|---|---|---|---|---|---|---|---|
| 6 | NE | NYJ | 16.5 | 9.12 | 8.0 | 17.0 | 64% | 11% |
| 1 | JAX | CLE | 16.5 | 8.90 | 8.0 | 17.0 | 56% | 12% |
| 2 | LAC | LV | 17.0 | 8.75 | 8.0 | 17.0 | 58% | 11% |
| 2 | SEA | ARI | 17.2 | 8.66 | 8.0 | 17.0 | 54% | 13% |
| 6 | LA | ARI | 17.0 | 8.62 | 8.0 | 16.0 | 61% | 13% |

Every one is an opponent implied total between 16.5 and 18.0. **That is the
whole signal**, and there is no second criterion worth applying.

Note the spread: the best matchup on the board has a median of **8** and a 90th
percentile of **17**. Even a perfect matchup is a coin flip between a dull week
and a good one, with an 11% chance of an outright bust. This is why the
distribution is the product and the point estimate is not.

**On double counting**, which the brief warned about: implied team total
correlates 0.817 with the spread and 0.574 with the game total, and
`implied_team_total == (total + spread) / 2` to machine precision. A defense
facing a low implied total, a large spread and a low opposing passing-TD prop
has not received three pieces of evidence. It has received one.

---

## 8. Does a second defense pay for its pick?

Judged on D/ST evidence alone. Defense A is held fixed; its best *drafted*
partner is compared against simply streaming the best undrafted defense each
week.

**Mean gain: +1.47 points over weeks 1–6 (se 0.24), positive for 16 of 16
defenses.** Consistently positive, and small.

| D/ST | A + streamer | A + drafted D2 | Gain |
|---|---|---|---|
| NYG | 41.6 | 44.9 | +3.3 |
| KC | 44.6 | 47.8 | +3.3 |
| SEA | 44.0 | 46.4 | +2.4 |
| BUF | 41.5 | 43.4 | +1.9 |
| DET | 45.1 | 46.7 | +1.7 |
| PHI | 44.0 | 44.9 | +0.9 |

Set against the measured value of the pick it costs. Defenses go around picks
84–160, where a non-D/ST pick returns **3.9 to 9.6 points above replacement**
over a full season.

**The direction is clear even though the units are not perfectly commensurable**
— six weeks of D/ST against a full season of a skill player. The second drafted
defense is not obviously worth its pick, and is probably slightly negative.
[inference]

---

## 9. Limitations

| Limitation | Effect |
|---|---|
| Six weeks only | The 2026 board cannot extend past week 7; lines do not exist |
| +0.25 pt optimism | Mostly league scoring drift, not model error; recency weighting made it worse |
| Tails ~5 pp thin | Tail probabilities in the pair table modestly understate |
| Defenses drawn independently within a week | The mechanical link between two sides of one game is ignored; matters only if you start two defenses from the same game |
| 16 drafted defenses | The marginal-value test rests on 16 comparisons |
| Closing lines | nflverse carries closing lines; lineups get set earlier |
| Preseason ratings excluded by evidence | If they matter in ways this panel cannot see, the board misses it |

---

## 10. What to actually do

1. **Do not draft a defense early.** Nothing about defensive quality predicts
   well enough to justify it, and defensive touchdowns are literally random.
2. **Draft one defense late, or none at all.** The second drafted defense buys
   about +1.47 points across six weeks for a pick worth several times that.
3. **Stream on opponent implied team total, and only that.** It beats every
   model built here, including the ones that know defensive form, opponent
   offence quality and the opposing quarterback.
4. **Target 16.5–18.0 implied opponent totals.** That is the whole board.
5. **Expect variance.** The best matchup available has a median of 8 and an 11%
   bust probability. Streaming is a distribution, not a plan.
6. **Ignore last season's defensive rankings.** Half-life ~6.5 weeks, and they
   lose to the market even in week 1.

---

## Reproducing

```bash
python dst-streaming/scripts/dst001_build_and_validate.py   # scoring validation
python dst-streaming/scripts/dst002_predictors.py           # predictors vs trivial rule
python dst-streaming/scripts/dst004_opposing_qb.py          # opposing-QB test
python dst-streaming/scripts/dst005_time_decay.py           # decay half-life
python dst-streaming/scripts/dst006_distributions.py        # calibration
python dst-streaming/scripts/dst007_pairing_engine.py       # the 2026 pair table
```
