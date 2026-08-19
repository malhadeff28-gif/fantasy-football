# QB-021 — Skeptic pass on the QB-016 market increment

**Wave 2 · Skeptic Agent · 2026-08-18**
Script: `qb-pairing/scripts/wave2_skeptic_market.py`
Outputs: `qb-pairing/outputs/qb021_*.csv`
Target: **QB-016** (`research/wave1/QB-016_market_vs_consensus.md`)

---

## Verdict summary

| # | Attack | Verdict |
|---|---|---|
| 1 | Stronger consensus proxy | **SURVIVES** — increment does not shrink (+1.85 → +1.72…+2.04 pp) |
| 2 | Hindsight ceiling | **SURVIVES, WEAKENED** — +1.35 pp against a realised-season oracle, **+0.55 pp** against the binding ceiling |
| 3 | Pre-registered week 10–18 concentration | **BROKEN** — does not replicate on held-out seasons (DiD +0.03 pp, t 0.15) |
| 4 | Redundancy / one variable | **SURVIVES** — +1.66 pp on implied team total alone; `spread_abs` is a net negative |
| 5 | Uncertainty under four clusterings | **SURVIVES** — every interval excludes zero; widest SE is 0.51, not 0.35 |
| 6 | Placebo: any weekly variable? *(added by the skeptic)* | **SURVIVES, strengthened** — non-market weekly context is worth +0.34 pp (t 0.98) |

**Headline, restated with its baseline attached and its ceiling bounded:**
the betting market is worth **+1.85 pp** (SE 0.35) over the QB-016 proxy at a
61.12% baseline, **+1.35 pp** (SE 0.19) over a hindsight leave-one-out realised
season projection at a 63.20% baseline, and **+0.55 pp** (SE 0.24, t 2.26,
5 of 6 seasons) over the strongest hindsight-fitted preseason projection that
can exist, at a 64.74% baseline. It is never zero and never negative in the
pooled data, but between 27% and 70% of the headline is baseline-weakness.

**One claim is broken outright:** the week 10–18 concentration.

---

## 0. How the attack was constructed

`wave1_market_vs_consensus.py` is **imported unmodified** and its panel builder,
ridge and pair rule are called directly — this is an attack on the same code,
not a re-implementation. The QB-016 headline reproduces to two decimals on the
first line of output:

```
panel rows 4,268   decided within-week pairs 49,026
QB-016 proxy  base 61.12%  market +1.85 pp (t 5.29)
```

Everything below is layered on that.

---

## 1. Attack 1 — build a deliberately stronger proxy

### 1.1 The ladder

Six honest rungs, nested from P2 upward. Every input is preseason-only:
prior-season aggregates come from finished seasons, the roster snapshot
(`nflreadpy.load_rosters`) is preseason, and birth date, entry year and draft
pick are fixed before a career starts.

| rung | what it adds | # features |
|---|---|---|
| `P0_prev_ppg_only` | prior-season PPG + missingness flag | 2 |
| `P1_adp_only` | preseason ADP only | 3 |
| `P2_qb016` | **the spec under attack**, verbatim | 8 |
| `P3_plus_efficiency` | prior-season YPA, TD rate, INT rate, CPOE, EPA/play, sack rate, pass/rush volume (S−1 and S−2) | 24 |
| `P4_plus_continuity_age` | roster continuity, team's prior-season QB production, age and age², experience and experience², rookie flag, draft pick, first-round flag | 37 |
| `P5_plus_multiseason` | three-season games-weighted averages (3:2:1) of PPG, YPA, TD rate, EPA/play, rushing yards, CPOE, plus S−3 | 47 |

### 1.2 The increment as a function of proxy strength — the actual deliverable

| proxy | OOS Spearman | proxy MAE | baseline pair acc | **market increment** | SE | t | +/6 |
|---|---|---|---|---|---|---|---|
| P0 prev PPG only | 0.460 | 6.297 | 59.48% | **+2.95 pp** | 0.60 | 4.89 | 6 |
| P1 ADP only | 0.581 | 6.197 | 60.92% | **+2.03 pp** | 0.58 | 3.50 | 6 |
| **P2 QB-016** | 0.554 | 6.192 | **61.12%** | **+1.85 pp** | 0.35 | 5.29 | 6 |
| P3 + efficiency | 0.574 | 6.186 | 61.22% | **+1.72 pp** | 0.33 | 5.20 | 6 |
| P4 + continuity/age | **0.577** | **6.183** | **61.30%** | **+1.85 pp** | 0.16 | 11.47 | 6 |
| P5 + multiseason | 0.570 | 6.186 | 61.03% | **+2.04 pp** | 0.23 | 8.86 | 6 |

OLS across the six honest rungs: `Δ = 41.28 − 0.644 × baseline_acc`, r = −0.986.
That relationship is real but is carried entirely by the two deliberately weak
anchors. **Within the P2–P5 cluster there is no trend at all**: 39 extra
preseason features move the baseline by 0.18 pp and the increment by ±0.2 pp,
in both directions.

**Verdict: SURVIVES.** The +1.85 pp is not a weak-baseline artefact *within the
space of preseason information*. Making the proxy genuinely better at the task
(P4: best OOS Spearman, best MAE, best baseline pair accuracy) leaves the
increment at exactly +1.85 pp.

### 1.3 The uncomfortable part: preseason information is already exhausted

The reason a stronger proxy does not kill the increment is that a materially
stronger *preseason* proxy could not be built. Adding efficiency, continuity,
age curves, draft capital and three-season weighting moves out-of-sample rank
correlation with realised season PPG from 0.554 to 0.577 and pairwise baseline
accuracy from 61.12% to 61.30%. That is the whole gain available from 39
additional preseason variables. This is evidence *for* QB-016's mechanism, not
against it — but it also means attack 1 alone cannot settle the question, which
is why attack 2 matters more.

### 1.4 A finding QB-016 should not have leaned on: the r = 0.867 validation

QB-016 validates the proxy at Pearson r = 0.867 against the real CBS 2026
board (n = 77 board rows, 24 of them drafted). Recomputed here alongside the
board's own raw ingredients, over those 24 drafted QBs:

| predictor of the CBS 2026 board | Pearson | Spearman |
|---|---|---|
| **−log(ADP), on its own** | **0.845** | **0.793** |
| −ADP QB rank, on its own | 0.791 | 0.793 |
| prior-season PPG, on its own | 0.579 | 0.535 |
| P1 proxy (ADP features only) | 0.821 | 0.793 |
| **P2 proxy (QB-016)** | **0.858** | **0.812** |
| P5 proxy (strongest) | 0.793 | 0.767 |

The QB-016 proxy beats raw ADP as a predictor of the CBS board by **0.013
Pearson and 0.019 Spearman**. The r = 0.867 headline is very close to
*"CBS and the draft market agree with each other"*, which was already known.
It is weak evidence that the proxy is a good stand-in for a projection, and
it is not evidence at all that the proxy is a good projection.

Two further points against quoting 0.867:

* **The relevant statistic is Spearman, not Pearson.** Pair accuracy is a rank
  criterion; a monotone transform of a projection scores identically. On rank,
  the QB-016 proxy agrees with CBS at 0.787 over the full 77-QB board and 0.812
  over the drafted pool, not 0.867.
* **The stronger proxies are *less* CBS-like while being *more* accurate.**
  P4 has the best out-of-sample rank correlation with realised outcomes (0.577)
  and the worst CBS agreement (0.715 Spearman). "Similar to CBS" and "good at
  the task" are different axes, and they point in opposite directions here.
  The instruction to confirm the stronger proxy beats r = 0.867 could not be
  satisfied and should not have been expected to be: it is the wrong target.

---

## 2. Attack 2 — the hindsight ceiling *(the decisive test)*

Three deliberately-cheating baselines, each an upper bound on what any public
preseason projection could achieve. None is admissible as a finding; all exist
only to bound the market's increment from above.

| ceiling | construction | legitimacy |
|---|---|---|
| `H1_hindsight_fit` | the 47-feature preseason model, ridge **refitted in-sample on the graded season** | the **binding** ceiling: strictly better than any real projection, uses no outcome the projection could not in principle rank |
| `H2_oracle_loo` | the QB's realised season mean **excluding the graded week** | a clean oracle: knows the season, not the week |
| `H2b_oracle_loo_shrunk` | H2 shrunk toward the season mean by 4 games | removes small-sample noise from H2 |
| `H3_oracle_full` | the QB's realised season mean **including the graded week** | **not a valid ceiling** — mechanically contaminated, reported only to show what contamination does |

### 2.1 Results

| baseline | baseline pair acc | **market increment** | SE | t | +/6 | over baseline+form |
|---|---|---|---|---|---|---|
| P2 QB-016 proxy | 61.12% | **+1.85 pp** | 0.35 | 5.29 | 6 | +0.99 pp |
| P5 strongest honest | 61.03% | **+2.04 pp** | 0.23 | 8.86 | 6 | +1.32 pp |
| H2b oracle, LOO shrunk | 62.02% | **+1.88 pp** | 0.20 | 9.61 | 6 | +1.54 pp |
| **H2 oracle, LOO** | **63.20%** | **+1.35 pp** | 0.19 | 7.19 | 6 | +1.30 pp |
| **H1 hindsight-fitted** | **64.74%** | **+0.55 pp** | 0.24 | **2.26** | **5** | +0.70 pp |
| H3 oracle, contaminated | 68.52% | +0.13 pp | 0.09 | 1.41 | 4 | +0.21 pp |

Per-season increment at the two ceilings (pp):

| season | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| H2 oracle LOO | +0.57 | +1.64 | +1.93 | +1.30 | +1.23 | +1.45 |
| H1 hindsight fit | **−0.27** | +0.34 | +1.35 | +0.84 | +0.11 | +0.93 |

### 2.2 Is it just oracle noise?

The obvious objection: a leave-one-out season mean is a noisy estimate of a
quarterback's true level, so the market may only be beating the noise. Tested
by restricting to pairs where **both** quarterbacks made ≥ 10 starts that
season, where the LOO mean is estimated from ≥ 9 games (n = 27,991 pairs):

| baseline | all pairs | both ≥ 10 starts |
|---|---|---|
| H2 oracle LOO | +1.35 pp (t 7.19) | **+1.39 pp** (t 8.43, 6/6) |
| H1 hindsight fit | +0.55 pp (t 2.26) | +0.58 pp (t 1.72, 5/6) |
| P2 QB-016 proxy | +1.85 pp (t 5.29) | +1.94 pp (t 3.71, 6/6) |

The increment is unchanged. It is not oracle noise.

### 2.3 The market's weight barely moves

Standardised ridge coefficients in `baseline + market`, by season:

| season | ITT vs P2 proxy | ITT vs H2 oracle | total vs P2 | total vs H2 |
|---|---|---|---|---|
| 2020 | 1.68 | 1.48 | 0.73 | 0.49 |
| 2021 | 1.49 | 1.22 | 0.72 | 0.51 |
| 2022 | 1.45 | 1.26 | 0.61 | 0.46 |
| 2023 | 1.51 | 1.31 | 0.63 | 0.45 |
| 2024 | 1.49 | 1.30 | 0.67 | 0.48 |
| 2025 | 1.59 | 1.34 | 0.56 | 0.39 |

Replacing the projection with an oracle that **knows the season's answer**
costs implied team total roughly 14% of its standardised weight. Its
information is not something a better projection contains.

### 2.4 Verdict on attack 2

**SURVIVES, WEAKENED.**

* Against the honest LOO oracle: +1.35 pp, SE 0.19, 6 of 6 seasons — 73% of
  the headline retained.
* Against the binding hindsight-fitted ceiling: **+0.55 pp, SE 0.24, t 2.26,
  5 of 6 seasons, and negative in 2020** — 30% of the headline retained, and
  only marginally distinguishable from zero on six seasons.
* On close calls, the +4.63 pp figure becomes +3.34 pp at the LOO oracle and
  +1.84 pp at the binding ceiling (each measured on the slice that baseline
  itself finds close; on the *fixed* QB-016 close set the ceilings give +1.80
  and +0.94 pp).
* H3 shows what contamination does: allowing the baseline to see 1/17th of the
  graded week's own outcome adds 5.3 pp of baseline accuracy and collapses the
  market's increment to +0.13 pp. It is reported to make the point that a
  "ceiling" which peeks at the answer is not a ceiling.

**The claim "the market carries information no preseason projection can have"
is upheld. The claim that this is worth ~1.85 pp is upheld only against a
realistically achievable projection; the irreducible floor is closer to
+0.55 pp.**

---

## 3. Attack 3 — pre-registering the week 10–18 claim

### 3.1 Pre-registration, fixed before touching the held-out data

> **H1** The market's increment over the QB-016 proxy is **larger in weeks ≥ 10
> than in weeks 1–9** (one-sided, α = 0.05).
> **H2** The market's increment in weeks ≥ 10 is positive.
> **Test set:** seasons **2014–2019**, which QB-016 never used. The panel is
> extended back to 2012 (`nflreadpy` player stats to 2010, FFC ADP to 2010,
> nflverse closing lines complete from 2008) so those seasons have two prior
> seasons and real preseason ADP.
> **Statistic:** per-season delta, SE across the six held-out seasons (5 df),
> plus a (season, week) cluster bootstrap on the late-week increment.

Extended panel: 7,135 QB-weeks, 2012–2025; 89,759 decided pairs over test
seasons 2014–2025.

### 3.2 Result

| proxy | season set | early (1–9) | late (≥10) | **DiD (late − early)** | SE | t | seasons DiD > 0 |
|---|---|---|---|---|---|---|---|
| P2 QB-016 | **HELD-OUT 2014–2019** | +2.37 pp | +2.41 pp | **+0.03 pp** | 1.10 | **0.15** | **3 / 6** |
| P2 QB-016 | generating 2020–2025 | +0.15 pp | +2.74 pp | +2.59 pp | 0.42 | 6.18 | 6 / 6 |
| P5 strongest | **HELD-OUT 2014–2019** | +3.00 pp | +2.83 pp | **−0.17 pp** | 0.70 | **−0.12** | **3 / 6** |
| P5 strongest | generating 2020–2025 | +0.49 pp | +2.81 pp | +2.32 pp | 0.47 | 4.94 | 6 / 6 |

Three-bucket view, matching QB-016 §4:

| proxy | set | 1–4 | 5–9 | 10–18 |
|---|---|---|---|---|
| P2 | HELD-OUT 2014–2019 | +1.25 | **+3.42** | +2.41 |
| P2 | generating 2020–2025 | +0.06 | +0.23 | **+2.74** |

**H1 fails.** On seasons QB-016 never saw, the difference between late and
early is +0.03 pp with a 95% interval of roughly [−2.65, +2.99] and the sign
splits 3–3 across seasons. On the held-out data the market's largest bucket is
weeks 5–9, not 10–18. The concentration is a property of 2020–2025, not of the
mechanism.

**H2 survives** (+2.41 pp late, SE 0.59, t 4.09, 6/6, bootstrap CI
[1.35, 3.49]) — but it survives trivially, because the early increment is
+2.37 pp on the same seasons. "The market helps late" is true only in the
sense that the market helps everywhere.

**Verdict: BROKEN.** QB-016's §4 week-bucket table should be struck, not merely
flagged exploratory. The mechanism QB-016 offered for it — "a decaying preseason
projection" — is not supported: on held-out seasons the baseline also decays
(59.82% early → 58.08% late) without the increment growing.

### 3.3 Bonus — the headline claim replicated out of sample

The same extended run grades QB-016's *main* claim on seasons it never used:

| proxy | season set | baseline | +market | **Δ** | SE | t | +/n |
|---|---|---|---|---|---|---|---|
| P2 QB-016 | **HELD-OUT 2014–2019** | 58.96% | 61.35% | **+2.39 pp** | 0.46 | 5.15 | 6/6 |
| P2 QB-016 | generating 2020–2025 | 61.55% | 63.00% | +1.45 pp | 0.52 | 2.79 | 5/6 |
| P2 QB-016 | all extended 2014–2025 | 60.38% | 62.25% | +1.88 pp | 0.36 | 5.33 | 11/12 |
| P5 strongest | **HELD-OUT 2014–2019** | 58.70% | 61.62% | **+2.92 pp** | 0.51 | 5.65 | 6/6 |
| P5 strongest | all extended 2014–2025 | 60.31% | 62.54% | +2.23 pp | 0.37 | 6.14 | 12/12 |

The core finding replicates cleanly on twelve seasons and 89,759 pairs. Two
caveats that cut against over-reading it:

1. The held-out increment is **larger** partly because the baseline there is
   **weaker** (58.96% vs 61.55%) — 2014–2016 train on only two or three prior
   seasons. This is the §1.2 relationship again, not independent evidence.
2. **Specification sensitivity worth recording:** merely extending the training
   window from 2018+ to 2012+ moves the 2020–2025 increment from **+1.85 pp to
   +1.45 pp** on essentially the same pairs. The panel's form and career
   features see more history, the proxy trains on more seasons, and the point
   estimate moves 0.4 pp. QB-016 does not report this sensitivity. The sign and
   significance are unaffected; the second decimal is not real.

---

## 4. Attack 4 — redundancy, verified independently

### 4.1 The correlation matrix, recomputed from raw panel columns

Computed with `np.corrcoef` on the panel rather than calling
`markets.market_redundancy`, n = 4,268:

| a | b | r | QB-016 reported |
|---|---|---|---|
| implied_team_total | team_spread | **0.817** | 0.817 |
| team_spread | implied_opp_total | **−0.818** | −0.818 |
| total_line | implied_opp_total | 0.578 | 0.578 |
| implied_team_total | total_line | **0.574** | 0.574 |
| implied_team_total | implied_opp_total | −0.336 | −0.336 |
| spread_abs | everything else | \|r\| ≤ 0.006 | \|r\| < 0.007 |

Reproduces exactly. And the identity is not approximate:

```
max |implied_team_total - (total_line + team_spread)/2| = 0.00e+00
```

Implied team total **is** the spread and total, rearranged. There are two
market numbers here, not three, and one derived restatement of them.

### 4.2 Does the increment survive on one variable?

All measured against the same QB-016 proxy baseline of **61.12%**, n = 49,026:

| specification | acc | **Δ vs baseline** | SE | t | boot 95% CI | Δ close calls | +/6 |
|---|---|---|---|---|---|---|---|
| all three (QB-016) | 62.97% | **+1.85 pp** | 0.35 | 5.29 | [1.12, 2.61] | +4.63 pp | 6 |
| **implied team total alone** | 62.78% | **+1.66 pp** | 0.39 | 4.25 | [0.94, 2.35] | +4.11 pp | 6 |
| implied team total + total | 63.01% | **+1.89 pp** | 0.37 | 5.12 | [1.18, 2.64] | +4.72 pp | 6 |
| total line alone | 61.96% | +0.84 pp | 0.27 | 3.10 | [0.25, 1.40] | +2.17 pp | 5 |
| team spread alone | 61.68% | +0.56 pp | 0.24 | 2.34 | [0.11, 1.05] | +1.58 pp | 5 |
| implied opp total alone | 61.15% | +0.03 pp | 0.11 | 0.33 | [−0.10, 0.17] | +0.08 pp | 3 |
| **spread_abs alone** | 61.11% | **−0.01 pp** | 0.08 | −0.05 | [−0.21, 0.19] | −0.02 pp | 3 |

**Verdict: SURVIVES, and QB-016's own redundancy claim is confirmed and
sharpened.** Implied team total alone delivers 90% of the three-variable
increment with a bootstrap interval that excludes zero. `spread_abs` is not
merely useless — dropping it makes the model *better* (+1.89 vs +1.85 pp).

**Recommendation to the model: use `implied_team_total`, optionally
`total_line`, and delete `spread_abs`.** `projection.py` currently carries all
three in `FEATURES`; the third earns nothing.

---

## 5. Attack 5 — uncertainty under four clusterings

SE 0.35 came from six test seasons (5 df). Recomputed with a (season, week)
block bootstrap, a season-cluster bootstrap, and a **player** cluster bootstrap
(multiway: resample players, weight each pair by `m_i × m_j`), 2,000 / 2,000 /
1,000 replicates.

| contrast | Δ | SE seasons | SE (season,week) | SE season | **SE player** | all CIs exclude 0 |
|---|---|---|---|---|---|---|
| market on P2 proxy | **+1.85 pp** | 0.35 | 0.38 | 0.32 | **0.51** | **yes** |
| market on P2 proxy, close calls | **+4.63 pp** | 0.78 | 0.81 | 0.67 | **1.08** | **yes** |
| market on P2 proxy + form | **+0.99 pp** | 0.30 | 0.30 | 0.27 | 0.36 | **yes** |
| market on P5 strongest proxy | **+2.04 pp** | 0.23 | 0.29 | 0.22 | 0.40 | **yes** |
| market on hindsight LOO oracle | **+1.35 pp** | 0.19 | 0.32 | 0.16 | 0.37 | **yes** |
| projection on form + market | −0.16 pp | 0.19 | 0.18 | 0.17 | 0.27 | no |

95% intervals for the headline:

| clustering | 95% CI |
|---|---|
| across seasons (t, 5 df) | [0.95, 2.75] |
| (season, week) blocks | [1.10, 2.57] |
| season clusters | [1.25, 2.43] |
| **players (widest)** | **[0.90, 2.84]** |

Leave-one-season-out range of the pooled delta: **[1.65, 2.11]** — no single
season carries it. For the close-call figure the jackknife range is
[4.16, 5.09]; for market-on-proxy+form, [0.71, 1.11].

**Verdict: SURVIVES.** Player clustering is the binding one and widens the SE
by 44% (0.35 → 0.51), which QB-016 did not test. The interval still excludes
zero comfortably.

**Correction to the secondary claim.** QB-016's "given form and the market, the
projection is worth −0.16 pp" is correctly reported as insignificant, but the
player-clustered interval is **[−0.70, +0.37]**. The data are consistent with
the projection being worth up to +0.37 pp. "Indistinguishable from zero on six
seasons" is supportable; "worth nothing" is over-stated.

---

## 6. Attack 6 — placebo (added by the skeptic, not requested)

The proxy is **constant within a player-season**; the market varies weekly. Any
week-varying variable would therefore add something. So the market was raced
against a non-market weekly context set — `is_home`, `indoors`, `rest`,
`div_game`, opponent-defence adjustment — on the same baseline.

| contrast | baseline | augmented | Δ | SE | t | +/6 |
|---|---|---|---|---|---|---|
| **non-market weekly context** on proxy | 61.12% | 61.46% | **+0.34 pp** | 0.33 | **0.98** | **3** |
| market on proxy | 61.12% | 62.97% | **+1.85 pp** | 0.35 | 5.29 | 6 |
| market on proxy + context | 61.46% | 62.89% | **+1.44 pp** | 0.23 | 6.16 | 6 |
| context on proxy + market | 62.97% | 62.89% | **−0.08 pp** | 0.20 | −0.42 | 2 |

**Verdict: SURVIVES, and the finding is strengthened.** Weekly variation per se
buys +0.34 pp with t = 0.98 and 3 of 6 seasons — nothing. The market buys
+1.44 pp *after* that context is already in the model, and the context is fully
subsumed by the market (−0.08 pp). This is not "any weekly signal"; it is the
market specifically.

---

## 7. What broke, what bent, what held

**Broke**
* The **week 10–18 concentration** (+3.26 pp late vs +0.74 early). Held-out
  DiD = +0.03 pp, t 0.15, 3 of 6 seasons. Strike it.

**Bent**
* **The effect size is more baseline-dependent than +1.85 pp suggests.** Against
  the binding hindsight ceiling it is +0.55 pp (t 2.26, negative in 2020).
  Always quote the baseline; the honest range is +0.55 to +2.04 pp.
* **The r = 0.867 CBS validation is weak.** Raw −log(ADP) alone reaches 0.845.
  The proxy adds 0.013 Pearson over the draft market. Quote Spearman (0.787
  board / 0.812 drafted) and describe it as "ADP-shaped like CBS", not as a
  validated projection.
* **The point estimate is specification-sensitive at ±0.4 pp.** Extending the
  training window 2018+ → 2012+ moves the 2020–2025 increment from +1.85 to
  +1.45 pp.
* **The secondary claim is over-stated.** "The projection is worth −0.16 pp"
  should read "is indistinguishable from zero, 95% CI [−0.70, +0.37]".
* **`spread_abs` should be deleted** from `MARKET` and from
  `models/projection.py`'s `FEATURES`. Dropping it improves the increment.

**Held**
* The market adds after **any** preseason projection that can actually be built
  (+1.72 to +2.04 pp across six proxy specifications, 6 of 6 seasons each).
* The market adds after a **hindsight oracle that knows the season's answer**
  (+1.35 pp, t 7.19, 6/6; +1.39 pp on well-estimated pairs).
* The market adds after **projection + in-season form** (+0.99 pp, all four
  clusterings exclude zero).
* The market's contribution is **one number**: implied team total alone gives
  +1.66 pp of the +1.85 pp.
* The claim **replicates out of sample** on six seasons QB-016 never touched
  (+2.39 pp, t 5.15, 6/6).
* It is **not a generic weekly-variable effect** (non-market weekly context:
  +0.34 pp, t 0.98).

---

## 8. Limitations of this attack

1. **Closing lines remain untested.** `spread_line` / `total_line` from
   nflverse are closing prices. They are pregame — no look-ahead — but they are
   the most informed pregame price, and no opening line exists in the cache. A
   manager setting a Sunday-morning lineup has a slightly earlier number. This
   is QB-016's limitation 5, and QB-021 could not resolve it either. It is the
   single largest remaining threat to the finding and needs a data purchase or
   a prospective log to settle.
2. **The hindsight ceilings are constructions, not projections.** H1 is the
   best in-sample linear read of a season using preseason features; a different
   functional form could sit higher and would leave the market less room.
   +0.55 pp is a floor estimate under *this* ceiling, not a theorem.
3. **Held-out seasons 2014–2019 have thinner training data.** 2014 trains on
   two seasons. That weakens the baseline and, per §1.2, inflates the
   increment. The out-of-sample replication is real but its magnitude is not
   comparable to the 2020–2025 figure.
4. **ADP windows in 2011–2013 close on or one day before the Thursday opener.**
   Those seasons are training-only for the held-out test (which starts in
   2014, window closing 2014-09-01 against a 2014-09-04 opener), so the
   pre-registered result is unaffected, but the extended panel is not free of
   the issue in its earliest training rows.
5. **De facto starters only** (≥ 10 attempts), inherited from `panel.py`. The
   separate problem of identifying the starter pregame is untouched, and the
   market's known strength there is not measured.
6. **The proxies are ridge-linear.** A non-linear preseason projection
   (gradient boosting on the P5 feature set) was not tried; QB-017 found
   non-linearity worth little on this panel, but that is an assumption carried
   in, not a result reproduced here.
7. **Multiplicity within this attack.** The six headline contrasts were fixed
   before running. The proxy ladder is a specification curve, not six
   hypothesis tests, and the week-bucket test is genuinely pre-registered. The
   market-subset table in §4.2 is exploratory and its individual t-statistics
   carry no correction.

---

## 9. Recommended disposition of QB-016

**KEEP, with amendments.** The market increment survives a stronger baseline,
a hindsight ceiling, four clusterings, an independent redundancy check, a
non-market placebo and an out-of-sample replication on six unseen seasons.

Required edits to the QB-016 write-up:

1. Delete or retract §4's week-bucket table and the "concentrates late" claim.
2. Restate the headline as baseline-bracketed: **+1.85 pp against a defensible
   public-projection proxy (61.12% baseline), +1.35 pp against a realised-season
   oracle (63.20%), +0.55 pp against the strongest projection that could
   possibly exist (64.74%).**
3. Replace "validated at r = 0.867 against the real CBS board" with the
   Spearman figures and the observation that raw ADP alone reaches 0.845 /
   0.793 — the validation shows the proxy is CBS-shaped, not that it is good.
4. Soften the secondary claim to "indistinguishable from zero, CI
   [−0.70, +0.37]".
5. Drop `spread_abs` from the market feature set.

---

## 10. Reproduction

```
./.venv/Scripts/python.exe qb-pairing/scripts/wave2_skeptic_market.py
```

Runs end to end in **11 seconds** on a single BLAS thread with warm caches.
Measured peak working set **719 MB** (peak commit 929 MB), driven by the
extended 2012–2025 panel build — run at most one other heavy agent alongside it.
Network is used once to cache FFC ADP (2010–2026), nflverse rosters
(2010–2026), nflverse player stats (2010–2025 and 2011–2025) and the CBS 2026
board; every subsequent run is offline from parquet, and the bootstraps are
seeded, so the numbers above reproduce exactly.

| output | contents |
|---|---|
| `qb021_proxy_ladder.csv` | attack 1 + 2 — increment vs proxy strength, every rung |
| `qb021_ceiling.csv` | attack 2 — ceilings, with the ≥ 10-start subset |
| `qb021_cbs_proxy_validation.csv` | proxy vs CBS 2026, plus the raw ADP ingredients |
| `qb021_proxy_oos_quality.csv` | per-season out-of-sample proxy rank correlations |
| `qb021_ladder_by_season.csv` | per-season increment for every rung |
| `qb021_ladder_error_metrics.csv` | MAE / RMSE for all 52 fitted model specs |
| `qb021_market_coefficients.csv` | standardised market coefficients by baseline |
| `qb021_prereg_week_late.csv` | attack 3 — the pre-registered DiD |
| `qb021_prereg_week_buckets.csv` | attack 3 — three-bucket view, both season sets |
| `qb021_extended_replication.csv` | attack 3 bonus — headline on held-out seasons |
| `qb021_market_redundancy.csv` | attack 4 — correlation matrix, recomputed |
| `qb021_market_variable_subsets.csv` | attack 4 — every market subset |
| `qb021_uncertainty.csv` | attack 5 — four clusterings, CIs, jackknife |
| `qb021_weekly_placebo.csv` | attack 6 — non-market weekly context |
