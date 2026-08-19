# QB Pairing Research

**2026 fantasy football — can two quarterbacks beat one good one?**

League: 12-team, 1QB/2RB/3WR/1TE/K/DST, no flex, 7 bench, standard 4-point
passing TDs · Backtest: 2018–2025, walk-forward · Last revised: 2026-08-18

> **Revision notice.** An adversarial re-analysis (Wave 2) broke the central
> economic finding of the first version of this report. Retired figures are
> struck through rather than deleted, so the record shows what was claimed,
> what broke it, and why. Four bugs in the analysis code were found and fixed in
> the process. See [§14](#14-what-the-adversarial-pass-found).

Conclusions are labelled **[evidence]** (measured here), **[inference]**
(a reasonable reading), or **[hypothesis]** (proposed, not established).

---

## 1. Executive summary

**The thesis is not supported, and neither is the fallback position.**

The project set out to test whether two quarterbacks with complementary
schedules beat one expensive quarterback. Three measurements dismantle the
mechanism:

1. **Schedule difficulty barely predicts anything.** Adding opponent difficulty
   to a model that already knows recent form improves weekly start/sit accuracy
   by **0.06 percentage points**; on top of the betting market, **−0.01 pp**.
   [evidence]
2. **At draft time it is weaker still.** Team defensive ratings persist
   year-to-year at **r = 0.206**, and a prior-season rating explains **0.39%**
   of quarterback variance. [evidence]
3. **There is no complementarity to find.** Across 2,163 pairs, two
   quarterbacks' weekly outcomes correlate at **+0.006**. Their results are
   already independent. [evidence]

The first version then argued a second quarterback still paid for itself as
*insurance*, at ~~+9.7 points per season alongside an elite QB1 and +30.1
alongside a mid-priced one~~. **That was wrong.** Those figures compared
strategies that did not share the same QB1, so most of the gap was the QB1
upgrade. Holding QB1 fixed, a second quarterback **costs about 11 points a
season**, negative at all seven price points tested. [evidence]

**What survives:** the betting market carries real weekly information —
**+1.85 pp** of start/sit accuracy over a public projection, and **+0.55 pp**
even against a projection fitted with hindsight. The weekly decision itself is
worth **+5.0 points a season**, about 12% of the theoretically available upside.

**Recommendation: stop.** Draft one quarterback, use the bench slot elsewhere,
do not build a tool. §19 gives the specifics.

---

## 2. Thesis

> A fantasy manager can avoid spending heavily on a single premium QB by
> drafting two quarterbacks whose draft cost, complementary schedules,
> individual matchup characteristics, and weekly start decisions produce better
> value than one elite quarterback.

| Claim | Verdict |
|-------|---------|
| Schedules can be chosen to complement each other | **Rejected** [evidence] |
| Opponents affect different QBs differently | **Rejected** [evidence] |
| A manager can pick the right QB weekly | **Weakly supported** — 62.9% vs 59.8% [evidence] |
| Two cheap QBs beat one expensive one on value | **Rejected** — −10.6/season [evidence] |

---

## 3. League format and why it matters

- **One QB slot** means a second quarterback never adds points directly, only
  optionality. His entire value is the difference between his week and QB1's,
  on weeks you choose correctly, plus coverage when QB1 is out.
- **No flex** removes the usual escape valve; the bench slot is a pure cost.
- **4-point passing TDs** compress the position. QB12 averaged **237 season
  points** over the weeks this league actually plays.
- **The league finishes in week 16.** NFL weeks 17–18 are fantasy-irrelevant
  here — a fact that turned out to matter (§14).

---

## 4. Data-source inventory

Full detail in [`DATA_SOURCES.md`](DATA_SOURCES.md). Every source verified
against a live endpoint.

The material finding: **historical closing betting lines are free**, at 100%
coverage for 2007–2025 in nflverse schedules, removing the assumed dependency
on a paid odds provider. League scoring matched nflverse `fantasy_points`
exactly — 0.0 maximum difference across 1,389 QB-weeks.

Structural limit: **2026 forward lines stop after week 7**; weeks 14–16 have
lines for 4 of 47 games.

---

## 5. Betting-market methodology

Market inputs are **decomposed rather than used raw**: implied team total
carries the scoring environment, spread magnitude the game script, game total
the pace. The sign convention was verified empirically (`spread_line` positive
when home is favoured, correlation +0.443 with home margin).

**The market is the most useful weekly input, and it is essentially one
number.** Implied team total correlates 0.150 with the form residual, more than
double any other feature — and correlates **0.817 with the spread** and **0.574
with the game total**. It is not three signals:
`implied_team_total == (total_line + spread) / 2` to machine precision.

Consequently `spread_abs` was **removed from the model**: worth −0.01 pp, and
dropping it improved the market increment from +1.85 to +1.89 pp.

**Does the market beat a public projection?** Yes. Against a documented
consensus proxy, **+1.85 pp** (player-clustered SE 0.48, 95% CI [0.97, 2.87],
positive in 6 of 6 seasons), rising to **+4.63 pp** on close calls. Against a
projection *fitted with hindsight on the same season* — an upper bound no real
projection can beat — it still adds **+0.55 pp** (t = 2.25). [evidence]

The reverse also holds: given form and the market, adding the published
projection is worth −0.16 pp with a CI of [−0.69, +0.39] — **indistinguishable
from zero**, a weaker and more accurate statement than the first version's
"worth nothing".

**Unresolved threat:** nflverse carries *closing* lines. No opening line exists
in the archive, and a manager setting a lineup early in the week sees a
different number. This caveat applies to the entire market finding.

---

## 6. Prediction-market methodology

Kalshi carries free QB-level series (`KXNFLPASSTDS`, `KXNFLPASSATT`) — the same
data sportsbook APIs gate behind $99/month. **But they cannot be used here:**
queried 2026-08-17 those series had zero open markets (they activate in season),
and neither Kalshi nor Polymarket publishes a free price history, so neither can
be backtested. Polymarket's August NFL inventory is novelty markets.

**Disposition:** forward-only collection, so a price history exists next season.
Nothing in the thesis depends on them.

---

## 7. NFL scheduling research

Each effect measured against the form residual — actual minus the
quarterback's own 8-game average.

| Effect | Size | t | Verdict |
|--------|------|---|---------|
| Wind ≥ 15 mph | **−1.48** | −2.53 | real, but see below |
| Home field | **+1.32** | +4.38 | real, and universal |
| Away | −0.69 | −4.11 | real |
| Indoors | +0.65 | +2.88 | real |
| Outdoors | −0.47 | −3.28 | real |
| ~~Wind 8–14 mph~~ | ~~−0.57~~ **−0.45** | **−1.29** | **broken** |
| Division game | −0.33 | −1.65 | not significant |
| Short week (≤4 days) | +0.67 | +1.37 | not significant |
| Coming off bye | −0.59 | −1.21 | not significant |
| Weeks 14–18 | −0.02 | −0.07 | **no effect** |

**Rest does not matter**, including coming off a bye, which contradicts common
practice. **Weeks 14–18 score like any other weeks**, settling the
playoff-weighting question by evidence. And **weather is real but much smaller
than first reported** — the original figures were raw bucket means rather than
contrasts, with standard errors treating two quarterbacks in one game as
independent (§14).

The claim that weather was the largest scheduling effect is **false**: home
field is larger and applies to *every* game rather than 7% of them. Wind ≥15 mph
hits 7.1% of QB-weeks, and only 18.5% of within-week pairs are close enough for
a 1.5-point tiebreak to change the decision — an **upper bound of 0.66 points
per season**, against +5.0 for the entire start/sit decision.

### Game script: large, real, and useless

Pregame, competitiveness does not shift opportunity: pass and rush attempts are
flat across spread buckets (32.4 / 32.6 / 32.1 and 3.7 / 3.7 / 3.7). But how a
game *actually* unfolds matters enormously — losing big is worth **−3.81
points** (t = −17.8). It cannot be forecast from the closing line. A large
effect and a usable edge are different things.

---

## 8. QB-specific matchup effects

Archetypes from k-means on prior-only play style are interpretable — pocket
high-volume (Goff, Mahomes, Herbert; rushing share 0.097), downfield (Mayfield,
Allen, Stafford; aDOT 8.86), rushing (Jackson, Hurts, Fields; rushing share
0.307).

**No granularity of matchup adjustment beats using none at all:**

| Granularity | MAE | Pair accuracy |
|-------------|-----|---------------|
| **None** | **5.9183** | **64.17%** |
| League-wide | 5.9188 | 64.07% |
| Archetype | 5.9193 | 64.18% |
| Individual (per QB) | 5.9455 | 63.67% |

Two independent diagnostics confirm it. **Split-half reliability**: for 41
quarterbacks with 40+ games, matchup sensitivity in two random halves of their
own games correlates at **r = +0.065**. And a **random-slopes mixed model**
(lme4, 3,842 QB-weeks, 69 quarterbacks) finds the per-QB slope on opponent
difficulty does not earn its place — LRT **p = 0.130**, AIC flat, BIC worse, and
**the fit is singular**, meaning the data cannot identify separate slope
variance at all. Random-slope SD is 0.194 against a residual SD of 7.41.

**New result:** quarterbacks do not differ in *market* sensitivity either
(p = 0.495, also singular), so the implied-team-total coefficient applies
league-wide with no per-player tuning anywhere. [evidence]

---

## 9. Proposed QB-week model

Ridge regression, walk-forward. Standardised coefficients:

| Feature | Coefficient |
|---------|------------|
| Implied team total | **+1.427** |
| Rushing attempts (form) | **+1.081** |
| Career points/game | +0.966 |
| Game total | +0.569 |
| Recent form | +0.548 |
| Indoors | +0.320 |
| Home | +0.250 |
| Pass attempts (form) | +0.147 |

Opponent adjustment, QB×matchup interactions and `spread_abs` are **excluded by
evidence**, each measured neutral or negative.

**The linear form is not a limitation.** Gradient-boosted trees were fitted on
identical folds and every variant lost: LightGBM +0.0292 MAE and −0.56 pp,
XGBoost +0.0216 / −0.27 pp, residual boosting +0.0186 / −0.39 pp. The
decomposition is informative — univariate curvature costs 0.029 MAE and
*interactions add +0.0005 on top of that*. Trees cannot match linear main
effects at this sample size; offered `num_leaves ∈ {3,7,15}` the tuner picked 3
in five of six folds.

**Advanced tracking data adds nothing either.** Twenty-two features from PFR
advanced stats, Next Gen Stats and play-by-play were worth **−0.13 pp** in
aggregate. The mechanism is redundancy, not noise: six of them are *more* stable
year-to-year than fantasy points per game itself, and none helps, because how a
quarterback plays is already contained in the 8-game mean of what he scored.
**Stability is necessary but not sufficient.** [evidence]

A genuine football finding fell out of that work: **pressure faced is a
quarterback trait, not an offensive-line one.** At QB level it persists at
r = 0.526 against r = 0.320 at team level, and time to throw says the same
(0.660 vs 0.462). Pressure faced is mostly a function of how long he holds the
ball. [evidence]

---

## 10. Pair-complementarity framework

The framework exists in `src/qbpair/pairing/engine.py` and works. **The concept
it was built to exploit does not.**

- Correlation of two quarterbacks' pregame implied team totals: **+0.008**
- Correlation of their actual weekly fantasy points: **+0.006**

Outcomes are already essentially independent, so the diversification benefit of
holding two quarterbacks is the same whichever two you hold. The pregame matchup
signal has a standard deviation of 2.02 fantasy points against actual outcomes
at 8.09.

---

## 11. Draft-cost framework

Costs measured from real drafts, 2018–2025, over the weeks **this league
actually plays** (1–16):

| Picks | Mean usable PAR | Hit rate |
|-------|----------------|----------|
| 1–12 | 67.15 | 84.0% |
| 13–24 | 43.38 | 74.7% |
| 25–36 | 25.61 | 65.3% |
| 37–48 | 26.50 | 59.6% |
| 49–60 | 19.08 | 52.5% |
| 61–72 | 13.13 | 45.3% |
| 73–84 | 10.00 | 39.7% |
| 97–108 | 7.62 | 25.7% |
| 121–132 | 6.77 | 22.0% |
| 133–144 | 5.40 | 20.4% |

An independent reconstruction that never opened the original module reproduced
this: identical ADP input (max difference 0.0) and **14 of 15 bucket means
inside a season-clustered bootstrap interval**.

**Two presentation corrections it forced.** From pick 61 onward the **median**
pick returns exactly zero — the means above are carried by a minority of hits.
And buckets 6 through 13 are **not statistically distinguishable from one
another**, so quoting them as point estimates overstates the precision. The
load-bearing figure is **7.92 usable PAR for picks 106–135, 95% CI [5.0,
11.5]** — an interval overlapping the retired +9.7, meaning even on its own
terms that figure was never distinguishable from break-even.

---

## 12. Bench-slot opportunity cost

```
net value = QB points delivered − QB12 replacement − expected return of picks spent on QBs
```

The honest marginal holds QB1 **fixed** and adds a rank-18 partner:

| QB1 ADP rank | Alone | With QB2 | **QB2 adds** |
|---|---|---|---|
| 1 | 75.5 | 67.4 | **−8.1** |
| 2 | 80.9 | 71.1 | **−9.7** |
| 4 | 43.4 | 33.4 | **−9.9** |
| 6 | 87.0 | 75.9 | **−11.1** |
| 8 | 41.4 | 27.3 | **−14.1** |
| 10 | 75.5 | 70.9 | **−4.6** |
| 12 | 38.7 | 22.1 | **−16.7** |

**Pooled −10.6 points per season, negative at 7 of 7 price points.** [evidence]

~~Marginal value of QB2: +9.7 alongside an elite QB1, +30.1 alongside a
mid-priced one.~~ Retired — those differenced strategies with different QB1s.

The **insurance argument is also retired**. Decomposed weekly, bye coverage is
worth +1.0 and the weekly choice +1.7 points a season against a 5.8-point pick
cost — and admitting *more* injured QB1s to the sample *lowers* the estimate.
Injuries explain 8.8% of the original figure. The waiver wire supplies that
coverage free.

---

## 13. Regular season versus playoff weighting

**The correct playoff weight is zero.** [evidence]

Weeks 14–18 show no scoring effect (−0.02, t = −0.07). Prior-season defensive
ratings — the only playoff-schedule input available in August — explain 0.39%
of variance. And 2026 playoff weeks have lines for 4 of 47 games. A
playoff-schedule bonus would sort pairs on noise. The engine still *reports* the
weeks 14–16 split; it just should not be *scored*.

---

## 14. What the adversarial pass found

Every surviving finding was handed to an agent instructed only to break it. The
attacks began by **reproducing** everything — the pick-value curve to 0.0000
from a rebuild that never opened the original module, the strategy table to the
decimal, all three marginals. The failures below are real, not misreadings.

### The economics comparison was invalid

"Late pair" drafts ADP ranks 10 and 18; "single value QB" drafts rank 8.
Differencing them and attributing the gap to the second quarterback charges QB2
with the effect of changing QB1. Rank-matched, the sign reverses everywhere.

### No uncertainty had been reported, and it swamps the effect

The +9.7 carried SE 4.8 and CI [−3.8, +23.1], p = 0.116. The +30.1 carried
SE 34.0 and an interval **189 points wide**. Neither was ever distinguishable
from zero, and both had been quoted to one decimal place.

### A false positive that cleared its first null

SHAP surfaced `career_fp_mean × implied_team_total`, which passed a raw
within-week permutation null at family-wise **p = 0.000**. It is noise, caught
three ways: a scale-free share statistic (observed interaction share 14.6%
against a null mean of 29.2%, p = 1.000 — the real model is *half* as
interaction-driven as a noise-fitting one); an exhaustive scan of all 171
pairwise products with a max-statistic null (best gains 0.14% of baseline error
at p = 0.360; 37/171 improved against 34.1 expected by chance); and a temporal
split where all five top candidates hurt. **A single null is not enough when the
statistic is scale-dependent.**

### A second exploratory finding that did not replicate

The claim that the market's edge concentrates in weeks 10–18 (+3.26 pp vs
+0.74) was pre-registered and tested on held-out seasons 2014–2019: **+0.03 pp,
t = 0.15, 3 of 6 seasons**, CI [−2.68, +3.00]. On the generating set it had been
+2.60 pp at t = 6.27. The largest held-out bucket is weeks 5–9.

### Four bugs in the analysis code

1. **Pick-cost lookup compared a float ADP against integer bucket edges**, so an
   ADP of 48.1 fell between the 37–48 and 49–60 buckets and was charged
   **nothing** — 8% of all picks. The identical fix already existed in a later
   script and had never been backported, which is exactly how it survived.
2. **Season totals counted NFL weeks 17–18**, which this league does not play,
   inflating the pick-value curve ~4% at the top and ~8% in round two.
3. **A non-deterministic group-by** let floating-point summation order drift
   prior-season scoring by ~1e-15 — enough to flip a ridge alpha, so results
   were not bit-reproducible.
4. **Nine scripts wrote to a cwd-relative output path**, silently writing to a
   stray directory after the repository was restructured.

### What survived

Survivorship bias (2.0% of drafted players fail to join, worth 0.22 points),
mean-versus-median pick cost, 5% trimming, and — importantly — the market
increment, which held against both a 47-feature strengthened baseline and a
hindsight-fitted ceiling. Replacement level was also *measured* rather than
assumed: streaming actually lands at the **56.3rd percentile** of weekly
starters, not the median.

---

## 15. Benchmark methodology

Weekly start/sit rules, 1,717 pair-seasons, decisions locked pregame:

| Decision rule | pts/season | vs always-QB1 | Accuracy |
|--------------|-----------|---------------|----------|
| Always QB1 | 327.3 | — | 59.84% |
| ADP consensus | 327.3 | +0.0 | 59.84% |
| Recent form | 327.8 | +0.5 | 60.41% |
| Market | 328.0 | +0.6 | 60.41% |
| **Model** | **332.3** | **+5.0** | **62.91%** |
| Oracle | 367.9 | +40.5 | 100% |

The model captures **12.3%** of available upside. "Start your better preseason
quarterback" is already right 59.8% of the time.

**Streaming deserves emphasis.** It never had a losing season across the
backtest while *every* drafted strategy did, and it beat drafting a single
mid-priced quarterback outright.

**Benchmark limitation:** CBS publishes no historical archive — confirmed at
runtime, where the 2022/2025/2026 URLs return an identical 77-row table. A
documented consensus proxy was necessary. Note its r = 0.867 correlation with
the 2026 CBS board is *not* strong validation: raw ADP alone achieves 0.845, so
the proxy adds 0.013 and is merely ADP-shaped.

---

## 16. Limitations

| Limitation | Effect |
|-----------|--------|
| Eight-season window | Thin for interactions; mitigated with ridge, archetype-level effects, split-half and hierarchical testing |
| Starter identification excluded | Panel is "given he started"; deployment needs depth-chart/injury logic |
| Closing lines only | No opening line exists in the archive; managers set lineups earlier |
| All eligible pairs, not selected pairs | Measures the average pair; selection was never shown to add value |
| No historical player props | Prop features are forward-only |
| Prediction markets unbacktestable | No free price archive exists |
| Weather 38–75% populated | Missingness is benign (2,652 of 2,654 readings are outdoor games) but coverage is uneven |
| Six test seasons | Standard errors are large; several strategy differences are not distinguishable |

---

## 17. Findings that contradict the thesis

1. **Schedule complementarity does not exist to exploit** — weekly outcomes
   correlate +0.006.
2. **Opponent difficulty adds nothing once the market is known**, and almost
   nothing without it.
3. **QB-specific matchup effects do not replicate** — split-half r = +0.065,
   and the hierarchical model cannot even identify the variance.
4. **The draft-time schedule signal is ~0.39% of variance.**
5. **The QB2 economics reverse under a valid comparison** — −10.6/season.
6. **The insurance mechanism is refuted by its own data** — injuries explain
   8.8% of it.
7. **Nonlinearity does not help**; 83% of weekly variance is unforecastable
   from this panel.
8. **Advanced tracking data adds nothing** despite being highly stable.
9. **The weather effect was mis-estimated** and is worth ≤0.66 points/season.
10. **Two exploratory findings failed replication** — one SHAP interaction, one
    late-season market concentration.

**What does not contradict it:** the market genuinely carries weekly
information, and the weekly decision is genuinely worth ~5 points a season.
Those are the only load-bearing positives left.

---

## 18. Open modelling decisions

| Decision | Status |
|----------|--------|
| Backtest window 2018–2025 | Settled; extended 2007–2025 available |
| Sportsbook props | **Not purchased.** Free-first, by decision |
| Playoff weighting | **Resolved to zero** by evidence |
| Insurance premium | **Retired** — the finding it was calibrated on is broken |
| Starter identification | **Open** — needed before live deployment |
| Opening vs closing lines | **Open** — the largest unresolved threat to the market finding |

---

## 19. Recommended next build phase

**Verdict: stop.** Do not build the interactive tool.

A tool's two jobs would be the weekly start/sit call — worth 5 points a season
over a rule you apply in your head — and the pair ranking, now known to be
negative. Neither justifies a frontend.

What the evidence supports:

- **Draft one quarterback and do not draft a second.** Held at a fixed QB1, the
  backup costs ~11 points a season at every price point tested.
- **Use the roster spot on a skill position** — precisely the alternative the
  backup was failing to beat.
- **Do not pay for schedule.** One point per game between extremes, and
  unforecastable in August.
- **Cover the bye from the waiver wire.** Worth ~+1.0 points a season, and free.
- **Consider streaming** if you dislike variance. It never had a losing season.
- **The market is the one real weekly signal** — if you check anything on Sunday
  morning, check your quarterback's implied team total.
- **Do not spend another hour** on schedule modelling, matchup interactions, or
  playoff weighting.

The most valuable output of this project is the list of things not to bother
with — and the demonstration that a finding nobody attacks is not a finding.

---

## Reproducing

```bash
python qb-pairing/scripts/walk_one_qb_week.py 2024 11 "Josh Allen"   # leakage audit
python qb-pairing/scripts/phase2_signal.py                # feature signal
python qb-pairing/scripts/phase3_scheduling.py            # scheduling effects
python qb-pairing/scripts/phase4_qb_matchup.py            # matchup granularity
python qb-pairing/scripts/phase6_draft_cost.py            # economics (rank-matched)
python qb-pairing/scripts/phase7_backtest.py              # decision-rule benchmarks
python qb-pairing/scripts/wave1_hierarchical_matchup.py   # lme4, needs R
python qb-pairing/scripts/wave2_skeptic_economics.py      # the attack that broke it
```
