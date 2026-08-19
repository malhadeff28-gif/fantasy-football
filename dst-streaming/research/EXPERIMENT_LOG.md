# D/ST Streaming - Experiment Log

Every meaningful experiment gets an ID and a verdict. Machine-readable mirror
lives in `experiments.csv`.

Verdicts: **KEEP** / **REJECT** / **UNCERTAIN**.

---

## Wave 0 - foundation (2026-08-18)

| ID | Question | Result | Effect | Verdict |
| --- | --- | --- | --- | --- |
| DST-001 | Can D/ST scoring be reconstructed and validated? | Yes | tiers monotonic, top seasons recognisable | **KEEP** |
| DST-002 | What predicts weekly D/ST points? | The opponent, and only the opponent | market alone 61.01% | **KEEP** |
| DST-003 | Are defensive outcomes stable traits or noise? | Mostly noise | def TD r=-0.075 | **KEEP** |
| DST-002b | Does any model beat the trivial rule? | **No** | +0.00 pp | **REJECT** |

### DST-001 - scoring validated

4,222 defense-weeks, 2018-2025. (CORRECTION 2026-08-18: this was first logged as 8,096, a figure never actually observed - the dst001 output was piped through `tail` and the panel-size line scrolled past unseen. Verified: 2,127 REG games x 2 = 4,254 possible, 4,222 retained after joining market context.) Points-allowed tiers behave monotonically:
shutouts (n=44) average 20.07 fantasy points, 35-plus allowed (n=499) average
-0.43. Top seasons are historically recognisable - 2018 CHI 180, 2019 NE 208,
2021 DAL 178, 2023 DAL/BAL 171 - which is the check that matters, because a
silent scoring bug would corrupt every experiment downstream.

CBS/Yahoo and ESPN points-allowed tiers correlate at 0.9899, so the tier choice
shifts the level but is unlikely to reorder rankings.

**Where D/ST points actually come from, per game:** sacks 2.40, interceptions
1.54, fumble recoveries 1.04, points-allowed tier 0.48, defensive TD 0.41,
special-teams TD 0.21, safety 0.05. Sacks dominate the mean; the points-allowed
tier (range -4 to +10) drives the tails. Floor and ceiling have different
sources.

### DST-003 - defensive performance barely persists

Season-to-season correlation of team rates (n = 222 team-seasons):

| Metric | r(N, N+1) | Verdict |
| --- | --- | --- |
| Points allowed | 0.279 | weak |
| Total D/ST points | 0.270 | weak |
| Sacks | 0.195 | essentially random |
| Interceptions | 0.171 | essentially random |
| Fumble recoveries | 0.051 | essentially random |
| **Defensive touchdowns** | **-0.075** | **literally random** |

The brief asked whether historical defensive touchdowns are predictive or
mostly random. They are random, and the point estimate is slightly *negative*.
Sacks are the largest single scoring component (2.40 pts/game) and persist at
only 0.195 - so even the biggest contributor is close to a coin flip year over
year.

### DST-002 - the opponent is the entire signal

Walk-forward, train on seasons < S, test on S, S = 2020-2025:

| Model | MAE | Pair accuracy |
| --- | --- | --- |
| M0 league mean | 4.314 | n/a |
| **M1 market only (implied opponent total)** | **4.134** | **61.01%** |
| M2 defensive form only | 4.287 | 53.61% |
| M3 form + opponent offence | 4.190 | 58.12% |
| M4 form + market | 4.135 | 60.81% |
| M5 everything | 4.116 | 60.95% |

**A single variable - the opponent implied team total - beats every richer
model.** Defensive form on its own reaches 53.61%, barely above a 50% coin
flip, which is exactly what DST-003 predicts: there is little stable defensive
quality to measure. Adding form and opponent-offence features to the market
makes ranking accuracy slightly *worse*.

### DST-002b - the model does not beat the thirty-second rule

The pre-registered benchmark was: each week start whichever rostered defense
faces the lowest opponent implied total. That rule scores **61.01%**. The best
model scores **61.01%**. The difference is **+0.00 pp**.

**The trivial rule is the strategy.** This was named as the bar before any
modelling was fitted, precisely so the result could not be rationalised
afterwards. The consequence is constructive rather than defeatist: the pairing
engine should rank matchups by opponent implied total and spend its effort on
distributions, bye structure and draft cost, where genuine decisions remain.

---

## DST-004 - does the opposing quarterback add anything?

| ID | Question | Result | Effect | Verdict |
| --- | --- | --- | --- | --- |
| DST-004 | Does opposing-QB identity beat team-level offence? | No | +0.10 pp | **REJECT** |
| DST-004b | Do QB traits correlate with D/ST outcomes? | Barely | best r = 0.030 | **REJECT** |
| DST-004c | Does the proposed interaction exist? | Detectable, negligible | 0.17% of baseline error | **UNCERTAIN** |

This is the hypothesis the brief emphasised most: that a defense facing a
turnover-prone or sack-prone quarterback is a better start than the
team-level matchup alone suggests. Panel of 4,036 defense-weeks (95.6%
retained after joining each defense to the quarterback it faced, with all
QB features strictly prior).

| Feature set | MAE | Pair accuracy |
| --- | --- | --- |
| **market only (opponent implied total)** | **4.127** | **60.81%** |
| baseline (market + opponent offence + home) | 4.130 | 60.66% |
| baseline + opposing QB | 4.092 | 60.77% |
| opposing QB only | 4.152 | 57.92% |

**Adding the opposing quarterback is worth +0.10 pp**, and the single market
variable still beats the model that knows who the quarterback is. A model
built only on quarterback traits reaches 57.92%, closer to a coin flip than
to the market.

Correlations with the residual from the team-level baseline - what the
quarterback adds beyond who he plays for - are all negligible: sack rate
0.030, games of experience -0.026, fantasy form 0.020, interception rate
0.016, fumble rate -0.012. For scale, implied team total correlates 0.150
with the QB-side residual in the other lab.

### The interaction search

Every pairwise product was tested rather than only the one the brief
proposed, so the winner is judged against the whole search. The proposed
combination - strong pass rush plus a poor pressure-to-sack profile plus
elevated interception expectation - **did not win**. The best product was
opponent generosity times the defence own interception form, at +0.0070 MAE.

Against a max-statistic permutation null (target shuffled within week, best
of the entire search recomputed each time) the null averaged +0.0033 and
peaked at +0.0059, so the observed value cleared it in 10 of 10 draws.

**Two honest caveats that make this UNCERTAIN rather than KEEP.** Ten
permutations resolve a p-value no finer than 0.1, so p = 0.000 overstates
what was actually measured. And the effect is **0.17% of baseline error** -
detectable is not the same as useful. Given DST-002 already showed the
trivial rule cannot be beaten, an interaction worth a thousandth of a
fantasy point does not change any decision.

### Consequence

Three independent routes - defensive form, opponent offence quality, and now
opposing-QB identity - all fail to improve on knowing the opponent implied
team total. That convergence is itself the finding: **for D/ST, the market
has already priced everything a manager could look up.**

---

## DST-005 - how fast does preseason knowledge decay?

| ID | Question | Result | Verdict |
| --- | --- | --- | --- |
| DST-005 | Half-life of preseason defensive knowledge | ~6.5 weeks, CI [1.8, >50] | **UNCERTAIN** |
| DST-005b | Should defenses be drafted on last year rankings? | No | **REJECT** |

**Half-life of preseason knowledge: about 6.5 weeks, 95% CI [1.8, more than
50].** That interval is the honest answer - a single week correlation rests on
roughly 200 defense-weeks, so its standard error is about half the effect
size. Point estimates: own D/ST points per game 6.5 weeks, own stinginess 6.6,
opponent prior offence 17.1.

**The market shows no decay at all** - the lowest-AIC functional form is a
constant. That is the sanity check that the method works: a forward-looking
price should not decay the way a backward-looking rating does.

Because an exponential fit is a modelling choice, the same question was asked
form-free: correlation drops from **+0.139 in weeks 1-6 to +0.058 in weeks
7-17**, a drop of +0.082 [+0.013, +0.151], P(drop > 0) = 0.988. Preseason
defensive knowledge loses more than half its value after week 6.

### Is weeks 1-6 the right window? Yes on decay grounds, but the framing
### oversells what is inside it

The decay is real and concentrated exactly there - if anything weeks 1-4 is
sharper. But **the preseason signal is weak everywhere**: r about 0.14 and 55%
pairwise accuracy even in weeks 1-6. And in every window including week 1 the
market beats every preseason rating - **preseason at its best (weeks 1-3,
55.1%) loses to the market at its worst (weeks 1-3, 58.6%)**.

So the case for weeks 1-6 rests on the original argument in the plan - that
lookahead betting lines exist for that window and vanish afterwards - and NOT
on prior-season team quality. **Drafting defenses on last year rankings is
rejected.**

In-season ratings turned out flat rather than decaying (own defense |r| about
0.09, opponent offence |r| about 0.15 across all ten lags), the same
opponent-dominates asymmetry DST-002 found.

## DST-006 - outcome distributions, and they are calibrated

| ID | Question | Result | Verdict |
| --- | --- | --- | --- |
| DST-006 | Can D/ST outcome distributions be calibrated? | Yes | **KEEP** |

3,230 walk-forward out-of-sample defense-weeks across 107 weeks. Coverage of
the published percentiles is **10.2 / 25.8 / 51.9 / 77.0 / 91.0** against a
nominal 10 / 25 / 50 / 75 / 90 - within 0.2 to 2.0 pp everywhere, against a
binomial standard error of 0.53 pp. **21 of 23 reliability bins contain the
prediction**, which is what chance produces at 95%. P(top-12) is calibrated in
every bin and P(bust) in every bin.

Discrimination is real as well as calibration: AUC for top-12 is 0.632 against
0.510 for climatology.

A PIT Kolmogorov-Smirnov test formally rejects uniformity (KS 0.0297,
p = 0.007), but that is power at n = 3,230 - the largest CDF departure anywhere
is 3 pp. Reported rather than hidden.

**The component design was earned, not asserted.** Paired CRPS bootstrapped
over weeks, every rival worse with an interval clearing zero: independent
components +0.030, total-only +0.011, climatology +0.145, Gaussian copula
+0.006. The agent switched from a Gaussian to an **empirical copula** after
diagnosing the Gaussian version as 3.5% too narrow and under-skewed; the
empirical version is less code and fixes it (simulated sd 5.66 and skew 0.87
against actual 5.53 and 0.85).

**One documented defect:** every variant is +0.25 points optimistic, but
climatology carries +0.19 of that - so the bulk is the league scoring less in
test seasons than training ones, not model error. Recency weighting was tested
as a fix, made things marginally worse, and defaults off.

**Thresholds are derived, not picked:** bust is 2 points or fewer (the median
weekly DST-24 line - the waiver wire beat you), spike is 14 or more (the 89th
percentile, past the median weekly DST-6 line of 11).

---

## DST-007 - pairing engine and the 2026 board

| ID | Question | Result | Verdict |
| --- | --- | --- | --- |
| DST-007 | Can every two-defense pair be scored pregame? | Yes, 496 pairs | **KEEP** |
| DST-008 | Does a second DRAFTED defense pay for its pick? | +1.47 pts over 6 weeks | **UNCERTAIN, likely no** |

496 combinations scored across 2026 weeks 1-6. The weekly starter is chosen
pregame on expected points; the outcome is drawn from the DST-006 simulator,
so each pair inherits a distribution rather than a point estimate.

**Best overall: KC + NE**, 47.8 expected points across weeks 1-6, and it tops
both the floor list (10th percentile 31.0) and the ceiling list (90th
percentile 66.0). **There is no floor-versus-ceiling trade-off to exploit** -
pairs that are good are good at both ends, because both ends are driven by the
same thing, facing bad offences. **Best first four weeks: KC + LAC** at 32.6.
**Best inexpensive: NE + SF**, within 1.6 points of the top pair.

**Honest caveat: New England appears in nine of the top ten pairs.** That is a
schedule artifact, not a claim about the defense - they draw the Jets in week
6 and the Raiders in week 5. The table ranks schedules, as intended, but must
not be read as a scouting report.

### DST-008 - the second drafted defense

Holding defense A fixed, the best drafted partner beats streaming the best
undrafted defense by **+1.47 points across weeks 1-6 (se 0.24), positive for
16 of 16 defenses**. Consistently positive and small.

Defenses go around picks 84-160, where a non-D/ST pick returns 3.9 to 9.6
points above replacement over a full season. The units are not perfectly
commensurable - six weeks of D/ST against a full season of a skill player -
but the direction is clear: **the second drafted defense is probably not worth
its pick.** Marked UNCERTAIN rather than REJECT because of that unit mismatch.

Judged on D/ST evidence alone, per the isolation instruction. Note it lands in
the same place as the QB2 result by an entirely independent route.

### Deliverable

`DST_STREAMING_RESEARCH.md` written: pair table (W1-4, W1-6, cost), best
overall / inexpensive / floor / ceiling, market signals, double-counting
analysis, limitations and recommendations.
