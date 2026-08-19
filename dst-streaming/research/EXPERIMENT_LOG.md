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

8,096 defense-weeks, 2018-2025. Points-allowed tiers behave monotonically:
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
