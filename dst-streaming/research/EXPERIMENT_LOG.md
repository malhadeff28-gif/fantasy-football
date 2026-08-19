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
