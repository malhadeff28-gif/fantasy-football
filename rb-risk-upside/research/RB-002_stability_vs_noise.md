# RB-002 — What is stable, what is noise, and what is stable but useless

**Verdict: KEEP the finding, REJECT every candidate predictor.**

The stability results are strong and reusable. The *usefulness* test is a clean
negative: **not one of nineteen workload or efficiency metrics beats "last
season's points per game" out of sample once multiple testing is accounted for.**
The QB lab's trap reproduces on running backs exactly as predicted.

Reproduce: `./.venv/Scripts/python.exe rb-risk-upside/scripts/rb002_stability.py`

Sample: 464 player-season pairs, 168 players, 2018→2025, both seasons ≥ 8 games.

---

## 1. Stability: r(season *t*, season *t+1*), bootstrap 95% CI (2,000 reps)

| Metric | kind | r | 95% CI | Spearman |
|---|---|---|---|---|
| carries per game | role | **0.772** | [0.728, 0.809] | 0.771 |
| touches per game | role | **0.772** | [0.732, 0.809] | 0.772 |
| carry share | role | 0.757 | [0.712, 0.799] | 0.762 |
| snap share | role | 0.730 | [0.680, 0.775] | 0.727 |
| *fantasy pts per game* | *out* | *0.723* | *[0.674, 0.767]* | *0.729* |
| RB target share | role | 0.716 | [0.661, 0.759] | 0.683 |
| targets per game | role | 0.705 | [0.653, 0.751] | 0.678 |
| red-zone carry share | role | 0.696 | [0.641, 0.743] | 0.703 |
| red-zone carries/gm | role | 0.652 | [0.590, 0.706] | 0.669 |
| RZ targets/gm | role | 0.501 | [0.422, 0.574] | 0.475 |
| **goal-line carries/gm** | role | 0.493 | [0.412, 0.568] | 0.531 |
| implied team total | env | 0.445 | [0.363, 0.521] | 0.446 |
| fantasy pts per touch | eff | 0.321 | [0.160, 0.462] | 0.221 |
| yards per carry | eff | 0.314 | [0.177, 0.443] | 0.220 |
| TD per touch | eff | 0.269 | [0.071, 0.443] | 0.160 |
| **rushing TD per carry** | eff | **0.267** | [0.028, 0.473] | 0.181 |
| games played | out | 0.185 | [0.094, 0.272] | 0.233 |
| TD per goal-line carry | eff | 0.137 | [0.039, 0.238] | 0.158 |
| yards per reception | eff | 0.038 | [−0.094, 0.168] | 0.102 |
| catch rate | eff | −0.035 | [−0.184, 0.138] | 0.021 |

The volume/efficiency split is stark and clean: **every opportunity measure sits
between 0.49 and 0.77; every efficiency measure sits between −0.04 and 0.32.**

Changing teams costs about 0.13 of correlation:

| | carries/gm | snap share | fp/gm |
|---|---|---|---|
| stayed (n=350) | 0.798 [.754,.838] | 0.754 [.702,.800] | 0.741 [.685,.788] |
| switched (n=114) | 0.671 [.565,.758] | 0.657 [.535,.755] | 0.607 [.490,.710] |

This matters directly: **Montgomery is the one mover of the four**, and his prior
role is the least reliable guide to his next one.

---

## 2. Touchdown rate is mostly noise. Goal-line volume is not.

Three independent tests, all agreeing.

**Season to season.** Rushing TD per carry: r = **0.267** [0.028, 0.473] — the
lower bound is nearly zero. Goal-line carries per game: r = 0.493. Carries per
game: r = 0.772.

**Split-half within one season** (odd vs even weeks, same player-season, ≥ 40
carries, n = 383). This is the cleanest available noise test because everything
else about the player is held fixed:

| | r(first half, second half) | 95% CI |
|---|---|---|
| rushing TD rate | **+0.172** | [+0.073, +0.266] |
| yards per carry | +0.243 | [+0.139, +0.338] |
| goal-line carries/gm | +0.330 | [+0.227, +0.433] |
| carries per game | **+0.788** | [+0.747, +0.825] |

A player cannot even predict *his own* touchdown rate across the halves of a
single season at better than r = 0.17. Roughly 97% of the variance in a
half-season TD rate is not repeatable skill.

**Regression to the mean** (≥ 100 carries, n = 267):

| Group at season *t* | TD rate *t* | TD rate *t+1* |
|---|---|---|
| above-median TD rate (n=133) | 0.0454 | 0.0335 |
| below-median TD rate (n=134) | 0.0198 | 0.0298 |
| league mean *t+1* | — | 0.0317 |

A 2.3× gap in TD rate collapses to 1.1× the following season. Roughly 85% of the
gap disappears.

**Consequence, and it is the load-bearing modelling decision in this project:**
buy goal-line and total carries; never pay for a touchdown rate. RB-004's
simulator draws production from the pool of historical RB-games at a matched role
level, which forces exactly this regression rather than assuming it.

---

## 3. Stable is not the same as useful — the trap, reproduced

Walk-forward, chronological. For each test season, fit on every transition whose
*outcome* season is strictly earlier, then predict. Baseline model: last
season's points per game. Challenger: baseline + the metric. 331 out-of-sample
player-seasons.

Baseline RMSE = **3.396** points per game. Every challenger, in order:

| Metric | RMSE + metric | gain | p(gain > 0) | survives BH q=0.10 |
|---|---|---|---|---|
| touches per game | 3.366 | +0.031 | 0.979 | **no** |
| TD per touch | 3.370 | +0.026 | 0.991 | **no** |
| carries per game | 3.372 | +0.024 | 0.943 | **no** |
| carry share | 3.378 | +0.018 | 0.916 | **no** |
| rushing TD rate | 3.379 | +0.017 | 0.980 | **no** |
| goal-line carries/gm | 3.381 | +0.016 | 0.816 | **no** |
| fp per touch | 3.381 | +0.015 | 0.968 | **no** |
| snap share | 3.396 | +0.001 | 0.564 | no |
| *(11 more, all negative gains)* | | | | no |

**Nothing survives.** Nineteen metrics were tested; at a 5% bar one false winner
is expected by chance. Under Benjamini-Hochberg at q = 0.10, the smallest
threshold is 0.0053 and the best observed p is 0.009. Zero survivors.

Even taken at face value the best gain is 0.031 points per game — **0.9% of the
baseline error**, about 0.5 points across a full season. That is not a signal a
drafter can act on.

The result is not that workload does not matter. It is that **last season's
fantasy points per game already contains the workload information.** Carries,
snap share and carry share are ~0.75 stable *and* ~0.75 correlated with the
baseline; they are the same fact wearing different clothes. Red-zone carry share
actively *hurts* (gain −0.013, p = 0.993) — it is the most redundant of all.

This is the QB lab's finding, reproduced: stability is necessary, not sufficient,
and several of the most stable variables here added nothing.

---

## What this changes downstream

1. **Do not build a points-per-game regression on workload features.** It has
   been tried, walk-forward, and it does not beat one lagged column.
2. **The predictable thing is the role, and the role is the thing that spreads
   the outcome distribution.** So model the role distribution (RB-003) and let
   production follow from it, rather than projecting a point estimate.
3. **Treat TD rate as drawn from the league, not owned by the player** (RB-004).
4. **Discount Montgomery's prior season more than the others'** — team-switchers
   are measurably less predictable.

## Limitations

- 464 transitions across 7 season pairs is not a large sample; the efficiency CIs
  are wide enough that "rushing TD rate is exactly zero-stable" cannot be
  rejected either.
- The ≥ 8 game filter conditions on survival. Backs who lost their job in week 3
  are absent from both sides, so the stability numbers are, if anything,
  optimistic.
- The usefulness test used a linear baseline. A non-linear model was not tried;
  with a 0.9% ceiling on the available gain, it was not worth trying.
