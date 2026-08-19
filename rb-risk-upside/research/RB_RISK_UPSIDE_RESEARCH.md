# RB Risk / Upside Research

**2026 fantasy football — of four running backs drafted 22 picks apart, which has
the best floor and which the best risk-adjusted ceiling?**

League: 12-team, 1QB/2RB/3WR/1TE/K/DST, no flex, standard (non-PPR) scoring ·
Panel: 12,003 RB-weeks, 392 players, 2018–2025 · Simulation: 50,000 draws ·
Backtest: 139 player-seasons, 2022–2025, walk-forward · Last revised: 2026-08-19

Conclusions are labelled **[evidence]** (measured here), **[inference]**
(a reasonable reading), or **[hypothesis]** (proposed, not established).

Every figure below was re-derived from `rb-risk-upside/outputs/*.csv` and the two
parquet panels rather than copied from the RB-001…RB-004 write-ups.
[§11](#11-reconciliation-against-the-outputs) lists the thirteen places where a
source document and its own data disagreed, and which one this report follows.

---

## 1. Executive summary

**Both answers are the same player, and the premise the project was built on did
not hold.**

The brief hoped to find a back whose workload distribution is more positively
skewed than consensus assumes. **Three of the four are negatively skewed.** The
only positively-skewed distribution belongs to David Montgomery, at
skew **1.04** — and it is a long right tail on a **92-point** median, which is a
lottery ticket, not a mispricing. His expected value is the worst of the four by
54 points. There is no hidden ceiling here to buy. [evidence]

What the work does produce:

**Highest floor: Quinshon Judkins.** Median **185.1**, p10 **88.8**, p25
**133.1**, P(bust) **0.309** — best of the four on all four measures, and the
only one of the four with **zero** comparables who ended the season as a backup
(0 of 30, Wilson 95% [0.00, 0.11]). [evidence]

**Highest risk-adjusted ceiling: also Judkins.** First under all six
specifications — Sharpe 0.537, Sortino 1.267, omega 0.763, P(top-5) 0.198,
upside CVaR 305.4, and cost-adjusted omega-at-ADP 17.94. Rank spread across
specifications: **zero**. [evidence]

**The ranking is robust. Most of the margins are not.** Judkins first and
Montgomery fourth under all six risk metrics and all three role-transfer
specifications. But resampling the 30 comparables that every distribution rests
on gives P(Judkins beats Swift) = **0.89** and P(Swift beats Pollard) =
**0.79**. Only Montgomery separates with certainty — 1.00 against all three, in
every one of 200 resamples. [evidence]

**And the honest ceiling on all of it:** backtested over 139 player-seasons the
simulator's median has MAE **57.4** against **58.7** for a curve that knows
nothing but ADP rank. **A 2% improvement.** The point estimate is not sharper
than the market. What survives the calibration test is the *shape* — an unbiased
median (48.9% of outcomes below it), a calibrated interquartile range (49.6%),
and a bust probability that means something. **The distribution is the product.**
[evidence]

**Recommendation:** take Judkins over Swift as a lean, not a conviction. Fade
Montgomery hard — that is the one call this project can defend. §10 gives the
specifics.

---

## 2. The four players, and their verified 2026 situations

Nothing in this table is from recall. Sources: `nfl_data.load_players()`,
`nflreadpy.load_rosters([2024,2025,2026])`, `nfl_data.load_depth_charts()`,
`markets.load_ffc_adp(2026)`, `markets.load_cbs_projections(2026,"RB")`.

| Player | 2025 team | **2026 team** | Depth slot | ADP (FFC 12-tm std) | Pos rank |
|---|---|---|---|---|---|
| D'Andre Swift | CHI | CHI | RB1 | 41.9 | RB20 |
| Quinshon Judkins | CLE | CLE | RB1 | 49.1 | RB22 |
| David Montgomery | **DET** | **HOU** | RB1 | 54.6 | RB24 |
| Tony Pollard | TEN | TEN | RB1 | 63.7 | RB26 |

All four sit inside 22 picks, overall 42–64: rounds 4 through 6 of a 12-team
draft. Their simulated medians span **106 points**. The market prices them as
near-equivalents; the comparable evidence does not.

### Montgomery changed teams, Detroit to Houston

Confirmed three independent ways — the master player table's `latest_team`, the
2026 season roster, and the March 2026 depth chart — and corroborated by FFC ADP
listing him as HOU. Every other assignment is unchanged from 2025. This is the
single finding that justifies not trusting memory on team assignment. [evidence]

**And the depth-chart label is a snapshot from the week he signed.** The chart
that lists Montgomery as Houston's RB1 is dated **2026-03-14** — immediately
after his signing and before the 2026 draft. Woody Marks handled **196 carries, a
43.6% carry share, over 16 games** for Houston in 2025 (verified from the panel).
A March depth chart written the week a veteran signs is a weak signal about an
August committee. It is deliberately **not** used as a model feature; the room he
walks into is. [inference]

### The room each one walks into (2026 roster + 2025 usage, panel-verified)

| Player | Team | Competition that matters | That player's 2025 |
|---|---|---|---|
| Swift | CHI | Kyle Monangai (2025 R7, pick 233) | 169 carries, 33.3% carry share, **12 goal-line carries vs Swift's 14** |
| Judkins | CLE | Dylan Sampson (2025 R4, pick 126) | 65 carries, 40 targets, **0 goal-line carries** — a passing-down complement |
| Montgomery | HOU | Woody Marks (2025 R4, pick 116) | **196 carries, 43.6% share, 16 games** — an incumbent, not a backup |
| Pollard | TEN | Tyjae Spears (2023 R3) + Singleton (2026 R5, pick 165) | Spears 72 carries, 50 targets |

**Montgomery is the only genuinely ambiguous room, and Swift's is a quieter
version of the same problem** — Monangai took a third of Chicago's carries and 12
goal-line carries to Swift's 14. Swift holds the role but not the scoring role
exclusively. [evidence]

### 2023–2025 usage, recomputed from the panel

| Player | Season | Tm | G | Fin | Pts | Car/g | Carry share | RZ share | GL car/g | Impl. total |
|---|---|---|---|---|---|---|---|---|---|---|
| Swift | 2023 | PHI | 16 | RB23 | 160.3 | 14.3 | .470 | .349 | 0.88 | 25.5 |
| Swift | 2024 | CHI | 17 | RB19 | 172.5 | 14.9 | .587 | .517 | 0.59 | 20.2 |
| Swift | 2025 | CHI | 16 | **RB14** | 194.6 | 13.9 | .474 | .444 | 0.88 | 22.7 |
| Montgomery | 2023 | DET | 14 | RB9 | 191.2 | 15.6 | .530 | .557 | 1.43 | 25.4 |
| Montgomery | 2024 | DET | 14 | RB17 | 185.7 | 13.2 | .421 | .505 | 1.29 | 27.8 |
| Montgomery | 2025 | DET | 17 | **RB27** | 142.9 | **9.3** | **.357** | .380 | 1.00 | 26.9 |
| Judkins | 2025 | CLE | 14 | RB26 | 143.8 | 16.4 | **.657** | **.647** | 0.86 | **16.7** |
| Pollard | 2023 | DAL | 17 | RB18 | 167.6 | 14.8 | .536 | .569 | 1.06 | 26.0 |
| Pollard | 2024 | TEN | 16 | RB22 | 159.7 | 16.2 | .618 | .576 | 0.38 | 18.8 |
| Pollard | 2025 | TEN | 17 | RB23 | 152.8 | 14.2 | .640 | .422 | **0.24** | **17.0** |

Three facts drive everything downstream:

1. **Montgomery's role collapsed in 2025** — 9.3 carries per game, a 35.7% carry
   share, RB27, on an offence with a 26.9 implied team total. He is not a lead
   back coming off a lead-back season. He is a committee back who changed teams.
2. **Judkins owns the largest share of anyone here** — 65.7% of carries and 64.7%
   of red-zone carries as a rookie — **on the worst offence**, a 16.7 implied
   team total. Elite share, poor environment.
3. **Pollard has almost no goal-line role** — 0.24 goal-line carries per game,
   down from 1.06 in Dallas. Volume without the scoring plays caps his ceiling
   structurally rather than by luck. [inference]

---

## 3. The workload panel and its as-of guarantees

`rb-risk-upside/src/rb_features.py` → `data/rb_week_panel.parquet`

**12,003 RB-weeks, 392 players, 2018–2025, regular season only** — verified by
direct count. Sources joined: `load_player_stats` (rescored non-PPR via
`scoring.skill_points`), `load_snap_counts` (through a pfr↔gsis crosswalk),
play-by-play red-zone and goal-line counts, and `team_game_context` for spread,
total and implied team total.

Play-by-play is read one season at a time with 11 columns pruned from 372 — the
same memory discipline as `qb-pairing/scripts/wave1_advanced_qb_vars.py`, because
a whole pbp season does not fit alongside the rest at ~1.7 GB free.

### No look-ahead, and how it is proved rather than asserted

Every rolling statistic is `.shift(1).rolling_*(...).over("player_id")`. Market
variables are closing lines that exist before kickoff and are used unshifted.
`rb_features._demo()` is a runnable proof asserting two things:

1. every player's first-ever panel game has **null** for all rolling features, so
   nothing is seeded from the future;
2. for a player whose career begins inside the panel, `f_carries` at game *k*
   equals the mean of games *k−8 … k−1* exactly, to 1e-9, for every game.

Case 2 is deliberately restricted to players whose careers start inside the
panel: `build_panel` loads one extra prior season for form history, so a
veteran's first panel row legitimately carries prior-season form. An earlier
version of the check failed on exactly this — **the check was wrong, not the
panel**, and the distinction is worth recording.

### Coverage

| Column group | Missing | Why |
|---|---|---|
| `pbp_rush_td` | 10.2% | RB-games with zero rushing attempts. Correct, not a defect |
| rolling `f_*` | 2.4% | First career game of each player, by design |
| market context | 0.77% | A handful of games with no closing line |
| `snap_share` | 0.03% | — |

### Two shared-module bugs found by using the modules rather than trusting them

1. `nfl_data` had no `load_players` or `load_rosters_weekly` despite both being
   available in nflreadpy. Added and cached.
2. **`markets.load_cbs_projections` was silently broken for every position except
   QB.** The column schema was hard-coded to CBS's QB table layout, so an RB
   request returned an empty frame. Per-position schemas added for RB/WR/TE; the
   QB path is byte-identical. Tight ends needed their own 11-column schema —
   using the receiver layout filtered out every TE row.

**A data limit for the next lab:** `load_rosters_weekly` hard-refuses any season
after 2025. Weekly roster membership for the upcoming season does not exist. Use
`load_rosters(seasons=[2026])` for team assignment.

---

## 4. What is stable, what is noise, and what is stable but useless

**This section is a clean negative and it is the most transferable finding in the
project.** Not one of nineteen workload or efficiency metrics beats "last
season's points per game" out of sample once multiple testing is accounted for.
The QB lab's trap reproduces on running backs exactly. [evidence]

Sample: **464 player-season pairs, 168 players, 2018→2025**, both seasons ≥ 8
games.

### 4.1 Stability: r(season *t*, season *t+1*), bootstrap 95% CI, 2,000 reps

| Metric | kind | r | 95% CI | Spearman |
|---|---|---|---|---|
| carries per game | role | **0.772** | [0.729, 0.809] | 0.771 |
| touches per game | role | **0.772** | [0.733, 0.809] | 0.772 |
| carry share | role | 0.757 | [0.712, 0.799] | 0.762 |
| snap share | role | 0.730 | [0.679, 0.776] | 0.727 |
| *fantasy pts per game* | *out* | *0.723* | *[0.674, 0.767]* | *0.729* |
| RB target share | role | 0.716 | [0.662, 0.761] | 0.683 |
| targets per game | role | 0.705 | [0.652, 0.752] | 0.678 |
| red-zone carry share | role | 0.696 | [0.641, 0.743] | 0.703 |
| *fantasy pts, season total* | *out* | *0.662* | *[0.610, 0.712]* | *0.678* |
| red-zone carries/gm | role | 0.652 | [0.591, 0.707] | 0.669 |
| RZ targets/gm | role | 0.501 | [0.423, 0.574] | 0.475 |
| **goal-line carries/gm** | role | 0.493 | [0.412, 0.567] | 0.531 |
| implied team total | env | 0.445 | [0.363, 0.521] | 0.446 |
| fantasy pts per touch | eff | 0.321 | [0.161, 0.462] | 0.221 |
| yards per carry | eff | 0.314 | [0.177, 0.442] | 0.220 |
| TD per touch | eff | 0.269 | [0.071, 0.443] | 0.160 |
| **rushing TD per carry** | eff | **0.267** | [0.027, 0.471] | 0.181 |
| games played | out | 0.185 | [0.095, 0.272] | 0.233 |
| TD per goal-line carry | eff | 0.137 | [0.039, 0.238] | 0.158 |
| yards per reception | eff | 0.038 | [−0.095, 0.166] | 0.102 |
| catch rate | eff | −0.035 | [−0.184, 0.138] | 0.021 |

`implied_team_total` is n = 460; every other row is n = 464.

**The volume/efficiency split is stark and clean: every opportunity measure sits
between 0.493 and 0.772; every efficiency measure between −0.035 and 0.321.**
[evidence]

Changing teams costs 0.10–0.14 of correlation:

| | carries/gm | snap share | fp/gm |
|---|---|---|---|
| stayed (n = 349) | 0.801 [.756,.838] | 0.757 [.700,.801] | 0.742 [.690,.790] |
| switched (n = 115) | 0.665 [.566,.749] | 0.652 [.533,.755] | 0.605 [.483,.711] |

**Montgomery is the one mover of the four**, so his prior role is the least
reliable guide to his next one. [evidence]

### 4.2 Touchdown rate is mostly noise. Goal-line volume is not.

Three independent tests, all agreeing.

**Season to season.** Rushing TD per carry r = 0.267 [0.027, 0.471] — the lower
bound is essentially zero. Goal-line carries per game 0.493. Carries per game
0.772.

**Split-half within one season** (odd vs even weeks, ≥ 40 carries in each half,
n = 383). The cleanest available noise test, because everything else about the
player is held fixed:

| | r(half, half) | 95% CI |
|---|---|---|
| rushing TD rate | **+0.172** | [+0.074, +0.262] |
| yards per carry | +0.243 | [+0.148, +0.340] |
| goal-line carries/gm | +0.330 | [+0.233, +0.423] |
| carries per game | **+0.788** | [+0.744, +0.827] |

A player cannot predict **his own** touchdown rate across the halves of a single
season at better than r = 0.172. **97.0% of the variance in a half-season TD rate
is not repeatable skill** (1 − r²). [evidence]

**Regression to the mean** (≥ 100 carries at season *t*, n = 267):

| Group at season *t* | TD rate *t* | TD rate *t+1* |
|---|---|---|
| above-median (n = 133) | 0.0454 | 0.0335 |
| below-median (n = 134) | 0.0198 | 0.0298 |
| league mean *t+1* | — | 0.0317 |

A **2.29×** gap collapses to **1.12×** the following season. **85.7% of the gap
disappears.** [evidence]

**The load-bearing modelling decision follows directly: buy goal-line and total
carries; never pay for a touchdown rate.** The simulator draws production from
the pool of historical RB-games at a matched role level, which *forces* this
regression rather than assuming it in a prior.

### 4.3 Stable is not the same as useful — the trap, reproduced

Walk-forward and chronological. For each test season, fit on every transition
whose *outcome* season is strictly earlier, then predict. Baseline: last season's
points per game. Challenger: baseline + the metric. 331 out-of-sample
player-seasons. Baseline RMSE = **3.396** points per game.

| Metric | RMSE + metric | gain | P(gain > 0) | survives BH q=0.10 |
|---|---|---|---|---|
| touches per game | 3.366 | +0.0308 | 0.979 | **no** |
| TD per touch | 3.370 | +0.0261 | 0.992 | **no** |
| carries per game | 3.372 | +0.0242 | 0.944 | **no** |
| carry share | 3.378 | +0.0180 | 0.916 | **no** |
| rushing TD rate | 3.379 | +0.0173 | 0.980 | **no** |
| goal-line carries/gm | 3.381 | +0.0156 | 0.815 | **no** |
| fp per touch | 3.381 | +0.0150 | 0.967 | **no** |
| snap share | 3.396 | +0.0006 | 0.563 | no |
| *(11 more, every gain negative)* | | −0.001 to −0.038 | | no |

**Nothing survives.** Nineteen metrics tested; at a 5% bar 0.95 false winners are
expected by chance. Under Benjamini-Hochberg at q = 0.10 the smallest threshold
is **0.0053** and the best observed p is **0.008**. Zero survivors. [evidence]

Even taken at face value the best gain is **0.031 points per game — 0.9% of the
baseline error**, about 0.5 points across a full season. That is not something a
drafter can act on.

**The result is not that workload does not matter. It is that last season's
fantasy points per game already contains the workload information.** On these 464
transitions the role metrics correlate **0.86 to 0.93** with same-season points
per game — carries/gm 0.902, carry share 0.879, snap share 0.856, touches/gm
0.934. And every one of them is a *worse* lagged predictor of next season's
points than lagged points itself: carries/gm 0.686, snap share 0.634, carry share
0.669, touches/gm 0.706, against **0.723** for lagged fp/gm. They are the same
fact wearing different clothes, measured slightly less well. [evidence]

Red-zone carry share actively **hurts** — gain −0.0128, with P(gain > 0) = 0.007,
i.e. a 99.3% probability it makes the model worse. It is the most redundant of
the nineteen.

**This is the QB lab's finding reproduced on a different position: stability is
necessary, not sufficient.** Several of the most stable variables here added
nothing. [evidence]

### What it changed downstream

1. **No points-per-game regression on workload features.** It was tried,
   walk-forward, and it does not beat one lagged column.
2. **Model the role distribution and let production follow**, rather than
   projecting a point estimate. The role is the predictable thing *and* the thing
   that spreads the outcome distribution.
3. **Treat TD rate as drawn from the league, not owned by the player.**
4. **Discount Montgomery's prior season more than the others'** — team-switchers
   are measurably less predictable.

---

## 5. The role-scenario model, and how comparables avoid look-ahead

For every RB in every season 2019–2025 a **pre-season situation vector** is built
from information that existed before that season kicked off:

| Feature | weight | source |
|---|---|---|
| prior touches per game | 2.0 | season *t−1* rollup |
| prior carries per game | 1.5 | season *t−1* |
| prior snap share | 1.0 | season *t−1* |
| prior RB target share | 0.5 | season *t−1* |
| prior games played | 0.5 | season *t−1* |
| years of experience | 0.5 | roster |
| ADP positional rank | 1.5 | FFC pre-season |
| competition draft capital | 1.0 | best draft slot among the other backs in the room |
| competition prior touches/gm | 1.0 | the most any teammate handled last year |
| team change | +0.6 distance penalty | roster *t* vs *t−1* |

**Weights were fixed in advance** to reflect what a drafter actually looks at.
They were not tuned against outcomes and no search was run over them, so there is
no multiple-testing debt here.

Two guards, both asserted in `rb_simulate._demo()`:

1. The comparable pool for a target season contains **only outcome seasons
   strictly earlier** than the target's, so the procedure is usable walk-forward
   — and is in fact backtested that way in §7.
2. Outcomes are joined **after** the k nearest neighbours are selected. Selection
   cannot see what happened.

k = 30. Sensitivity to k is the main robustness check and is reported below.

### The four situation vectors (verified from the situation panel)

| | Swift (CHI) | Pollard (TEN) | Judkins (CLE) | Montgomery (HOU) |
|---|---|---|---|---|
| ADP / pos rank | 41.9 / RB20 | 63.7 / RB26 | 49.1 / RB22 | 54.6 / RB24 |
| prior touches/gm | 16.9 | 16.6 | **19.0** | **11.0** |
| prior snap share | 0.58 | 0.63 | 0.51 | **0.37** |
| prior games | 16 | 17 | 14 | 17 |
| years exp | 6 | 7 | **1** | 7 |
| best other back's draft slot | 115 | 81 | 126 | 116 |
| best other back's prior touches/gm | 11.7 | 10.5 | **7.8** | **14.5** |
| changed team | no | no | no | **yes** |

Judkins has the biggest prior role *and* the weakest incumbent competition.
Montgomery has the smallest prior role, a team change, and the strongest
incumbent competition — Woody Marks handled 14.5 touches per game for Houston
last year.

### Role scenarios (Wilson 95% intervals, n = 30 each)

Bands are named for what a draft room calls them, not carved out of the outcome
distribution, so the probabilities are informative rather than 25/50/25 by
construction.

| Band (touches/gm) | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| **bellcow** (18+) | 0.43 [.27,.61] | 0.33 [.19,.51] | **0.47** [.30,.64] | **0.07** [.02,.21] |
| **lead** (13–18) | 0.47 [.30,.64] | 0.47 [.30,.64] | 0.43 [.27,.61] | 0.30 [.17,.48] |
| **committee** (9–13) | 0.03 [.01,.17] | 0.07 [.02,.21] | 0.10 [.03,.26] | **0.40** [.25,.58] |
| **backup** (<9) | 0.07 [.02,.21] | 0.13 [.05,.30] | **0.00** [.00,.11] | 0.23 [.12,.41] |
| median pts if bellcow | 212 | 209 | 185 | 199 |
| median pts if lead | 141 | 148 | 149 | 143 |
| comps playing < 13 games | 23% | 23% | 20% | **33%** |

**The intervals are wide.** Thirty comparables is thirty comparables: every
mid-range probability here carries roughly ±0.17, and even the narrowest row in
the table — Judkins's 0 of 30 backups — spans 0.11. **Swift, Pollard and Judkins
overlap heavily and differences between them should not be read as real.
Montgomery does not overlap with any of them.** [evidence]

The one thing the bands show that a point projection cannot: **Judkins has zero
comparables who ended up a backup**, while Swift and Pollard each carry a real
7–13% chance of falling out of the role entirely.

### Downside / base / upside, and the skew question

Percentiles of the 30 comparables' realised seasons (20th / 50th / 80th):

| | touches/gm | season points | **skew** |
|---|---|---|---|
| Swift | 14.2 – 17.1 – 19.2 | 97 – 170 – 220 | 0.69 |
| Pollard | 13.3 – 16.7 – 18.8 | 70 – 157 – 208 | 0.58 |
| Judkins | 14.2 – 17.3 – 20.6 | 128 – 167 – 201 | 0.86 |
| Montgomery | 6.8 – 11.4 – 16.1 | 43 – 92 – 144 | **1.04** |

skew = (p80 − median) / (median − p20), computed on **season points**. Above 1
means the upside tail is longer than the downside tail.

**This is the negative result on the project's stated hypothesis, and it should
be read before anything else in the report.** The edge was supposed to come from
finding a back whose workload distribution is more positively skewed than
consensus assumes. **Three of the four are negatively skewed** — their downside
tail is longer than their upside, which is the normal shape for an established
back whose role can collapse but cannot easily double.

The only positively-skewed distribution belongs to **Montgomery**, and it is
skewed up from a base of **92 points**. A long right tail on a low median is a
lottery ticket, not a market inefficiency, and its expected value is still the
worst of the four by 54 points. [evidence]

**Judkins's 0.86 is the interesting number.** His distribution is the *tightest*
of the four — 128 to 201 covers the middle 60% — which is a floor story, not a
ceiling story. [inference]

### What actually happened to the comparables

**Montgomery's comp set is the one the market is not pricing:** an ageing back
coming off a reduced role who signs somewhere new. The **median such back
finished RB41/42**. Only **3 of 30 finished top-12** (James Conner 2021 RB5, Mark
Ingram 2019 RB8, Javonte Williams 2025 RB11) and only **7 of 30 finished top-24**.
Only **2 of 30 reached the bellcow band** at 18+ touches per game — Javonte
Williams 2025 and Aaron Jones 2024. That is the real upside case, and it is
roughly a one-in-fifteen event, not the base case. [evidence]

For contrast, the raw comparable finishes for the other three: Swift 13 of 30
top-12 (median RB16), Judkins 10 of 30 (median RB18.5), Pollard 9 of 30 (median
RB22).

**Note what that last paragraph does to the headline.** On raw comparable
finishes, *Swift's* neighbours did better than Judkins's. The simulator ranks
Judkins ahead because it applies each comparable's role *change* to the target's
own prior role, and Judkins's prior role (19.0 touches/gm) is the largest of the
four. Under the `absolute` specification — which uses the comparable's realised
role level directly — the two nearly converge (§7). **The Judkins-over-Swift lean
is a property of the role-transfer specification, not of the raw comparable
record.** [evidence]

### Robustness to k

Median comparable season points:

| k | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| 15 | 153 | 153 | 192 | 91 |
| 20 | 168 | 150 | 169 | 92 |
| **30** | **170** | **157** | **167** | **92** |
| 45 | 169 | 147 | 169 | 92 |
| 60 | 165 | 153 | 168 | 78 |

P(bellcow role, 18+ touches/gm):

| k | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| 15 | 0.33 | 0.20 | 0.53 | 0.00 |
| 20 | 0.35 | 0.20 | 0.45 | 0.00 |
| **30** | **0.43** | **0.33** | **0.47** | **0.07** |
| 45 | 0.47 | 0.29 | 0.47 | 0.07 |
| 60 | 0.45 | 0.32 | 0.47 | 0.07 |

**The conclusion that survives every k is Montgomery's.** He is last under both
statistics at all five neighbourhood sizes, trailing the third-placed back by
56 to 101 median points, and his bellcow probability never exceeds 0.07.

**The Judkins-over-Swift ordering does not survive every k, and the write-up's
claim that "the ordering never changes" is too strong.** On P(bellcow) Judkins ≥
Swift > Pollard at all five sizes. On median comparable points Judkins leads at
k = 15, 20 and 60, **Swift leads at k = 30**, and they tie at k = 45; Swift and
Pollard tie at k = 15. This is the same instability §6.4 quantifies at 89%
confidence, showing up in a different statistic. Swift's bellcow probability is
also the widest-ranging figure in the table (0.33 to 0.47) because his nearest
neighbours are the most heterogeneous. [evidence]

---

## 6. The Monte Carlo distributions

### 6.1 The generative model, and why it is shaped this way

§4 dictated the architecture. Volume persists (r ≈ 0.77); efficiency and
touchdown rate barely do (split-half r = 0.172). So:

1. **Role** is drawn from the comparables — pick one weighted by inverse
   distance, take its realised role change and its games played.
2. **Production** is drawn from the pool of *historical RB-games at the implied
   role level*, not from a per-player efficiency estimate. This mechanically
   regresses touchdown rate and yards per carry to the league, which is what the
   stability evidence says happens.
3. **The finish bar is itself random.** Each draw samples its own RB12/RB24
   threshold from the eight observed seasons, so year-to-year movement of the bar
   is inside the answer rather than averaged away.

Thresholds, 2018–2025, with 2018–2020 scaled by 17/16 for a like-for-like season
(sd is over the 8 observed seasons, population convention):

| | mean | min | max | sd |
|---|---|---|---|---|
| RB3 | 279.2 | 224.4 | 312.2 | 29.0 |
| RB5 | 245.9 | 218.7 | 269.2 | 19.7 |
| RB12 | 191.9 | 177.6 | 201.1 | 9.0 |
| RB24 | 146.6 | 138.3 | 158.0 | 5.7 |
| RB36 | 108.8 | 97.3 | 118.4 | 7.2 |

### 6.2 Convergence — measured, not guessed

Five independent seeds at each size. The columns are the spread of the estimate
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
**50,000 is used for headroom.** Doubling again to 100,000 leaves P(top-12) error
flat (0.0030 → 0.0030) and the median error nearly flat (0.32 → 0.29); only the
p90 error still halves (0.80 → 0.41). **500,000 draws would buy nothing that
matters**, because the error is dominated by the 30-comparable sample, not the
draw count. §6.4 quantifies that. [evidence]

### 6.3 Season outcome distributions, 50,000 draws each

| | Team | ADP | mean | MC se | p10 | p25 | **median** | p75 | p90 | sd | games | tch/gm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Quinshon Judkins** | CLE | 49.1 | 184.1 | 0.31 | **88.8** | **133.1** | **185.1** | 233.8 | **274.5** | 69.9 | 14.6 | 18.4 |
| **D'Andre Swift** | CHI | 41.9 | 158.9 | 0.34 | 59.5 | 110.8 | 159.6 | 205.7 | 257.1 | 75.0 | 14.0 | 16.3 |
| **Tony Pollard** | TEN | 63.7 | 142.2 | 0.31 | 46.7 | 92.2 | 147.5 | 187.8 | 228.6 | 69.4 | 13.7 | 15.0 |
| **David Montgomery** | HOU | 54.6 | 88.0 | 0.23 | 27.1 | 46.0 | 78.8 | 126.6 | 161.5 | 51.2 | 13.5 | 10.0 |

Finish probabilities (thresholds resampled per draw):

| | P(top-3) | P(top-5) | P(top-12) | P(top-24) | **P(bust)** | P(disaster) |
|---|---|---|---|---|---|---|
| Judkins | 0.110 | **0.204** | **0.463** | **0.691** | **0.309** | 0.160 |
| Swift | 0.073 | 0.129 | 0.315 | 0.580 | 0.420 | 0.243 |
| Pollard | 0.034 | 0.070 | 0.234 | 0.505 | 0.495 | 0.307 |
| Montgomery | 0.001 | 0.002 | 0.031 | 0.156 | **0.844** | 0.662 |

**Definitions, and why these and not others:**

- **Bust = finishes outside the top 24 RBs.** This league starts 12 teams × 2 RBs
  with no flex, so exactly 24 RBs start and RB24 *is* the replacement line —
  `scoring.LEAGUE.replacement_rank["RB"]` returns 24 for precisely this reason. A
  pick spent in rounds 4–6 that does not return a startable RB returned nothing;
  the roster spot could have held a waiver-wire RB24. This is a stricter and more
  honest bar than "finished below ADP", which lets a pick succeed by beating a
  low expectation.
- **Disaster = outside the top 36** — not even a useful bench asset where three
  RBs per team is a normal allocation.
- **League-winning = a top-5 finish.** Not top-12: a top-12 RB drafted in round 4
  is a good pick, but a good pick does not win a league. Top-3 is reported
  alongside for readers who want the stricter bar; it changes no ordering.

**One presentational trap.** Two P(top-5) figures appear in this report and they
are different quantities. The table above (0.204 / 0.129 / 0.070 / 0.002) resamples
the RB5 threshold on every draw. The risk-metric table in §7 (0.198 / 0.124 /
0.063 / 0.001) uses the **fixed** mean bar of 245.9 points. Both are correct, and
the resampled version is larger for everyone because the survival curve is convex
that far out in the tail, so averaging over a moving bar sits above its value at
the mean bar. No ordering depends on which is used.

### 6.4 How much of the ranking is actually real

The Monte Carlo error is negligible. The real uncertainty is that each
distribution rests on **30 comparable seasons**. Resampling those 30 with
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

**The honest statement of confidence:**

- Judkins, Swift and Pollard each beat Montgomery: **certain** — 1.00 in all 200
  resamples.
- Judkins > Pollard: **97%**.
- Judkins > Swift: **89%** — likely, not settled.
- Swift > Pollard: **79%** — a lean, nothing more. [evidence]

### 6.5 Is the simulator calibrated? Walk-forward backtest

Every RB with a top-40 positional ADP in 2022–2025, simulated using only seasons
strictly earlier than his own. **139 player-seasons.**

| season | n | p10–p90 coverage | p25–p75 coverage | frac below median | MAE sim | MAE ADP curve |
|---|---|---|---|---|---|---|
| 2022 | 34 | 0.853 | 0.559 | 0.559 | 48.0 | 48.7 |
| 2023 | 36 | 0.750 | 0.500 | 0.528 | 54.6 | 58.9 |
| 2024 | 37 | 0.730 | 0.405 | 0.459 | **67.9** | **66.6** |
| 2025 | 32 | 0.688 | 0.531 | 0.406 | 58.4 | 60.0 |
| **all** | **139** | **0.755** | **0.496** | **0.489** | **57.4** | **58.7** |

This is the most important table in the project.

**Good:** the median is unbiased — 48.9% of outcomes fall below it against a
nominal 50% — and the interquartile range is almost exactly right, 49.6% against
50%. **The distributions are honest where it matters most.** [evidence]

**Mildly bad:** p10–p90 coverage is 75.5% against a nominal 80%. The tails are
about 5 percentage points too thin; real seasons produce slightly more extreme
outcomes than 30 comparables can show. **Every tail probability in §6.3 — P(top-5),
P(disaster) — should be read as a modest understatement.** [inference]

**The honest ceiling: MAE 57.4 for the simulator against 58.7 for a curve that
knows nothing but ADP rank. A 2% improvement**, and the simulator is *worse* than
the ADP curve in one of the four test seasons (2024: 67.9 vs 66.6). **The
simulator does not know who will score more points than the market does.** This
is entirely consistent with §4, where nineteen workload metrics could not beat one
lagged column either. [evidence]

What the simulator *does* add over the ADP curve is a calibrated shape: an ADP
curve gives one number; this gives a floor, a ceiling and a bust probability that
survive a coverage test. **That is the deliverable.**

---

## 7. Risk-adjusted upside, under six specifications

Bars: RB24 = 146.6, RB12 = 191.9, RB5 = 245.9. The ADP baseline is fitted on
**772 RB seasons, 2018–2025**, points against log ADP positional rank, **R² =
0.465**.

| | ADP bar | sharpe | sortino | **omega** | P(top-5) | CVaR-up | **omega@ADP** |
|---|---|---|---|---|---|---|---|
| Judkins | 106.8 | **0.537** | **1.267** | **0.763** | **0.198** | **305.4** | **17.94** |
| Swift | 113.0 | 0.164 | 0.265 | 0.328 | 0.124 | 294.4 | 4.63 |
| Pollard | 96.1 | −0.064 | −0.085 | 0.156 | 0.063 | 262.3 | 4.98 |
| Montgomery | 101.2 | −1.146 | −0.765 | 0.005 | 0.001 | 184.4 | 0.55 |

**What each one is for, and why it exists:**

- **sharpe** = (mean − RB24) / sd. The textbook baseline. It punishes upside
  variance exactly as hard as downside variance, which is the wrong objective for
  a late-round pick where the whole point is buying a tail. Included as the thing
  to beat, not as the answer.
- **sortino** = (mean − RB24) / downside deviation. Same idea, but only shortfall
  below replacement counts as risk. Closer to what a drafter feels; still a
  mean-based summary and still blind to how big the upside is.
- **omega** = E[points above RB12] / E[shortfall below RB12]. **The headline.** In
  redraft a bust is *droppable*: a back who finishes RB40 is cut in week 6 and
  replaced from waivers, so the realised loss is capped near the replacement line
  while the gain is not capped at all. Omega is the only specification here whose
  asymmetry matches that structural asymmetry. RB12 is the reference because that
  is the bar for "this pick gave me a genuine weekly starter advantage".
- **P(top-5)** — the raw league-winning probability with no risk adjustment at
  all. Included precisely so the reader can see how much work the adjustment is
  doing. Answer: none of the ordering depends on it.
- **CVaR-up** = mean of the best decile. The size of the prize when it lands, as
  opposed to the chance of it landing.
- **omega@ADP** — the same gain/shortfall ratio, but each player measured against
  what a back at *his own* ADP has historically returned. The only specification
  that charges each player his own price; the other five compare all four against
  a common bar even though they cost 22 picks apart.

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
Pollard is 22 picks cheaper, so once each back is charged his own price the
cheaper, worse player is marginally better value. That flip is small — 4.98 vs
4.63 — and sits well inside the §6.4 uncertainty, where Swift > Pollard is only a
79% lean. [evidence]

### Robustness to the largest modelling choice

How a comparable's role transfers to the target is the single biggest judgement
call in the project. Three specifications, 50,000 draws each:

| median | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| **ratio** (comp's role *change* on his own prior role) | 159.6 | 147.5 | **185.1** | 78.8 |
| **absolute** (comp's realised role level used directly) | 170.8 | 151.9 | **173.3** | 93.7 |
| **blend** (geometric mean) | 164.9 | 149.1 | **177.5** | 86.8 |

| P(top-12) | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| ratio | 0.315 | 0.234 | 0.463 | 0.031 |
| absolute | 0.354 | 0.257 | 0.376 | 0.082 |
| blend | 0.325 | 0.229 | 0.410 | 0.034 |

Judkins first and Montgomery last under all three. **The `absolute`
specification is the most generous available reading of Montgomery** — it says
the situation determines the workload regardless of the man — and it still leaves
him at P(top-12) = 0.082 and P(bust) = 0.767. His conclusion does not depend on
the specification.

**Swift is the one whose position does.** Under `absolute` he closes to within
0.022 of Judkins on P(top-12) (0.354 vs 0.376) and his median rises 11 points to
170.8 while Judkins's falls 12 to 173.3 — a 25-point gap becomes a 2.5-point one.
Read together with the raw comparable finishes in §5, where Swift's neighbours
actually did better, **Judkins over Swift should be held as a lean at roughly 89%
confidence and abandoned without regret if anything changes in either backfield.**
[inference]

---

## 8. Where this disagrees with consensus

CBS publishes under PPR; components were rescored to this league with
`scoring.Scoring()`.

| | ADP | CBS proj. | CBS (rescored) | sim mean | difference | CBS's percentile in our distribution |
|---|---|---|---|---|---|---|
| Judkins | 49.1 / RB22 | 276 att, 973 yds, 10 TD | 178.3 | 184.1 | **+5.8** | 0.46 |
| Pollard | 63.7 / RB26 | 227 att, 992 yds, 6 TD | 150.7 | 142.2 | −8.5 | 0.52 |
| Swift | 41.9 / RB20 | 247 att, 1117 yds, 9 TD | 202.2 | 158.9 | **−43.3** | **0.74** |
| Montgomery | 54.6 / RB24 | 263 att, 1012 yds, 9 TD | 172.3 | 88.0 | **−84.3** | **0.93** |

Two disagreements are large enough to act on.

**Montgomery (0.93).** CBS's projection sits at the **93rd percentile** of our
distribution: it is close to our *best case*, treated as the base case. The
mechanism is visible in the data rather than argued. CBS projects **263 carries
for a back who took 158 last year**, moved teams, and joins a room where Woody
Marks handled 196 carries and a 43.6% share in 2025. The market has priced the
March depth chart. The comparable evidence says the median outcome for this exact
profile — ageing, reduced role, new team — is **RB41/42**, and only 3 of 30 such
backs finished top-12. [evidence]

**Swift (0.74).** CBS projects **9 rushing TDs**. Swift scored 9 rushing TDs on
**223 carries** in 2025 — a **4.04%** rate against a **3.2%** league mean (0.0317,
§4.2) — while Kyle Monangai took 12 of Chicago's goal-line carries to Swift's 14.
§4.2 says a TD rate above the mean regresses roughly 85% of the way back.
**Consensus is projecting the good version of a coin flip.** [inference]

CBS's own positional ranks sharpen the picture: it lists Swift **RB13** against
an RB20 ADP and Pollard **RB31** against an RB26 ADP. **CBS is more bullish on
Swift than the market is, and we are less bullish than either.**

Judkins (0.46) and Pollard (0.52) sit essentially at our median. **We do not
disagree with consensus about them at all.**

### The availability assumption, isolated

**CBS projects all four for a healthy 17 games.** Our simulator does not — it
draws games played from the comparables, which averaged 13.2 to 14.2. Stripping
that difference out:

| | CBS | sim mean if 17 games | sim mean all-in | cost of missed games |
|---|---|---|---|---|
| Swift | 202.2 | 182.6 | 158.9 | **23.7** |
| Pollard | 150.7 | 167.1 | 142.2 | **24.9** |
| Judkins | 178.3 | 210.7 | 184.1 | **26.6** |
| Montgomery | 172.3 | 106.7 | 88.0 | **18.7** |

**Roughly 19 to 27 points per back — 12.6% to 17.5% of what a healthy 17-game
season would return — is quietly assumed away by any projection that pencils in
17 games.** [evidence]

On a like-for-like 17-game basis we are *above* CBS on Judkins (+32.4) and
Pollard (+16.4), and the Swift gap shrinks from −43.3 to −19.6. **Most of our
apparent pessimism is just counting injuries.** The genuine disagreement — the
one that survives the like-for-like adjustment, at −65.6 even with 17 healthy
games granted — is Montgomery. [evidence]

---

## 9. Limitations

| Limitation | Effect |
|---|---|
| **Thirty comparables per player** | The binding constraint on everything. Every mid-range role probability in §5 carries a ±0.15–0.18 Wilson interval; every median in §6.3 sits inside a resample interval **47 to 57 points wide**. Swift, Pollard and Judkins are not separable within it |
| **The point estimate is barely better than ADP** | MAE 57.4 vs 58.7 over 139 player-seasons — 2%, and worse than ADP in 2024. Do not use the medians as projections |
| **Tails are ~5 pp too thin** | p10–p90 coverage 0.755 against a nominal 0.80. P(top-5) and P(disaster) both understate. And at n = 139 the coverage estimate itself carries roughly ±0.07, so 0.755 is not confidently distinguishable from 0.80 — the tail-thinness conclusion is a lean, not a certainty |
| **P(top-N) treats the rest of the position as fixed** | The other ~60 backs are not simulated jointly. A season where the whole position is unusually healthy or unusually injured enters only through the resampled threshold |
| **Comparable pool is 2019–2025** | Seven years of NFL committee-usage norms, inherited whole. If usage patterns shift, the pool is stale before the model knows it |
| **No injury feature beyond prior games played** | A specific known injury, a surgery, a durability history — none of it is visible to this model. Games played is only r = 0.185 stable season to season, so this is the weakest-modelled component of the distribution |
| **The competition feature is crude** | Best other back in the room by draft slot and prior touches. It cannot see coaching intent, contract structure, or a training-camp report |
| **Montgomery's March depth chart is excluded by judgement, not measurement** | The choice not to feed the RB1 label into the model is defensible (§2) but it is a judgement call, and it is the call that drives the largest disagreement with consensus. If Houston genuinely intends a Montgomery-led backfield, this report is wrong about him |
| **Judkins is the weakest comparable match** | He is a second-year player and most of his neighbours are third- or fourth-year backs, because there are few second-year backs with a 65% carry share to compare him to |
| **The Judkins-over-Swift ordering is specification-dependent** | It holds under all six risk metrics but nearly vanishes under the `absolute` role transfer, and reverses on the raw comparable finish record (§5) |
| **§4's usefulness test used a linear baseline** | No non-linear model was tried. With a 0.9% ceiling on the available gain, it was not worth trying — but that is an argument, not a measurement |
| **The ≥ 8 game filter conditions on survival** | Backs who lost their job in week 3 are absent from both sides of every stability transition. The stability numbers in §4.1 are, if anything, optimistic |
| **464 transitions across 7 season pairs is not a large sample** | The efficiency CIs are wide enough that "rushing TD rate is exactly zero-stable" cannot be rejected either |
| **Closing lines only** | Market context in the panel is closing, not opening. Inherited from the QB lab and unresolved there too |

---

## 10. What to actually do

**First, the frame that governs everything below.** The gap between Judkins's
median and Swift's is **25.5 points**. The simulator's own out-of-sample error is
**57.4 points**. **A 25-point median gap sits comfortably inside a 57-point error
bar.** Treat the ordering as a tiebreaker, not a plan. [evidence]

1. **Fade David Montgomery.** This is the one call the project can genuinely
   defend. The 106-point gap from Judkins is nearly two MAEs; it holds under all
   six risk metrics, all three role specifications, all five values of k, and in
   1.00 of 200 comparable resamples. P(bust) = 0.844, P(disaster) = 0.662, and
   consensus sits at the 93rd percentile of our distribution. Even the most
   generous specification available leaves him at P(top-12) = 0.082. [evidence]
2. **Prefer Judkins at RB22 ADP** — best floor (p10 88.8, p25 133.1, P(bust)
   0.309) and best risk-adjusted ceiling (first under all six specifications,
   rank spread zero). Hold it at **89% confidence over Swift**, not as a
   conviction, and note that his edge shrinks to almost nothing under the
   `absolute` role transfer.
3. **Swift is second, and he is also the most expensive.** He goes **7 picks
   before Judkins** at ADP 41.9 and is 79% likely to beat Pollard. Our −43.3 gap
   to CBS is not an indictment of the player: **more than half of it — 23.7 of
   43.3 points — is the 17-game assumption**, and most of the rest is 9 projected
   rushing TDs off a 4.04% rate that regresses ~85% of the way to 3.2%. Discount
   the projection, not the player — but note you are paying the highest price of
   the four for the second-best distribution.
4. **Take Pollard only at his price.** He is the worst of the three on every raw
   measure but the cheapest by 22 picks, and he wins the one specification that
   charges each back his own price (omega@ADP 4.98 vs Swift's 4.63). That flip is
   inside the noise; it is a reason to be indifferent, not a reason to prefer him.
5. **Discount every 17-game projection you read by 19–27 points.** That is the
   measured cost of the availability assumption, and it is silently baked into
   the entire consensus board. On a like-for-like basis this work is *more*
   optimistic than CBS on two of the four.
6. **Never pay for a touchdown rate.** A player cannot predict his own TD rate
   across halves of the same season at better than r = 0.172, and 85.7% of the
   gap between high- and low-TD-rate backs regresses away by the next season. Buy
   goal-line carries and total carries instead — they persist at 0.493 and 0.772.
7. **Do not build a workload model.** Nineteen metrics were tested walk-forward
   against a single lagged column and none survived multiple-testing correction.
   The best possible gain was 0.9% of RMSE. This is the QB lab's finding
   reproduced, and it is worth more than any of the four picks above.
8. **Use the distributions, not the medians.** The point estimate is 2% better
   than an ADP curve; the calibrated floor, ceiling and bust probability are the
   product. If you take one number from this report, take **P(bust)** — Judkins
   0.309, Swift 0.420, Pollard 0.495, Montgomery 0.844.

**The most valuable output of this project is the negative.** The premise —
a back whose workload distribution is more positively skewed than consensus
assumes — did not survive contact with the data. Three of four are negatively
skewed, the fourth's positive skew sits on a 92-point base, and the resulting
point estimates are no sharper than the draft board they were meant to beat.

---

## 11. Reconciliation against the outputs

Every figure in this report was checked against `outputs/*.csv`,
`data/rb_week_panel.parquet` and `data/rb003_comparables.parquet`, and the
non-CSV figures (17-game means, the ADP baseline fit, CBS percentiles, the
pairwise resample matrix) were re-derived by re-running the relevant simulator
calls read-only. **The RB-004 numbers reproduce exactly.** Thirteen items in the
source write-ups did not reconcile; where a source disagrees with its data, this
report follows the data.

| # | Source | Claim | What the data says |
|---|---|---|---|
| 1 | RB-003 §3 | Montgomery's two comps who "became lead backs" were **Ingram 2019, Johnson 2020** | Wrong pair. The two who reached 18+ touches/gm are **Javonte Williams 2025** and **Aaron Jones 2024**. Ingram (15.4) and Johnson (16.1) both landed in the 13–18 band. Three of thirty finished top-12: Conner 2021 RB5, Ingram 2019 RB8, Javonte Williams 2025 RB11 |
| 2 | RB-003 §3 | The "nearest" comparable lists | Curated, not the k-nearest by distance. Swift's list names James Cook 2025 (10th) and Nick Chubb 2022 (11th) while omitting Conner 2020 (7th, RB25), Gibson 2022 (8th, RB35) and Pacheco 2024 (9th, RB63); Judkins's omits Edwards-Helaire 2021 (6th, RB38); Pollard's omits Jones 2023, Swift 2025 and Elliott 2022 (6th–8th). **The omissions skew toward poor finishes and the inclusions toward good ones.** All named players are in the k=30 sets; the ordering is not as presented |
| 3 | RB-002 §1 | stayed n = 350 / switched n = 114; stayed r = 0.798 / 0.754 / 0.741, switched 0.671 / 0.657 / 0.607 | Recompute gives **349 / 115** and stayed **0.801 / 0.757 / 0.742**, switched **0.665 / 0.652 / 0.605**. Point estimates are deterministic, so this is a genuine mismatch. Conclusion (switching costs 0.10–0.14 of correlation) unchanged |
| 4 | RB-002 §3 | "the best observed p is 0.009" | **0.008** (1 − 0.992 for TD-per-touch). BH threshold 0.10/19 = 0.0053 confirmed; zero survivors either way |
| 5 | RB-002 §3 | red-zone carry share "gain −0.013, **p = 0.993**" | The CSV column is **P(gain > 0) = 0.007**. 0.993 is P(gain < 0) — the figure quoted contradicts the header of the table it sits under |
| 6 | RB-002 §3 | role metrics are "~0.75 correlated with the baseline" | Understated. Within-season r(role, fp/gm) on the 464 transitions is **0.856 to 0.934**. The redundancy argument is stronger than the write-up claims |
| 7 | RB-002 §1 | 11 of 21 bootstrap CI bounds | Differ from the CSV in the third decimal (e.g. carries/gm [0.728, 0.809] vs CSV [0.729, 0.809]) — bootstrap jitter across runs. All point estimates match exactly. CSV values used here |
| 8 | RB-002 §1 | the stability table | Omits `fp_total` (r = 0.662 [0.610, 0.712]), which is in the CSV. Restored in §4.1 |
| 9 | RB-001 §1 | Swift 2024 implied team total **20.3**; Pollard 2024 carries/gm **16.3** | Panel gives **20.2** and **16.2** (260 carries / 16 games = 16.25). Rounding |
| 10 | RB-003 §2, §4 | Pollard prior touches/gm **16.7**; Judkins committee-band Wilson interval **[.04,.26]** | **16.6** (16.647); the Wilson interval for 3/30 is **[.03,.26]**. Rounding |
| 11 | RB-004 §1 | threshold sd column | Population sd (ddof = 0) over 8 seasons. The sample sd would be ~7% larger (RB3 31.0 rather than 29.0). Not an error; the convention is now stated |
| 12 | RB-004 §3 / §5 | two different **P(top-5)** figures (0.204 vs 0.198 for Judkins) | Both correct and both in CSVs, but they are different quantities — resampled threshold vs the fixed 245.9 bar. The write-ups do not say so. Flagged in §6.3 |
| 13 | RB-003 §6 | "The ordering never changes. Judkins ≥ Swift > Pollard ≫ Montgomery under every k from 15 to 60" | Contradicted by the table directly beneath it, which reproduces exactly. On median comparable points **Swift leads Judkins at k = 30** (170 vs 167), they tie at k = 45, and Swift ties Pollard at k = 15. Montgomery's last place *is* invariant under every k. Corrected in §5 |

Additionally, RB-004's backtest section does not mention that **2024 is the one
season where the simulator loses to the ADP curve** (MAE 67.9 vs 66.6). It is in
`rb004_backtest.csv` and is stated in §6.5 here.

Three figures could not be checked against any artefact in the repository and are
carried forward on the source's authority: the ADP collection window
(2026-08-11 to 2026-08-18, 1,466 drafts), the depth-chart snapshot date
(2026-03-14), and the draft slots of the competing backs.

---

## Reproducing

```bash
./.venv/Scripts/python.exe rb-risk-upside/scripts/rb001_panel.py       # panel + situations
./.venv/Scripts/python.exe rb-risk-upside/scripts/rb002_stability.py   # stability + the negative
./.venv/Scripts/python.exe rb-risk-upside/scripts/rb003_scenarios.py   # comparables + role bands
./.venv/Scripts/python.exe rb-risk-upside/scripts/rb004_simulate.py    # Monte Carlo + risk + backtest
```
