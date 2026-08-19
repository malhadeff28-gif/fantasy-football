# QB-023 — Independent replication of the pick-value curve (QB-012)

**Verdict: REPLICATES**, with two documented specification divergences — one of
which corrects the top of the curve down by ~4% and the QB2 cost band up from
"5–8" to **7.9 points (95% CI 5.0–11.5)**.

Script: `qb-pairing/scripts/wave2_weather_and_replication.py`
Evidence: `qb-pairing/outputs/qb023_*.csv`

---

## Independence protocol

`qb-pairing/src/qbpair/benchmarks/draft_value.py` was **neither read nor
imported**. The curve was rebuilt from:

- `shared/market-data/markets.py` → `load_ffc_adp(season, teams=12, fmt="standard")`,
  fetched live from the Fantasy Football Calculator API for 2018–2025
- `shared/nfl-data/nfl_data.py` → `load_player_stats(2018..2025)`
- `shared/utilities/scoring.py` for the league definition

Disclosure: after the independent curve was finalised and written to CSV, the
*caller* script `scripts/phase6_draft_cost.py` was read to confirm which
function produced the committed file. It calls `dv.pick_value_curve(SEASONS)`
and reveals nothing about the internals. `draft_value.py` itself was never
opened.

## Specification, derived from the league format rather than from the committed code

| choice | value | rationale |
|---|---|---|
| replacement levels | **QB12, RB24, WR36, TE12** | 12 teams × 1QB/2RB/3WR/1TE starters |
| replacement basis | Nth-best realised season total at the position, that season | the last startable player in the league |
| scoring | nflverse `fantasy_points` (standard, 4-pt passing TD) | verified league scoring |
| fantasy season | **NFL weeks 1–16** | league regular season 1–13 + playoffs 14–16 (`scoring.League`) |
| PAR | season points − replacement | |
| usable PAR | max(0, PAR) | a below-replacement pick returns nothing usable |
| hit | PAR > 0 | |
| pick | floor(ADP), buckets of 12 | |
| drafted but never played | counted as **0**, not dropped | dropping them is survivorship bias |
| K / DST | excluded (no `player_stats` coverage) | 214 of 1,574 ADP rows |

Scoring cross-check: reconstructing `fantasy_points` from components via
`scoring.skill_points` matches nflverse exactly on all but 93 of ~44,000
skill-player weeks; every mismatch is exactly −6.0 or −12.0, i.e. kick/punt
return touchdowns, which nflverse credits and `skill_points` does not. Immaterial
at the season-total level.

## Data integrity

| check | result |
|---|---|
| ADP rows fetched, all positions, 2018–2025 | 1,574 |
| ADP rows that are K or DST | 214 |
| skill-player ADP rows | 1,360 |
| matched to nflverse by normalised name + position + season | 1,332 (**97.9%**) |
| duplicate name keys within a position-season | **0** |
| ADP identical to the committed cache (`ffc_adp_all_2018_2025.parquet`) | **yes** — 1,574 vs 1,574 rows, all 1,574 keys join, max abs ADP difference **0.0** |

**The draft-cost inputs are byte-identical.** Any divergence lives entirely on
the realised-points and join side.

Replacement levels, weeks 1–16 (season points):

| season | QB | RB | WR | TE |
|---|---|---|---|---|
| 2018 | 272.1 | 135.5 | 102.8 | 79.1 |
| 2019 | 248.8 | 128.1 | 108.5 | 75.6 |
| 2020 | 267.2 | 123.4 | 108.9 | 85.8 |
| 2021 | 240.0 | 121.6 | 94.2 | 84.7 |
| 2022 | 236.1 | 126.2 | 98.4 | 75.6 |
| 2023 | 246.3 | 131.9 | 106.6 | 85.4 |
| 2024 | 237.0 | 138.0 | 104.0 | 81.6 |
| 2025 | 241.3 | 133.4 | 97.1 | 86.5 |

---

## The independent curve

Uncertainty is a 1,000-draw bootstrap **resampling whole seasons**, because
players inside a season share a replacement level and a scoring environment.
Point estimates alone would be misleading here.

| picks | n | mean usable PAR | se (iid) | 95% season-bootstrap CI | median | hit rate |
|---|---|---|---|---|---|---|
| 1–12 | 102 | **66.88** | 5.30 | [59.7, 74.7] | 56.97 | 0.83 |
| 13–24 | 98 | **45.80** | 4.53 | [39.7, 52.0] | 34.10 | 0.76 |
| 25–36 | 103 | 27.40 | 3.21 | [22.0, 32.5] | 16.90 | 0.66 |
| 37–48 | 101 | 28.47 | 3.47 | [23.0, 34.0] | 16.20 | 0.59 |
| 49–60 | 96 | 21.39 | 3.20 | [13.9, 29.5] | 3.09 | 0.53 |
| 61–72 | 96 | 13.85 | 2.41 | [10.1, 17.6] | 0.00 | 0.44 |
| 73–84 | 95 | 10.45 | 1.97 | [7.1, 13.6] | 0.00 | 0.37 |
| 85–96 | 100 | 13.81 | 2.83 | [8.3, 19.8] | 0.00 | 0.37 |
| 97–108 | 89 | **9.39** | 2.15 | [8.2, 10.5] | 0.00 | 0.29 |
| 109–120 | 86 | 9.13 | 2.40 | [2.5, 17.7] | 0.00 | 0.22 |
| 121–132 | 101 | 6.33 | 1.56 | [3.3, 10.4] | 0.00 | 0.21 |
| 133–144 | 66 | 5.33 | 1.74 | [2.1, 9.2] | 0.00 | 0.18 |
| 145–156 | 111 | **4.83** | 1.28 | [2.9, 7.9] | 0.00 | 0.19 |
| 157–168 | 114 | 1.70 | 0.60 | [0.4, 3.9] | 0.00 | 0.11 |
| 169–180 | 2 | 0.00 | — | — | 0.00 | 0.00 |

The median column is the part the mean hides: **from pick 61 onward the median
draft pick returns exactly zero usable points above replacement.** The curve is
a mean over a distribution that is mostly zeros with a thin tail of hits.

---

## Bucket-by-bucket comparison with `outputs/phase6_pick_value_curve.csv`

| picks | n (indep) | n (committed) | independent | committed | diff | % | committed inside 95% CI |
|---|---|---|---|---|---|---|---|
| 1–12 | 102 | 100 | 66.88 | 69.99 | −3.11 | −4.4% | yes |
| 13–24 | 98 | 87 | 45.80 | 47.24 | −1.44 | −3.0% | yes |
| 25–36 | 103 | 95 | 27.40 | 27.11 | +0.29 | +1.1% | yes |
| 37–48 | 101 | 89 | 28.47 | 27.95 | +0.53 | +1.9% | yes |
| 49–60 | 96 | 80 | 21.39 | 21.25 | +0.14 | +0.7% | yes |
| 61–72 | 96 | 75 | 13.85 | 13.15 | +0.70 | +5.3% | yes |
| 73–84 | 95 | 78 | 10.45 | 11.00 | −0.56 | −5.1% | yes |
| 85–96 | 100 | 84 | 13.81 | 10.34 | +3.46 | +33.5% | yes |
| 97–108 | 89 | 70 | 9.39 | 7.75 | +1.64 | +21.2% | **no** (season-clustered); yes on iid CI [5.2, 13.6] |
| 109–120 | 86 | 68 | 9.13 | 6.57 | +2.56 | +39.0% | yes |
| 121–132 | 101 | 82 | 6.33 | 8.27 | −1.94 | −23.4% | yes |
| 133–144 | 66 | 49 | 5.33 | 5.37 | −0.04 | −0.8% | yes |
| 145–156 | 111 | 89 | 4.83 | 4.49 | +0.34 | +7.7% | yes |
| 157–168 | 114 | 95 | 1.70 | 2.48 | −0.78 | −31.5% | yes |
| 169–180 | 2 | 1 | 0.00 | 0.00 | 0.00 | — | yes |

**14 of 15 committed values fall inside the independent 95% season-clustered
bootstrap interval; 15 of 15 fall inside the iid interval.** The single
exception (97–108) is an artefact of an unusually tight bootstrap interval in
that bucket — between-season variance happens to be small there — and the
committed 7.75 sits comfortably inside the ordinary interval [5.2, 13.6].

The eye-catching percentage gaps are all in buckets where the absolute values
are single digits and the intervals span a factor of three or more. **Reporting
this curve as a set of point estimates was the mistake; buckets 6 through 13 are
not statistically distinguishable from one another in either version.**

---

## Where the divergence comes from

### 1. The fantasy calendar (explains the top of the curve)

| week window | 1–12 | 13–24 | 97–108 | 145–156 |
|---|---|---|---|---|
| weeks 1–13 (league regular season) | 54.57 | 34.29 | 7.58 | 4.77 |
| **weeks 1–16 (reg + playoffs) — primary** | **66.88** | **45.80** | **9.39** | **4.83** |
| weeks 1–17 | 67.77 | 47.08 | 9.40 | 5.18 |
| weeks 1–18 (all NFL regular season) | 69.52 | 49.31 | 9.22 | 5.17 |
| *committed* | *69.99* | *47.24* | *7.75* | *4.49* |

The committed top-of-draft values (70.0, 47.2) sit at the weeks 1–17/1–18 end of
this range. **This league's season ends in week 16** (`League.playoff_weeks =
(14, 15, 16)`). Points scored in NFL weeks 17 and 18 cannot be used by anyone in
this league, and star players are frequently rested in week 18. Counting them
inflates the value of an early pick by roughly **4%** (66.9 → 69.5) and of a
second-round pick by **8%** (45.8 → 49.3).

No single week window reproduces all four committed numbers — 1–18 matches picks
1–12, 1–16 matches 13–24, 1–13 matches 97–108 — so the calendar is not the only
difference.

### 2. The player universe (explains the late buckets)

Identical ADP input, different retention:

| join | players retained of 1,360 drafted |
|---|---|
| naive raw-string name match | 1,260 (92.6%) |
| **this rebuild** (normalised name + position + season) | **1,332 (97.9%)** |
| committed curve | 1,142 (84.0%) |

The shortfall is not uniform — it grows with pick number:

| bucket | drafted | naive join | this rebuild | committed |
|---|---|---|---|---|
| 1–12 | 102 | 98 | 101 | 100 |
| 49–60 | 96 | 91 | 96 | 80 |
| 97–108 | 89 | 83 | 89 | 70 |
| 145–156 | 111 | 101 | 106 | 89 |

That gradient (98% retained at the top, 80% at pick 150) is the signature of a
name-matching shortfall on obscure late-round players, not of a deliberate
filter. **Sensitivity check:** rebuilding the curve on the naive-join universe —
deliberately throwing away the 72 hardest-to-match players — moves no bucket by
more than **1.3 points**. So coverage of this magnitude does not by itself
explain the late-bucket gaps; those gaps are sampling noise on 66–114
observations per bucket.

### 3. Pick rounding (immaterial)

floor / round / ceil of ADP moves buckets by at most ~2 points and never changes
an ordering. Full grid in `qb023_spec_grid.csv` (24 specifications).

---

## Additional attacks on the curve

### Look-ahead in the benchmark

The replacement level is a **realised end-of-season rank**. Nobody knows on
draft day who QB12 will be. Re-running with the **prior season's** replacement
level — which a drafter did know:

| picks | same-season repl. | prior-season repl. | diff |
|---|---|---|---|
| 1–12 | 66.88 | 64.22 | −2.66 |
| 13–24 | 45.80 | 44.81 | −0.99 |
| 25–36 | 27.40 | 25.48 | −1.92 |
| 49–60 | 21.39 | 23.05 | +1.66 |
| 97–108 | 9.39 | 8.47 | −0.92 |
| 109–120 | 9.13 | 6.45 | −2.68 |
| 145–156 | 4.83 | 4.63 | −0.20 |

Differences of −2.7 to +1.7, unsigned and well inside the bootstrap intervals.
The ex-post replacement level is a defensible accounting convention here, not a
material look-ahead. It should still be labelled as ex-post.

### Survivorship in the ADP universe

The curve only sees players who were drafted. Players who finished above
replacement without ever being drafted are invisible:

| position | top-N slots (8 seasons) | filled by an undrafted player | share |
|---|---|---|---|
| QB | 96 | 8 | 8.3% |
| RB | 192 | 12 | 6.3% |
| WR | 288 | 34 | 11.8% |
| TE | 96 | 27 | **28.1%** |

This is **correct** for costing a *pick* — an undrafted player costs no pick. It
is **wrong** if the curve is read as the value of a *roster spot*, because more
than a quarter of startable tight ends and one in eight startable receivers were
free. Any strategy comparison that charges a draft pick for a roster spot that
could have been filled from waivers is overcharging.

### The number QB-012 actually leans on

QB-012 argues a second quarterback goes around picks 106–135, where the forgone
alternative is "roughly 5–8 points", and that this is the low bar QB2 must clear.

Independent measurement of exactly that band:

> **picks 106–135: n = 235, mean usable PAR = 7.92 (se 1.30, 95% season-bootstrap
> [4.95, 11.46]), hit rate 0.21**

The central estimate is at the **top** of the published 5–8 range and the
interval reaches **11.5**. On the league-consistent weeks 1–13 window it would be
roughly 20% lower again. The qualitative conclusion is unchanged — the bar is
low — but the bar is higher and considerably less certain than "5–8" implies.
Against QB-012's marginal QB2 value of **+9.7 points alongside an elite QB1**,
the pick cost's own 95% interval [5.0, 11.5] **overlaps that figure**. The elite-
QB1 case for a second quarterback is not distinguishable from break-even once
the pick cost is reported with uncertainty. The mid-priced-QB1 case (+30.1)
survives comfortably.

---

## Verdict

**REPLICATES.**

- The curve reproduces independently in shape, magnitude and every ordering that
  matters. 14/15 committed bucket means fall inside the independent 95%
  season-clustered bootstrap interval; 15/15 fall inside the iid interval.
- The ADP input is provably identical (max abs difference 0.0 across 1,574 rows).
- Two specification divergences are identified and quantified:
  1. **the committed curve appears to count NFL weeks 17–18**, which this league
     does not play — top-of-draft value overstated by roughly 4–8%;
  2. **the committed curve retains 84% of drafted skill players versus 97.9%
     here**, with the shortfall concentrated in late picks — demonstrated by
     sensitivity test to be worth at most 1.3 points per bucket.
- The material corrections are to the *presentation*, not the *result*: buckets
  6–13 are statistically indistinguishable from one another and should never
  have been quoted as point estimates; the QB2 pick cost is **7.9 points
  [5.0, 11.5]**, not "5–8"; and the +9.7 elite-QB1 marginal value does not clear
  its own cost interval.

Recommended log change: QB-012 stays **KEEP** for the mid-priced-QB1 conclusion,
but the elite-QB1 marginal value (+9.7) should be downgraded to **UNCERTAIN** —
it is inside the noise of the pick cost it is being compared against.
