# QB-020 — Attacking the QB-012 / QB-011 economics

**Wave 2 · Skeptic Agent · 2026-08-18**
Script: `qb-pairing/scripts/wave2_skeptic_economics.py`
Outputs: `qb-pairing/outputs/qb020_*.csv` (19 files)
Targets: QB-012 (+9.7 elite / +30.1 mid, net of pick cost), QB-011 (+12.0 gross)

---

## Verdict

| Claim | Verdict | Revised number |
|---|---|---|
| **+30.1 alongside a mid-priced QB1** | **BROKEN** | It is not a QB2 effect at all. Rank-matched it is **−12.0** (se 7.6) or **−6.0** (se 8.7). The published figure compares a *different QB1*. |
| **+9.7 alongside an elite QB1** | **BROKEN as a point estimate** | Reproduces exactly, but it is one arbitrary cell. Move the partner two ADP slots and it becomes −5.9 or −4.3. Rank-matched and pooled: **−4.2 pts/season (95% CI [−11.2, +2.5])**. |
| **Mechanism: "availability insurance"** | **BROKEN** | Decomposed: coverage **+1.0**, weekly choice **+1.7** pts/season at the median streamer. Admitting *more* injured QB1s **lowers** the measured gain. |
| **+12.0 gross (QB-011)** | **WEAKENED, survives directionally** | **+12.0 (season-cluster se 4.0, CI [+4.3, +20.2])** at the median streamer; **+8.4** at the measured streamer level; **+2.1** at the 75th percentile. Gross of draft cost, so ≈ **+4 to +6 net**. |
| Pick-value curve itself | **SURVIVES** | Replicates to 0.0000 from an independent rebuild. Survivorship correction moves it 0.22 pts. |

**One-line summary.** The economics reproduce exactly, and then fall apart under
every dimension that was never varied. The load-bearing number for the draft
recommendation is not +9.7 / +30.1; it is **approximately zero, and probably
slightly negative, with an uncertainty band wider than the effect.**

---

## 0. Replication first — the attacks land on the real numbers

Nothing below is an argument about a number I failed to reproduce.

**Pick-value curve, rebuilt from raw `nflreadpy` player stats and the FFC ADP
endpoint without calling `benchmarks/draft_value.py`** (own name normaliser, own
replacement ranks re-derived from `LEAGUE`): identical in every one of 15 pick
buckets, `n` identical, **max absolute difference 0.0000**. [evidence]
(`qb020_curve_replication.csv`)

**Phase 6 strategy table**, recomputed inside this script:

```
premium + late partner  53.0   premium (QB1 only)  43.3   late pair        42.3
streaming               36.3   strong value + comp 20.5   single value QB  12.2
```

Identical to the published table in `QB_PAIRING_RESEARCH.md` §12, including sd
and worst-season. The marginals reproduce at **+9.7** and **+30.1**. [evidence]

**Phase 2c pair table**: `model` and `oracle` totals match
`outputs/phase2c_pair_value.csv` to 0.0 on all **1,106** pairs; the mean gain
reproduces at **+12.0**. [evidence]

---

## 1. ATTACK 5 (run first, because it dominates everything) — sample size

Phase 6 averages over **five seasons**. Not six — see §7.1. Every headline is a
mean of five paired numbers, so the honest standard error is a paired
season-level SE on n = 5, df = 4.

| Claim | est | sd | se | 95% CI | p |
|---|---|---|---|---|---|
| +9.7 elite | +9.7 | 10.8 | **4.8** | **[−3.8, +23.1]** | 0.116 |
| +30.1 mid | +30.1 | 76.1 | **34.0** | **[−64.3, +124.5]** | 0.426 |

**Neither published number is statistically distinguishable from zero.** The
+30.1 has a confidence interval 189 points wide — it cannot distinguish "QB2 is
worth thirty points" from "QB2 costs you sixty". The director's prior of 15–20
point SEs on season-level quantities was, if anything, optimistic. [evidence]

For QB-011 the reported sample is 1,106 pairs, which looks large and is not.
Every pair inside a season shares the same 17–18 weeks of outcomes and the same
quarterbacks recur across hundreds of pairs. Treating pairs as independent gives
se = 0.93; a **season-cluster bootstrap** (2,000 draws, resampling whole seasons)
gives **se = 4.0**, 4.3× larger. [evidence]

| Streamer pctile | mean gain | iid se (wrong) | cluster se (right) | 95% CI |
|---|---|---|---|---|
| 50 | +12.0 | 0.93 | **4.0** | **[+4.3, +20.2]** |

QB-011 is the **only** headline in this family that clears zero. [evidence]

---

## 2. ATTACK 1 — replacement level

### 2.1 The sweep

Weekly replacement is the Nth percentile of that week's de-facto starters. It was
set at the median and never varied.

| Streamer pctile | QB-012 elite (+9.7) | se | QB-012 mid (+30.1) | se | QB-011 (+12.0) | cluster se |
|---|---|---|---|---|---|---|
| 30th | **+18.8** | 7.3 | +47.4 | 34.6 | **+20.2** | 4.2 |
| 40th | **+14.9** | 6.2 | +39.5 | 34.0 | **+16.4** | 4.0 |
| **50th (published)** | **+9.7** | 4.8 | +30.1 | 34.0 | **+12.0** | 4.1 |
| 60th | **+5.5** | 3.9 | +22.4 | 34.4 | **+8.4** | 3.8 |
| 75th | **−2.1** | 4.0 | +6.9 | 34.6 | **+2.1** | 3.7 |

All three collapse at the 75th percentile, exactly as the attack predicted. The
elite figure changes sign; QB-011 falls by 82%. Every gain is a linear function
of one unvalidated modelling choice, and the choice moves the answer by more than
the answer itself. [evidence]

### 2.2 But the 75th percentile is not a real opponent — so I measured the real one

Reporting "it dies at the 75th percentile" would be dishonest without saying what
percentile a streamer actually reaches. Phase 6 already contains the answer: its
own `streaming (no QB drafted)` strategy picks the best *pregame projection*
among QBs outside the top-19 ADP. Reading that strategy's realised weekly score
back into the same week's starter distribution converts the replacement level
from an assumption into a measurement.

| season | weeks | mean pctile | median pctile |
|---|---|---|---|
| 2020 | 17 | 49.9% | 48.5% |
| 2021 | 18 | 50.7% | 39.5% |
| 2022 | 18 | 62.1% | 64.4% |
| 2023 | 18 | 59.2% | 65.6% |
| 2024 | 18 | 57.8% | 58.7% |
| 2025 | 18 | 57.9% | 62.5% |
| **pooled** | **107** | **56.3%** | **59.5%** |

(`qb020_streamer_percentile.csv`)

**A real streamer lands at the 56th–60th percentile, not the 50th.** [evidence]
Two caveats, in both directions: the simulated streamer knows who started, which
flatters it; but its pool excludes the top-19 ADP QBs, some of whom are free
agents in a 12-team 1QB league, which penalises it. The defensible band is
**50th–60th**, with the published median sitting at the end of that band most
generous to the pairing thesis. [inference]

**Corrected headline at the measured replacement level (60th percentile):**

- QB-012 elite: **+5.5 (se 3.9, CI [−5.4, +16.4], p = 0.24)**
- QB-011: **+8.4 (cluster se 3.8, CI [+1.5, +16.0])**

**Attack 1 verdict: the numbers do not collapse at a realistic streamer, but they
lose roughly 40% of their magnitude and the elite figure loses what little
statistical footing it had.**

---

## 3. ATTACK 2 — the pick-value curve averages hits and misses

Four specifications of the cost of a pick (`qb020_pick_value_curves.csv`):

| Picks | mean (published) | median | hits only | zero-filled |
|---|---|---|---|---|
| 97–108 | 7.7 | **0.0** | 27.1 | 7.7 |
| 109–120 | 6.6 | **0.0** | 34.4 | 6.4 |
| 121–132 | 8.3 | **0.0** | 33.9 | 8.0 |

A QB2 goes around picks 106–135, so these three rows are the bar.

**The median usable PAR of a pick after round 5 is exactly zero.** More than half
of all picks from 61 onward return nothing startable. Under the median
specification a QB2 pick is free, and +9.7 becomes **+17.2 (se 4.5, p = 0.019)** —
the only specification in this report under which the elite claim is
conventionally significant.

Under `hits_only` (mean over players who finished above positional replacement)
+9.7 becomes **−15.6 (se 4.7)** and +30.1 becomes **+1.9**. I am not leaning on
this: conditioning on the pick working is not knowable on draft day, so
`hits_only` is an upper bound on cost, not a rival estimate of expected cost. It
is reported because the attack asked for it. [evidence, with the caveat stated]

### 3.1 A defect found while varying the curve: 8% of picks are charged nothing

`phase6_draft_cost.py::pick_cost_lookup` scans integer ranges
`[12k+1, 12k+12]` with `<=`, but ADP is a float. Any value strictly between
`12k+12` and `12k+13` matches no bucket and falls through to
`rows[-1]["mean_usable_par"]` — the deepest bucket, worth **0.0**. Those picks
are free.

**126 of 1,574 ADP values (8.0%) land in a gap.** Four of them are quarterbacks
phase 6 actually drafts:

| season | QB rank | player | ADP | charged | should be |
|---|---|---|---|---|---|
| 2020 | 18 | Jared Goff | 132.70 | 0.0 | 8.3 |
| 2021 | 18 | Deshaun Watson | 132.10 | 0.0 | 8.3 |
| 2023 | 18 | Russell Wilson | 132.30 | 0.0 | 8.3 |
| 2025 | 10 | Dak Prescott | 84.20 | 0.0 | 11.0 |

Three of the four are the rank-18 partner in `late pair` — the strategy the
+30.1 rests on. Repairing the lookup (`mean_fixed`):

- +9.7 → **+9.7** (its picks happen not to fall in gaps)
- +30.1 → **+24.6**; on the league calendar as well, **+15.3**

[evidence] (`qb020_cost_lookup_gap.csv`)

**Attack 2 verdict: the cost estimator moves +9.7 across a ±16-point range —
wider than the effect and wider than its own standard error. The mean-vs-median
choice alone is worth ±7.5 points. This is a live, unacknowledged source of
uncertainty, but neither *defensible* variant (mean, median, zero-filled) kills
the finding; only the inadmissible one does.**

---

## 4. ATTACK 4 — survivorship in the ADP set

Two distinct issues; one is negligible and one is real but bounded.

**4.1 Drafted players who never played are silently dropped.** `adp_with_outcomes`
inner-joins ADP to season totals, so a drafted player who never recorded a
regular-season stat line vanishes rather than entering at zero — a genuine upward
bias, because those are pure busts.

Magnitude: **23 of 1,165** drafted RB/WR/TE rows (2.0%) fail to join. Of those, 13
share a name key with another season (real players who recorded no REG stat
line); 10 never appear at all (rookie busts, camp cuts, or unresolved names).
Entering all of them at zero moves a pick's expected value by **0.22 points on
average** and moves +9.7 to **+9.9**. [evidence]

**4.2 The ADP list is shallower than a real draft.** A 12-team 16-round draft
fills 192 slots. The FFC list runs out earlier:

| season | players listed | deepest ADP | slots with no ADP entry |
|---|---|---|---|
| 2018–2023 | 192–226 | 166–175 | 0 (covered) |
| 2024 | 180 | 167 | 12 |
| **2025** | **147** | **166** | **45** |

So the curve has essentially nothing to say about the last two rounds in recent
seasons, and its final bucket (169–180) is n = 1. This does not touch the QB2
range (picks 106–135, n = 70–85 per bucket), which is well populated. [evidence]

**Attack 4 verdict: SURVIVES. The bias exists, runs in the predicted direction,
and is worth 0.22 points. It is the weakest of the five attacks.**

---

## 5. ATTACK 3 — is the gain a few extreme events, and is it really insurance?

### 5.1 Direct decomposition of the mechanism

Rather than infer the mechanism from a correlation, I decomposed the gain
week-by-week on the rank-matched grid (§6):

```
gross marginal = coverage (weeks QB1 did NOT start: pair points - the streamer)
               + choice   (weeks QB1 DID start:     pair points - QB1 himself)
```

| Streamer pctile | coverage | choice | gross | QB2 pick cost | **net** |
|---|---|---|---|---|---|
| 30th | +12.7 | +1.7 | +14.3 | 5.8 | **+8.5** |
| 40th | +7.3 | +1.7 | +9.0 | 5.8 | **+3.2** |
| **50th** | **+1.0** | **+1.7** | +2.6 | 5.8 | **−3.1** |
| 60th | −3.6 | +1.7 | −1.9 | 5.8 | **−7.7** |
| 75th | −13.0 | +1.7 | −11.4 | 5.8 | **−17.2** |

Coverage weeks: **3.7 per season** (bye plus missed starts).
(`qb020_mechanism_decomposition.csv`)

This is the most informative table in the report. [evidence]

- The **weekly start/sit choice is worth +1.7 pts/season** and is invariant to the
  streamer, as it must be. QB-013 claimed +4.9 on a different universe; on a
  rank-matched ADP-selectable universe it is a third of that.
- **The insurance is worth +1.0 pts/season at the median streamer and is negative
  at the measured one.** Not because QB1 never misses time — he misses 3.7 weeks —
  but because an ADP-rank-14-to-20 QB2 is *not better than the best quarterback
  you can stream*. The waiver wire already supplies the insurance, for free.
- The pick costs 5.8. The gross gain does not cover it.

**The published mechanism claim — "that value is availability insurance against
QB1 failing" — is refuted by its own data.** [evidence]

### 5.2 The injury split

Post-hoc diagnostic (it conditions on realised availability; it is an
explanation, not a draft rule).

Phase-6 grid, net marginal:

| Streamer pctile | QB1 missed 0–2 (n=179) | QB1 missed 3+ (n=97) |
|---|---|---|
| 40th | −0.0 | **+9.2** |
| **50th** | −2.6 | **−4.1** |
| 60th | −4.6 | **−13.4** |
| 75th | −8.7 | **−32.8** |

At a generous streamer the injury subset does pay. **At the median and above it
pays *less* than the healthy subset** — owning a mediocre QB2 you feel obliged to
start is worse than streaming.

Phase-2c pair sample, mean gain at the median streamer:

| sample | pairs | mean | cluster se | 95% CI |
|---|---|---|---|---|
| all pairs | 1,106 | +12.0 | 4.0 | [+4.3, +20.2] |
| 5% trimmed both tails | 994 | **+9.2** | 2.6 | [+4.6, +14.7] |
| QB1 missed 0–2 starts | 883 | +11.0 | 3.8 | [+4.7, +18.7] |
| QB1 missed 3+ starts | 223 | +16.2 | 5.6 | [+5.7, +27.2] |

Trimming the top and bottom 5% removes **23%** of the gain (+12.0 → +9.2) but the
finding does not live in the tails; the trimmed interval still clears zero.

The injury premium is +5.2 points on 20.2% of pairs, so **injuries explain
0.202 × 5.2 = 1.05 points, i.e. 8.8% of the +12.0.** The other 91% is earned on
pairs where QB1 missed at most two starts. [evidence]

### 5.3 The eligibility filter removes the very events the story needs

`phase2c_pair_value.py` requires **both** quarterbacks to have started 12+ games.
That excludes the largest QB1 failures by construction. Relaxing it:

| min games | pairs | mean gain | cluster se | max QB1 missed | share missing 3+ |
|---|---|---|---|---|---|
| 12 (published) | 1,106 | **+12.0** | 4.0 | 5 | 20.2% |
| 10 | 1,479 | +10.7 | 3.2 | 7 | 28.7% |
| 8 | 1,962 | +8.7 | 2.2 | 9 | 37.0% |
| 6 | 2,304 | **+7.5** | 2.3 | 11 | 39.9% |

Admitting twice as many injury cases **lowers** the estimate by 38%. If the
mechanism were insurance this would have gone the other way. [evidence]

**Attack 3 verdict: the gain is NOT driven by a few extreme events (trimming
costs 23%, not 100%), but the stated MECHANISM is wrong. The value is not
insurance. It is a small weekly-choice edge (+1.7) plus a coverage term that the
waiver wire already provides.**

---

## 6. STRUCTURAL — the +30.1 is not a marginal QB2 effect at all

This is not one of the five commissioned attacks. It is a defect found while
reading the code, and it is the most serious finding in the report.

`phase6_draft_cost.py` defines:

```python
"late pair":       [10, 18],   # ADP QB rank 10 + rank 18
"single value QB": [8],        # ADP QB rank 8
```

and then prints their difference as *"second cheap QB adds +30.1"*. **The two arms
do not share a QB1.** One drafts QB rank 10, the other QB rank 8. The difference
therefore contains the QB1 swap and its pick-cost saving, not just the partner.

Rank-matched, holding QB1 fixed (median streamer, mean pick cost, n = 5 seasons):

| comparison | net | se | 95% CI |
|---|---|---|---|
| as published: QB10+QB18 − QB8 alone | **+30.1** | 34.0 | [−64, +125] |
| **rank-matched: QB8+QB18 − QB8 alone** | **−12.0** | 7.6 | [−33.1, +9.2] |
| **rank-matched: QB10+QB18 − QB10 alone** | **−6.0** | 8.7 | [−30.2, +18.3] |

The sign reverses. **Every point of the +30.1 is QB1 selection, not the second
quarterback.** [evidence] (`qb020_marginal_grid.csv`)

---

## 7. Specification fragility of the +9.7

### 7.1 The panel is unbalanced and the drop is not random

`late pair` has no 2021 row — the rank-18 ADP name does not resolve to a panel
player. The script's `group_by("strategy").mean()` silently averages each strategy
over a *different* set of seasons. The published §12 table happens to use the
balanced 5 seasons; the script as written does not, so the two disagree. Either
way n = 5, and **2021 is the one season in the window where the elite QB1 started
every single game.** It is also the only season the pair loses:

| season | QB1 + QB2 | coverage | choice | cost | **net** |
|---|---|---|---|---|---|
| 2020 | Mahomes + Roethlisberger | +14.7 | 0.0 | 8.3 | **+6.4** |
| **2021** | **Mahomes + Lawrence** | **+1.9** | **0.0** | **8.3** | **−6.3** ← dropped |
| 2022 | Allen + Tagovailoa | +1.5 | 0.0 | 8.3 | **−6.8** |
| 2023 | Mahomes + Goff | +11.9 | +15.4 | 6.6 | **+20.7** |
| 2024 | Allen + Daniels | +6.2 | +11.8 | 7.7 | **+10.3** |
| 2025 | L. Jackson + Lawrence | +21.4 | +2.9 | 6.6 | **+17.7** |

Mean over the 5 reported seasons: **+9.7**. Over all 6 available: **+7.0**.
Three of six seasons are negative. Two seasons (2023, 2025) supply 79% of the
reported total. [evidence] (`qb020_published_cell.csv`)

### 7.2 The partner rank is an unjustified free parameter

`STRATEGIES` pairs the elite QB1 with **ADP QB rank 16**. Nothing in the research
document motivates 16 over 14, 18 or 20. Moving it:

| QB2 ADP rank | net (QB1 = rank 1, n seasons) | se | net (pooled QB1 ranks 1–12) | 95% CI |
|---|---|---|---|---|
| 14 | **−5.9** | 2.4 | **−15.5** | [−21.7, −10.4] |
| **16 (published)** | **+7.0** | 4.8 | **+10.7** | [−6.3, +28.9] |
| 18 | **−4.3** | 4.9 | **−10.6** | [−21.0, +2.2] |
| 20 | **−1.6** | 3.0 | **+1.6** | [−12.5, +17.1] |

**Two ADP slots either side of the published choice flips the sign.** Rank 16 in
this window happens to have drawn Jared Goff (2023), Jayden Daniels (2024) and
Trevor Lawrence (2025). That is draft-slot luck, not a structural property.
[evidence] (`qb020_partner_rank_sensitivity.csv`)

### 7.3 The dropped season is a QB2 bust — and correcting it does not help my case

The 2021 `late pair` row is absent because its rank-18 pick, **Deshaun Watson,
never played a snap in 2021**. That is not missing data. It is the single worst
possible outcome for a strategy that spends a pick on a second quarterback, and
phase 6 deletes the whole season rather than charging it.

Structurally this biases every pair strategy upward: the seasons deleted are
exactly the seasons the second quarterback returned zero. In *this* sample,
however, it does not work out that way, because 2021 was also a season the QB1
(Matthew Stafford, ADP 82) was excellent. Charging the bust properly:

- 2021 `late pair` = 345.8 QB pts, 19.3 pick cost → **net +66.4**, the best of the six
- `late pair` mean: +42.3 over 5 seasons → **+46.3 over 6**
- +30.1 → **+31.7** (se 27.8, CI [−39.8, +103.2], n = 6)

**The correction moves the number the wrong way for the attack, and it is
reported as found.** The defect is real and should be fixed; it happens not to be
what is wrong with the +30.1. [evidence] (`qb020_bust_seasons.csv`)

### 7.4 Phase 6 scores weeks this league does not play

`LEAGUE.regular_season_weeks` is 1–13 and `playoff_weeks` is 14–16, but phase 6
scores every week present in the panel, including weeks 17 and 18. On the league
calendar: +9.7 → **+8.1**, +30.1 → **+20.8**. Small, but it is points from games
no fantasy team plays. [evidence]

### 7.5 The two scripts disagree about what "the median" is

`engine.weekly_replacement` uses `.quantile(0.5)` (nearest interpolation);
`phase2c_pair_value.py` uses `.median()` (linear). On the same panel the two
differ by up to 1.7 pts/season on a single pair. Cosmetic, but it means the two
published figures are not on the same replacement scale.

---

## 8. The generalised estimate — what QB2 is actually worth

The published +9.7 is one cell of a grid. Here is the whole grid: for every
season 2020–2025, every QB1 at ADP QB rank 1–12, and every partner at rank
14/16/18/20, hold QB1 fixed and add the partner. 276 rank-matched cells, nothing
changing except the second quarterback, season-clustered bootstrap.

| Streamer pctile | mean cost (as published) | **mean cost, lookup repaired** | median cost | share of cells > 0 |
|---|---|---|---|---|
| 30th | +8.5 [+1.3, +15.4] | **+7.5 [−0.9, +14.3]** | +14.3 | 50.0% |
| 40th | +3.2 [−3.9, +9.6] | **+2.2 [−6.2, +8.7]** | +9.0 | 41.3% |
| **50th (published)** | −3.1 [−9.7, +3.6] | **−4.2 [−11.2, +2.5]** | +2.6 | **32.2%** |
| **60th (measured)** | −7.7 [−13.8, −1.1] | **−8.7 [−15.9, −2.3]** | −1.9 | 27.2% |
| 75th | −17.2 [−22.7, −11.3] | **−18.2 [−24.7, −12.1]** | −11.4 | 17.4% |

5% trimmed at the median streamer: **−5.0** (se 2.9), so the negative result is
not a tail artefact either.

By QB1 rank at the published specification, the net marginal is negative for **9
of 12 ranks** (rank 1: −1.0; the exceptions are rank 5 at +0.0, rank 9 at +5.0 and
rank 10 at +0.3, none of which excludes zero). There is no QB1 rank at which the
effect is reliably positive. (`qb020_marginal_by_qb1_rank.csv`,
`qb020_grid_bootstrap.csv`) [evidence]

**Revised headline: the marginal value of a second quarterback, net of the pick
it costs, is −4.2 pts/season (95% CI [−11.2, +2.5]) at the published replacement
level and −8.7 (95% CI [−15.9, −2.3]) at the measured one.** Only a third of the
276 rank-matched cells are positive.

Consistency check against QB-011: +12.0 gross at the median, less a 5.8-point
pick cost, is ≈ +6 net on the phase-2c universe (both QBs 12+ game starters,
selected on prior-season PPG rather than ADP), against −3.1 on the ADP-selectable
universe. The gap is the difference between "any two startable QBs" and "the two
quarterbacks a draft actually lets you have". The second is the decision-relevant
one.

---

## 9. Defects found in the target code

| # | File | Defect | Effect |
|---|---|---|---|
| 1 | `phase6_draft_cost.py` | `late pair` [10,18] vs `single value QB` [8] compared as a marginal QB2 effect | **+30.1 → −12.0 / −6.0**; sign reversal |
| 2 | `phase6_draft_cost.py` | Partner rank 16 unmotivated and outcome-determining | sign flips at 14, 18 |
| 3 | `phase6_draft_cost.py` | `group_by("strategy").mean()` over an unbalanced panel | script and paper disagree; drops the one healthy-QB1 season |
| 4 | `phase6_draft_cost.py` | Scores weeks 17–18, outside `LEAGUE` weeks 1–16 | +9.7 → +8.1 |
| 5 | `phase2c_pair_value.py` | `prev_ppg` joins the *projected* panel, which starts at the first test season, so **2020 is silently dropped**; the header still says 2020–2025 | n = 5, not 6 |
| 6 | `phase2c_pair_value.py` | `>= 12 games` eligibility removes the QB1 failures the insurance story rests on | estimate falls to +7.5 when relaxed |
| 7 | `draft_value.py` | `adp_with_outcomes` inner-joins, dropping drafted players who never played | +0.22 pts on a pick (immaterial) |
| 8 | `engine.py` vs `phase2c` | `.quantile(0.5)` (nearest) vs `.median()` (linear) | up to 1.7 pts/season |

None of 1–5 involve look-ahead. All selection remains ADP-based and all
projections remain walk-forward; these are specification and bookkeeping errors,
not leakage.

---

## 10. What this does to the recommendation

The draft recommendation currently rests on "QB2 clears its cost in both
regimes". It does not. [evidence]

What can still be said honestly:

1. **A late QB2 is cheap** — the forgone pick is worth 5–8 points, and its
   *median* return is exactly zero. That part of §11 survives intact. [evidence]
2. **Carrying a second quarterback is roughly break-even.** −3 to −8 points per
   season net, with a confidence interval that includes zero at the published
   replacement level. It is not a source of edge, and it is not a mistake either.
   [evidence]
3. **The decision should not be made on a measured points edge**, because there
   isn't one that survives five seasons of noise and four arbitrary modelling
   choices. It should be made on grounds this analysis does not price: bench
   scarcity, the cost of a mid-season waiver claim, and how much variance the
   manager wants. [inference]
4. **The insurance argument should be dropped.** The waiver wire supplies the
   same coverage for free; an owned QB2 outscores the streamer by 1.0 points a
   season at the median and loses to it at the measured level. [evidence]
5. **QB-011's +12.0 stands, with corrections**: +12.0 (cluster se 4.0) gross at
   the median streamer, +8.4 at the measured one, over 5 seasons not 6, and it is
   gross of draft capital. It measures the value of *having* two startable
   quarterbacks, not of *drafting* one. [evidence]

---

## Reproducing

```
./.venv/Scripts/python.exe qb-pairing/scripts/wave2_skeptic_economics.py
```

Runs in about 10 seconds warm (roughly 30 on the first run, which fetches ADP).
ADP is pulled once from the FFC public API and cached to
`shared/nfl-data/cache/raw/ffc_adp_all_2018_2025.parquet`; everything else reads
the existing nflverse cache. BLAS threads are pinned to 1 before numpy is
imported, and the season-cluster bootstrap is seeded (`default_rng(20260818)`),
so the CSVs are byte-reproducible.

Outputs (`qb-pairing/outputs/`):

| file | contents |
|---|---|
| `qb020_curve_replication.csv` | independent vs published pick-value curve |
| `qb020_pick_value_curves.csv` | four cost specifications by pick bucket |
| `qb020_adp_survivorship.csv` | unmatched drafted players by bucket |
| `qb020_adp_coverage.csv` | ADP list depth vs a 192-pick draft |
| `qb020_streamer_percentile.csv` | measured streamer percentile by season |
| `qb020_strategy_table.csv` | phase-6 table × quantile × cost spec × week window |
| `qb020_headline_uncertainty.csv` | claims ledger with se, CI, t, p |
| `qb020_marginal_grid.csv` | 276 rank-matched cells, full cross of specs |
| `qb020_grid_bootstrap.csv` | clustered bootstrap of the grid |
| `qb020_marginal_by_qb1_rank.csv` | net marginal by QB1 ADP rank |
| `qb020_partner_rank_sensitivity.csv` | net marginal by QB2 ADP rank |
| `qb020_published_cell.csv` | the +9.7 cell, season by season |
| `qb020_mechanism_decomposition.csv` | coverage vs choice |
| `qb020_pair_gains.csv` | 1,106 phase-2c pairs, gains at five quantiles |
| `qb020_pair_value_quantiles.csv` | QB-011 vs streamer sharpness |
| `qb020_pair_trim_and_injury.csv` | trimmed means and injury split |
| `qb020_pair_eligibility.csv` | min-games sensitivity |
