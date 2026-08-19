# QB-024 — Rerun of the slot-7 late-QB analysis under four corrections

**Wave 2 · Replication Agent · 2026-08-18**
Scripts: `qb-pairing/scripts/phase8b_late_qb_slot7.py` (updated),
`qb-pairing/scripts/qb024_slot7_attribution.py` (attribution harness)
Outputs: `qb-pairing/outputs/qb024_*.csv`
Target: the published slot-7 analysis (commit `e7b2538`), which predates
corrections C1–C4.

---

## Verdict

| Headline | Before | After | Moved |
|---|---|---|---|
| Historical plan ranking (premium > premium+late > late single > late pair) | 46.3 / 38.8 / 33.8 / 23.6 | 45.5 / 36.2 / 33.9 / 22.2 | **ordering unchanged, ≤ 2.6 pts** |
| Best-minus-worst spread | 22.7 | 23.4 | +0.7 |
| Unpaired SEs | 15.4 – 19.7 | 11.5 – 21.1 | comparable |
| **Paired** late-pair − late-single | not reported | **−11.7 (se 5.5, t = −2.13)** | new, and it is the only distinguishable difference |
| 2026 top-ranked pair NET | 66.8 | **2.9** | **−63.9** |
| Marginal value of QB2 in 2026 | +33.3 "insurance" | **−2.0 at best, 0 of 125 pairs positive** | sign flip |
| Recommended output | a pair table | **one quarterback, one pick, one probability** | changed |

**One line.** The historical plan comparison barely moved — the week-17-18 bug
was very nearly self-cancelling in this script because it inflated quarterback
points and the replacement level by almost the same amount — but the 2026
recommendation collapsed, because it rested entirely on an insurance premium
that no longer exists. **From pick 7 of 12 the honest output is a single
quarterback, and the rank-matched −10.6 the corrections imported is confirmed
independently here at −11.7 (se 5.5) on a completely different construction.**

---

## 0. Replication first

The pre-correction configuration (`V0`) reproduces the published table exactly,
to the last digit, including the worst-season column that was never in a CSV:

| plan | published NET | V0 rerun | published se | V0 se | published worst | V0 worst |
|---|---|---|---|---|---|---|
| premium QB1 only | 46.3 | **46.3** | 19.3 | **19.3** | −20.1 | **−20.1** |
| premium QB1 + late QB2 (~127) | 38.8 | **38.8** | 18.7 | **18.7** | −25.5 | **−25.5** |
| late single QB (at 79) | 33.8 | **33.8** | 15.4 | **15.4** | −4.9 | **−4.9** |
| late pair (79 + ~127) | 23.6 | **23.6** | 19.7 | **19.7** | −47.3 | **−47.3** |

The pre-correction pick curve and replacement levels are reconstructed by
monkeypatching `draft_value.season_totals` back to week 18, so every other step
— PAR, usable PAR, the ADP join, the buckets — is the committed code path, not a
re-implementation of it. The reconstruction returns the 2024 QB replacement
level at **282.0**, which is the exact number the QB-023 commit message records
for the pre-fix state. [evidence] (`qb024_replacement_levels.csv`)

Nothing below is an argument about a number that failed to reproduce.

---

## 1. Audit of the two functions flagged for inspection

### 1.1 `pick_cost` — **no bug. C2 is a no-op here.**

`phase6_draft_cost.py::pick_cost_lookup` charged 0.0 for any float ADP that fell
between two integer bucket edges. **This script never had that defect**: both of
its cost functions (`pick_cost` on the 2026 board and `cost` inside the
backtest) already did `int(min(max(round(a), lo), hi))` before scanning. The
phase-6 fix comment says as much — "the same fix was already in phase8 and was
never backported here".

Applying C2 therefore changes **exactly nothing**: `V1b` and `V2` are identical
in every cell of every table. To make the audit informative rather than merely
negative, the harness also runs the counterfactual `Xbug` — this script *with*
the phase-6 bug — which shows what was avoided:

| plan | correct | with the phase-6 bug | free points |
|---|---|---|---|
| premium QB1 only | 45.5 | 45.5 | 0.0 |
| premium QB1 + late QB2 | 36.2 | 37.0 | +0.8 |
| late single QB | 33.9 | 35.3 | +1.4 |
| late pair | 22.2 | 24.4 | +2.2 |

14 of 195 QB ADP values in 2018–2025 fall in a bucket gap, and they are
concentrated in the late-round range where a QB2 is drafted, so the bug would
have flattered exactly the strategies under test. The 2026 board was never
exposed at all: it charges pick *numbers* (79, 90, 103, …), which are integers.
[evidence] (`qb024_attribution_plans.csv`, config `Xbug`)

**Action taken:** the two duplicate lookups are collapsed into one
`cost_lookup()` — duplication is how the sibling bug survived in phase 6 — and
`_check_cost_lookup()` now asserts that every tenth of a pick from 1 to 168
returns a real bucket value. That check runs on every invocation.

### 1.2 `insurance_premium` — **removed, not re-derived.**

The function interpolated a premium of 15 points at ADP 16 up to 35 points at
ADP 84, then scaled it by `qb2_ppg / 18`. Those endpoints are the retired QB-012
marginals (+9.7 alongside an elite QB1, +30.1 alongside a mid-priced one)
grossed up. QB-020 showed the +30.1 was a comparison between strategies with
**different QB1s** and that rank-matched the marginal is **−10.6 pts/season,
negative at 7 of 7 QB1 ranks**.

**A premium cannot be re-derived from a negative marginal.** Re-anchoring the
same interpolation on −10.6 would have produced a function whose only honest
output is "subtract a roughly constant number", which is not a premium and not a
function of QB1's ADP — the interpolation over ADP was never measured, it was
asserted. So the term is deleted, and a partner is priced from the two pieces of
it that were actually measured:

```
marginal(QB2) = bye-week coverage  +  weekly choice edge  -  his pick cost
              = max(0, ppg2 - 16)  +  1.7                 -  cost(pick)
```

The choice edge is QB-020's rank-matched +1.7 pts/season, not the +4.9 this
script used (QB-013, measured on a universe that was not ADP-selectable, and
superseded). Bye coverage is a real, QB2-specific quantity in a forward season
because byes are known exactly.

This construction is deliberately the **optimistic** bound: CBS does not project
the QB2 busting, and the historical rank-matched estimate of the same object is
−10.6. On 2026 prices it still returns a negative marginal for **every one of
125 feasible pairs**, best case −2.0.

### 1.3 A defect found while rerunning: C1 is not safe to apply by halves

`season_totals` now stops at week 16, so the QB12 replacement level and the pick
curve are on the league calendar. **This script was still summing quarterback
points through week 18** — the panel runs to week 17 or 18 depending on season —
while subtracting a 16-week replacement level. That mismatch is worth **+29 to
+33 points on every plan** (config `V1` below): pure accounting, no football.

The fix is one line (`LAST_WEEK` truncates the backtest weeks), and it is
included in this rerun as `C1b`. Anyone who applied C1 and reran without it
would have reported that waiting on a quarterback is worth 63 points.

### 1.4 A second defect: the QB2 timing sweep is unbalanced

Cells at picks 151 and 162 average over **4 and 2 seasons** respectively, not 6,
because the FFC board runs out before those picks in recent seasons (QB-020 §4.2
documented the same shrinkage: 147 players listed in 2025). The published sweep
printed those cells alongside the six-season cells with no marker. The table now
prints `n` per cell, flags unbalanced cells, and computes the spread and the
median standard error over balanced cells only.

---

## 2. Attribution: what each correction did

Corrections applied cumulatively, so each column is the effect of exactly one
change. NET, 2020–2025 average, points.

| plan | V0 published | V1 +C1a (curve+repl to wk16) | V1b +C1b (QB pts to wk16) | V2 +C2 (cost lookup) | V3 +C3 (drop spread_abs) | V4 +C4 (insurance) | **total** |
|---|---|---|---|---|---|---|---|
| premium QB1 only | 46.3 | 78.5 | 45.5 | 45.5 | 45.5 | 45.5 | **−0.8** |
| premium QB1 + late QB2 | 38.8 | 72.2 | 35.8 | 35.8 | 36.2 | 36.2 | **−2.6** |
| late single QB (at 79) | 33.8 | 63.3 | 33.9 | 33.9 | 33.9 | 33.9 | **+0.1** |
| late pair (79 + ~127) | 23.6 | 54.3 | 23.8 | 23.8 | 22.2 | 22.2 | **−1.4** |
| *step size* | — | *+29.4 to +33.4* | *−29.4 to −36.4* | ***0.0*** | *0.0 to −1.6* | *0.0* | |

Supporting quantities, same configs:

| quantity | V0 | V4 |
|---|---|---|
| QB points, premium QB1 | 370.4 | 337.5 |
| QB points, late single QB | 317.8 | 288.5 |
| pick cost, premium QB1 | 51.0 | 47.3 |
| pick cost, late single QB | 10.9 | 9.9 |
| worst season, late pair | −47.3 | −29.8 |
| worst season, late single | −4.9 | −2.8 |

**Which correction mattered most, by table:**

* **Historical plans: C1, and only because of the half-applied trap.** Applied
  consistently, C1 is worth −0.8 to +0.2 points. The 8-season QB12 mean falls
  from 273.1 to 248.6 (−24.5) and the top of the pick curve from 70.0 to 67.1,
  but the quarterbacks lose two weeks of points at the same time — 370.4 → 337.5
  for a premium QB1 — and the two cancel to within 3 points. **The published
  historical conclusion was accidentally robust to the bug that fed it.**
* **C2: exactly zero** (see §1.1).
* **C3: small and confined to the pair rows** (0.0 to −1.6). Dropping
  `spread_abs` changes the projection, which changes only *which* of two
  quarterbacks gets started in a given week; single-QB plans have no such
  decision and do not move at all. The late pair's worst season moves −21.7 →
  −29.8, i.e. the sign of the change is not stable across seasons — consistent
  with a feature worth −0.01 pp.
* **C4: zero on the historical side** (it scores realised points and never used
  the premium) **and decisive on the 2026 board**.

### The 2026 board

| step | top pair | NET | what changed |
|---|---|---|---|
| V0 as published | Hurts + Dart (79/114) | **66.8** | — |
| V1 +C1a | Hurts + Dart | 68.5 | week-16 pick curve: picks 79/114 get *cheaper*, so NET rises |
| V1b +C1b | Hurts + Dart | 39.4 | CBS 17-game totals rebased onto the 16-week league season, replacement rebased with them |
| V4 +C4 | Hurts + Dart | **2.9** | insurance premium (33.3) deleted, choice edge 4.9 → 1.7 |

**−63.9 points, of which −36.5 is C4 and −29.1 is the calendar rebase.** The
`insurance` column of the old CSV was carrying half the reported value of every
pair. [evidence] (`qb024_attribution_2026_board.csv`,
`late_qb_slot7_pairs_2026.csv` for the before)

### The QB2 timing sweep

NET by the pick at which QB2 is taken, QB1 fixed at 79:

| QB2 at | V0 | V4 | n seasons |
|---|---|---|---|
| none | 33.8 | 33.9 | 6 |
| 90 | 47.4 | 47.4 | 6 |
| 103 | 25.7 | 25.7 | 6 |
| 114 | 26.9 | 28.9 | 6 |
| 127 | 23.6 | 22.2 | 6 |
| 138 | 41.5 | 36.5 | 6 |
| 151 | 37.0 | 35.6 | **4** |
| 162 | 20.0 | 22.5 | **2** |

Spread over balanced cells 23.8 → 25.2, median se ≈ 12. Still non-monotonic,
still one specific quarterback per cell per season. **The disclaimer stands
unchanged: no optimal QB2 pick can be read off this table.** The only cell that
matters is `none`, and it is mid-pack.

---

## 3. Standard errors — is anything distinguishable now?

The published claim was that six seasons give ±15–20 points per plan against a
22.7-point spread, so no difference is established. **That is still true for the
unpaired comparison, and it was the wrong test.**

| plan | NET | unpaired se | paired diff vs late single QB | paired se | t |
|---|---|---|---|---|---|
| premium QB1 only | 45.5 | 21.1 | +11.6 | 30.2 | 0.39 |
| premium QB1 + late QB2 | 36.2 | 20.9 | +2.3 | 28.8 | 0.08 |
| late single QB (at 79) | 33.9 | 11.5 | — | — | — |
| late pair (79 + ~127) | 22.2 | 16.0 | **−11.7** | **5.5** | **−2.13** |

Plans share seasons and often share quarterbacks, so their season-level errors
are correlated and the unpaired SEs overstate the uncertainty **of the
differences**. Pairing by season:

* **premium vs late single: +11.6 ± 30.2.** Not distinguishable from zero, and
  the paired SE is *larger* than the unpaired one — because the two plans hold
  different quarterbacks and their season errors are close to uncorrelated.
  Premium beats waiting by 90 in 2022 and 80 in 2024, and loses by 64 in 2023
  and 77 in 2025. Six seasons cannot separate them.
* **late pair vs late single: −11.7 ± 5.5, t = −2.13.** The two arms share the
  same QB1 by construction, so nearly all the common season variance cancels and
  what remains is the partner. **This is the same rank-matched contrast QB-020
  ran on phase 6, from a different slot-based construction, and it lands within
  1.1 points of the −10.6 that correction imported.** It is negative in 5 of 6
  seasons.

**So the conclusion is unchanged where the published report said it was
uncertain, and newly firm where it did not look:** which quarterback you take
first is not resolvable at n = 6; whether you take a second one is, and the
answer is no.

Caveat on the t-statistic: n = 6, df = 5, so |t| = 2.13 is p ≈ 0.086 on a
two-sided test — suggestive, not conventionally significant, and it is one of
three differences examined. It is reported as corroboration of an
independently-derived figure, not as a fresh discovery. [inference]

---

## 4. The revised 2026 recommendation

Availability is unchanged by any correction — it uses FFC's published ADP
standard deviation and a normal approximation — and the ADP was **refreshed
live** for this rerun: FFC 12-team standard, drafts from **2026-08-11 to
2026-08-18, n = 1,466**. ADP moves daily; rerun before drafting.

Because the QB2 marginal is negative at every price, the output is one
quarterback, not a pair.

### Which single quarterback, at which pick, with what probability

`take at` is the latest of your picks (79, 90, 103, 114, 127, 138) at which he
is still ≥ 55% likely to be on the board. NET is against the 12th-best
*projected* quarterback on the same CBS scale, over the 16 weeks this league
plays, minus the measured value of the pick spent.

| QB | ADP (sd) | take at | P(there) | pick cost | NET | bye |
|---|---|---|---|---|---|---|
| **Jalen Hurts** | 77.5 (13.0) | **79** | **45%** | 10.0 | **+4.9** | 10 |
| Jaxson Dart | 122.0 (16.8) | 114 | 68% | 5.8 | +1.9 | 8 |
| Trevor Lawrence | 93.3 (17.2) | 90 | 58% | 9.6 | +0.7 | 7 |
| Brock Purdy | 88.1 (11.7) | 79 | 78% | 10.0 | −2.0 | 8 |
| Patrick Mahomes | 100.0 (14.5) | 90 | 76% | 9.6 | −4.4 | 5 |
| Caleb Williams | 92.5 (15.1) | 90 | 57% | 9.6 | −4.8 | 10 |
| Justin Herbert | 107.1 (14.9) | 103 | 61% | 7.6 | −9.7 | 7 |

Gone before pick 79 whatever they project: Josh Allen (24), Drake Maye (47),
Lamar Jackson (53), Joe Burrow (55), Dak Prescott (65), Jayden Daniels (72).

**The plan.** Take **Jalen Hurts at pick 79** if he is there — 45%. If not,
**Brock Purdy** (78% at 79) is the same projection for the same pick. If both
are gone, wait: **Trevor Lawrence at 90** (58%), **Patrick Mahomes at 90**
(76%), then **Jaxson Dart at 114** (68%), who is the best points-per-pick on the
board and costs half as much. P(at least one of Hurts / Purdy / Stafford is
there at 79) = **93%**. Do not spend a second pick on a quarterback.

Robustness: Hurts, Lawrence, Dart and Purdy hold a top-5 slot under all five
settings tested (pick cost ×0.5/×1.0/×1.5, and shrinking every projection edge
to half and a quarter of CBS's spread). The pick-cost spread across rounds 7–12
is only 4.6 points, so this ranking is driven by projected points per game, not
by which late pick you spend. A replacement-level shift cannot reorder it at all
— it is a constant — which is why the old "insurance ×0.5/×1.5" grid was never a
real test of anything.

### Three honest caveats on the NET column

1. **It is a ranking, not a forecast of surplus.** The replacement is the 12th
   quarterback on the *projected* scale (280 points over 16 weeks). The realised
   QB12 has averaged **240** over 2021–2025. That 40-point gap is projection
   optimism — CBS gives every quarterback 17 games and no injuries — not
   available surplus.
2. **The top seven gettable quarterbacks span 15 points** (+4.9 to −9.7; the
   full gettable list of 18 runs down to −41.3), against projection error that
   QB-021 measured in points per game. The difference between Hurts and Purdy is
   inside the noise of the projections that generated it; the difference between
   either of them and Jordan Love is not.
3. **The 55% availability threshold is a choice, not a measurement.** It is the
   only free parameter left in the recommendation.

---

## 5. What did not change

* No look-ahead anywhere. Every historical pick is chosen by ADP rank alone;
  every weekly start is chosen on the pregame projection. The 2026 board uses
  only preseason ADP and preseason projections.
* The availability model (normal around ADP with FFC's published sd).
* The historical *ordering* of the four plans.
* The disclaimer on the QB2 timing sweep.
* The downside argument for waiting, which survives and is now cleaner: the late
  single QB's worst season is **−2.8** against **−20.8** for premium QB1. Waiting
  is still defensible as risk reduction rather than as an edge.

## 6. Files

| file | contents |
|---|---|
| `outputs/qb024_single_qb_slot7_2026.csv` | the recommendation: every QB, take-at pick, availability at each of your six picks, NET |
| `outputs/qb024_pairs_slot7_2026.csv` | all 125 feasible pairs with the partner priced from measured components |
| `outputs/qb024_plan_comparison_slot7.csv` | corrected historical plan table |
| `outputs/qb024_plan_paired_diffs_slot7.csv` | paired season differences and their SEs |
| `outputs/qb024_plan_by_season_slot7.csv` | season-by-season NET, all four plans |
| `outputs/qb024_qb2_timing_sweep_slot7.csv` | corrected sweep, with n per cell |
| `outputs/qb024_attribution_plans.csv` | every plan under all seven configs |
| `outputs/qb024_attribution_sweep.csv` | the sweep under all seven configs |
| `outputs/qb024_attribution_2026_board.csv` | 2026 top-3 under each correction |
| `outputs/qb024_pick_curve_shift.csv` | pick curve, week 18 vs week 16 |
| `outputs/qb024_replacement_levels.csv` | QB12 by season, week 18 vs week 16 |

Reproduce with:

```
.venv/Scripts/python.exe qb-pairing/scripts/phase8b_late_qb_slot7.py 7 12
.venv/Scripts/python.exe qb-pairing/scripts/qb024_slot7_attribution.py
```
