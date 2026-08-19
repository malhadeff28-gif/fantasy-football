# RB Risk / Upside - Experiment Log

Every meaningful experiment gets an ID and a verdict. Machine-readable mirror
lives in `experiments.csv`.

Verdicts: **KEEP** / **REJECT** / **UNCERTAIN**.

---

## Wave 0 - workload, scenarios and simulation (2026-08-18)

| ID | Question | Result | Verdict |
| --- | --- | --- | --- |
| RB-001 | Can an as-of RB workload panel be built? | Yes | **KEEP** |
| RB-002 | Do workload metrics predict beyond lagged points? | **No, none of 19 survive** | **KEEP finding / REJECT predictors** |
| RB-003 | Do the four backs have distinct role distributions? | Yes | **KEEP** |
| RB-004 | Are simulated outcome distributions usable? | Yes, with a stated ceiling | **KEEP with ceiling** |

### Verified 2026 situations, from data rather than memory

| Player | Team | ADP | Competition in the room |
| --- | --- | --- | --- |
| D'Andre Swift | CHI | 41.9 / RB20 | Kyle Monangai took 33% of carries and 12 goal-line carries to Swift 14 in 2025 |
| Quinshon Judkins | CLE | 49.1 / RB22 | Dylan Sampson, 65 carries and 40 targets, a passing-down complement |
| **David Montgomery** | **HOU** (was DET) | 54.6 / RB24 | Woody Marks, 196 carries, 43.6% share, 16 games in 2025 |
| Tony Pollard | TEN | 63.7 / RB26 | Tyjae Spears plus 2026 fifth-round rookie Singleton |

**Montgomery changed teams**, Detroit to Houston, confirmed three ways (player
table, 2026 roster, March 2026 depth chart) and corroborated by ADP. He is also
the ambiguous case: the RB1 depth-chart label comes from a snapshot dated the
week he signed, and Marks is a real incumbent.

Two data limits found: `load_rosters_weekly` hard-refuses 2026, and
`markets.load_cbs_projections` was silently broken for every position except
quarterback because it hard-coded the QB column layout. Both fixed in `shared/`.
Tight ends turned out to need their own 11-column schema; using the receiver
layout filtered out every TE row.

### The answers

**Highest floor: Judkins.** Median 185, 10th percentile 89, 25th percentile
133, P(bust) 0.31 - best of the four on all four measures. He also has zero
comparables who ended the season as a backup.

**Highest risk-adjusted ceiling: also Judkins.** First on all six
specifications - Sharpe, Sortino, omega, P(top-5), upside CVaR, and
cost-adjusted omega at ADP. Rank spread zero.

**The ranking is robust; the margins mostly are not.** Judkins first and
Montgomery fourth under all six metrics and all three role-transfer
specifications. But resampling the 30 comparables gives **P(Judkins beats
Swift) = 0.89** and **P(Swift beats Pollard) = 0.79**. Only Montgomery
separates with certainty (1.00 against everyone). The single metric flip is
Swift against Pollard on omega-at-ADP, where the 22-pick discount edges it.

**The premise did not hold.** The brief hoped to find a back whose workload
distribution is more positively skewed than consensus assumes. Three of the
four are *negatively* skewed. The only positively skewed one is Montgomery at
1.04 - but that is a long right tail on a 92-point base, which is a lottery
ticket rather than a market inefficiency.

### Where this disagrees with consensus

- **Montgomery.** The CBS projection of 172.3 sits at the **93rd percentile** of
  the simulated distribution. The median here is 79, P(bust) 0.84. Comparables
  for an ageing back with a reduced role who changed teams median out around
  RB40.
- **Swift.** CBS projects 9 rushing touchdowns off a 4.0% rate that RB-002
  shows regresses roughly 85% of the way back to 3.2%.
- **Availability.** CBS gives all four backs 17 games, silently assuming away
  **19 to 27 points each**. On a like-for-like 17-game basis this work is
  *above* CBS on Judkins (+32) and Pollard (+16).

### Two findings worth more than the picks

**RB-002 is a clean negative, and it reproduces the QB trap exactly.** Nineteen
workload metrics were tested walk-forward against a lagged-points baseline and
**none survives Benjamini-Hochberg correction at q = 0.10**. The best gain was
0.9% of RMSE. Volume is highly stable - carries per game correlate 0.77 season
to season - but stability is not usefulness, which is precisely what QB-018
found for pressure and tracking data. Touchdown rate is mostly noise:
split-half within a single season r = **0.17**, and a 2.3x gap in touchdown
rate collapses to 1.1x the following year.

**RB-004 states its own ceiling.** Backtested over 139 player-seasons the
median is unbiased (48.9% of outcomes below it against a nominal 50%) and the
interquartile range is calibrated (49.6%), but 10th-to-90th coverage is 0.755
against 0.80, so the tails are about 5 pp thin and tail probabilities
understate. And **MAE is 57.4 against 58.7 for a naive ADP curve - a 2%
improvement.** The distribution is the product; the point estimate is not
sharper than the market. **A 25-point gap between two medians sits inside a
57-point error bar**, which is the honest frame for every comparison above.

---

## Synthesis pass, 2026-08-19 - and a weakened conclusion

`RB_RISK_UPSIDE_RESEARCH.md` written. Every figure was re-derived from the
output CSVs, the panel parquet and a read-only rerun of the simulator rather
than transcribed from the Wave 0 write-ups. **All RB-004 numbers reproduce
exactly.** Thirteen discrepancies were found in the source documents and
corrected; two of them matter.

### 1. RB-003 named the wrong comparables

The claim that two of thirty Montgomery comparables became lead backs cited
**Ingram 2019 and Johnson 2020**. The two who actually reached 18-plus touches
per game are **Javonte Williams 2025 and Aaron Jones 2024**; Ingram (15.4) and
Johnson (16.1) both landed in the 13-18 band. Corrected counts: three of
thirty finished top-12, seven of thirty top-24.

### 2. The ranking is NOT invariant, contradicting its own table

RB-003 stated that the ordering never changes across comparable-pool sizes.
The table directly beneath it disagrees: on median comparable points **Swift
leads Judkins at k = 30** (170 against 167), they tie at k = 45, and Swift ties
Pollard at k = 15. **Only Montgomery in last place is invariant.**

### 3. The raw comparables actually favour Swift

A genuine addition, not a correction. On raw comparable finishes **Swift leads
- 13 of 30 top-12, median RB16 - against Judkins at 10 of 30, median RB18.5.**
The Judkins-first ranking comes from the ratio-mode role transfer applied to
his larger prior role. That is the same instability the 89% resample figure
measures, now surfacing in a second independent statistic.

### Other corrections

- The published nearest-comparable lists were **curated rather than k-nearest**,
  and the omissions skew toward poor finishes while the inclusions skew toward
  good ones. That is a selection bias in the direction of the conclusion.
- RB-002 stayed/switched table does not reproduce: n = 349/115 and r =
  0.801/0.665, not 350/114 and 0.798/0.671.
- Best observed p is 0.008, not 0.009.
- The red-zone carry share p of 0.993 contradicts its own column header,
  where P(gain > 0) = 0.007.
- Correlation with the baseline is 0.856-0.934, not the stated ~0.75.
- Two undisclosed ambiguities now surfaced: two different P(top-5) definitions
  are in use, and **2024 is the one backtest season where the simulator loses
  to the ADP curve**.

### Revised bottom line

Judkins still has both the best floor and the best risk-adjusted ceiling under
all six specifications - but at 89% confidence over Swift, inside a 57-point
error bar, with the raw comparables pointing the other way and the ordering
flipping at k = 30.

**The only separation this work can defend is fading Montgomery.** Everything
finer than that is inside the noise.
