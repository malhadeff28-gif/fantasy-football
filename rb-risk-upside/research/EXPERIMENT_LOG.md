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
