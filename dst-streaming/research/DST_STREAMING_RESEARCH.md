# D/ST Streaming Research

**Two defenses, rotated across weeks 1–6, for a 12-team league**

Scoring: CBS/Yahoo points-allowed tiers · Backtest 2018–2025, walk-forward ·
2026 board rebuilt 2026-08-19 · Panel: 4,222 defense-weeks (4,188 after feature
lags, which is what DST-002 fits on)

> **Critic pass, 2026-08-19.** Every figure was re-derived from its source CSV or
> by re-running its script: **33 claim groups covering 181 individual numbers,
> of which 14 needed correction.** Changed claims are marked **[corrected]**,
> **[weakened]** or **[added]** inline rather than silently edited; the full
> verification log is the appendix. One data bug was found and fixed: the 2026
> ADP join dropped the Rams (FFC `LAR` never matched nflverse `LA`), which
> priced a pick-118 defense as a free waiver-wire streamer.
> `dst007_pairing_engine.py` is repaired and the board re-run; every number below
> is from the repaired run.

---

## 1. Executive summary

**The strategy works, but not for the reason it was proposed, and the edge is
about a point and a half.**

Four independent lines of attack land in the same place: **the opponent's
implied team total is the entire signal.** Defensive form, opponent-offence
quality, opposing-QB identity, and prior-season defensive ratings each add
essentially nothing on top of it.

| Finding | Result |
|---|---|
| Best model *that adds features* vs. the trivial rule | **−0.06 pp** **[corrected]** |
| Opposing-QB identity, on top of team-level | +0.10 pp |
| Defensive form alone | 53.61% (coin flip = 50%) |
| Defensive touchdowns, season to season | **r = −0.075** (random) |
| Preseason defensive knowledge, half-life | ~6.5 weeks, CI [1.8, >50] |
| Drafting a second defense vs. streaming one | **+1.61 pts over weeks 1–6**, se 0.24 **[corrected]** |

**[corrected]** The first row previously read "+0.00 pp". That number is real but
tautological: the script picks the best model by pair accuracy, and the winner
*is* M1, a ridge regression on the single market variable — which ranks
identically to the trivial rule. The honest statement is that the best model
which knows anything beyond the market (M5, everything) scores 60.95% against
the rule's 61.01%. Adding features makes ranking **worse**, by 0.06 pp.

**[corrected]** The second-defense gain was +1.47 before the ADP join was
repaired; the Rams sitting in the streaming pool inflated the streaming
baseline. Repaired: **+1.61 (se 0.24, 95% CI [1.15, 2.07]), positive for 17 of
17 defenses.**

**Recommendation:** draft one defense late if you want one, and stream the other
slot. The pair table below is real and usable, but the second *drafted* defense
buys +1.61 points over six weeks against an alternative pick worth 1.31–5.19
points over the same six weeks — a net loss everywhere before pick ~145 and a
wash after ~157 (see §8, which now compares like with like).

---

## 2. Why weeks 1–6 — and the honest version of that argument

The window was chosen because forecasting gets harder further out. That is true,
but **not for the reason originally assumed.**

**What is true:** 2026 betting lines exist for **93 of 93 games in weeks 1–6**,
7 of 14 in week 7, and **12 of 165 in weeks 8–18** **[corrected: was "8 of
158"; the schedule carries 165 games in that span and 12 of them are priced]**.
Coverage is total through week 6, half in week 7 and scattered after. The one
predictor that works effectively stops existing after week 6. That alone
justifies the window.

**What is not true:** that preseason knowledge of team quality survives into
week 6 and can be drafted on. Preseason defensive ratings decay with a half-life
of about **6.5 weeks (95% CI [1.8, >50])** — but they are weak *everywhere*, at
r ≈ 0.14 and 55% pairwise accuracy even in weeks 1–6. And in every window
including week 1, **preseason at its best (55.1%) loses to the market at its
worst (58.6%)**.

The sanity check that the decay method works: **the market shows no decay at
all**, best fit by a constant. A forward-looking price should not decay the way
a backward-looking rating does.

> **Do not draft defenses on last season's rankings.** [evidence]

---

## 3. What actually predicts a D/ST week

Walk-forward, train on seasons before S, test on S, S = 2020–2025:

| Model | MAE | Pair accuracy |
|---|---|---|
| League mean | 4.314 | — |
| **Opponent implied total alone** | **4.134** | **61.01%** |
| Defensive form only | 4.287 | 53.61% |
| Form + opponent offence | 4.190 | 58.12% |
| Everything | 4.116 | 60.95% |

A single variable beats every richer model, and adding features makes ranking
slightly *worse*.

**The benchmark was pre-registered before any model was fitted:** each week,
start whichever rostered defense faces the lowest opponent implied total. That
rule scores **61.01%**.

**[corrected]** The report previously read "the best model scores 61.01%, the
gap is +0.00 pp". That is circular: the best-scoring model *is* M1, a ridge fit
on the one market variable, which produces the same within-week ranking as the
rule and therefore the same accuracy by construction. Stated non-circularly:
**the best model that knows anything the rule does not (M5) scores 60.95%, a gap
of −0.06 pp.** Every feature set that adds defensive form, opponent offence or
QB identity ranks worse than the thirty-second rule. The conclusion is
unchanged and slightly harsher than before.

### The opposing quarterback does not help either

The brief emphasised opposing-QB vulnerability most of all. Adding the opposing
quarterback's prior interception rate, sack rate, fumble rate, form and
experience is worth **+0.10 pp** — and the market-only model still beats the
model that knows who the quarterback is. A QB-only model reaches 57.92%.

Every QB trait correlates with the team-level residual at |r| < 0.031. The
brief's proposed combination — strong pass rush plus poor pressure-to-sack plus
elevated interception expectation — **did not win** an exhaustive pairwise
interaction search. The best product that did win is worth **0.17% of baseline
error**: detectable and useless.

### What is trait and what is noise

| Metric | r(season N, N+1) |
|---|---|
| Points allowed | 0.279 |
| Total D/ST points | 0.270 |
| Sacks | 0.195 |
| Interceptions | 0.171 |
| Fumble recoveries | 0.051 |
| **Defensive touchdowns** | **−0.075** |

Answering the brief directly: **defensive touchdowns are random**, with a
slightly negative point estimate. Even sacks — the largest scoring component at
2.40 points per game — persist at only 0.195.

---

## 4. Where D/ST points come from

Per game, on average:

| Component | Points/game |
|---|---|
| Sacks | **2.40** |
| Interceptions | 1.54 |
| Fumble recoveries | 1.04 |
| Points-allowed tier | 0.48 |
| Defensive TD | 0.41 |
| Special-teams TD | 0.21 |
| Safeties + blocked kicks | 0.20 **[added]** |
| **Total** | **6.28** |

**[added]** The safety-and-blocked-kick row was missing, so the table did not
reconcile: the six original rows sum to 6.09 against a panel mean of 6.28. It is
a larger contributor than the special-teams touchdown and is drawn empirically
by the simulator rather than modelled.

Sacks dominate the **mean**; the points-allowed tier (range −4 to +10) and the
6-point defensive touchdown drive the **tails**. Floor and ceiling have
different sources, which is why the model simulates components separately rather
than fitting one distribution to the total.

---

## 5. The distributions are calibrated

3,230 walk-forward out-of-sample defense-weeks:

| Nominal percentile | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| **Observed coverage** | 10.2 | 25.8 | 51.9 | 77.0 | 91.0 |

Within 0.2–2.0 pp everywhere against a binomial standard error of 0.53 pp.
**22 of 23 reliability bins contain the prediction** **[corrected: was 21]**,
which is what chance produces at 95%. The single genuine miss is P(spike) in the
0.05–0.10 bin (n = 1,274, predicted 0.077, observed 0.057, CI [0.046, 0.071]).
DST-006 also counts the P(top-6) 0.40–0.50 bin as a miss, but its interval
[0.199, 0.439] does contain the predicted 0.436 — narrowly. `dst006_reliability.csv`
is the authority here and DST-006's prose should be reconciled to it.
Discrimination is real as well as calibration: AUC for top-12 is 0.632 against
0.510 for climatology.

Reported rather than hidden: a PIT Kolmogorov–Smirnov test formally rejects
uniformity (p = 0.007), but that is power at n = 3,230 — the largest CDF
departure anywhere is 3 pp. And every variant runs **+0.25 points optimistic**,
of which climatology carries +0.19, so the bulk is league scoring drift between
train and test seasons rather than model error.

Thresholds are derived, not chosen: **bust ≤ 2 points** (the median weekly
DST-24 line — the waiver wire beat you), **spike ≥ 14** (89th percentile).

---

## 6. The 2026 pair table

All 496 two-defense combinations scored across weeks 1–6, with the weekly
starter chosen **pregame** on expected points and the outcome drawn from the
calibrated simulator.

### Best overall, weeks 1–6

| Pair | ADP | W1–4 | W1–6 | p10 | p90 | Top-12 wks | Bad wks |
|---|---|---|---|---|---|---|---|
| **KC + NE** | 143/125 | 30.4 | **47.8** | **31.0** | **66.0** | 4.05 | 0 |
| DET + NE | 126/125 | 29.3 | 46.7 | 30.0 | 64.0 | 3.97 | 0 |
| NE + SEA | 125/84 | 29.0 | 46.4 | 29.0 | 65.0 | **4.14** | 0 |
| NE + SF | 125/undrafted | 28.7 | 46.2 | 29.0 | 64.0 | 4.07 | 0 |
| NE + PIT | 125/140 | 28.6 | 46.0 | 29.0 | 64.0 | 4.05 | 0 |

### Best first four weeks

| Pair | ADP | W1–4 | W1–6 |
|---|---|---|---|
| **KC + LAC** | 143/147 | **32.6** | 45.9 |
| JAX + SF | 151/undrafted | 32.1 | 44.8 |
| PIT + SF | 140/undrafted | 32.1 | 44.3 |
| LAC + SF | 147/undrafted | 31.9 | 44.8 |
| DET + LAC | 126/147 | 31.9 | 44.5 |

### Best inexpensive (both late or undrafted)

**[corrected]** This section previously read that NE + SF and CHI + NE "land
within 1.6 points of the best pair on the board at a fraction of the cost."
Two things were wrong. CHI + NE is **2.1** points back, not 1.6. And the
cheap-pair filter (combined ADP ≥ 260) is headed by **KC + NE**, which is the
best pair on the whole board *and* qualifies as inexpensive at 143/125. There is
no cost/quality trade to make here:

| Pair | ADP | W1–6 | Gap to best |
|---|---|---|---|
| **KC + NE** | 143/125 | **47.8** | — |
| NE + SF | 125/undrafted | 46.2 | −1.6 |
| NE + PIT | 125/140 | 46.0 | −1.8 |
| CHI + NE | undrafted/125 | 45.7 | −2.1 |

### Floor and ceiling

**KC + NE tops both lists** — highest 10th percentile (31.0) *and* highest 90th
percentile (66.0). Across all 496 pairs, r(p10, p90) = **0.978** (Spearman
0.973), so the claim that there is no floor-versus-ceiling trade-off verifies.

**[weakened]** But it is a much weaker finding than "pairs that are good are
good at both ends." The p90 − p10 width is **28 to 36 points for every one of
the 496 pairs** (mean 33.4). The pair distributions are near-translates of each
other: they differ in location and barely at all in shape. That is a property of
the simulator, not a discovery about defenses — DST-006 carries no
defense-specific quality term, so every defense facing the same implied
opponent total is given the same distribution. After regressing out the mean,
the residual correlation between p10 and p90 is **−0.17**, a faint trade-off
inside the integer rounding on both columns. The correct statement: *this model
cannot express a floor/ceiling trade-off, and nothing in the data forced it to.*

### The honest caveat about this table

**New England appears in eight of the top ten pairs** **[corrected: was nine]**.
That is not a claim about the New England defense. It is a schedule artifact:
they face the Jets in week 6 (16.5 implied) and the Raiders in week 5 (18.0).
The table ranks *schedules*, exactly as intended — but do not read it as a
scouting report.

**[weakened] The top of this table is one defense and a plateau, and the pair
table is worth less than its precision suggests.** The ten best pairs span
47.8 down to 45.5 — **2.3 points across ten pairs**, or 0.4 points per week
spread over two starting slots. The top twenty-five span 3.1. The best NE-free
pair (KC + LAC, 45.9) is 1.9 points behind the best pair overall. Among the
eight NE pairs in the top ten the spread is 1.1 points, which is the entire
measurable contribution of the *partner*. Set that against the 33-point p10–p90
width of any single pair: choosing correctly within the top ten moves the
expected total by about 7% of one pair's own outcome spread. The operative
instruction is "roster a defense with a soft weeks 1–6 schedule"; which partner
you attach to it is close to a free choice.

---

## 7. The market signals that matter

Best single matchups in weeks 1–6, and what they have in common:

| Wk | D/ST | vs | Opp implied | Exp | p50 | p90 | P(top-12) | P(bust) |
|---|---|---|---|---|---|---|---|---|
| 6 | NE | NYJ | 16.5 | 9.12 | 8.0 | 17.0 | 64% | 11% |
| 1 | JAX | CLE | 16.5 | 8.90 | 8.0 | 17.0 | 56% | 12% |
| 2 | LAC | LV | 17.0 | 8.75 | 8.0 | 17.0 | 58% | 11% |
| 2 | SEA | ARI | 17.2 | 8.66 | 8.0 | 17.0 | 54% | 13% |
| 6 | LA | ARI | 17.0 | 8.62 | 8.0 | 16.0 | 61% | 13% |

Every one is an opponent implied total between 16.5 and 18.0. **That is the
whole signal**, and there is no second criterion worth applying.

Note the spread: the best matchup on the board has a median of **8** and a 90th
percentile of **17**. Even a perfect matchup is a coin flip between a dull week
and a good one, with an 11% chance of an outright bust. This is why the
distribution is the product and the point estimate is not.

### What P(top-12) is a probability *of* **[added]**

P(top-12) ranks a defense against **all 32** defenses that week. A 12-team
league rosters only 12–24 of them, so the natural worry is that the published
number is optimistic. It is not — it is conservative, and it is also answering a
different question than a manager usually asks.

| Field the defense is ranked against | NE wk 6 | mean over the pool |
|---|---|---|
| All 32 defenses (as published) | 64% | 42% |
| The 24 highest-projected (2 rostered per team) | 71% | 50% |
| The 16 highest-projected | 88% | 75% |
| The 12 highest-projected | 100% | 100% |

Shrinking the field **removes competitors**, so P(top-12) rises: +8 pp on
average at 24 rostered, +29 pp at 16. Against a 12-defense field the statistic
degenerates entirely — every started defense is trivially "top 12". So the
published figure understates rather than overstates the chance of a top-12
*score*.

The real caution is the opposite one. If every manager streams on the same
signal, the twelve defenses actually started each week are the twelve best
matchups, and "top-12" stops discriminating between them. The competitive
question — *did I beat the other managers' defenses* — has much soberer answers:

| Wk | D/ST | P(top-12 of 32) | P(best of the 12 started) | P(top half of the 12 started) |
|---|---|---|---|---|
| 6 | NE | 64% | **16%** | 64% |
| 1 | JAX | 56% | 14% | 62% |
| 2 | LAC | 58% | 11% | 58% |
| 2 | SEA | 54% | 12% | 57% |
| 6 | LA | 61% | 13% | 61% |

The best matchup on the entire 2026 board is the week's best defense one time in
six, and beats the median streamed defense about 64% of the time. Read
P(top-12) as "this scored like a startable defense", never as "this won me the
position".

**On double counting**, which the brief warned about: implied team total
correlates **0.820** with the spread and **0.576** with the game total
**[corrected: was 0.817 and 0.574; recomputed on the 4,222-row panel]**, and
`implied_team_total == (total + spread) / 2` exactly — the maximum absolute
deviation over 4,222 rows is 0.0. A defense facing a low implied total, a large
spread and a low opposing passing-TD prop has not received three pieces of
evidence. It has received one.

---

## 8. Does a second defense pay for its pick?

Judged on D/ST evidence alone. Defense A is held fixed; its best *drafted*
partner is compared against simply streaming the best undrafted defense each
week.

**Mean gain: +1.61 points over weeks 1–6 (se 0.24, 95% CI [1.15, 2.07]),
positive for 17 of 17 defenses.** Consistently positive, and small.

**[corrected]** Two things changed here. The published figure was +1.47 over 16
defenses; the Rams were missing because FFC's `LAR` never joined nflverse's
`LA`, so a pick-118 defense sat in the "free streamer" pool and inflated the
baseline every drafted defense was measured against. And the table below
previously showed six rows that looked like the top six but were not — it
carried the first five plus PHI, which is twelfth, and omitted the four smallest
gains entirely because they were below the console's print cutoff. Full list:

| D/ST | A + streamer | A + drafted D2 | Gain |
|---|---|---|---|
| NYG | 41.6 | 44.9 | +3.29 |
| KC | 44.6 | 47.8 | +3.27 |
| DET | 43.9 | 46.7 | +2.79 |
| CLE | 41.9 | 44.4 | +2.47 |
| SEA | 44.0 | 46.4 | +2.40 |
| BUF | 41.5 | 43.4 | +1.92 |
| PIT | 44.3 | 46.0 | +1.68 |
| NE | 46.2 | 47.8 | +1.67 |
| LA | 43.9 | 45.3 | +1.40 |
| DEN | 41.1 | 42.4 | +1.24 |
| MIN | 44.4 | 45.6 | +1.18 |
| HOU | 42.6 | 43.7 | +1.12 |
| LAC | 44.8 | 45.9 | +1.07 |
| PHI | 44.0 | 44.9 | +0.87 |
| BAL | 44.8 | 45.5 | +0.72 |
| TEN | 43.4 | 43.5 | +0.17 |
| JAX | 44.8 | 44.9 | +0.07 |

The bottom of that list matters: for JAX and TEN the second drafted defense buys
essentially nothing, because their best undrafted partner is already as good as
their best drafted one.

### The units did not match. Now they do. **[corrected]**

The previous version compared "+1.47 points over six weeks" against "3.9 to 9.6
points above replacement over a full season" and conceded the units were "not
perfectly commensurable". They were not comparable at all, and three separate
things were wrong with the cost side:

1. Defenses go at picks **84.1 to 172.3**, not 84–160. Five of the seventeen go
   after pick 156.
2. The 3.9–9.6 range is the mean usable PAR of buckets 85–156 only. Across the
   buckets that actually span the D/ST ADP range the figure is **1.8 to 9.6**,
   and 10.0 for the bucket pick 84 rounds into.
3. "A full season" is weeks 1–16 in `phase6_pick_value_curve.csv`, not 18 — the
   league ends in week 16.

**The like-for-like comparison.** The pick-value curve was recomputed on the
same code path with the season truncated at week 6 and replacement levels
re-derived on six-week totals, so both sides now measure the same window:

| Pick range | Mean usable PAR, wks 1–16 | Mean usable PAR, **wks 1–6** | Median | Hit rate, wks 1–6 |
|---|---|---|---|---|
| 85–96 | 9.65 | **3.97** | 0.0 | 33% |
| 97–108 | 7.62 | **5.19** | 0.0 | 25% |
| 109–120 | 5.82 | **3.17** | 0.0 | 25% |
| 121–132 | 6.77 | **4.79** | 0.0 | 32% |
| 133–144 | 5.40 | **3.23** | 0.0 | 23% |
| 145–156 | 3.92 | **2.70** | 0.0 | 16% |
| 157–168 | 1.77 | **1.31** | 0.0 | 13% |

Note that across picks 85–168 the six-week share is **59.5% of the sixteen-week
value**, not the 37.5% uniform proration would predict — late picks front-load
what usable value they have. Anyone who had prorated by 6/16 would have
understated the alternative pick by more than a third. It had to be computed,
not assumed (`dst008_pick_cost_six_weeks.py`).

**Verdict: the conclusion survives, but it was overstated and it is not
universal.** A second defense returns **+1.61** over weeks 1–6. The pick it
costs returns **1.31 to 5.19** over the same weeks. Subtracting:

| Pick range | Alternative pick, wks 1–6 | Second defense | Net |
|---|---|---|---|
| 85–144 | 3.17 to 5.19 | +1.61 | **−1.6 to −3.6** |
| 145–156 | 2.70 | +1.61 | **−1.1** |
| 157–168 | 1.31 | +1.61 | **+0.3** |

- Before pick ~145 the second defense loses by 1.6 to 3.6 points. The original
  direction holds, and this is where SEA, DEN, HOU, LAR, MIN, NE, DET, PHI, PIT
  and KC are drafted.
- After pick ~157 — BAL, BUF, CLE, TEN, NYG on this board — the second defense
  wins, narrowly. NYG has both the latest ADP (172.3) and the largest gain on
  the board (+3.29), though at 172.3 it sits in a bucket with n = 1 and is
  really only bounded by the 157–168 figure above it.
- **"Probably slightly negative" was too strong across the range and backwards at
  the end of it.** The point estimate is −1.6 to −3.6 before pick 145 and
  **+0.3 after pick 157.**

**And the two distributions are not comparable even at matched units.** Median
usable PAR is **0.0 in every bucket in both windows**, with six-week hit rates
of 13–33% across the D/ST ADP span — so between two-thirds and seven-eighths of
the time the alternative pick produces no startable player at all in weeks 1–6.
Its mean is carried entirely by a minority of hits. The +1.61 is close to a
certainty (17 of 17 defenses positive, CI [1.15, 2.07]). A mean of a
right-skewed lottery and a near-sure thing are not interchangeable at equal
expectation, and a manager who cares about their week-1-to-6 record should not
treat them as such.

**What would be needed for a genuinely clean answer, which this evidence cannot
supply.** The D/ST side is *censored*, not complete: +1.61 is the value of a
second defense over the only six weeks that can be priced, and the defense keeps
playing for twelve more. Extending it needs a weeks 7–18 board, which needs a
non-market predictor, and DST-002 and DST-005 both say the best one available is
worth 53–55% pairwise — a coin flip. Meanwhile `usable_par` is `max(0, ·)`
applied to a *season* total, so it cannot be sliced by week without redefining
replacement, which is what was done above, and a six-week replacement bar is not
the same bar as a sixteen-week one. Two things would close the gap and neither
exists here: (a) lookahead lines, or any predictor beating 55%, for weeks 7–18;
and (b) the alternative pick's return expressed as a distribution rather than a
mean, so a certainty-equivalent comparison is possible. Until then the honest
claim is directional with a magnitude of 1.6–3.6 points against for a defense
drafted before pick 145, and no measurable case against one drafted after pick
157. [inference]

---

## 9. Limitations

| Limitation | Effect |
|---|---|
| Six weeks only | The 2026 board cannot extend past week 6; only 12 of 165 later games are priced |
| +0.25 pt optimism | Mostly league scoring drift, not model error; recency weighting made it worse |
| Percentiles run 1–2 pp optimistic **[corrected]** | Was stated as "tails ~5 pp thin", which nothing in DST-006 supports. Observed coverage misses nominal by 0.2–2.0 pp; the largest CDF departure anywhere is 3 pp. P(bust) is nudged down and P(spike) up by 1–2 pp |
| No defense-specific quality term **[added]** | Every defense facing the same implied opponent total gets an identical distribution. This is why all 496 pairs have the same 28–36 point width and why the pair table can only rank schedules |
| Defenses drawn independently within a week | The mechanical link between two sides of one game is ignored; matters only if you start two defenses from the same game |
| 17 drafted defenses **[corrected]** | The marginal-value test rests on 17 comparisons. It was 16 only because of the `LAR`/`LA` join failure, now repaired |
| Provider team codes **[added]** | `markets.TEAM_FIXUP` exists but `load_ffc_adp` does not apply it. Any other script joining FFC ADP to nflverse codes has the same latent bug |
| Closing lines | nflverse carries closing lines; lineups get set earlier |
| Preseason ratings excluded by evidence | If they matter in ways this panel cannot see, the board misses it |
| P(top-12) is measured against 32 defenses **[added]** | Conservative as a scoring benchmark, and not a competitive one — see §7 |

---

## 10. What to actually do

1. **Do not draft a defense early.** Nothing about defensive quality predicts
   well enough to justify it, and defensive touchdowns are literally random.
2. **Draft one defense late, or none at all.** The second drafted defense buys
   +1.61 points across six weeks against an alternative pick worth 1.31–5.19
   over the same six weeks — a loss of 1.6–3.6 points if you spend a pick before
   ~145, and a wash-to-slight-gain if you spend one after ~157. **[corrected]**
3. **Stream on opponent implied team total, and only that.** It beats every
   model built here, including the ones that know defensive form, opponent
   offence quality and the opposing quarterback.
4. **Target 16.5–18.0 implied opponent totals.** That is the whole board.
5. **Expect variance.** The best matchup available has a median of 8 and an 11%
   bust probability. Streaming is a distribution, not a plan.
6. **Ignore last season's defensive rankings.** Half-life ~6.5 weeks, and they
   lose to the market even in week 1.
7. **Do not agonise over which pair.** **[added]** The top ten pairs span 2.3
   points across six weeks, against a 33-point outcome spread within any one of
   them. Get a defense with a soft weeks 1–6 schedule; the partner is close to a
   free choice.

---

## Reproducing

```bash
python dst-streaming/scripts/dst001_build_and_validate.py   # scoring validation
python dst-streaming/scripts/dst002_predictors.py           # predictors vs trivial rule
python dst-streaming/scripts/dst004_opposing_qb.py          # opposing-QB test
python dst-streaming/scripts/dst005_time_decay.py           # decay half-life
python dst-streaming/scripts/dst006_distributions.py        # calibration
python dst-streaming/scripts/dst007_pairing_engine.py       # the 2026 pair table
python dst-streaming/scripts/dst008_pick_cost_six_weeks.py  # pick cost, weeks 1-6
```

---

## Appendix — verification log, 2026-08-19

Every figure in this report was traced to a CSV on disk or re-derived by
re-running its script. **33 claim groups, 181 individual numbers, 14
corrections.** DST-002, DST-004 and DST-007 were re-run from scratch; all three
reproduce bit-identically, so the seeds are doing their job. The pattern in the
misses is worth naming: nine of the fourteen are transcription artifacts of
reading a truncated console block — a count taken by eye (nine of ten, 21 of 23),
a table row picked from the bottom of a print cutoff (PHI), a range read off the
buckets that happened to be visible (3.9 to 9.6).

| # | Claim | Source | Result |
|---|---|---|---|
| 1 | Panel 4,222 defense-weeks | `dst_data.build_dst_panel` | PASS |
| 2 | Model table, 5 MAEs and 4 accuracies | `dst002_model_comparison.csv` | PASS (9/9) |
| 3 | "Best model vs. trivial rule +0.00 pp" | same | **MISMATCH — circular; best feature-adding model is −0.06 pp** |
| 4 | Season-to-season r, all six metrics | `dst002` re-run | PASS (6/6, on the 4,188-row feature panel) |
| 5 | Opposing QB +0.10 pp, QB-only 57.92%, market-only still ahead | `dst004_model_comparison.csv` | PASS (3/3) |
| 6 | Every QB trait \|r\| < 0.031 | `dst004_qb_trait_correlations.csv` | PASS (max 0.0296) |
| 7 | Best interaction worth 0.17% of baseline error | `dst004_interactions.csv` | PASS (0.0070 / 4.130) |
| 8 | Lines: 93 of 93 in weeks 1–6 | `nfl_data.team_game_context` | PASS |
| 9 | Lines: "8 of 158 in weeks 8–18" | same | **MISMATCH — 12 of 165** |
| 10 | Half-life 6.5 wk CI [1.8, >50]; r ≈ 0.14; 55.1% vs 58.6%; market constant | DST-005 | PASS (4/4) |
| 11 | Component points/game, six rows | panel | PASS (6/6) |
| 12 | Component table completeness | panel | **INCOMPLETE — safeties/blocks 0.20 missing; rows summed to 6.09 vs 6.28** |
| 13 | 3,230 OOS weeks; coverage 10.2/25.8/51.9/77.0/91.0; SE 0.53 pp | `dst006_coverage.csv` | PASS (7/7) |
| 14 | "21 of 23 reliability bins" | `dst006_reliability.csv` | **MISMATCH — 22 of 23** |
| 15 | AUC 0.632 vs 0.510; BSS 0.052; KS p 0.007; bias +0.25 / +0.19 | `dst006_variant_comparison.csv` | PASS (6/6) |
| 16 | Bust ≤ 2 = median DST-24; spike ≥ 14 = 89th pct | DST-006 | PASS |
| 17 | 496 pairs scored | `dst007_pairs_2026.csv` | PASS |
| 18 | Best-overall table, 5 pairs × 7 fields | same | PASS (35/35) |
| 19 | Best-first-four table, 5 pairs × 4 fields | same | PASS (20/20; rows 4 and 5 are a tie printed in the other order) |
| 20 | "NE + SF and CHI + NE within 1.6 of the best pair" | same | **MISMATCH — CHI + NE is 2.1 back, and the best pair is itself in the cheap bucket** |
| 21 | KC + NE tops both floor and ceiling (31.0 / 66.0) | same | PASS |
| 22 | "No floor-versus-ceiling trade-off" | same | PASS (r = 0.978) but **weakened — it is a model property, not a finding** |
| 23 | "NE in nine of the top ten pairs" | same | **MISMATCH — eight of ten** |
| 24 | Weekly table, 5 rows × 8 fields | `dst007_weekly_2026.csv` | PASS (40/40) |
| 25 | r(implied team total, spread) = 0.817 | panel | **MISMATCH — 0.820** |
| 26 | r(implied team total, game total) = 0.574 | panel | **MISMATCH — 0.576** |
| 27 | `implied_team_total == (total + spread)/2` exactly | panel | PASS (max deviation 0.0) |
| 28 | Marginal: +1.47, se 0.24, 16 of 16 | `dst007_marginal.csv` | PASS as published, but **superseded — the ADP join dropped the Rams; repaired: +1.61, se 0.24, 17 of 17** |
| 29 | Marginal table, 6 rows × 3 fields | same | PASS (18/18) but **MISLEADING — presented as the top six; the sixth row is twelfth and the four smallest gains were never shown** |
| 30 | "Defenses go around picks 84–160" | `phase6_pick_value_curve.csv` | **MISMATCH — 84.1 to 172.3** |
| 31 | "3.9 to 9.6 points above replacement over a full season" | same | **MISMATCH — 1.8 to 9.6 across the actual ADP span, and "full season" is weeks 1–16** |
| 32 | "Tails ~5 pp thin" | DST-006 | **MISMATCH — unsupported; the sourced figure is 1–2 pp** |
| 33 | Exec-summary "pick worth roughly 4–7 points" | `phase6_pick_value_curve.csv` | **MISMATCH — no bucket range in the file produces 4–7** |

**Pre-registration, checked.** `RESEARCH_PLAN.md` states the benchmark verbatim
("each week, start whichever rostered defense faces the lowest opponent implied
team total") and was committed in `22e4243`, one commit before the wave-0
modelling in `f4ce203`. The claim holds on commit ordering, which is the
strongest evidence available for it.

**New work.** `dst008_pick_cost_six_weeks.py` fills the DST-008 slot the plan
already reserved for draft cost against the measured pick-value curve. It ships
with an assert-based `--demo` self-check.

**Reproducibility note.** `dst007_pairing_engine.py` is seeded (`fit` at 7,
`distributions` at 11); a clean re-run produced byte-identical CSVs before the
ADP repair, which is how the repair's effect could be isolated to the join.
