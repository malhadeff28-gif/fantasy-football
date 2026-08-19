# QB Pairing - Experiment Log

Every meaningful experiment gets an ID and a verdict. Machine-readable mirror
lives in `experiments.csv`.

Verdicts: **KEEP** (survives out-of-sample and critique) / **REJECT** (does not)
/ **UNCERTAIN** (underpowered or unresolved).

---

## Wave 0 - foundation and first pass (2026-08-17)

Detailed narrative in `RESEARCH_LOG.md`; summarised here as formal records.

| ID | Question | Result | Effect | OOS | Verdict |
| --- | --- | --- | --- | --- | --- |
| QB-001 | Can a QB-week be rebuilt from pregame data only? | Yes, leakage audit passes | n/a | n/a | KEEP |
| QB-002 | Does opponent difficulty predict QB fantasy points? | Almost nothing | +0.06 pp pair accuracy | yes | REJECT |
| QB-003 | Does schedule add anything on top of the market? | No | -0.01 pp | yes | REJECT |
| QB-004 | Does a QB-quality-controlled defense rating rescue it? | No | MAE 6.023 vs 6.023 | yes | REJECT |
| QB-005 | Does prior-season defense predict current-season QB output? | Barely | r=0.063, 0.39% variance | yes | REJECT |
| QB-006 | Do two QBs weekly outcomes correlate? | No | r=+0.006 over 2,163 pairs | n/a | REJECT |
| QB-007 | Do QB-specific matchup effects replicate? | No | split-half r=+0.065 | yes | REJECT |
| QB-008 | Which scheduling factors are real? | Venue and home/away; wind revised down | wind -1.48, t=-2.53 | yes | **UNCERTAIN (QB-022)** |
| QB-009 | Does game competitiveness shift opportunity pregame? | No | attempts flat across spreads | yes | REJECT |
| QB-010 | Do weeks 14-18 differ from other weeks? | No | -0.02, t=-0.07 | yes | REJECT |
| QB-011 | What is a second QB worth? | +12.0 pts/season | 25.4% of ceiling | yes | KEEP |
| QB-012 | Does QB2 clear its draft cost? | ~~Yes~~ **No** | ~~+9.7 / +30.1~~ **-11.6 pooled** | yes | **BROKEN by QB-020** |
| QB-013 | What is the weekly start/sit decision worth? | +4.9 pts/season | 12% of available upside | yes | KEEP |
| QB-014 | Is the 2026 pair ranking robust? | No | reorders 1st to 123rd | n/a | REJECT |
| QB-015 | Does waiting until round 7 for QB1 work? | Indistinguishable | +33.8 vs +46.3, SE ~19 | yes | UNCERTAIN |

### Open questions carried into Wave 1

- Do betting markets add information **after a public fantasy projection is
  already known**? Wave 0 compared markets against recent form, not against a
  consensus projection. This is the sharper question.
- Does nonlinear structure exist that a regularised linear model missed?
- Do pressure, blitz, aDOT, scramble rate and red-zone rushing variables add
  anything beyond box-score form?
- Do the Wave 0 negative results survive an independent reconstruction?

---

## Wave 1 - deep research (2026-08-18)

| ID | Question | Result | Effect | OOS | Verdict |
| --- | --- | --- | --- | --- | --- |
| QB-016 | Do markets add information after a public projection is known? | Yes | +1.85 pp pair accuracy (SE 0.35, t=5.28) | yes, 6/6 seasons | **KEEP** |
| QB-016b | Does the projection add anything given form + market? | No | -0.16 pp (SE 0.19) | yes | **REJECT** |
| QB-019 | Do QBs differ in matchup sensitivity? (hierarchical) | No, unidentifiable | LRT p=0.130, singular fit | n/a | **REJECT** |
| QB-019b | Do QBs differ in market sensitivity? (never tested before) | No | LRT p=0.495, singular fit | n/a | **REJECT** |
| QB-017 | Does nonlinear structure exist beyond ridge? | No, every tree model lost | -0.27 to -0.56 pp | yes | **REJECT** |
| QB-017b | Do the top SHAP interactions survive a proper null? | No, false positive | p=0.360 on exhaustive scan | yes | **REJECT** |
| QB-018 | Do pressure/blitz/NGS/scramble/RZ variables add value? | No | -0.13 pp for all 22 | yes | **REJECT** |
| QB-018b | Is pressure faced an offensive-line trait? | No, it is a QB trait | QB r=0.526 vs team r=0.320 | n/a | **KEEP** |

### QB-016 detail (Market Agent)

The question was posed backwards, and the reversal is the finding. Markets add
**+1.85 pp** of start/sit accuracy over a consensus-projection proxy, rising to
**+4.63 pp** on close calls. But given recent form *and* the market, adding the
published projection is worth **-0.16 pp**. The preseason projection is the
redundant input, not the market.

Mechanism, model-free: implied team total correlates 0.150 with the form
residual and 0.141 with the consensus residual. An August projection cannot
contain a week-11 injury, depth chart or weather, so no better preseason
projection absorbs it. Consistent with this, the gain concentrates in weeks
10-18 (+3.26 pp vs +0.74 pp early) - a static projection decaying while the
market updates. Flagged exploratory, no multiplicity correction.

The market contribution is **essentially one number**: implied team total
correlates 0.817 with team spread and 0.574 with game total. Not three signals.

Proxy validation: the consensus proxy correlates **r = 0.867** with the real
CBS 2026 board, the one season where both exist. CBS historical archive
confirmed dead at run time (2022/2025/2026 URLs return an identical 77-row
table), so the proxy was genuinely necessary rather than a shortcut.

Independent replication: the agent rebuilt the Wave 0 pair set from scratch and
reproduced form -> form+market at **+1.46 pp**, matching QB-004 exactly.

Carry-forward caveat: the effect size depends on baseline strength. Always
quote it with its baseline attached.

### QB-019 detail (Research Director, R/lme4)

Random-slopes mixed models on 3,842 QB-weeks and 69 quarterbacks. Neither a
per-QB slope on opponent difficulty (p=0.130) nor on implied team total
(p=0.495) earns its place; AIC flat, BIC worse for both, and **both fits are
singular**, meaning the data cannot identify separate slope variance at all.

Random-slope SD on opponent difficulty is 0.194 against a residual SD of 7.41.
This is the strongest available statement that QB-specific matchup effects do
not exist, and it supersedes the informal split-half test.

New result: quarterbacks do not differ in market sensitivity either, so the
implied-team-total coefficient can be applied league-wide with no per-player
tuning anywhere in the model.


### QB-018 detail (Data Agent) - REJECT

22 advanced features built from PFR advanced stats, Next Gen Stats and
play-by-play, all shift(1)-then-roll, walk-forward 2020-2025:

| Feature block | Delta pair acc | 95% CI | Delta MAE |
| --- | --- | --- | --- |
| + pressure/blitz/hurry/sack | +0.04 pp | [-0.16, +0.23] | -0.0056 |
| + NGS tracking | -0.11 pp | [-0.47, +0.22] | -0.0002 |
| + scramble / red-zone rush | -0.10 pp | [-0.31, +0.10] | +0.0084 |
| + opponent defence pressure profile | -0.13 pp | [-0.28, +0.01] | -0.0017 |
| **+ all 22** | **-0.13 pp** | [-0.52, +0.23] | +0.0045 |
| advanced features only (no box score) | -3.45 pp | [-4.46, -2.49] | +0.3272 |

Season-to-season standard deviation is **1.90 pp**, fourteen times the largest
effect. Best single addition is aggressiveness at +0.07 pp [-0.05, +0.20],
which is the maximum of 22 draws. Best residual correlation is air-yards
differential at r = -0.042, the only CI excluding zero out of 22 scanned -
exactly one false positive at the 5% level. A LightGBM check gives -0.50 pp,
so this is not a linear-model artefact.

Leakage audit: recomputing a rolling pressure feature by explicit history
filtering agrees with the vectorised version to 0.00e+00, asserted in-script.

**Mechanism for the null: redundancy, not noise.** Six of these variables are
MORE stable season to season than fantasy points per game itself (designed
rushes 0.892, scramble rate 0.788, red-zone rushes 0.785, time to throw 0.660
vs fantasy points 0.555) and none of them helps. Downfield aggression
duplicates form_adot; designed-rush and red-zone usage duplicate form_rush_att
and rush_fp_share. How a quarterback plays is already contained in the 8-game
mean of what he scored. **Stability is necessary but not sufficient.**

**QB-018b, a genuine football finding that contradicts the brief.** The brief
assumed pressure faced would be a stable offensive-line property while sacks
taken would be the noisy outcome. The data reverses it: at team level pressure
faced (0.320) is LESS stable than sacks taken (0.375), and at QB level pressure
faced (0.526) travels with the quarterback far more than with his team (0.320).
Time to throw tells the same story directly - 0.660 at QB level against 0.462
at team level. Pressure faced is mostly a function of how long the quarterback
holds the ball, not of who is blocking.

**Data coverage discovered (now documented in the shared loader):** PFR
advanced stats exist for 2018-2025 only and are keyed by pfr_player_id with a
100%-complete crosswalk to gsis; NGS `week == 0` rows are season aggregates
that silently corrupt rolling features, and NGS enforces a 15-attempt minimum
so 10-14 attempt starts are structurally unreachable.


### QB-017 detail (Nonlinear Modeling Agent) - REJECT, and a caught false positive

Ridge baseline reproduced at R2 0.1726, MAE 5.9233, pairwise accuracy 63.78%.
Every tree model lost to it, walk-forward:

| Model | delta MAE vs ridge | P(better) | delta accuracy |
| --- | --- | --- | --- |
| GBM additive (depth-1 stumps) | +0.0286 | 0.07 | -0.49 pp |
| GBM full (lightgbm) | +0.0292 | 0.07 | -0.56 pp |
| GBM full (xgboost) | +0.0216 | 0.14 | -0.27 pp |
| GBM on ridge residuals | +0.0186 | 0.17 | -0.39 pp |
| 50/50 ridge + GBM blend | -0.0062 | 0.71 | +0.08 pp |

**The depth-1 versus depth-free split is the informative decomposition.**
Univariate curvature costs 0.029 MAE and interactions add +0.0005 on top of
that - i.e. nothing. The failure is not that interactions hurt; it is that
trees cannot even match linear main effects at this sample size. The tuner
agrees: offered num_leaves in {3, 7, 15} with honest inner folds, it chose 3
in five of six folds. Ridge is also the best calibrated (slope 0.970 vs 0.922).

### The false positive, and how it was caught

SHAP surfaced `career_fp_mean x implied_team_total` (0.1206) and
`form_fp_mean x implied_team_total`. Both cleared a raw within-week shuffle
null at family-wise **p = 0.000 and p = 0.015**. Under a normal workflow that
ships as a finding.

It is a false positive, caught three independent ways:

1. **Scale-free share statistic.** Observed interaction share 14.6% against a
   null mean of 29.2%, p = 1.000. The real model is *half* as interaction-driven
   as a noise-fitting one. The raw SHAP magnitude looked large only because a
   signal-rich model has more total SHAP mass to distribute - comparing raw
   magnitudes against a null fitted on noise is the trap.
2. **Exhaustive out-of-sample scan.** All 171 pairwise products added to ridge,
   walk-forward, with a max-statistic null over the entire search. The SHAP
   winners buy +0.0013 and +0.0001 MAE, p = 1.000. The best of all 171
   (`form_cpoe x rest`) gains +0.0081, which is 0.14% of baseline error, at
   **p = 0.360**. 37 of 171 pairs improved MAE against 34.1 expected by chance.
3. **Honest temporal split.** Discover on seasons <= 2021, grade on 2022-2025:
   all five top candidates *hurt*.

**Methodological lesson, the most important in the project so far:** had this
stopped after the first permutation null, it would have shipped a false
positive with p = 0.000 attached. A single null is not enough when the
statistic itself is scale-dependent.

### Bearing on Wave 0

Wave 0 hand-picked four interactions and found them slightly harmful. All four
sit in the negative range here, so that conclusion was correct - but Wave 0 was
not *entitled* to it on the evidence it had. The exhaustive scan plus
max-statistic null is what actually settles the question.

### The ceiling is real, and where progress must come from

Residual sd 7.4 against outcome sd 8.1: roughly **83% of weekly QB variance is
unforecastable from this panel**, and the strongest detectable interaction is
two orders of magnitude too small to matter. Combined with QB-018 (advanced
tracking variables add nothing because they are redundant with box-score form),
the conclusion is that progress requires **new information** - pregame starter
and injury status, weather forecasts, player props - not a richer functional
form or more features derived from the same box scores.


---

## Wave 2 - adversarial (2026-08-18)

| ID | Target | Outcome | Verdict |
| --- | --- | --- | --- |
| QB-020 | QB-012 economics (+9.7 / +30.1) | Sign reverses when QB1 is held fixed | **BROKEN** |
| QB-021 | QB-016 market increment | Survives a stronger baseline and a hindsight ceiling; sub-claims corrected | **KEEP with amendments** |
| QB-022 | QB-008 weather effects | Estimator was wrong; effect real but small and rare | **WEAKENED** |
| QB-023 | Independent economics replication | Curve replicates; two divergences found | **REPLICATES** |

### QB-020 - the headline economics do not survive

Everything reproduced exactly first, which matters: the pick-value curve to
0.0000 from a rebuild that never calls `draft_value.py`, the phase-6 table to
the decimal, and the +9.7 / +30.1 / +12.0 marginals. The attacks therefore land
on real numbers rather than on a misreading.

**Fatal, structural.** The +30.1 was never a QB2 effect. `late pair` uses ADP QB
ranks 10 + 18 while `single value QB` uses rank 8 - a *different QB1*. Most of
the gap was the QB1 change. Holding QB1 fixed reverses the sign:

| QB1 rank | Alone | With rank-18 QB2 | QB2 adds |
| --- | --- | --- | --- |
| 1 | 43.3 | 36.3 | **-7.0** |
| 2 | 49.4 | 39.1 | **-10.3** |
| 4 | 13.2 | 2.7 | **-10.5** |
| 6 | 56.9 | 45.2 | **-11.7** |
| 8 | 12.2 | -2.5 | **-14.7** |
| 10 | 46.1 | 37.4 | **-8.7** |
| 12 | 9.6 | -8.5 | **-18.2** |

Pooled **-11.6 pts/season, negative for 7 of 7 ranks**. This was reproduced
independently in the corrected `phase6_draft_cost.py`, not merely accepted from
the agent.

**Uncertainty was never reported and swamps the effect.** +9.7 has SE 4.8, CI
[-3.8, +23.1], p = 0.116. +30.1 has SE 34.0, CI [-64.3, +124.5] - an interval
189 points wide. Neither published figure was ever distinguishable from zero.
QB-011's pair-level SE of 0.93 is also wrong because pairs share seasons and
quarterbacks; the season-cluster SE is 4.0, over four times larger.

**The insurance mechanism is refuted by its own data.** Decomposed weekly:
coverage is worth +1.0 and the weekly choice +1.7 pts/season, against a 5.8
point pick cost. Relaxing the >=12 games filter to admit *more* injured QB1s
*lowers* the estimate (+12.0 to +7.5). Injuries explain only 8.8% of the +12.0.
The waiver wire supplies the insurance for free.

**Specification fragility.** Partner rank was unmotivated and outcome-
determining: rank 14 gives -5.9, rank 16 gives +7.0, rank 18 gives -4.3, rank
20 gives -1.6. Two seasons supply 79% of the total.

**Two real bugs in my code, both confirmed and fixed.**
1. `pick_cost_lookup` compared a float ADP against integer bucket edges, so 8%
   of picks matched no bucket and were charged **0.0**. The identical fix was
   already in `phase8_rank_2026.py` and had never been backported - which is
   precisely how it survived.
2. An unresolved ADP name deleted the strategy-season instead of charging a
   bust. 2021's rank-18 pick was Deshaun Watson, who never played.

**Attacks the finding survived:** survivorship (2.0% unjoined, worth 0.22 pts),
mean-versus-median pick cost (median gives +17.2, the only significant cell in
the report), and 5% trimming (QB-011 falls +12.0 to +9.2 but still clears zero).

**Replacement level, measured rather than assumed.** The agent measured what
phase 6's own streaming strategy actually achieves - the **56.3rd percentile**
of weekly starters, median 59.5th - rather than accepting my median assumption
or the 75th-percentile stress test. At the honest 60th percentile, +9.7 becomes
+5.5 and +12.0 becomes +8.4.

### Revised position

| Claim | Published | Revised |
| --- | --- | --- |
| Marginal QB2, elite QB1 | +9.7 | **-4.2, CI [-11.2, +2.5]** |
| Marginal QB2, mid QB1 | +30.1 | **-12.0 (se 7.6) / -6.0 (se 8.7)** |
| QB-011 gross | +12.0 | **+12.0, cluster CI [+4.3, +20.2]**; ~+4 to +6 net |
| Weekly start/sit | +4.9 | **+1.7** on an ADP-selectable universe |

**The draft recommendation cannot rest on +9.7 / +30.1.** The honest number is
approximately zero and probably slightly negative, with an uncertainty band
wider than the effect. QB-011 survives directionally and is the only member of
this family that clears zero. The insurance argument is retired.


### QB-021 - the market increment survives, three of my claims did not

**Attack 1, stronger baseline: survives completely.** A six-rung proxy ladder
added prior-season efficiency (YPA, TD and INT rate, CPOE, EPA per play, sack
rate, rush volume), roster continuity, team offensive environment, an
age/experience/draft-capital curve and three-season weighted averages - 47
features against QB-016's 8. The increment did not shrink:

| Proxy | OOS Spearman | Baseline acc | Market increment |
| --- | --- | --- | --- |
| prev PPG only | 0.462 | 59.49% | +2.95 pp |
| ADP only | 0.532 | 60.93% | +2.02 pp |
| QB-016 spec | 0.556 | 61.13% | **+1.85 pp** (t 5.30) |
| + continuity/age | **0.577** | **61.30%** | **+1.85 pp** (t 11.46) |
| + multiseason | 0.570 | 61.04% | +2.04 pp (t 8.85) |

**39 extra preseason features buy 0.18 pp of baseline.** Preseason information
is already exhausted at the simple spec, which is itself a finding.

**Attack 2, the hindsight ceiling: survives, but this is where it weakens.**

| Ceiling | Baseline acc | Increment | t | Seasons positive |
| --- | --- | --- | --- | --- |
| QB-016 proxy | 61.13% | +1.85 pp | 5.30 | 6/6 |
| Leave-one-out season oracle | 63.21% | +1.35 pp | 7.22 | 6/6 |
| **Hindsight-refitted preseason model (binding)** | 64.75% | **+0.55 pp** | **2.25** | **5/6** |
| Full-season oracle (contaminated, invalid) | 68.53% | +0.13 pp | 1.41 | 4/6 |

**73% of the headline survives the LOO oracle; 30% survives the binding
ceiling.** Implied team total keeps about 86% of its standardised coefficient
even when the baseline is an oracle that already knows the season answer.

### Corrections to what I previously reported

1. **The r = 0.867 CBS validation was near-worthless.** Raw negative-log ADP
   alone correlates 0.845 with the CBS board; the proxy adds 0.013. It showed
   the proxy is *ADP-shaped* like CBS, not that it is a good projection. The
   rank statistic (0.787) is the relevant one, not Pearson. Stronger proxies
   are *less* CBS-like while being *more* accurate.
2. **The week 10-18 concentration is BROKEN.** Pre-registered and tested on
   held-out seasons 2014-2019 (panel extended back to 2012; QB-016 never saw
   them): difference-in-differences **+0.03 pp, SE 1.10, t 0.15, 3 of 6
   seasons**, CI [-2.68, +3.00]. On the generating set it had been +2.60 pp,
   t 6.27, 6/6. A textbook exploratory false positive, and the largest
   held-out bucket is weeks 5-9, not 10-18.
3. **"The projection is worth -0.16 pp" was over-stated.** Player-clustered CI
   is [-0.69, +0.39]. The correct phrasing is *indistinguishable from zero*,
   not *worth nothing*.

### Revised numbers

- Headline **+1.85 pp**. Binding uncertainty is the **player-clustered SE of
  0.48**, 38% wider than the 0.35 I reported. Widest 95% CI [0.97, 2.87].
  Excludes zero under all four clusterings.
- **Implied team total alone: +1.66 pp** (t 4.25, 6/6) - 90% of the
  three-variable effect. `implied_team_total == (total_line + spread) / 2` to
  machine zero.
- **`spread_abs` is worth -0.01 pp and dropping it improves the model to
  +1.89 pp.** Removed from `models/projection.py`; phase 7 accuracy rose
  62.89% to 62.91% and capture 12.0% to 12.3%.
- Out-of-sample replication on the unseen 2014-2019 seasons: **+2.39 pp**
  (t 5.15, 6/6).
- Sensitivity not previously reported: extending the training window from
  2018+ to 2012+ moves the 2020-2025 increment from +1.85 to +1.45 pp.
- **Placebo check (agent addition):** non-market weekly context - home,
  indoors, rest, division, opponent defence - is worth +0.34 pp, t 0.98, 3/6,
  and is fully subsumed by the market. So this is not a generic
  weekly-variable effect; it is specifically market information.

### Bugs found and fixed

`prior_season_stats` used an unordered group_by, letting floating-point
summation order vary between runs and drifting prior-season PPG by about
1e-15 - enough to flip a RidgeCV alpha and change reported accuracy. QB-016
was therefore not bit-reproducible. Fixed by sorting the input and setting
`maintain_order=True`.

### Unresolved threat

The betting lines in the cache are **closing** lines; no opening line exists
in nflverse. A manager setting a lineup early in the week sees a different
number. This is unchanged from QB-016 and remains the largest open question
about the practical value of the increment.

### QB-022 - the weather effect was mis-estimated

Every attack aimed at missingness failed. Of 2,654 rows carrying a wind
reading, 2,652 are already outdoor games, so domes never contaminated the
estimate. Restricted to outdoor games, rows with and without a reading are
identical on the outcome (form-residual difference -0.020, t = -0.05) and on
every market variable. The gap is a vendor or scrape artefact concentrated in
early weeks and two bad seasons (2022 at 51% coverage). Inverse-probability
reweighting moves the estimate only from -1.608 to -1.517, and it survives
stadium, team and season fixed effects plus a permutation null corrected for
the threshold search (p = 0.0040).

**What broke it was my own estimator, on two counts.**

1. The published figure was a **raw bucket mean, not a contrast**, and its
   standard errors treated two quarterbacks in the same game as independent.
   n = 304 QB-weeks is really **153 games**. Corrected: **-1.484, se 0.586,
   t = -2.53, CI [-2.63, -0.33]** against the published -1.61, t = -3.45.
2. **The wind 8-14 mph claim is BROKEN**: -0.448, se 0.348, **t = -1.29**,
   CI [-1.13, +0.23]. The published t of -2.39 came entirely from measuring
   against zero and double-counting games.

It is unstable across eras, -0.835 (se 0.766) in 2018-21 against -2.374
(se 0.908) in 2022-25, and significant in only 3 of 8 seasons.

**The claim that weather was the largest scheduling effect is false.** Home
field is +1.32 (t = 4.38) on the same sample and applies to every game rather
than 7% of them.

**Actionable? Technically yes, practically no.** nflverse records observed
game-time weather rather than the forecast, but the forecast was public: the
game total is 1.20 points lower in high-wind games (t = -2.96), and about 80%
of the effect survives a realistic 2-4 mph forecast error. There is no double
counting either, since controlling for implied team total leaves the
coefficient essentially unchanged (-1.484 to -1.490).

Frequency is what kills it. Wind at or above 15 mph hits **7.1%** of QB-weeks,
and only **18.5%** of within-week pairs are close enough on form for a
1.48-point tiebreak to change the decision. **Upper bound: 0.66 points per
season**, against +4.9 for the entire start/sit decision.

### QB-023 - the pick-value curve replicates, with two divergences

Rebuilt from the shared loaders with replacement levels derived independently.
The ADP input is provably identical to the cache (1,574 rows, maximum absolute
difference **0.0**), and **14 of 15 committed bucket means fall inside a 95%
season-clustered bootstrap interval** (15 of 15 under an iid interval).

**Divergence 1, confirmed and now fixed: the curve was counting NFL weeks
17-18**, which this league does not play. Stopping at week 16 moved the curve
exactly as predicted: picks 1-12 from 69.99 to **67.15** (-4%), picks 13-24
from 47.24 to **43.38** (-8%). The 2024 QB replacement level fell from 282.0
to 237.0.

**Divergence 2: player-universe retention.** The committed path keeps 84.0% of
drafted skill players against 97.9% in the replication, the shortfall growing
from 2% at picks 1-12 to 20% at 145-156, the signature of name-join failures.
A sensitivity test moved no bucket by more than 1.3 points, so this does not
explain the late-bucket gaps; those are noise.

**The substantive correction is to presentation.** From pick 61 onward the
**median** pick returns exactly zero, and buckets 6 through 13 are not
statistically distinguishable from one another in either version. Quoting them
as point estimates was the error. The load-bearing figure is **7.92 points
above replacement for picks 106-135, 95% CI [5.0, 11.5]**, not the 5-to-8
range previously stated.

### Where the two skeptics disagree, and how it resolves

QB-023 concluded that the mid-priced-QB1 case (+30.1) survives comfortably,
because its interval does not overlap the pick cost. **QB-020 supersedes it.**
QB-023 attacked the pick-cost input while holding my strategy definitions
fixed; QB-020 showed the comparison itself was invalid, because late pair and
single value QB do not share a QB1. No pick-cost correction can rescue a
contrast that was never measuring QB2. Rank-matched after the week-17-18 fix,
the marginal is **-10.6 points per season, negative at 7 of 7 QB1 ranks**.

The narrower QB-023 point still lands: picks 106-135 return 7.92 points above
replacement with an interval that **overlaps** the retired +9.7, so even on
its own terms that figure was never distinguishable from break-even.
