# DST-005 - Time decay of preseason and in-season knowledge

**Verdict: KEEP** (the decay is real and the weeks 1-6 window survives)
**Sub-verdict: REJECT** using prior-season defensive ratings as a draft-time
selection tool - they are too weak to build on even in week 1.

Script: `scripts/dst005_time_decay.py`
Outputs: `outputs/dst005_time_decay.png`, `dst005_half_lives.csv`,
`dst005_preseason_curve.csv`, `dst005_buckets.csv`, `dst005_contrast.csv`

---

## The question

The whole project is framed around weeks 1-6. The plan justifies that on
*information availability* - lookahead betting lines exist for 100% of the 2026
weeks 1-6 games and essentially none after. DST-005 tests the other half of the
premise, which had never been checked: does what we know in August about a
defense, and about the offences it will face, still predict anything by week 6?

Nothing was assumed. No decay percentage was picked; the curve was estimated.

## Design

Panel: 4,222 defense-weeks, 2018-2025 regular season, from
`src/dst_data.build_dst_panel`. Preseason analysis uses target seasons
2019-2025 (7 seasons), since a rating frozen before week 1 of season S needs
season S-1 to exist.

Four ratings, every one oriented so that **higher should mean more D/ST
points**, which makes every curve expected-positive and a decay a fall toward
zero:

| Rating | Definition | Frozen? |
| --- | --- | --- |
| `pre_def_dst` | own D/ST points per game in season S-1 | yes, before week 1 |
| `pre_def_sting` | minus own points allowed per game in season S-1 | yes |
| `pre_opp_weak` | minus that week's opponent's points per game in S-1 | yes |
| `mkt` | minus this week's implied opponent total | no, re-priced weekly |

`mkt` is the reference line, not a competitor: it is re-priced every week and so
has no decay to measure. It is drawn to show what the preseason ratings are
losing ground *against*.

**Uncertainty** comes from a cluster bootstrap over team-seasons (n=224
clusters, 400 replicates) for correlations, because all seventeen of one team's
weeks share one frozen rating and are not independent observations. Pair
accuracy is bootstrapped over season-weeks instead, because a week is a single
whole ranking problem.

Weeks are capped at 17: 2019-2020 are 17-week seasons, so week 18 would rest on
five seasons rather than seven.

## Result 1 - preseason knowledge decays, and it starts weak

Half-life from a weighted nonlinear fit of `r(week) = r0 * exp(-(week-1)/tau)`,
with the half-life refitted inside every bootstrap replicate:

| Frozen preseason rating | r(week 1) | half-life | 95% CI | P(half-life <= 6 wk) | best form |
| --- | --- | --- | --- | --- | --- |
| own D/ST pts per game | +0.205 | **6.5 wk** | [1.8, >50] | 0.44 | exponential |
| own stinginess (-pts allowed) | +0.185 | **6.6 wk** | [2.1, >50] | 0.46 | exponential |
| opponent offence (-prior PPG) | +0.137 | **17.1 wk** | [6.8, >50] | 0.01 | exponential |
| market this week (reference) | +0.172 | no decay | - | 0.00 | **constant** |

"best form" is the lowest-AIC choice among exponential-to-zero,
exponential-to-asymptote, linear and constant. For the market, **constant wins**
- no decay is detectable, which is exactly what should happen to a price that is
re-struck every week.

The 95% intervals are wide enough that the point estimates must not be quoted
alone. A single week's correlation rests on ~200 defense-weeks, so its standard
error is around 0.07 - about half the size of the effect being measured. The
honest summary is **a half-life of roughly six to seven weeks for own-defense
preseason ratings, with the data unable to exclude anything from two weeks to no
decay at all.**

## Result 2 - the same conclusion without assuming a curve shape

Because an exponential fit is a modelling choice, the decay was also measured
form-free, as the drop in correlation between the early and late windows:

| Rating | r(W1-6) | r(W7-17) | drop | 95% CI | P(drop > 0) |
| --- | --- | --- | --- | --- | --- |
| own D/ST pts per game | +0.139 | +0.058 | **+0.082** | [+0.013, +0.151] | 0.988 |
| own stinginess | +0.126 | +0.051 | **+0.075** | [+0.006, +0.140] | 0.980 |
| opponent offence | +0.168 | +0.104 | +0.064 | [-0.003, +0.136] | 0.963 |
| market (reference) | +0.265 | +0.302 | -0.037 | [-0.101, +0.019] | 0.098 |

Own-defense preseason knowledge loses **more than half its correlation** between
the first six weeks and the rest of the season, and that drop clears zero at 95%
for both own-defense ratings. Opponent-offence knowledge decays more slowly and
the drop does not quite clear zero. The market goes the other way, which is a
useful sanity check on the method: a variable that should not decay does not.

## Result 3 - the decision metric, window by window

Correlation is not the decision. The decision is "start this defense or that
one", so within-week pairwise ranking accuracy is reported alongside:

| Window | rating | r | 95% CI | pair accuracy | 95% CI |
| --- | --- | --- | --- | --- | --- |
| W1-3 | own D/ST pts | +0.152 | [+0.086, +0.213] | 55.11% | [52.70, 57.48] |
| W1-3 | opponent offence | +0.180 | [+0.117, +0.250] | 56.90% | [53.88, 59.71] |
| W1-3 | **market** | **+0.253** | [+0.194, +0.325] | **58.64%** | [56.16, 60.84] |
| W4-6 | own D/ST pts | +0.125 | [+0.039, +0.206] | 54.76% | [51.80, 57.49] |
| W4-6 | opponent offence | +0.154 | [+0.081, +0.219] | 55.73% | [53.76, 58.02] |
| W4-6 | **market** | **+0.280** | [+0.203, +0.348] | **60.01%** | [57.42, 62.00] |
| W7-10 | own D/ST pts | +0.063 | [-0.012, +0.135] | 53.24% | [50.97, 55.46] |
| W7-10 | opponent offence | +0.085 | [+0.017, +0.145] | 52.71% | [50.34, 54.79] |
| W7-10 | **market** | **+0.276** | [+0.214, +0.342] | **60.53%** | [57.36, 63.80] |
| W11-17 | own D/ST pts | +0.055 | [+0.005, +0.107] | 51.97% | [49.98, 53.81] |
| W11-17 | opponent offence | +0.114 | [+0.070, +0.156] | 54.74% | [52.92, 56.39] |
| W11-17 | **market** | **+0.316** | [+0.276, +0.363] | **62.43%** | [59.93, 64.93] |

Two things are visible at once.

1. Preseason ratings really are worth more early. Own-defense pair accuracy
   falls 55.1% -> 54.8% -> 53.2% -> 52.0% across the four windows.
2. **In every single window, including week 1, the market beats every preseason
   rating.** A frozen preseason defensive rating at its very best (weeks 1-3,
   55.1%) is worse than the market at its very worst (weeks 1-3, 58.6%).

## Result 4 - in-season knowledge barely decays, because there is barely any

An N-games-ago rating predicting the game N games later, pooled across the
sample. Gaps of 1 to 10 games, ratings built on a 4-game window (and a 1-game
window for contrast):

| Curve | r(gap = 1) | r(gap = 10) | half-life | 95% CI | best form |
| --- | --- | --- | --- | --- | --- |
| own defense, 4-game D/ST rating | +0.089 | +0.060 | 9.5 wk | [2.2, >50] | exponential |
| own defense, single-game rating | +0.046 | +0.034 | 10.6 wk | [3.1, >50] | **constant** |
| opponent offence, 4-game PPG | -0.154 | -0.137 | 27.2 wk | [8.3, >50] | **constant** |

(The opponent-offence sign is the expected direction: a stronger opponent
offence means fewer D/ST points conceded, so |r| is the signal.)

The in-season picture is flat, not decaying. A four-week-old read on an
opponent's offence is worth essentially as much as last week's read. But both
curves sit at |r| <= 0.16 the whole way, so this is a flat line at a low level,
not durable knowledge. Note also that the opponent's offence (|r| = 0.15) is
consistently worth about **twice** the defense's own form (|r| = 0.09) - the
same asymmetry DST-002 found.

## Does this justify weeks 1-6?

**Yes, but not for the reason a reader would assume, and the honest version is
less flattering than the framing suggests.**

- Preseason defensive knowledge does halve at about week six. The
  weeks 1-6 window is where whatever preseason edge exists is concentrated, and
  the form-free contrast confirms it at P(drop > 0) = 0.988. On decay grounds,
  weeks 1-6 is not optimistic - it is roughly right, and if anything weeks 1-4
  is the sharper cut.
- But the preseason signal is weak everywhere. r = 0.14 and 55% pair accuracy in
  weeks 1-6 is a coin flip with a lean. **A defense drafted in August because it
  was good last year is barely better than a defense drafted at random**, in
  week 1 as much as in week 10.
- The information that actually works - the opponent implied total - does not
  decay at all, and beats preseason ratings in every window. So the case for
  weeks 1-6 rests on the plan's original argument (lookahead lines exist for that
  window and vanish after it), not on prior-season team quality.

The constructive reading for DST-007: **pick the pair on schedule and market,
not on last year's defensive rankings.** The pairing engine should treat
prior-season defensive quality as a tie-breaker at most.

## Limitations

- nflverse carries **closing** lines, not the August lookahead lines a drafter
  actually sees. The `mkt` reference is therefore the upper bound of what the
  market can do in-week; the August price for a week-5 game is noisier than the
  flat line drawn here. This experiment cannot measure how much noisier, and
  that gap is the single most important open question for DST-007.
- Half-life intervals are wide. The point estimates are usable for framing, not
  for tuning a decay weight in a downstream model.
- The in-season lag curves pool over week-of-season, so a lag-10 pair is
  necessarily drawn from later in the season than a lag-1 pair. That confounds
  lag with calendar position; the effect is small given how flat both curves are,
  but it is not zero.
- Prior-season ratings ignore roster turnover, coaching change and schedule
  strength. A better preseason rating might decay differently. What is measured
  here is the decay of the *naive* preseason rating that a manager actually uses.

## Data note for the director

`build_dst_panel(2018..2025)` returns **4,222** defense-weeks, not the 8,096
recorded against DST-001 in `EXPERIMENT_LOG.md`. 4,222 is consistent with the
underlying source (nflverse team stats has 4,254 regular-season team-games over
that span; 32 are lost joining to games with market context). Every DST-005 and
DST-006 number is computed on the 4,222-row panel. The log entry appears to be a
misstatement of the row count rather than a different dataset, but it should be
reconciled since three experiments now quote it.
