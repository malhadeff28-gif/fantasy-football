# QB Pairing Research

**2026 fantasy football — can two quarterbacks beat one good one?**

Research period: 2026-08-17 · League: 12-team, 1QB/2RB/3WR/1TE/K/DST, no flex,
7 bench, standard 4-point passing TDs · Backtest: 2018–2025, walk-forward

Every conclusion below is labelled **[evidence]** (measured in this repository),
**[inference]** (a reasonable reading of the evidence), or **[hypothesis]**
(proposed, not established).

---

## 1. Executive summary

**The thesis as posed is not supported. A weaker version of it is.**

The project set out to test whether two quarterbacks with complementary
schedules can beat one expensive quarterback. Three findings dismantle the
mechanism:

1. **Schedule difficulty barely predicts anything.** Adding opponent difficulty
   to a model that already knows a quarterback's recent form improves weekly
   start/sit accuracy by **0.06 percentage points**. Added on top of the betting
   market it contributes **−0.01 pp** — nothing. [evidence]
2. **At draft time it is weaker still.** Team defensive ratings persist
   year-to-year at only **r = 0.206**, and a prior-season rating explains
   **0.39%** of the variance in quarterback outcomes. The gap between the
   easiest and hardest third of schedules is **1.00 fantasy point per game**.
   [evidence]
3. **There is no complementarity to find.** Across 2,163 pairs, the correlation
   between two quarterbacks' weekly outcomes is **+0.006**, and between their
   pregame environments **+0.008**. Their results are already independent, so
   there is no correlation structure for a pairing model to exploit. [evidence]

**What survives is real but different.** Carrying a second quarterback is worth
roughly **+9.7 points per season** alongside an elite QB1 and **+30.1** alongside
a mid-priced one, net of the draft pick it costs. But that value is
**availability insurance against QB1 failing** — not weekly matchup
exploitation. [evidence for the magnitudes, inference for the mechanism]

The weekly start/sit decision, which was supposed to be the engine of the
strategy, is worth **+4.9 points per season** over simply starting your better
quarterback every week — **12% of the theoretically available upside**. [evidence]

**Recommendation: MODIFY, leaning stop.** Do not build the interactive tool.
Section 19 gives the specific alternative.

---

## 2. Thesis

As stated at the outset:

> A fantasy manager can avoid spending heavily on a single premium QB by
> drafting two quarterbacks whose draft cost, complementary schedules,
> individual matchup characteristics, and weekly start decisions produce better
> value than one elite quarterback.

This decomposes into four testable claims, and they fared very differently:

| Claim | Verdict |
|-------|---------|
| Schedules can be chosen to complement each other | **Rejected** [evidence] |
| Opponents affect different QBs differently | **Rejected** [evidence] |
| A manager can pick the right QB weekly | **Weakly supported** — 62.9% vs 59.8% baseline [evidence] |
| Two cheap QBs can beat one expensive one on value | **Partly supported** — but not for the proposed reason [evidence] |

---

## 3. League format and why it matters

12 teams, one starting quarterback, no flex, seven bench slots, standard
scoring with 4-point passing touchdowns.

Three consequences shape everything:

- **One QB slot** means a second quarterback never adds points directly; he only
  adds *optionality*. His entire value is the difference between his week and
  QB1's week, on weeks you choose correctly, plus coverage when QB1 is out.
- **No flex** removes the usual escape valve, so the bench slot is a pure cost.
- **12 teams with 7 bench slots** means 192 players rostered. The waiver wire
  stays deep at quarterback, which is why streaming performs as well as it does
  (§15).
- **4-point passing TDs** compress the quarterback position. QB12 averaged
  **272 season points** across 2018–2025, and the spread from QB1 to QB12 is
  narrow relative to the spread at running back.

---

## 4. Data-source inventory

Full detail in [`DATA_SOURCES.md`](DATA_SOURCES.md). Every source was verified
against a live endpoint, not from documentation.

The material finding: **historical closing betting lines are free.** nflverse's
schedule table carries spread, total, moneyline and odds at 100% coverage for
2007–2025, which removed the project's assumed dependency on a paid odds
provider and set the backtest window.

League scoring matched nflverse `fantasy_points` exactly — **0.0 maximum
difference across 1,389 QB-weeks** — so no custom scorer was needed.

A structural limit worth restating: **2026 forward lines stop after Week 7**.
Weeks 14–16 have lines for 4 of 47 games. Market data therefore cannot inform
the preseason playoff evaluation the brief requested.

---

## 5. Betting-market methodology

Market inputs are **decomposed rather than used raw**, per the brief's warning
that a favourite is not automatically the better fantasy quarterback:

- **implied team total** = `total_line / 2 + team_spread / 2` — the scoring
  environment;
- **spread magnitude** — expected game script;
- **game total** — overall pace and scoring expectation.

The sign convention was verified empirically rather than assumed: `spread_line`
is positive when the home team is favoured (correlation with home margin
**+0.443**, 2015–2025).

**Result: the market is the single most useful weekly input.** Implied team
total correlates **0.150** with the form residual, more than double any other
feature, and adding market data lifts start/sit accuracy by **+1.46 pp** — the
largest single gain measured anywhere in this project. [evidence]

But note the ceiling this implies. The best available weekly signal moves
accuracy from 62.5% to 64.0%. Weekly quarterback scoring is mostly irreducible
noise: the best model reaches **R² ≈ 0.18**.

---

## 6. Prediction-market methodology

Kalshi carries genuinely QB-level series — `KXNFLPASSTDS`, `KXNFLPASSATT`,
`KXNFLSEASONPASSYDS` — free, which is notable since sportsbook APIs gate
passing-TD props behind $99/month.

**They cannot be used for this research.** [evidence]

- Queried 2026-08-17, `KXNFLPASSTDS` returned **zero open markets**; these
  series activate in-season.
- Neither Kalshi nor Polymarket publishes a free historical price archive, so
  **neither can be backtested at all**.
- Polymarket's August NFL inventory is novelty markets ("Tush Push banned for
  2026?").

For game-level information both venues are largely redundant with the
sportsbook lines already free in nflverse. [inference]

**Disposition:** retained as forward-only collection so a price history exists
next season; nothing in the thesis depends on them. Sportsbook props were
adopted separately by user decision (§18).

---

## 7. NFL scheduling research

Each effect measured against the form residual — actual minus the
quarterback's own 8-game average — so a good quarterback on a good team cannot
masquerade as a scheduling effect. Marked where |t| ≥ 2.

| Effect | Size (fantasy pts) | t | Verdict |
|--------|-------------------|---|---------|
| Wind ≥ 15 mph | **−1.61** | −3.45 | real |
| Wind 8–14 mph | −0.57 | −2.39 | real |
| Away | −0.69 | −4.11 | real |
| Indoors | +0.65 | +2.88 | real |
| Home | +0.44 | +2.52 | real |
| Outdoors | −0.47 | −3.28 | real |
| Division game | −0.33 | −1.65 | not significant |
| Short week (≤4 days) | +0.67 | +1.37 | not significant |
| Off bye (14+ days) | −0.59 | −1.21 | not significant |
| Weeks 14–18 | −0.02 | −0.07 | **no effect** |

Three results deserve emphasis:

- **Rest does not matter**, including coming off a bye, which contradicts
  common practice. [evidence]
- **Weather is the largest scheduling effect** and is not knowable at draft
  time — and `temp`/`wind` are only 38–75% populated, so it is hard to use even
  in-season.
- **Time of season has no effect.** Weeks 14–18 score like any other weeks,
  which bears directly on playoff weighting (§13).

### Game competitiveness

The brief hypothesised that a close game raises aggression and rushing
opportunity. **Not supported pregame.** [evidence]

Pass and rush attempts are flat across spread buckets (32.4 / 32.6 / 32.1 and
3.7 / 3.7 / 3.7 for toss-ups, moderate and lopsided games), and expected
competitiveness has no significant effect on the residual.

*Actual* game script matters enormously — losing big is worth **−3.81 points**
(t = −17.8), winning big **+2.07** — but it is not forecastable from the
pregame line. This is the clearest example in the project of a large real
effect that is useless for decisions. [evidence]

---

## 8. QB-specific matchup effects

The brief's central concept, tested at three levels of granularity.

K-means on prior-only play style produces interpretable archetypes:

| Archetype | n | rush share | aDOT | pass att | exemplars |
|-----------|---|-----------|------|----------|-----------|
| 0 pocket | 1,840 | 0.097 | 7.15 | 34.6 | Goff, Mahomes, Cousins, Herbert |
| 1 downfield | 1,379 | 0.143 | 8.86 | 33.3 | Mayfield, Allen, Wilson, Stafford |
| 2 rushing | 1,049 | 0.307 | 8.08 | 27.4 | Jackson, Hurts, Fields |

**No granularity of matchup adjustment beats using none at all:** [evidence]

| Matchup granularity | MAE | pair accuracy |
|--------------------|-----|---------------|
| **none** | **5.9183** | **64.17%** |
| league-wide | 5.9188 | 64.07% |
| archetype | 5.9193 | 64.18% |
| individual | 5.9455 | 63.67% |

The decisive diagnostic is split-half reliability. For 41 quarterbacks with ≥40
games, matchup sensitivity measured in two random halves of their own games
correlates at **r = +0.065** — explaining 0.4% of variance. A quarterback's
apparent matchup sensitivity does not replicate against himself, so per-QB
coefficients fit noise. **Rejected as unstable complexity.** [evidence]

---

## 9. Proposed QB-week model

Ridge regression, walk-forward (train on all prior seasons, test on the next),
never in-sample. Standardised coefficients, 2018–2024:

| Feature | Coefficient |
|---------|------------|
| implied team total | **+1.427** |
| rushing attempts (form) | **+1.081** |
| career points/game | +0.966 |
| game total | +0.569 |
| recent form | +0.548 |
| indoors | +0.320 |
| home | +0.250 |
| pass attempts (form) | +0.147 |
| spread magnitude | +0.022 |

Opponent adjustment and QB×matchup interactions are **excluded by evidence**,
not omitted by oversight — including them made the model marginally worse.

That rushing attempts rank second is worth noting: rushing volume is a stable
floor, and it is the most durable individual trait in the model. [inference]

Ridge rather than gradient boosting is deliberate — with 8 seasons the risk of
over-fitting is live, and a regularised linear model makes the marginal
contribution of each feature readable.

---

## 10. Pair-complementarity framework

The framework exists in `src/qbpair/pairing/engine.py` and works: it scores any
pair week by week, handles byes and unavailability, counts coverage failures,
reports divergence, and separates regular-season from Weeks 14–16.

**The concept it was built to exploit does not exist.** [evidence]

Measured directly across 2,163 pairs:

- correlation of two quarterbacks' pregame implied team totals: **+0.008**
- correlation of their actual weekly fantasy points: **+0.006**

Two quarterbacks' outcomes are already essentially independent. There is no
systematic anti-correlation to discover, which means **the diversification
benefit of holding two quarterbacks is the same whichever two you hold.**
Selecting partners for "complementary schedules" adds essentially nothing on
top of that.

Within-week dispersion makes the same point: the pregame matchup signal has a
standard deviation of 2.02 fantasy points, while actual weekly outcomes have a
standard deviation of 8.09. The signal is small relative to the noise it is
trying to sort.

---

## 11. Draft-cost framework

Costs are measured, not assumed. Using real FFC drafts 2018–2025, expected
**usable points above replacement** by pick range:

| Picks | Mean usable PAR | Hit rate |
|-------|----------------|----------|
| 1–12 | 70.0 | 86% |
| 13–24 | 47.2 | 71% |
| 25–36 | 27.1 | 63% |
| 49–60 | 21.3 | 55% |
| 73–84 | 11.0 | 37% |
| 97–108 | 7.7 | 29% |
| 121–132 | 8.3 | 24% |
| 145–156 | 4.5 | 18% |

A second quarterback typically goes around picks 106–135, where the forgone
alternative is worth roughly **5–8 points**. That is the bar QB2 must clear,
and it is a low bar — which is the main reason the pairing strategy survives at
all. [evidence]

---

## 12. Bench-slot opportunity cost

Everything is expressed in points above replacement so quarterback points and
the forgone pick share units:

```
net value = QB points delivered − QB12 replacement − expected return of picks spent on QBs
```

Like-for-like across the five seasons where every strategy is present:

| Strategy | QB pts | vs QB12 | pick cost | **NET** | sd | worst |
|----------|--------|---------|-----------|---------|-----|-------|
| premium + late partner | 386.4 | 112.3 | 59.3 | **53.0** | 43.0 | −2.3 |
| premium (QB1 only) | 369.2 | 95.1 | 51.8 | **43.3** | 52.3 | −20.1 |
| late pair | 328.6 | 54.5 | 12.2 | **42.3** | 62.4 | −44.9 |
| streaming (no QB drafted) | 310.4 | 36.3 | 0.0 | **36.3** | 20.3 | **+11.4** |
| strong value + complement | 328.6 | 54.4 | 33.9 | **20.5** | 36.7 | −34.5 |
| single value QB | 300.6 | 26.5 | 14.3 | **12.2** | 24.9 | −23.7 |

**Marginal value of adding QB2: +9.7 alongside an elite QB1, +30.1 alongside a
mid-priced one.** [evidence]

The second quarterback clears its cost in both regimes, and clears it by far
more when QB1 is cheaper and likelier to fail — which is the clearest
quantitative signature that the mechanism is insurance. [inference]

QB2 also cuts downside: worst season **−2.3** with a partner versus **−20.1**
without. As risk management it is more convincing than as an offensive strategy.

---

## 13. Regular season versus playoff weighting

The brief asked for regular-season performance to dominate, with Weeks 14–16
adding a bonus, and for the exact weighting to come from sensitivity analysis
rather than a guess.

**The analysis says the correct playoff weight is zero, and applying one would
be decoration.** [evidence]

- Weeks 14–18 show **no scoring effect** (−0.02, t = −0.07, §7).
- Prior-season defensive ratings — the only playoff-schedule input available in
  August — explain **0.39%** of quarterback variance.
- 2026 playoff weeks have betting lines for **4 of 47 games**, so there is no
  market view to substitute.

A playoff-schedule bonus would therefore be sorting pairs on noise. The engine
still reports the Weeks 14–16 split separately, because it is worth *seeing*;
it just should not be *scored*.

---

## 14. Historical-validation methodology

Full detail in [`METHODOLOGY.md`](METHODOLOGY.md). The essentials:

- **Two independent no-look-ahead implementations** cross-checked against each
  other (explicit per-week filtering, and vectorised shift-then-roll), agreeing
  to 0.0. Look-ahead is the one failure mode that would invalidate everything
  here and it is silent when it happens.
- **Walk-forward only** — train on seasons before S, test on S. A random split
  over panel data would leak the future into the past.
- **An explicit leakage audit** asserting the latest input date precedes
  kickoff.
- **Decisions locked before outcomes are seen**, always.

Panel: 4,373 QB-weeks, 2018–2025, ≥10 pass attempts.

---

## 15. Benchmark methodology

Weekly start/sit rules, 1,717 pair-seasons, all decisions locked pregame:

| Decision rule | pts/season | vs always-QB1 | accuracy |
|--------------|-----------|---------------|----------|
| always QB1 | 327.3 | — | 59.84% |
| ADP consensus | 327.3 | +0.0 | 59.84% |
| recent form | 327.8 | +0.5 | 60.41% |
| market | 328.0 | +0.6 | 60.41% |
| **model** | **332.2** | **+4.9** | **62.89%** |
| oracle | 367.9 | +40.5 | 100% |

**The model captures 12.0% of the available upside.** [evidence]

Note that "always start your better preseason quarterback" is already right
59.8% of the time. Sophistication buys three percentage points.

**Streaming deserves separate emphasis.** It never had a losing season across
the backtest (worst +11.4, sd 20.3) while *every* drafted strategy did, and it
beat drafting a single mid-priced quarterback outright (36.3 vs 12.2). For a
manager who dislikes variance, streaming is the quietly strong option. [evidence]

**Benchmark limitation:** CBS and FantasyPros publish current rankings but no
accessible historical archive, so they cannot be scored over past seasons.
Preseason ADP served as the historical consensus benchmark; CBS is compared
directly on the 2026 board.

---

## 16. Limitations

| Limitation | Effect |
|-----------|--------|
| **8-season window** (user decision) | Thin for interaction terms; mitigated with ridge, archetype-level effects, split-half testing |
| **Starter identification excluded** | Panel is "given he started"; deployment needs depth-chart/injury logic |
| **Pairs evaluated are all eligible pairs** | Measures the average pair; §17 addresses whether selection helps |
| **No historical player props** | Prop features are forward-only; prop history begins 2023-05, covering 3 of 8 backtest seasons |
| **Prediction markets unbacktestable** | No free price archive anywhere |
| **Replacement = weekly median** | A sharper streamer raises the baseline and shrinks the measured gain |
| **Injuries not modelled explicitly** | Appear implicitly as QB1 failure — precisely where the value is |
| **Insurance premium is interpolated** | An estimate, not a measurement; §17 shows the 2026 ranking is not robust to it |
| **Weather data 38–75% populated** | The largest scheduling effect is the least reliably available |

---

## 17. Findings that contradict the thesis

Collected deliberately, since this is where the project's value lies.

1. **Schedule complementarity does not exist to be exploited.** Weekly outcomes
   of two quarterbacks correlate at +0.006. [evidence]
2. **Opponent difficulty adds nothing once you know the market**, and almost
   nothing even without it (+0.06 pp). [evidence]
3. **QB-specific matchup effects do not replicate** against the same
   quarterback's own games (r = +0.065). [evidence]
4. **The draft-time schedule signal is close to zero** — 0.39% of variance,
   1.00 point per game between easiest and hardest thirds. [evidence]
5. **The weekly decision is worth ~5 points a season**, 12% of the available
   upside. [evidence]
6. **Game competitiveness does not shift opportunity pregame.** Attempts are
   flat across spread buckets. [evidence]
7. **A whole tercile of playoff schedule is worth nothing measurable.**
   Weeks 14–18 show no effect. [evidence]
8. **The 2026 ranking is not robust.** Varying the two estimated inputs over a
   reasonable range reorders it completely: [evidence]

   | Assumption | Top pair | Best Allen pair rank |
   |-----------|----------|---------------------|
   | insurance ×0.0 | Josh Allen + Daniel Jones | 1 |
   | insurance ×1.0 | Jalen Hurts + Jaxson Dart | 33 |
   | pick cost ×0.5 | Josh Allen + Jaxson Dart | 1 |
   | pick cost ×1.5 | Jaxson Dart + Daniel Jones | 123 |

   The gaps between pairs are smaller than the uncertainty in the assumptions.
   **This is the most important single caveat in the report.**

9. **57.1% of pairs beat a single QB — barely better than a coin flip**, with
   quartile outcomes from −6.5 to +21.9 points. [evidence]

**What does *not* contradict the thesis:** carrying a second quarterback is
still net positive (+9.7 to +30.1), and it meaningfully reduces downside. The
practice is fine. The *reasoning* behind it was wrong.

---

## 18. Open modelling decisions

| Decision | Status |
|----------|--------|
| Backtest window 2018–2025 | **Settled** by user; extended 2007–2025 available for robustness |
| Sportsbook props ($99/mo) | **Adopted** by user; adapter built, awaiting `ODDS_API_KEY`. Covers only 3 of 8 backtest seasons — keep the prop layer additive |
| Playoff weighting | **Resolved to zero** by evidence (§13) rather than by choice |
| Insurance premium | **Open.** Interpolated from two Phase 6 anchors; drives the 2026 ranking and is its weakest link |
| QB2 bye-week waiver strategy | **Open.** Shared-bye pairs are excluded rather than penalised numerically |
| Starter identification | **Open.** Needs depth charts and injury reports before any live deployment |
| Streaming realism | **Open.** Modelled as the weekly median starter; a real manager's waiver access may differ |

---

## 19. Recommended next build phase

**Verdict: MODIFY, leaning stop. Do not build the interactive tool (Phase 10).**

The case against it is straightforward. A tool's job would be to deliver the
weekly start/sit recommendation and the pair ranking. The weekly
recommendation is worth **+4.9 points per season** over a rule you can apply in
your head ("start your better quarterback"), and the pair ranking **reorders
completely** under reasonable changes to its own assumptions. Neither justifies
a frontend.

What the research does support, in order of value per unit of effort:

1. **A one-paragraph draft rule** (available now, no build required):
   - Draft one quarterback you like, then take a cheap second one late.
   - The second QB is worth more the *cheaper* your first one is — its job is
     insurance, so pair it with the fragile starter, not the safe one.
   - **Do not pay for schedule.** It is worth about one point per game between
     schedule extremes and cannot be forecast in August anyway.
   - Only hard constraint: **avoid a shared bye week.** That is the one piece of
     genuine 2026-specific pair information that is both knowable and material.
   - If you dislike variance, streaming is a legitimate choice — it never had a
     losing season in the backtest.

2. **A thin weekly helper** — a script that prints the projection for your two
   quarterbacks each week using implied team totals. This is where the +4.9
   lives, it reuses `models/projection.py` unchanged, and it is perhaps an
   hour's work. No UI.

3. **Forward-collect what could not be backtested** — log Kalshi QB markets and
   Odds API props weekly from Week 1. In a year there would be a prop price
   history that does not exist today, and the one genuinely untested hypothesis
   (that QB-level props beat game-level lines) could finally be evaluated.

**What not to do:** do not spend effort refining the schedule model, the
matchup interactions, or the playoff weighting. Those were tested carefully and
found empty. The most valuable output of this project is knowing not to spend
the next month on them.

---

## Reproducing

```bash
python scripts/walk_one_qb_week.py 2024 11 "Josh Allen"   # leakage audit
python scripts/phase2_signal.py            # feature signal
python scripts/phase2b_schedule_test.py    # schedule + ceiling
python scripts/phase2c_pair_value.py       # realistic pair value
python scripts/phase3_scheduling.py        # scheduling effects
python scripts/phase4_qb_matchup.py        # matchup granularity
python scripts/phase6_draft_cost.py        # strategy economics
python scripts/phase7_backtest.py          # decision-rule benchmarks
python scripts/phase8_rank_2026.py         # 2026 board + sensitivity
```
