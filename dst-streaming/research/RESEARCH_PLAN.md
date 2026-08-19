# D/ST Streaming - research plan (staged, launches after Wave 2)

**Goal:** identify two defenses that can be drafted together and rotated across
roughly weeks 1-6 so that we repeatedly start a defense facing a favourable
fantasy environment. Explicitly an early-season strategy.

## Why weeks 1-6 specifically - already verified

The premise is not a guess. Checked against the live 2026 schedule:

| Weeks | 2026 games with betting lines |
| --- | --- |
| 1-6 | **93 / 93 (100%)** |
| 7 | 7 / 14 |
| 8-18 | 8 / 158 |

Market information exists for the entire target window and essentially vanishes
after it. That is the quantitative justification for the early-season framing,
and it is a genuine structural advantage the QB project did not have.

## What transfers from the QB lab

Three findings carry over and shape the design before a line of modelling code
is written:

1. **The market is roughly one number, not several.** QB-016 measured implied
   team total at r=0.817 with spread and 0.574 with game total. A defense must
   not be credited three times for one market opinion. `markets.market_redundancy`
   exists to measure this; run it before combining anything.
2. **Stability is necessary but not sufficient.** QB-018 found six variables
   more stable year-to-year than fantasy points itself, none of which helped,
   because they were redundant with simpler features. Expect the same trap with
   defensive traits - test increment, never assume it from persistence.
3. **A single permutation null is not enough.** QB-017 cleared a false positive
   at p=0.000 on its first null and only caught it with a scale-free statistic,
   an exhaustive out-of-sample scan, and a temporal split. Any interaction
   search here gets the same three-way treatment.

## Experiment queue

| ID | Question |
| --- | --- |
| DST-001 | Reconstruct D/ST scoring historically and validate against known season finishes |
| DST-002 | What predicts weekly D/ST points? Baseline hierarchy: prior form, opponent offense quality, implied opponent total, home/away |
| DST-003 | Are defensive touchdowns predictable, or essentially random? Separate stable traits from noisy outcomes |
| DST-004 | Does opposing-QB identity add information beyond team offense quality? |
| DST-005 | Time decay - how fast does preseason knowledge of offense and defense quality lose predictive value? Estimate from data, do not assume a percentage |
| DST-006 | Full outcome DISTRIBUTION per defense-week: 10th/25th/50th/75th/90th, P(top-12), P(top-6), P(bust), P(spike) |
| DST-007 | Pairing engine: every two-defense combination scored week by week, starter chosen pregame |
| DST-008 | Draft cost and roster cost, using the measured pick-value curve |
| DST-009 | Consensus comparison - does this beat simply streaming the lowest opponent implied total? |

## The benchmark that matters

The honest test is not "did we find good defenses". It is whether the model
beats a fantasy manager who does the obvious thing: **each week, start whichever
rostered defense faces the lowest opponent implied team total.** That rule takes
thirty seconds and needs no model. If the research cannot beat it, say so.

## Distribution emphasis

D/ST scoring is violently asymmetric - a defensive touchdown is a rare, large,
largely random event. Mean projections are the wrong object. Every deliverable
reports the distribution, and the pairing engine is scored on
P(at least one option is top-12) as well as on expected points.

## Deliverable

`dst-streaming/research/DST_STREAMING_RESEARCH.md`, including the pair table
(W1-W6 starters, W1-4 and W1-6 totals, cost), best overall / best inexpensive /
highest floor / highest ceiling pairs, QB-driven matchup advantages, market
signals, and model uncertainty. Not simply the two best real-world defenses.
