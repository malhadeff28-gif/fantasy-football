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
| QB-008 | Which scheduling factors are real? | Wind, venue, home/away only | wind -1.61 pts, t=-3.45 | yes | KEEP |
| QB-009 | Does game competitiveness shift opportunity pregame? | No | attempts flat across spreads | yes | REJECT |
| QB-010 | Do weeks 14-18 differ from other weeks? | No | -0.02, t=-0.07 | yes | REJECT |
| QB-011 | What is a second QB worth? | +12.0 pts/season | 25.4% of ceiling | yes | KEEP |
| QB-012 | Does QB2 clear its draft cost? | Yes | +9.7 elite / +30.1 mid QB1 | yes | KEEP |
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
| QB-017 | Does nonlinear structure exist beyond ridge? | agent running | | | pending |
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
