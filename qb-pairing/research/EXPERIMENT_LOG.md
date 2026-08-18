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

In progress. Records appended as experiments complete.
