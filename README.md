# QB Pairing Lab

Research project testing whether two quarterbacks with complementary schedules
can beat one expensive quarterback in a 12-team, 1-QB fantasy football league.

**Headline result: the thesis is not supported, but a weaker version of it is.**

Full report: [`research/QB_PAIRING_RESEARCH.md`](research/QB_PAIRING_RESEARCH.md)

---

## What was found

Three measurements dismantle the proposed mechanism:

- **Schedule difficulty adds +0.06 pp** of weekly start/sit accuracy over
  knowing a quarterback's recent form, and **−0.01 pp** on top of the betting
  market.
- **At draft time it is weaker still** — team defensive ratings persist
  year-to-year at r = 0.206, and a prior-season rating explains 0.39% of QB
  variance.
- **There is no complementarity to exploit.** Two quarterbacks' weekly outcomes
  correlate at **+0.006**. They are already independent.

QB-specific matchup effects were tested at three levels of granularity. None
beat using no matchup adjustment at all, and per-QB sensitivity fails a
split-half reliability test (r = +0.065).

**What survives:** carrying a second quarterback is worth **+9.7 pts/season**
alongside an elite QB1 and **+30.1** alongside a mid-priced one, net of the pick
it costs — but as *availability insurance against QB1 failing*, not as matchup
exploitation. The weekly start/sit decision is worth **+4.9 pts/season**, about
12% of the theoretically available upside.

The 2026 pair ranking **is not robust**: it reorders completely under
reasonable changes to its own assumptions. See §17 of the report.

---

## Layout

```
research/   QB_PAIRING_RESEARCH.md  the report and its go/modify/stop verdict
            DATA_SOURCES.md         verified sources, coverage, costs, gaps
            METHODOLOGY.md          how the modelling and validation work
            RESEARCH_LOG.md         chronological record of findings
src/qbpair/ config, data adapters, features, model, pairing engine, benchmarks
scripts/    one runnable study per research phase
outputs/    generated CSVs
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
python scripts/walk_one_qb_week.py 2024 11 "Josh Allen"
```

Data comes from free public sources (nflverse, Fantasy Football Calculator,
CBS, Kalshi). Sportsbook props need `ODDS_API_KEY` in the environment; nothing
else requires credentials, and no secrets are committed.

## Design notes

- **No look-ahead** is enforced by two independent implementations that are
  cross-checked against each other, plus an explicit leakage audit. It is the
  one failure mode that would invalidate every result here, and it is silent
  when it happens.
- **The model is lean by evidence.** Opponent adjustment and QB×matchup
  interactions are excluded because including them measured slightly worse.
- **All validation is walk-forward.** Train on prior seasons, test on the next.
