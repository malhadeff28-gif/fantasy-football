# Fantasy Football Quant Research Lab

Three quantitative research projects for a 12-team league, over one shared NFL
data foundation. The goal is to find small, real edges hidden in a large search
space by doing work that would be unreasonable to calculate by hand.

| Project | Question | Status |
| --- | --- | --- |
| [`qb-pairing/`](qb-pairing/) | Can two quarterbacks beat one expensive one? | Phase 1-9 complete; deep wave in progress |
| [`dst-streaming/`](dst-streaming/) | Which two defenses can be rotated through weeks 1-6? | Starting |
| [`rb-risk-upside/`](rb-risk-upside/) | Swift / Pollard / Judkins / Montgomery: floor vs risk-adjusted upside | Queued |

Shared foundation in [`shared/`](shared/): nflverse ingestion and cache, market
data (ADP, CBS, sportsbook, Kalshi), league scoring and evaluation utilities.

See [CLAUDE.md](CLAUDE.md) for layout rules, methodology standards, and the
verified environment.

## Headline result so far

The QB pairing thesis was tested across eight seasons and **the proposed
mechanism does not exist**: two quarterbacks weekly outcomes correlate at
+0.006, opponent difficulty adds +0.06 percentage points of start/sit accuracy,
and per-QB matchup sensitivity fails a split-half reliability test at r=0.065.
An earlier version concluded a second quarterback still paid for itself as
insurance. An adversarial pass broke that too: the apparent gain compared
strategies that did not share the same QB1. Held fixed, a backup costs about
11 points a season, negative at all seven price points tested.

Full brief: [`qb-pairing/research/QB_PAIRING_RESEARCH.md`](qb-pairing/research/QB_PAIRING_RESEARCH.md)

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python qb-pairing/scripts/walk_one_qb_week.py 2024 11 "Josh Allen"
```

All data is free and public. Only the optional sportsbook-props adapter needs a
key (`ODDS_API_KEY`); nothing is committed and nothing else costs money.
