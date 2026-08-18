# Fantasy Football Quant Research Lab

Three separate quantitative research projects over one shared NFL data
foundation. Target: a 12-team league of experienced managers.

## Layout

```
qb-pairing/       QB pairing and draft strategy
dst-streaming/    D/ST streaming pairs, weeks 1-6
rb-risk-upside/   RB workload distributions and risk-adjusted upside
shared/
  nfl-data/       nflverse ingestion + parquet cache (all labs read this)
  market-data/    ADP, CBS consensus, sportsbook, Kalshi
  utilities/      league scoring, evaluation, draft value
```

Each project owns `research/ data/ src/ outputs/ scripts/ tests/`.

**Project-specific modelling never goes in `shared/`.** A QB pairing model
belongs in `qb-pairing/`. A defensive streaming model belongs in
`dst-streaming/`. An RB workload simulation belongs in `rb-risk-upside/`.

## Importing shared code

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import bootstrap; bootstrap.setup()
import nfl_data, markets, scoring, evaluation
```

## The standard

The research is the product; the picks are one output. Every finding must be
inspectable: what was tested, why, on what data, what failed, what survived,
accuracy, distributions, confidence, alternative specifications, comparison
with consensus.

Do not report "this is a favourable matchup". Report why, by how much, which
variables drive it, how confident we are, what happened historically in similar
situations, whether it survives out-of-sample, and whether it changes under
alternate specifications.

## Non-negotiable methodology

- **No look-ahead.** Historical predictions use only information that existed
  before the prediction date. Walk-forward or chronological validation only;
  never a random split over panel data. Never select historical starters using
  actual fantasy scores. Never use end-of-season ratings to predict week 3.
- **Complexity must earn its place.** A model that does not beat the simpler
  one out of sample is rejected, not tuned until it wins.
- **Market variables overlap.** Spread, total, implied team total and player
  props largely encode one opinion. Measure redundancy before combining them;
  `markets.market_redundancy` exists for this.
- **Multiple testing is real.** Interaction searches need a permutation null or
  a holdout, not a p-value from the search itself.
- **Agreement between agents is not evidence.** The data decides.

## Environment (verified 2026-08-18)

- Python 3.12.10 in `.venv`; polars, pandas, numpy, scipy, scikit-learn,
  statsmodels, matplotlib, seaborn, lightgbm, xgboost, shap, pyarrow, nflreadpy
- 16 CPU cores but **~1.8 GB free RAM** — memory, not CPU, is the binding
  constraint. Cap concurrent heavy agents at 3. Pin BLAS threads to 1 before
  importing numpy; OpenBLAS fails to allocate otherwise.
- `gh` 2.95 authenticated; node 24.17; no R
- Subagents, git worktrees, `/batch` and `/loop` all verified available
- ECC, Ponytail and OmniRoute plugins installed. OmniRoute is NOT configured;
  its `setup` needs interactive input the agent shell cannot supply

## Cost rule

Free and public sources first. **Do not spend money or subscribe to paid data
without asking.** The Odds API adapter is env-var gated and inert without
`ODDS_API_KEY`; nothing else costs anything.

## Research memory

Every project keeps `research/EXPERIMENT_LOG.md` and `research/experiments.csv`.
Findings must land in repository files, not evaporate inside a subagent
context. Every meaningful experiment gets an ID and a KEEP / REJECT / UNCERTAIN
verdict.

## Scoring

Standard, 4-point passing TDs. Defined once in `shared/utilities/scoring.py`.
The QB implementation is verified against nflverse `fantasy_points` to 0.0
maximum absolute difference across 1,389 QB-weeks. D/ST uses CBS/Yahoo
points-allowed tiers, with ESPN tiers available for sensitivity testing.
