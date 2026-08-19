"""DST-006: full outcome distributions for a defense-week.

Mean projections are the wrong object for D/ST. The mean is 6.28 points, but a
quarter of defense-weeks land at 2 or below and a tenth clear 14, and the two
tails are driven by different things: sacks carry the middle, the points-allowed
tier (-4 to +10) and the 6-point defensive touchdown carry the ends.

So the total is not modelled directly. Each scoring component is modelled on its
own and the components are recombined, because:

  * they have genuinely different shapes - counts with different dispersion, a
    step function of an opponent's score, and a near-Bernoulli 6-point event;
  * the points-allowed tier is a known deterministic map of a quantity we can
    model well (the opponent's score, which is exactly what the betting market
    prices), so modelling the score and applying the map is strictly better than
    fitting a distribution to the tier;
  * DST-002 established that one market number carries the whole signal, so each
    component gets the same single predictor and nothing else.

Components are NOT independent - a shutout comes with sacks and interceptions
attached (corr(sacks, points allowed) = -0.36, corr(INT, def TD) = +0.31).
Sampling them independently would throw away the joint upper tail, which is the
only part a streamer cares about. Dependence is restored by resampling whole
rows of the training PIT matrix: each simulated game reuses the relative
position a real game occupied inside every one of its own marginals at once.
That is an empirical copula, it needs less code than a Gaussian one, and it
keeps joint-tail behaviour a Gaussian copula cannot represent. Out of sample it
beats both the Gaussian and the independent version on CRPS and on every
calibration check (DST-006), which is why it is the default rather than an
assumption.

Public API:
    fit(train_df)                    -> DstSim
    distributions(sim, df, n_sims)   -> one row per defense-week with
                                        expected/median/p10/p25/p75/p90 and
                                        P(top-12), P(top-6), P(bust), P(spike)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))
import bootstrap                                    # noqa: E402
bootstrap.setup()
import scoring                                      # noqa: E402

# The only predictor, per DST-002: one market number, not several.
MARKET = "implied_opp_total"

# Count components and their scoring weight. `def_tds` and `special_teams_tds`
# are pooled: both are 6 points, both are rare, and splitting them would fit two
# near-empty models instead of one thin one.
COUNTS = {"def_sacks": 1.0, "def_interceptions": 2.0,
          "fumble_recovery_opp": 2.0, "td_any": 6.0}
NB_COMPONENTS = ("def_sacks",)      # only sacks are meaningfully overdispersed

# Bust and spike thresholds, both derived rather than picked:
#   2 pts  = the median weekly 24th-best D/ST score, i.e. the waiver wire would
#            have beaten you, and the 26th percentile of all defense-weeks.
#   14 pts = the 89th percentile, and clearly past the median weekly 6th-best
#            score (11), i.e. a start that wins the position for the week.
BUST = 2.0
SPIKE = 14.0

# Default dependence handling. "empirical" resamples whole rows of the training
# PIT matrix, which reproduces joint-tail behaviour a Gaussian copula cannot and
# needs less code than one; "gaussian" and "independent" exist so the choice is
# scored out of sample in dst006_distributions.py rather than asserted.
COPULA = "empirical"

# Tier map taken by calling shared scoring, never by re-listing thresholds here,
# so the league definition cannot silently drift between modules.
_TIER = np.array([scoring.DSTScoring.points_allowed(float(v))
                  for v in range(0, 121)])


@dataclass
class DstSim:
    """Fitted component models plus the dependence structure between them."""

    count_beta: dict = field(default_factory=dict)   # name -> coef vector
    count_use_market: dict = field(default_factory=dict)
    nb_alpha: dict = field(default_factory=dict)
    pa_beta: np.ndarray = None
    pa_resid: np.ndarray = None                      # sorted residual pool
    misc_pool: np.ndarray = None                     # safety / blocked-kick pts
    u_pool: np.ndarray = None                        # training PIT, n_train x k
    chol: np.ndarray = None                          # gaussian copula only
    order: tuple = ()                                # copula column order
    market_mean: float = 0.0
    copula: str = COPULA


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    """Pool the two 6-point scores and recover the leftover base points."""
    td = pl.col("def_tds").fill_null(0) + pl.col("special_teams_tds").fill_null(0)
    return df.with_columns(td.alias("td_any")).with_columns(
        (pl.col("_dst_base")
         - pl.col("def_sacks").fill_null(0)
         - 2 * pl.col("def_interceptions").fill_null(0)
         - 2 * pl.col("fumble_recovery_opp").fill_null(0)
         - 6 * pl.col("td_any")).alias("misc_pts"))


def _design(x: np.ndarray, centre: float, use_market: bool) -> np.ndarray:
    if use_market:
        return np.column_stack([np.ones_like(x), x - centre])
    return np.ones((len(x), 1))


def fit(train: pl.DataFrame, copula: str = COPULA,
        half_life: float | None = None) -> DstSim:
    """Fit every component on `train` only. No row from a later week is used.

    copula: "empirical" | "gaussian" | "independent".
    half_life: seasons, for an exponential recency weight on training rows.
        None weights every season equally, which is the default because the
        weighted variant was measured and changed nothing (see DST-006).
    """
    import statsmodels.api as sm

    t = _prep(train)
    x = t[MARKET].to_numpy().astype(float)
    centre = float(x.mean())
    sim = DstSim(market_mean=centre, copula=copula)
    rng = np.random.default_rng(7)

    s = t["season"].to_numpy().astype(float)
    w = (np.ones(len(s)) if half_life is None
         else 0.5 ** ((s.max() - s) / half_life))
    w = w / w.mean()                     # keep sum(w) = n so AIC stays comparable

    for name in COUNTS:
        y = t[name].fill_null(0).to_numpy().astype(float)
        if name in NB_COMPONENTS:
            # method of moments for the dispersion, then a GLM at fixed alpha
            m1 = np.average(y, weights=w)
            v1 = np.average((y - m1) ** 2, weights=w)
            a = max((v1 - m1) / m1 ** 2, 1e-6)
            sim.nb_alpha[name] = a
            fam = sm.families.NegativeBinomial(alpha=a)
        else:
            fam = sm.families.Poisson()
        best, best_aic = None, np.inf
        for use in (False, True):
            m = sm.GLM(y, _design(x, centre, use), family=fam,
                       freq_weights=w).fit()
            if m.aic < best_aic:
                best, best_aic, best_use = m.params, m.aic, use
        sim.count_beta[name] = np.asarray(best, float)
        sim.count_use_market[name] = best_use

    # points allowed: linear in the market, then an empirical residual pool.
    # Residual spread is near constant across the fitted range (sd 8.7 -> 9.6
    # from the lowest to the highest quintile of fitted values), so a single
    # pool is adequate and keeps the lumpy 3-and-7 shape of real NFL scores.
    pa = t["points_allowed"].to_numpy().astype(float)
    X = _design(x, centre, True)
    sw = np.sqrt(w)[:, None]
    sim.pa_beta = np.linalg.lstsq(X * sw, pa * sw[:, 0], rcond=None)[0]
    resid = pa - X @ sim.pa_beta
    p = w / w.sum()
    sim.pa_resid = np.sort(rng.choice(resid, size=len(resid), p=p))
    sim.misc_pool = rng.choice(t["misc_pts"].to_numpy().astype(float),
                               size=t.height, p=p)

    # Training PIT matrix: where each observed game sat inside its own fitted
    # marginal. Randomised PIT for the counts, rank PIT for the points-allowed
    # residual. Unweighted - any drift is in the component levels, not in how
    # the components hang together.
    sim.order = tuple(COUNTS) + ("pa",)
    mu = _means(sim, x)
    cols = []
    for name in COUNTS:
        y = t[name].fill_null(0).to_numpy().astype(float)
        d = _count_dist(sim, name, mu[name])
        lo, hi = d.cdf(y - 1), d.cdf(y)
        cols.append(lo + rng.random(len(y)) * (hi - lo))
    cols.append((stats.rankdata(resid) - 0.5) / len(resid))
    sim.u_pool = np.clip(np.column_stack(cols), 1e-6, 1 - 1e-6)

    if copula == "gaussian":
        Z = stats.norm.ppf(sim.u_pool)
        C = np.corrcoef(Z.T)
        sim.chol = np.linalg.cholesky(C + 1e-8 * np.eye(len(C)))
    return sim


def _means(sim: DstSim, x: np.ndarray) -> dict:
    out = {}
    for name in COUNTS:
        X = _design(x, sim.market_mean, sim.count_use_market[name])
        out[name] = np.exp(X @ sim.count_beta[name])
    return out


def _count_dist(sim: DstSim, name: str, mu):
    if name in sim.nb_alpha:
        n = 1.0 / sim.nb_alpha[name]
        return stats.nbinom(n, n / (n + mu))
    return stats.poisson(mu)


def _simulate(sim: DstSim, week: pl.DataFrame, n_sims: int,
              rng: np.random.Generator) -> np.ndarray:
    """(n_sims, n_teams) matrix of simulated D/ST fantasy points."""
    n = week.height
    x = week[MARKET].to_numpy().astype(float)
    mu = _means(sim, x)
    k = len(sim.order)

    if sim.copula == "empirical":
        U = sim.u_pool[rng.integers(0, len(sim.u_pool), n_sims * n)]
    elif sim.copula == "gaussian":
        Z = rng.standard_normal((n_sims * n, k)) @ sim.chol.T
        U = stats.norm.cdf(Z)
    else:
        U = rng.random((n_sims * n, k))
    U = np.clip(U, 1e-9, 1 - 1e-9).reshape(n_sims, n, k)

    total = np.zeros((n_sims, n))
    for j, name in enumerate(COUNTS):
        d = _count_dist(sim, name, np.broadcast_to(mu[name], (n_sims, n)))
        total += COUNTS[name] * d.ppf(U[:, :, j])

    m = len(sim.pa_resid)
    idx = np.clip((U[:, :, k - 1] * m).astype(int), 0, m - 1)
    X = _design(x, sim.market_mean, True)
    pa = np.rint(X @ sim.pa_beta + sim.pa_resid[idx])
    total += _TIER[np.clip(pa, 0, 120).astype(int)]

    # ponytail: misc (safety + blocked kick) is 0.20 pts/gm and nonzero in 9% of
    # games; drawn independently rather than joined to the copula. Fold it in if
    # blocked-kick-heavy special teams ever become a real streaming edge.
    total += rng.choice(sim.misc_pool, size=(n_sims, n))
    return total


_PCTS = [10, 25, 50, 75, 90]


def distributions(sim: DstSim, df: pl.DataFrame, n_sims: int = 3000,
                  seed: int = 11, keep_sims: bool = False):
    """Percentile and probability set for every defense-week in `df`.

    `df` may span many weeks; each (season, week) is simulated jointly so that
    P(top-12) and P(top-6) are ranks against that week's actual field. Those two
    columns are null for a week with fewer than 12 or 6 defenses.

    Returns a DataFrame with season, week, team plus exp_pts, sd, p10, p25, p50,
    p75, p90, p_top12, p_top6, p_bust, p_spike. With keep_sims=True it also
    returns the raw (n_sims, n_rows) ensemble aligned to the output rows, which
    the calibration script needs and the pairing engine does not.
    """
    rng = np.random.default_rng(seed)
    frames, ens = [], []
    for _key, wk in df.sort(["season", "week"]).group_by(
            ["season", "week"], maintain_order=True):
        S = _simulate(sim, wk, n_sims, rng)
        q = np.percentile(S, _PCTS, axis=0)
        rank = (-S).argsort(axis=1).argsort(axis=1) + 1     # 1 = best that week
        n = wk.height
        frames.append(wk.select("season", "week", "team", "opponent", MARKET)
                      .with_columns(
            pl.Series("exp_pts", S.mean(axis=0)),
            pl.Series("sd", S.std(axis=0)),
            *[pl.Series(f"p{p}", q[i]) for i, p in enumerate(_PCTS)],
            pl.Series("p_top12", (rank <= 12).mean(axis=0) if n >= 12
                      else np.full(n, np.nan)),
            pl.Series("p_top6", (rank <= 6).mean(axis=0) if n >= 6
                      else np.full(n, np.nan)),
            pl.Series("p_bust", (S <= BUST).mean(axis=0)),
            pl.Series("p_spike", (S >= SPIKE).mean(axis=0)),
        ))
        if keep_sims:
            ens.append(S)
    out = pl.concat(frames)
    return (out, np.hstack(ens)) if keep_sims else out


def demo() -> None:
    """Self-check: fit on 2018-2021, simulate 2022, assert it is sane."""
    sys.path.insert(0, str(REPO / "dst-streaming" / "src"))
    import dst_data

    d = dst_data.build_dst_panel(list(range(2018, 2023)))
    tr = d.filter(pl.col("season") < 2022)
    te = d.filter(pl.col("season") == 2022)
    sim = fit(tr)
    out = distributions(sim, te, n_sims=2000)

    assert out.height == te.height
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all()
    for c in ("p_top12", "p_top6", "p_bust", "p_spike"):
        assert out[c].min() >= 0 and out[c].max() <= 1, c
    # P(top-12) must average to roughly 12 / field size each week
    share = out.group_by(["season", "week"]).agg(
        pl.col("p_top12").sum().alias("s"), pl.len().alias("n"))
    assert np.allclose(share["s"].to_numpy(), 12.0, atol=0.01), "top-12 mass"
    # the mean of the simulated distribution must track reality in aggregate
    gap = out["exp_pts"].mean() - te["dst_points"].mean()
    assert abs(gap) < 1.0, f"mean off by {gap:.2f}"
    # the tier map must be the shared one, not a local copy
    assert _TIER[0] == scoring.DSTScoring.points_allowed(0)
    assert _TIER[35] == scoring.DSTScoring.points_allowed(35)
    print(f"ok: {out.height} defense-weeks, mean gap {gap:+.3f} pts, "
          f"P(bust) {out['p_bust'].mean():.3f} vs actual "
          f"{(te['dst_points'] <= BUST).mean():.3f}")


if __name__ == "__main__":
    demo()
