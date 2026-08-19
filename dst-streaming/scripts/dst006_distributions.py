"""DST-006: full D/ST outcome distributions, and whether they are calibrated.

Fit on seasons < S, simulate season S, never the reverse. Four candidate
distributions are scored against each other so the component design has to earn
its place rather than be asserted:

  A  components + Gaussian copula   the proposal (src/dst_distributions.py),
                                    training rows down-weighted with a 3-season
                                    half-life
  B  components, independent        same marginals, dependence deleted
  C  total only                     one regression on the total plus a residual
                                    bootstrap - the "just fit the total" rival
  D  climatology                    the pooled historical distribution, same for
                                    every defense every week - the no-information
                                    floor that anything must beat
  E  A with equal season weights    what A looked like before the recency weight,
                                    kept in the table so that choice is scored
                                    rather than asserted

Calibration is checked three ways, because each catches a different failure:
randomised PIT uniformity (is the whole shape right), quantile coverage (are the
percentiles we publish honest), and reliability curves for the four event
probabilities the pairing engine will actually consume.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys                                          # noqa: E402
from pathlib import Path                            # noqa: E402

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import polars as pl                                 # noqa: E402
from scipy import stats                             # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import dst_data                                     # noqa: E402
import dst_distributions as dd                      # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_width_chars(150)

SEASONS = list(range(2018, 2026))
TEST = list(range(2020, 2026))
N_SIMS = 3000
BAR = "=" * 84
RNG = np.random.default_rng(606)


# ------------------------------------------------------------------ metrics
def crps(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score from an ensemble, per column.

    CRPS = E|X-y| - 0.5 E|X-X'|, using the sorted-sample identity so the second
    term is O(m log m) rather than O(m^2).
    """
    m = S.shape[0]
    xs = np.sort(S, axis=0)
    t1 = np.abs(xs - y).mean(axis=0)
    w = (2 * np.arange(m) - m + 1).reshape(-1, 1)
    return t1 - (w * xs).sum(axis=0) / m ** 2


def pit(S: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Randomised PIT, which is the right transform for a discrete outcome."""
    lo = (S < y).mean(axis=0)
    hi = (S <= y).mean(axis=0)
    return lo + rng.random(len(y)) * (hi - lo)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def reliability(p: np.ndarray, y: np.ndarray, edges) -> list[dict]:
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < edges[-1] else (p >= a) & (p <= b)
        if m.sum() < 25:
            continue
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = stats.beta.ppf([0.025, 0.975], [k + 0.5, k + 0.5],
                                [n - k + 0.5, n - k + 0.5])
        rows.append({"lo": a, "hi": b, "n": n, "pred": float(p[m].mean()),
                     "obs": k / n, "obs_lo": float(lo), "obs_hi": float(hi)})
    return rows


# --------------------------------------------------------- rival simulators
def fit_total_only(train: pl.DataFrame):
    """Rival C: regress the total on the market, bootstrap the residuals."""
    x = train[dd.MARKET].to_numpy().astype(float)
    y = train["dst_points"].to_numpy().astype(float)
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"b": b, "centre": float(x.mean()), "res": y - X @ b}


def sim_total_only(model, wk: pl.DataFrame, n: int, rng) -> np.ndarray:
    x = wk[dd.MARKET].to_numpy().astype(float)
    X = np.column_stack([np.ones_like(x), x - model["centre"]])
    return (X @ model["b"])[None, :] + rng.choice(model["res"], size=(n, wk.height))


# -------------------------------------------------------------------- main
def main() -> int:
    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    d = dst_data.build_dst_panel(SEASONS)
    print(f"panel: {d.height} defense-weeks, {d['season'].n_unique()} seasons")

    print(f"\n{BAR}\n DST-006: WHY MODEL COMPONENTS RATHER THAN THE TOTAL\n{BAR}")
    dp = d.with_columns((pl.col("def_tds").fill_null(0)
                         + pl.col("special_teams_tds").fill_null(0)).alias("td_any"))
    print(f"{'component':<24}{'mean':>8}{'var/mean':>10}{'pts/gm':>9}   shape used")
    print("-" * 84)
    for c, w in dd.COUNTS.items():
        v = dp[c].fill_null(0).to_numpy().astype(float)
        shape = "negative binomial" if c in dd.NB_COMPONENTS else "Poisson"
        print(f"{c:<24}{v.mean():>8.3f}{v.var() / v.mean():>10.3f}"
              f"{v.mean() * w:>9.3f}   {shape}")
    pa = dp["points_allowed"].to_numpy().astype(float)
    print(f"{'points_allowed':<24}{pa.mean():>8.3f}{'-':>10}"
          f"{dp['_dst_pa'].mean():>9.3f}   linear + empirical residuals -> tier map")
    print(f"{'misc (safety/blocks)':<24}{'-':>8}{'-':>10}"
          f"{dp['_dst_base'].mean() - sum(dp[c].fill_null(0).mean() * w for c, w in dd.COUNTS.items()):>9.3f}"
          f"   empirical draw")
    cols = list(dd.COUNTS) + ["points_allowed"]
    M = np.column_stack([dp[c].fill_null(0).to_numpy().astype(float) for c in cols])
    C = np.corrcoef(M.T)
    print("\nComponents are correlated, so they cannot be sampled independently:")
    print(f"{'':<22}" + "".join(f"{c[:10]:>12}" for c in cols))
    for i, c in enumerate(cols):
        print(f"{c[:21]:<22}" + "".join(f"{C[i, j]:>12.3f}" for j in range(len(cols))))

    y_all = d["dst_points"].to_numpy()
    print(f"\nThresholds. bust <= {dd.BUST:.0f} pts: {(y_all <= dd.BUST).mean():.1%} "
          f"of defense-weeks, and {dd.BUST:.0f} is the median weekly 24th-best "
          f"score - the waiver wire would have beaten you.")
    print(f"spike >= {dd.SPIKE:.0f} pts: {(y_all >= dd.SPIKE).mean():.1%} of "
          f"defense-weeks, past the median weekly 6th-best score of 11 - a start "
          f"that wins the position.")

    # ------------------------------------------------- walk-forward simulation
    print(f"\n{BAR}\n DST-006: WALK-FORWARD OUT-OF-SAMPLE SIMULATION\n{BAR}")
    variants = ["A copula", "B independent", "C total-only", "D climatology",
                "E equal-weight"]
    acc = {v: {"crps": [], "pit": [], "top12": [], "top6": [],
               "bust": [], "spike": []} for v in variants}
    keep, actual, is12, is6, isb, issp = [], [], [], [], [], []
    sim_pool, example = [], None

    for s in TEST:
        tr = d.filter(pl.col("season") < s)
        te = d.filter(pl.col("season") == s)
        sim_a = dd.fit(tr)
        sim_b = dd.fit(tr, independent=True)
        sim_e = dd.fit(tr, half_life=None)
        mod_c = fit_total_only(tr)
        clim = tr["dst_points"].to_numpy().astype(float)
        for (_, wkno), wk in te.group_by(["season", "week"], maintain_order=True):
            seed = int(s * 100 + wkno)
            rng = np.random.default_rng(seed)
            y = wk["dst_points"].to_numpy().astype(float)
            n = wk.height
            rk = stats.rankdata(-y, method="average")
            actual.append(y)
            is12.append((rk <= 12).astype(float) if n >= 12 else np.full(n, np.nan))
            is6.append((rk <= 6).astype(float) if n >= 6 else np.full(n, np.nan))
            isb.append((y <= dd.BUST).astype(float))
            issp.append((y >= dd.SPIKE).astype(float))

            ens = {}
            frm, ens["A copula"] = dd.distributions(sim_a, wk, N_SIMS, seed, True)
            _, ens["B independent"] = dd.distributions(sim_b, wk, N_SIMS, seed, True)
            ens["C total-only"] = sim_total_only(mod_c, wk, N_SIMS, rng)
            ens["D climatology"] = rng.choice(clim, size=(N_SIMS, n))
            keep.append(frm)
            sim_pool.append(ens["A copula"][::30].ravel())
            if (s == 2024 and wkno == 5) or example is None:
                example = (wk, frm)

            for v, S in ens.items():
                rank = (-S).argsort(axis=1).argsort(axis=1) + 1
                acc[v]["crps"].append(crps(S, y))
                acc[v]["pit"].append(pit(S, y, rng))
                acc[v]["top12"].append((rank <= 12).mean(axis=0) if n >= 12
                                       else np.full(n, np.nan))
                acc[v]["top6"].append((rank <= 6).mean(axis=0) if n >= 6
                                      else np.full(n, np.nan))
                acc[v]["bust"].append((S <= dd.BUST).mean(axis=0))
                acc[v]["spike"].append((S >= dd.SPIKE).mean(axis=0))
        print(f"  season {s}: trained on {tr.height} rows, simulated {te.height}")

    cat = lambda xs: np.concatenate(xs)                              # noqa: E731
    A = {k: cat(v) for k, v in zip(["y", "t12", "t6", "bust", "spike"],
                                   [actual, is12, is6, isb, issp])}
    P = {v: {k: cat(acc[v][k]) for k in acc[v]} for v in variants}
    summ = pl.concat(keep)
    print(f"\nout-of-sample defense-weeks scored: {len(A['y'])}")

    # -------------------------------------------------- headline comparison
    print(f"\n{BAR}\n DST-006: WHICH DISTRIBUTION IS BEST OUT OF SAMPLE\n{BAR}")
    base = float(np.mean(P["D climatology"]["crps"]))
    print(f"{'variant':<16}{'CRPS':>8}{'skill':>8}{'KS(PIT)':>9}{'p(KS)':>8}"
          f"{'Brier12':>9}{'BSS12':>8}{'AUC12':>8}{'Brier bust':>12}")
    print("-" * 90)
    rows = []
    m12 = np.isfinite(A["t12"])
    b12_clim = float(np.mean((P["D climatology"]["top12"][m12] - A["t12"][m12]) ** 2))
    for v in variants:
        c = float(np.mean(P[v]["crps"]))
        ks = stats.kstest(P[v]["pit"], "uniform")
        b12 = float(np.mean((P[v]["top12"][m12] - A["t12"][m12]) ** 2))
        bb = float(np.mean((P[v]["bust"] - A["bust"]) ** 2))
        a12 = auc(P[v]["top12"][m12], A["t12"][m12])
        print(f"{v:<16}{c:>8.3f}{1 - c / base:>8.3f}{ks.statistic:>9.4f}"
              f"{ks.pvalue:>8.3f}{b12:>9.4f}{1 - b12 / b12_clim:>8.3f}"
              f"{a12:>8.3f}{bb:>12.4f}")
        rows.append({"variant": v, "crps": c, "crps_skill": 1 - c / base,
                     "ks_pit": float(ks.statistic), "ks_p": float(ks.pvalue),
                     "brier_top12": b12, "bss_top12": 1 - b12 / b12_clim,
                     "auc_top12": a12, "brier_bust": bb})
    print("\nskill / BSS are versus climatology; positive means better. KS p above")
    print("0.05 means the PIT is not distinguishable from uniform, i.e. calibrated.")

    # ------------------------------------------------------ quantile coverage
    print(f"\n{BAR}\n DST-006: PUBLISHED PERCENTILES, OUT-OF-SAMPLE COVERAGE\n{BAR}")
    print("A p10 should sit above the actual score 10% of the time, and so on.\n")
    print(f"{'variant':<16}" + "".join(f"{f'p{q}':>12}" for q in dd._PCTS))
    print("-" * 76)
    cov_rows = []
    for v in variants:
        u = P[v]["pit"]
        line = f"{v:<16}"
        rec = {"variant": v}
        for q in dd._PCTS:
            f = float(np.mean(u <= q / 100))
            se = np.sqrt(f * (1 - f) / len(u))
            line += f"{f * 100:>8.1f}%".rjust(12)
            rec[f"p{q}"] = f
            rec[f"p{q}_se"] = float(se)
        print(line)
        cov_rows.append(rec)
    print(f"{'nominal':<16}" + "".join(f"{q:>11}%" for q in dd._PCTS))
    print(f"\n(binomial SE at n={len(P['A copula']['pit'])} is about "
          f"{np.sqrt(0.1 * 0.9 / len(P['A copula']['pit'])) * 100:.2f} pp at p10)")

    # ---------------------------------------------------------- reliability
    print(f"\n{BAR}\n DST-006: RELIABILITY OF THE EVENT PROBABILITIES (variant A)\n{BAR}")
    edges = np.array([0, .05, .1, .15, .2, .3, .4, .5, .7, 1.0])
    rel_all = {}
    for ev, mask in (("top12", m12), ("top6", np.isfinite(A["t6"])),
                     ("bust", np.ones(len(A["y"]), bool)),
                     ("spike", np.ones(len(A["y"]), bool))):
        key = {"top12": "t12", "top6": "t6", "bust": "bust", "spike": "spike"}[ev]
        rel = reliability(P["A copula"][ev][mask], A[key][mask], edges)
        rel_all[ev] = rel
        print(f"\nP({ev}):")
        print(f"{'bin':<14}{'n':>7}{'predicted':>12}{'observed':>11}{'95% CI':>18}")
        for r in rel:
            print(f"[{r['lo']:.2f},{r['hi']:.2f})".ljust(14)
                  + f"{r['n']:>7}{r['pred']:>12.3f}{r['obs']:>11.3f}"
                  + f"   [{r['obs_lo']:.3f},{r['obs_hi']:.3f}]")

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    u = P["A copula"]["pit"]
    ax[0, 0].hist(u, bins=20, range=(0, 1), color="#4c78a8",
                  edgecolor="white", density=True)
    ax[0, 0].axhline(1.0, color="k", ls="--", lw=1.2)
    ks = stats.kstest(u, "uniform")
    ax[0, 0].set_title(f"PIT of the total, variant A  (KS={ks.statistic:.4f}, "
                       f"p={ks.pvalue:.3f})")
    ax[0, 0].set_xlabel("randomised PIT"); ax[0, 0].set_ylabel("density")

    w, xs = 0.2, np.arange(len(dd._PCTS))
    for i, v in enumerate(variants):
        vals = [np.mean(P[v]["pit"] <= q / 100) * 100 for q in dd._PCTS]
        ax[0, 1].bar(xs + (i - 1.5) * w, vals, w, label=v)
    ax[0, 1].plot(xs, dd._PCTS, "k_", ms=34, mew=2, label="nominal")
    ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels([f"p{q}" for q in dd._PCTS])
    ax[0, 1].set_ylabel("% of actuals below"); ax[0, 1].legend(fontsize=7.5)
    ax[0, 1].set_title("quantile coverage, out of sample")

    for a, evs, ttl in ((ax[1, 0], ["top12", "top6"], "P(top-12) / P(top-6)"),
                        (ax[1, 1], ["bust", "spike"],
                         f"P(bust <= {dd.BUST:.0f}) / P(spike >= {dd.SPIKE:.0f})")):
        for ev, col in zip(evs, ["#4c78a8", "#e45756"]):
            r = rel_all[ev]
            p = [x["pred"] for x in r]
            o = [x["obs"] for x in r]
            lo = [max(x["obs"] - x["obs_lo"], 0.0) for x in r]
            hi = [max(x["obs_hi"] - x["obs"], 0.0) for x in r]
            a.errorbar(p, o, yerr=[lo, hi], fmt="o-", color=col, capsize=3,
                       ms=5, label=f"P({ev})")
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.set_xlabel("predicted"); a.set_ylabel("observed frequency")
        a.set_title(f"reliability: {ttl}"); a.legend(fontsize=8)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
    fig.suptitle("DST-006 calibration, walk-forward 2020-2025 (variant A: "
                 "components + copula)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "dst006_calibration.png", dpi=140)
    plt.close(fig)

    # example week + shape check
    fig, ax = plt.subplots(1, 2, figsize=(14, 6),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    wk, frm = example
    o = np.argsort(-frm["exp_pts"].to_numpy())
    lab = frm["team"].to_numpy()[o]
    yv = np.arange(len(o))
    ax[0].hlines(yv, frm["p10"].to_numpy()[o], frm["p90"].to_numpy()[o],
                 color="#c8d4e3", lw=7)
    ax[0].hlines(yv, frm["p25"].to_numpy()[o], frm["p75"].to_numpy()[o],
                 color="#4c78a8", lw=7)
    ax[0].plot(frm["p50"].to_numpy()[o], yv, "|", color="white", ms=9, mew=2)
    ax[0].plot(wk["dst_points"].to_numpy()[o], yv, "D", color="#e45756", ms=5,
               label="actual")
    ax[0].axvline(dd.BUST, color="grey", ls=":", lw=1)
    ax[0].axvline(dd.SPIKE, color="grey", ls=":", lw=1)
    ax[0].set_yticks(yv); ax[0].set_yticklabels(lab, fontsize=7)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("D/ST fantasy points")
    ax[0].set_title(f"{wk['season'][0]} week {wk['week'][0]}: p10-p90 (light), "
                    "p25-p75 (dark), median, actual")

    bins = np.arange(-6, 32, 1)
    ax[1].hist(A["y"], bins=bins, density=True, alpha=0.55, color="#e45756",
               label="actual, 2020-2025")
    ax[1].hist(np.concatenate(sim_pool), bins=bins, density=True,
               histtype="step", lw=2, color="#4c78a8",
               label="simulated, variant A (pooled OOS)")
    ax[1].set_xlabel("D/ST fantasy points"); ax[1].set_ylabel("density")
    ax[1].set_title("shape check: the skew and the -4 / +10 tier tails")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dst006_example_week.png", dpi=140)
    plt.close(fig)

    summ.write_csv(out / "dst006_oos_distributions.csv")
    pl.DataFrame(rows).write_csv(out / "dst006_variant_comparison.csv")
    pl.DataFrame(cov_rows).write_csv(out / "dst006_coverage.csv")
    pl.DataFrame([{"event": e, **r} for e, rl in rel_all.items() for r in rl]
                 ).write_csv(out / "dst006_reliability.csv")
    print(f"\nwrote {out / 'dst006_calibration.png'}")
    print(f"wrote {out / 'dst006_example_week.png'}")
    print("wrote dst006_oos_distributions.csv, dst006_variant_comparison.csv,")
    print("      dst006_coverage.csv, dst006_reliability.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
