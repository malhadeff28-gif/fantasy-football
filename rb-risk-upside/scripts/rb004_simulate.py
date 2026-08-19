"""RB-004: full-season Monte Carlo, convergence, and risk-adjusted upside.

Distributions first, ranking last. The point of the exercise is that the back
with the best floor and the back with the best risk-adjusted ceiling need not
be the same player, so both are reported and neither is collapsed into a single
number until the metric doing the collapsing has been justified and stress
tested against alternatives.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import rb_data as R  # noqa: E402
import rb_features as F  # noqa: E402
import rb_simulate as S  # noqa: E402
from rb003_scenarios import build_targets  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(170)
BAR = "=" * 78
K = 30
N_SIMS = 50_000
COLORS = {"D'Andre Swift": "#1f77b4", "Tony Pollard": "#ff7f0e",
          "Quinshon Judkins": "#2ca02c", "David Montgomery": "#d62728"}


def risk_metrics(totals: np.ndarray, rb24: float, rb12: float,
                 rb5: float, adp_bar: float | None = None) -> dict:
    """Five specifications, each with a stated reason for existing.

    sharpe    - textbook mean/sd. Punishes upside variance as hard as downside
                variance, which is exactly wrong for a late-round pick, so it
                is included as the baseline to beat, not as the answer.
    sortino   - same idea but only downside deviation counts. Closer to what a
                drafter feels, still a mean-based summary.
    omega     - E[gain above the useful-starter bar] / E[shortfall below it].
                In redraft a bust is droppable and replaceable off waivers, so
                losses are truncated in a way symmetric measures cannot see.
                This is the headline for RISK-ADJUSTED UPSIDE.
    p_top5    - raw probability of the league-winning outcome. No risk
                adjustment at all, included so the reader can see how much of
                the ranking the adjustment is doing.
    cvar_up   - mean of the best decile, the size of the prize when it lands.
    omega_adp - the same gain/shortfall ratio but measured against what a back
                at THAT ADP has historically returned. The only specification
                that charges each player his own draft price; the other five
                compare all four against a common bar even though they cost
                22 picks apart.
    """
    gain = np.maximum(totals - rb12, 0).mean()
    loss = np.maximum(rb12 - totals, 0).mean()
    dd = np.sqrt(np.mean(np.minimum(totals - rb24, 0) ** 2))
    m = {
        "sharpe": (totals.mean() - rb24) / totals.std(),
        "sortino": (totals.mean() - rb24) / dd if dd > 0 else np.nan,
        "omega": gain / loss if loss > 0 else np.inf,
        "p_top5": float((totals >= rb5).mean()),
        "cvar_up": float(totals[totals >= np.percentile(totals, 90)].mean()),
    }
    if adp_bar is not None:
        g = np.maximum(totals - adp_bar, 0).mean()
        l_ = np.maximum(adp_bar - totals, 0).mean()
        m["omega_adp"] = g / l_ if l_ > 0 else np.inf
    return m


def backtest(sit: pl.DataFrame, out: pl.DataFrame, pool: tuple, th: dict,
             adp_fn, seasons=(2022, 2023, 2024, 2025), n: int = 3_000) -> pl.DataFrame:
    """Run the whole pipeline as it would have been run before each season.

    Nothing from the test season or later is available: comparables come only
    from earlier outcome seasons. Reports interval coverage, which is the thing
    that actually matters for a risk study, and accuracy against the ADP curve.
    """
    joined = sit.join(out, left_on=["gsis_id", "season"],
                      right_on=["player_id", "season"], how="inner")
    rows = []
    for yr in seasons:
        cand = joined.filter((pl.col("season") == yr) & (pl.col("adp_rb_rank") <= 40)
                             & pl.col("prior_touches_pg").is_not_null())
        for r in cand.iter_rows(named=True):
            t = S.Target(name=r["full_name"], player_id=r["gsis_id"],
                         team=r["team"], season=yr,
                         prior={**{c: float(r[c]) for c in S.MATCH},
                                "switched": int(r["switched"])},
                         adp=float(r["adp"]), adp_rb_rank=int(r["adp_rb_rank"]))
            try:
                c = S.comparables(t, sit, out, k=K)
            except ValueError:
                continue
            if c.height < 10:
                continue
            s = S.simulate(t, c, pool, th, n_sims=n, seed=7 + yr)
            actual = float(r["out_fp_total"])
            rows.append({"season": yr, "name": t.name, "actual": actual,
                         "median": s["median"], "p10": s["p10"], "p90": s["p90"],
                         "p25": s["p25"], "p75": s["p75"],
                         "adp_pred": adp_fn(t.adp_rb_rank)})
    b = pl.DataFrame(rows)
    summ = []
    for yr in list(seasons) + ["all"]:
        s = b if yr == "all" else b.filter(pl.col("season") == yr)
        if s.height == 0:
            continue
        ins = ((s["actual"] >= s["p10"]) & (s["actual"] <= s["p90"])).mean()
        iqr = ((s["actual"] >= s["p25"]) & (s["actual"] <= s["p75"])).mean()
        below = (s["actual"] < s["median"]).mean()
        mae_m = float((s["actual"] - s["median"]).abs().mean())
        mae_a = float((s["actual"] - s["adp_pred"]).abs().mean())
        summ.append({"season": str(yr), "n": s.height,
                     "cover_p10_p90": round(float(ins), 3),
                     "cover_p25_p75": round(float(iqr), 3),
                     "frac_below_median": round(float(below), 3),
                     "mae_sim": round(mae_m, 1), "mae_adp_curve": round(mae_a, 1)})
    return pl.DataFrame(summ)


def main() -> None:
    panel = F.load_or_build()
    ss = F.season_rollup(panel, min_games=1)
    sit = S.situation_panel(ss, list(range(2018, R.TARGET_SEASON + 1)))
    out = S.outcome_panel(ss)
    targets = build_targets(sit)
    pool = S.game_pool(panel, ss)
    th = S.rank_thresholds(ss)

    print(BAR, "\n 1. FINISH THRESHOLDS (17-game basis, 2018-2025)\n", BAR)
    for n, v in sorted(th.items()):
        print(f"  RB{n:<3} season points: mean {v.mean():6.1f}  "
              f"min {v.min():6.1f}  max {v.max():6.1f}  sd {v.std():5.1f}")
    print("\n  Each simulation draws its own threshold from this set, so the")
    print("  answer carries the year-to-year movement of the bar itself.")
    print("  2018-2020 totals are scaled by 17/16 for a like-for-like season.")

    comps = {t.name: S.comparables(t, sit, out, k=K) for t in targets}

    print("\n", BAR, "\n 2. CONVERGENCE - HOW MANY DRAWS ARE ACTUALLY NEEDED\n", BAR)
    probe = targets[0]
    conv = S.convergence(probe, comps[probe.name], pool, th, reps=5)
    print(f" {probe.name}, 5 independent seeds per size. Columns are the spread")
    print(" of the estimate ACROSS seeds, which is the Monte Carlo error.")
    print(conv)
    ok = conv.filter((pl.col("median_sd") < 1.0) & (pl.col("p_top12_sd") < 0.005))
    n_use = int(ok["n_sims"].min()) if ok.height else 100_000
    print(f"\n  first size meeting +/-1.0 pt on the median and +/-0.005 on")
    print(f"  P(top-12): n = {n_use}. Running {N_SIMS} for headroom; 500k would")
    print("  buy nothing measurable.")

    print("\n", BAR, "\n 3. SEASON OUTCOME DISTRIBUTIONS\n", BAR)
    sims = {}
    rows = []
    for t in targets:
        s = S.simulate(t, comps[t.name], pool, th, n_sims=N_SIMS)
        sims[t.name] = s
        se = s["sd"] / np.sqrt(N_SIMS)
        rows.append({
            "name": t.name, "team": t.team, "adp": t.adp,
            "adp_rb": t.adp_rb_rank,
            "mean": round(s["mean"], 1), "mc_se": round(se, 2),
            "p10": round(s["p10"], 1), "p25": round(s["p25"], 1),
            "median": round(s["median"], 1), "p75": round(s["p75"], 1),
            "p90": round(s["p90"], 1), "sd": round(s["sd"], 1),
            "games": round(s["mean_games"], 1), "touch_pg": round(s["mean_role"], 1),
        })
    dist = pl.DataFrame(rows)
    print(dist)

    print("\n finish probabilities (Monte Carlo standard error < 0.003 at "
          f"n={N_SIMS}):")
    prob = pl.DataFrame([{
        "name": n, "P_top3": round(s["p_top3"], 3), "P_top5": round(s["p_top5"], 3),
        "P_top12": round(s["p_top12"], 3), "P_top24": round(s["p_top24"], 3),
        "P_bust_out_of_top24": round(s["p_bust24"], 3),
        "P_disaster_out_of_top36": round(s["p_bust36"], 3),
    } for n, s in sims.items()])
    print(prob)
    print("\n DEFINITIONS, and why:")
    print("  bust  = finishes outside the top 24 RBs. In a 12-team, 2-RB, no-flex")
    print("          league exactly 24 RBs start, so RB24 IS the replacement line")
    print("          (scoring.LEAGUE.replacement_rank). A pick in rounds 4-6 that")
    print("          does not return a startable RB has returned nothing.")
    print("  disaster = outside the top 36: not even a useful bench asset.")
    print("  league-winning = top-5 RB finish. That is the outcome that turns a")
    print("          round-4/5 pick into a round-1 asset and is what actually")
    print("          separates a title team; top-12 is merely a good starter.")

    print("\n", BAR, "\n 4. AGAINST THE MARKET\n", BAR)
    try:
        cbs = R.cbs_nonppr(2026).filter(pl.col("key").is_in(list(R.TARGETS)))
        cbs_map = dict(cbs.select("key", "cbs_std_points").iter_rows())
    except Exception as e:
        print(f" CBS unavailable: {e}")
        cbs_map = {}
    cmp_rows = []
    for t in targets:
        s = sims[t.name]
        c = cbs_map.get(R.norm(t.name))
        cmp_rows.append({
            "name": t.name, "adp": t.adp, "adp_rb": t.adp_rb_rank,
            "cbs_pts": round(c, 1) if c else None,
            "sim_mean": round(s["mean"], 1),
            "sim_median": round(s["median"], 1),
            "vs_cbs": round(s["mean"] - c, 1) if c else None,
            "cbs_pctile_in_sim": round(float((s["totals"] < c).mean()), 3) if c else None,
        })
    cm = pl.DataFrame(cmp_rows)
    print(cm)
    print("\n cbs_pctile_in_sim is where the consensus projection falls inside the")
    print(" simulated distribution. 0.50 means consensus sits at our median; well")
    print(" above 0.50 means consensus is optimistic relative to what comparable")
    print(" situations actually produced.")
    print("\n like-for-like: CBS projects all four for a healthy 17 games. Ours")
    print(" does not. Stripping the availability difference out:")
    fs = pl.DataFrame([{
        "name": t.name, "cbs_pts": cbs_map.get(R.norm(t.name)),
        "sim_mean_if_17g": round(sims[t.name]["mean_if_17_games"], 1),
        "sim_mean_all_in": round(sims[t.name]["mean"], 1),
        "cost_of_missed_games": round(sims[t.name]["mean_if_17_games"]
                                      - sims[t.name]["mean"], 1),
    } for t in targets])
    print(fs)
    print(" the last column is what a season-long point projection silently")
    print(" ignores: roughly a fifth of the expected total for every one of them.")

    print("\n", BAR, "\n 4b. ROBUSTNESS: HOW THE ROLE IS TRANSFERRED\n", BAR)
    print(" ratio = comparable's role CHANGE applied to this player's own prior")
    print(" role; absolute = comparable's realised role level used directly;")
    print(" blend = geometric mean. This is the single biggest modelling choice.")
    rm_rows = []
    for mode in ("ratio", "absolute", "blend"):
        for t in targets:
            s = S.simulate(t, comps[t.name], pool, th, n_sims=N_SIMS,
                           role_mode=mode)
            rm_rows.append({"mode": mode, "name": t.name,
                            "median": round(s["median"], 1),
                            "p90": round(s["p90"], 1),
                            "P_top12": round(s["p_top12"], 3),
                            "P_bust": round(s["p_bust24"], 3)})
    rm = pl.DataFrame(rm_rows)
    for col in ("median", "P_top12", "P_bust"):
        print(f"\n {col} by role transfer mode:")
        print(rm.pivot(on="name", index="mode", values=col))

    print("\n", BAR, "\n 5. RISK-ADJUSTED UPSIDE, SIX SPECIFICATIONS\n", BAR)
    rb24, rb12, rb5 = th[24].mean(), th[12].mean(), th[5].mean()
    print(f" bars used: RB24 = {rb24:.0f}, RB12 = {rb12:.0f}, RB5 = {rb5:.0f} points")
    adp_fn, adp_r2, adp_n = S.adp_baseline(sit, out)
    print(f" ADP baseline fitted on {adp_n} RB seasons 2018-2025, "
          f"R^2 = {adp_r2:.3f} (ADP explains draft price, not much of outcome)")
    mrows = []
    for t in targets:
        bar = adp_fn(t.adp_rb_rank)
        m = risk_metrics(sims[t.name]["totals"], rb24, rb12, rb5, adp_bar=bar)
        mrows.append({"name": t.name, "adp_bar": round(bar, 1),
                      **{k: round(v, 3) for k, v in m.items()}})
    met = pl.DataFrame(mrows)
    print(met)

    print("\n ranking under each specification (1 = best):")
    METS = ("sharpe", "sortino", "omega", "p_top5", "cvar_up", "omega_adp")
    rank = met.select("name", *[pl.col(c).rank("ordinal", descending=True).alias(c)
                                for c in METS])
    print(rank)
    spread = (rank.select(pl.exclude("name")).max_horizontal()
              - rank.select(pl.exclude("name")).min_horizontal())
    print("\n rank spread across specifications (0 = the answer is robust):")
    print(rank.select("name").with_columns(pl.Series("spread", spread)))

    print("\n floor ranking (median, and p10 as the hard floor):")
    print(dist.select("name", "median", "p10", "p25")
          .sort("median", descending=True))

    print("\n", BAR, "\n 5b. HOW MUCH OF THAT RANKING IS REAL?\n", BAR)
    print(" The Monte Carlo error is tiny. The real uncertainty is that each")
    print(f" player's distribution rests on {K} comparable seasons. Resampling")
    print(" those 30 with replacement (200 reps) gives the honest interval:")
    bs_rows = []
    rng = np.random.default_rng(4242)
    draws = {}
    for t in targets:
        c = comps[t.name]
        meds, p12s = [], []
        for b in range(200):
            idx = rng.integers(0, c.height, c.height)
            cb = c[idx.tolist()]
            s = S.simulate(t, cb, pool, th, n_sims=4_000, seed=int(rng.integers(1e9)))
            meds.append(s["median"]); p12s.append(s["p_top12"])
        draws[t.name] = (np.array(meds), np.array(p12s))
        bs_rows.append({
            "name": t.name,
            "median": round(sims[t.name]["median"], 1),
            "median_lo": round(float(np.percentile(meds, 2.5)), 1),
            "median_hi": round(float(np.percentile(meds, 97.5)), 1),
            "P_top12": round(sims[t.name]["p_top12"], 3),
            "P_top12_lo": round(float(np.percentile(p12s, 2.5)), 3),
            "P_top12_hi": round(float(np.percentile(p12s, 97.5)), 3)})
    bsd = pl.DataFrame(bs_rows)
    print(bsd)

    print("\n pairwise: P(row's median > column's median) over the same resamples")
    names = [t.name for t in targets]
    pw = []
    for a in names:
        row = {"": a.split()[-1]}
        for b in names:
            row[b.split()[-1]] = ("-" if a == b else
                                  f"{float((draws[a][0] > draws[b][0]).mean()):.2f}")
        pw.append(row)
    print(pl.DataFrame(pw))
    bsd.write_csv(R.OUT / "rb004_comparable_uncertainty.csv")

    print("\n", BAR, "\n 6. IS THE SIMULATOR CALIBRATED? WALK-FORWARD BACKTEST\n", BAR)
    print(" Every drafted RB (ADP top 40 at the position) in 2022-2025, simulated")
    print(" using only seasons before his own. If the intervals are honest, ~80%")
    print(" of realised totals land between p10 and p90 and ~50% below the median.")
    bt = backtest(sit, out, pool, th, adp_fn)
    print(bt)

    dist.write_csv(R.OUT / "rb004_distributions.csv")
    prob.write_csv(R.OUT / "rb004_probabilities.csv")
    met.write_csv(R.OUT / "rb004_risk_metrics.csv")
    conv.write_csv(R.OUT / "rb004_convergence.csv")
    rm.write_csv(R.OUT / "rb004_role_mode_robustness.csv")
    bt.write_csv(R.OUT / "rb004_backtest.csv")

    # ---- plots ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for t in targets:
        ax.hist(sims[t.name]["totals"], bins=80, density=True, histtype="step",
                lw=2, label=f"{t.name} ({t.team})", color=COLORS[t.name])
    for v, lab in [(rb24, "RB24"), (rb12, "RB12"), (rb5, "RB5")]:
        ax.axvline(v, color="grey", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1] * 0.95, lab, fontsize=8, color="grey",
                ha="center")
    ax.set_xlabel("2026 season fantasy points (standard, non-PPR)")
    ax.set_ylabel("density")
    ax.set_title(f"Simulated season outcome distributions ({N_SIMS:,} draws each)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(R.OUT / "rb004_distributions.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    names = [t.name for t in targets]
    x = np.arange(len(names))
    for i, n in enumerate(names):
        s = sims[n]
        ax.plot([i, i], [s["p10"], s["p90"]], color=COLORS[n], lw=2.5, alpha=.45)
        ax.plot([i, i], [s["p25"], s["p75"]], color=COLORS[n], lw=8, alpha=.75)
        ax.plot(i, s["median"], "o", color="white", mec=COLORS[n], mew=2, ms=9)
        ax.text(i + .13, s["p90"], f"p90 {s['p90']:.0f}", fontsize=8, va="center")
        ax.text(i + .13, s["p10"], f"p10 {s['p10']:.0f}", fontsize=8, va="center")
        ax.text(i + .13, s["median"], f"med {s['median']:.0f}", fontsize=8,
                va="center", fontweight="bold")
    for v, lab in [(rb24, "RB24 replacement"), (rb12, "RB12"), (rb5, "RB5")]:
        ax.axhline(v, color="grey", ls="--", lw=1)
        ax.text(-0.45, v + 3, lab, fontsize=8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n.split()[-1]}\n{t.team} / RB{t.adp_rb_rank}"
                        for n, t in zip(names, targets)])
    ax.set_ylabel("season fantasy points")
    ax.set_title("Floor, median and ceiling (p10-p90 thin, p25-p75 thick)")
    fig.tight_layout()
    fig.savefig(R.OUT / "rb004_floor_ceiling.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for t in targets:
        s = sims[t.name]
        ax.errorbar(t.adp, s["mean"],
                    yerr=[[s["mean"] - s["p25"]], [s["p75"] - s["mean"]]],
                    fmt="o", color=COLORS[t.name], capsize=4, ms=9,
                    label=f"{t.name} (sim)")
        c = cbs_map.get(R.norm(t.name))
        if c:
            ax.plot(t.adp, c, "x", color=COLORS[t.name], ms=11, mew=2.5)
            ax.plot([t.adp, t.adp], [s["mean"], c], color=COLORS[t.name],
                    ls=":", lw=1)
    ax.axhline(rb24, color="grey", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], rb24 + 3, "RB24 replacement", fontsize=8,
            color="grey")
    ax.set_xlabel("FFC ADP, 12-team standard (earlier pick = more expensive)")
    ax.set_ylabel("season fantasy points")
    ax.set_title("Draft cost vs simulated value  (o = simulation mean with IQR, "
                 "x = CBS consensus)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R.OUT / "rb004_adp_vs_value.png", dpi=130)
    plt.close(fig)

    print(f"\nwritten: 6 csv + 3 png under {R.OUT}")


if __name__ == "__main__":
    main()
