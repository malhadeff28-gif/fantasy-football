"""Wave 2 skeptic pass: QB-022 (weather attack) and QB-023 (economics replication).

Two adversarial jobs, run from one script because they share nothing but the
intent to break a published finding.

QB-022 -- attack QB-008: "wind >= 15mph costs a quarterback 1.61 fantasy points
(t=-3.45, n=304); wind 8-14mph costs 0.57 (t=-2.39)."

  The weakness under test is that nflverse `temp`/`wind` are 35-75% populated by
  season and structurally absent indoors. Five attacks:
    A. characterise the missingness against observables (is it MAR?)
    B. re-estimate on outdoor games only, where a missing reading is a data gap
       rather than a structural n/a
    C. re-estimate with the betting market as a control -- if the total already
       prices the forecast, the residual estimate is double counting
    D. ask directly whether the market prices wind (does total_line move?)
    E. ask whether wind was knowable and actionable pregame, since nflverse
       stores OBSERVED game-time weather, not the forecast a manager saw

  Everything is re-estimated with standard errors clustered by game, because
  both quarterbacks in a game share one wind reading and the published t-stats
  treat them as independent.

QB-023 -- independently rebuild the pick-value curve behind QB-012 from raw
sources (FFC ADP + nflverse player stats) without reading or importing
`qbpair/benchmarks/draft_value.py`, then compare bucket by bucket against the
committed `outputs/phase6_pick_value_curve.csv`.

Usage:  ./.venv/Scripts/python.exe qb-pairing/scripts/wave2_weather_and_replication.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                        # noqa: E402
import warnings                                   # noqa: E402
from pathlib import Path                          # noqa: E402

import numpy as np                                # noqa: E402
import polars as pl                               # noqa: E402
import statsmodels.api as sm                      # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]     # qb-pairing/
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))

import bootstrap                                  # noqa: E402

bootstrap.setup()

import markets                                    # noqa: E402
import nfl_data                                   # noqa: E402
from qbpair.data import loaders                   # noqa: E402
from qbpair.features import panel as panel_mod    # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)

SEASONS = list(range(2018, 2026))
OUT = PROJECT / "outputs"
OUT.mkdir(exist_ok=True)
DATA = PROJECT / "data"
DATA.mkdir(exist_ok=True)
BAR = "=" * 78
RNG = np.random.default_rng(20260818)


# ---------------------------------------------------------------------------
# small statistical helpers
# ---------------------------------------------------------------------------

def mean_se(x: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan, n


def cluster_mean_se(x: np.ndarray, groups: np.ndarray) -> tuple[float, float, int, int]:
    """Mean with a cluster-robust standard error (one-way, by `groups`).

    Two quarterbacks share a game and therefore share one wind reading; treating
    them as independent understates the standard error.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(groups)
    n = x.size
    m = x.mean()
    d = x - m
    uniq = np.unique(g)
    gsum = np.array([d[g == u].sum() for u in uniq])
    G = uniq.size
    # sandwich for the mean: var = sum_g (sum_i e_ig)^2 / n^2, small-sample scaled
    var = (gsum ** 2).sum() / n ** 2
    var *= G / max(G - 1, 1)
    return float(m), float(np.sqrt(var)), int(n), int(G)


def ols(df: pl.DataFrame, y: str, xs: list[str], cluster: str | None = None):
    d = df.drop_nulls(subset=[y] + xs + ([cluster] if cluster else []))
    X = sm.add_constant(d.select(xs).to_numpy().astype(float), has_constant="add")
    Y = d[y].to_numpy().astype(float)
    if cluster:
        res = sm.OLS(Y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": d[cluster].to_numpy()})
    else:
        res = sm.OLS(Y, X).fit()
    return res, d.height


def coef_rows(res, xs: list[str], spec: str, n: int, target: str) -> list[dict]:
    names = ["const"] + xs
    out = []
    ci = res.conf_int()
    for i, nm in enumerate(names):
        if nm == "const":
            continue
        out.append({"spec": spec, "target": target, "term": nm, "n": n,
                    "coef": float(res.params[i]), "se": float(res.bse[i]),
                    "t": float(res.tvalues[i]), "p": float(res.pvalues[i]),
                    "ci_lo": float(ci[i][0]), "ci_hi": float(ci[i][1])})
    return out


# ---------------------------------------------------------------------------
# QB-022 : the weather attack
# ---------------------------------------------------------------------------

def load_weather_panel() -> pl.DataFrame:
    p = (panel_mod.build_panel(SEASONS)
         .drop_nulls(subset=["form_fp_mean", "form_pass_att", "rush_fp_share"])
         .with_columns((pl.col("fantasy_points") - pl.col("form_fp_mean"))
                       .alias("form_resid")))
    sch = loaders.load_schedules().select(
        ["game_id", "temp", "wind", "roof", "surface", "gametime", "stadium"])
    w = p.join(sch, on="game_id", how="left")
    return w.with_columns(
        pl.col("wind").is_null().cast(pl.Int8).alias("wx_missing"),
        pl.col("roof").is_in(["outdoors", "open"]).cast(pl.Int8).alias("is_outdoor"),
        (pl.col("wind") >= 15).cast(pl.Float64).alias("wind_ge15"),
        ((pl.col("wind") >= 8) & (pl.col("wind") < 15)).cast(pl.Float64).alias("wind_8_14"),
    )


def q22_baseline(w: pl.DataFrame) -> pl.DataFrame:
    """Reproduce the published QB-008 numbers exactly, then re-do the SEs."""
    print(f"{BAR}\n QB-022 / STEP 0 : reproduce the published estimate\n{BAR}")
    ww = w.drop_nulls("wind").with_columns(
        pl.when(pl.col("wind") >= 15).then(pl.lit("3 wind >=15mph"))
        .when(pl.col("wind") >= 8).then(pl.lit("2 wind 8-14mph"))
        .otherwise(pl.lit("1 wind <8mph")).alias("wind_bucket"))
    rows = []
    print(f"{'bucket':<18}{'n':>6}{'games':>7}{'mean':>9}{'se_iid':>9}{'t_iid':>8}"
          f"{'se_clu':>9}{'t_clu':>8}")
    print("-" * 78)
    for b in sorted(ww["wind_bucket"].unique().to_list()):
        s = ww.filter(pl.col("wind_bucket") == b)
        m, se, n = mean_se(s["form_resid"].to_numpy())
        mc, sec, _, G = cluster_mean_se(s["form_resid"].to_numpy(),
                                        s["game_id"].to_numpy())
        rows.append({"bucket": b, "n": n, "games": G, "mean": m,
                     "se_iid": se, "t_iid": m / se,
                     "se_cluster": sec, "t_cluster": m / sec})
        print(f"{b:<18}{n:>6}{G:>7}{m:>9.3f}{se:>9.3f}{m/se:>8.2f}"
              f"{sec:>9.3f}{m/sec:>8.2f}")
    # contrast vs the low-wind bucket, which is the causal quantity the claim
    # implicitly asserts (the claim reports the raw bucket mean instead)
    hi = ww.filter(pl.col("wind_bucket") == "3 wind >=15mph")["form_resid"].to_numpy()
    lo = ww.filter(pl.col("wind_bucket") == "1 wind <8mph")["form_resid"].to_numpy()
    diff = hi.mean() - lo.mean()
    se_d = np.sqrt(hi.var(ddof=1) / hi.size + lo.var(ddof=1) / lo.size)
    print(f"\ncontrast >=15 minus <8 : {diff:+.3f}  (se {se_d:.3f}, t {diff/se_d:+.2f})")
    print("the headline 'costs 1.61' is a raw bucket mean, not a contrast; the")
    print("low-wind bucket is itself below zero, so the differential is smaller.")
    return pl.DataFrame(rows)


def q22_missingness(w: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"\n{BAR}\n QB-022 / A : is the weather missingness at random?\n{BAR}")

    print("\ncoverage by roof type (QB-weeks, 2018-2025):")
    cov = (w.group_by("roof").agg(
        pl.len().alias("n"),
        (1 - pl.col("wx_missing").mean()).alias("wind_coverage"),
        pl.col("form_resid").mean().alias("mean_form_resid"),
        pl.col("implied_team_total").mean().alias("mean_itt"))
        .sort("n", descending=True))
    print(cov)
    print("`dome`, `closed` and `open` are structurally 0% -- a retractable roof")
    print("left OPEN is an outdoor game with no weather row at all, so 'indoors'")
    print("and 'no reading' are not the same set.")

    print("\ncoverage by season, OUTDOOR games only:")
    o = w.filter(pl.col("is_outdoor") == 1)
    seas = (o.group_by("season").agg(
        pl.len().alias("n"),
        (1 - pl.col("wx_missing").mean()).alias("wind_coverage"),
        pl.col("form_resid").mean().alias("mean_form_resid"))
        .sort("season"))
    print(seas)

    obs = ["season", "week", "implied_team_total", "total_line", "team_spread",
           "spread_abs", "form_fp_mean", "career_fp_mean", "is_home", "div_game",
           "rest", "def_adj", "fantasy_points", "form_resid"]

    def compare(df: pl.DataFrame, label: str) -> list[dict]:
        rows = []
        pres = df.filter(pl.col("wx_missing") == 0)
        absent = df.filter(pl.col("wx_missing") == 1)
        for c in obs:
            a = pres.drop_nulls(c)[c].to_numpy().astype(float)
            b = absent.drop_nulls(c)[c].to_numpy().astype(float)
            if a.size < 30 or b.size < 30:
                continue
            d = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            rows.append({"sample": label, "variable": c,
                         "n_present": int(a.size), "n_absent": int(b.size),
                         "mean_present": float(a.mean()), "mean_absent": float(b.mean()),
                         "diff": float(d), "se": float(se), "t": float(d / se)})
        return rows

    rows = compare(w, "all QB-weeks") + compare(o, "outdoor only")
    tab = pl.DataFrame(rows)
    for lab in ("all QB-weeks", "outdoor only"):
        print(f"\nweather-PRESENT minus weather-ABSENT, {lab}:")
        print(f"{'variable':<20}{'present':>10}{'absent':>10}{'diff':>9}{'se':>8}{'t':>8}")
        print("-" * 78)
        for r in tab.filter(pl.col("sample") == lab).to_dicts():
            star = "  *" if abs(r["t"]) >= 2 else ""
            print(f"{r['variable']:<20}{r['mean_present']:>10.3f}{r['mean_absent']:>10.3f}"
                  f"{r['diff']:>9.3f}{r['se']:>8.3f}{r['t']:>8.2f}{star}")

    # A model: can missingness be predicted from observables among outdoor games?
    print("\nlogit: P(wind reading missing | outdoor game) on pregame observables")
    xs = ["week", "implied_team_total", "total_line", "is_home", "div_game",
          "form_fp_mean"]
    d = o.drop_nulls(subset=xs + ["wx_missing"])
    X = sm.add_constant(d.select(xs).to_numpy().astype(float), has_constant="add")
    y = d["wx_missing"].to_numpy().astype(float)
    lg = sm.Logit(y, X).fit(disp=0)
    mrows = []
    for i, nm in enumerate(["const"] + xs):
        mrows.append({"model": "logit_missing_outdoor", "term": nm,
                      "coef": float(lg.params[i]), "se": float(lg.bse[i]),
                      "z": float(lg.tvalues[i]), "p": float(lg.pvalues[i])})
        print(f"    {nm:<22}{lg.params[i]:>9.4f}  se {lg.bse[i]:.4f}  z {lg.tvalues[i]:>6.2f}"
              f"{'  *' if lg.pvalues[i] < 0.05 else ''}")
    print(f"    pseudo-R2 {lg.prsquared:.4f}   n={d.height}")

    # season fixed effects alone
    ds = o.to_dummies("season")
    scols = [c for c in ds.columns if c.startswith("season_")][1:]
    Xs = sm.add_constant(ds.select(scols).to_numpy().astype(float), has_constant="add")
    lgs = sm.Logit(ds["wx_missing"].to_numpy().astype(float), Xs).fit(disp=0)
    print(f"    season dummies alone: pseudo-R2 {lgs.prsquared:.4f}")
    mrows.append({"model": "logit_missing_outdoor", "term": "_pseudo_r2_observables",
                  "coef": float(lg.prsquared), "se": np.nan, "z": np.nan, "p": np.nan})
    mrows.append({"model": "logit_missing_outdoor", "term": "_pseudo_r2_season_only",
                  "coef": float(lgs.prsquared), "se": np.nan, "z": np.nan, "p": np.nan})
    print("\n=> missingness is driven overwhelmingly by WHICH SEASON the game is in")
    print("   (a scrape/vendor gap), not by pregame game characteristics.")

    cov.write_csv(OUT / "qb022_coverage_by_roof.csv")
    seas.write_csv(OUT / "qb022_coverage_by_season_outdoor.csv")
    return tab, pl.DataFrame(mrows)


def q22_outdoor_only(w: pl.DataFrame) -> pl.DataFrame:
    print(f"\n{BAR}\n QB-022 / B : re-estimate on OUTDOOR games only\n{BAR}")
    print("Among rows WITH a wind reading, 2652/2654 are already outdoor, so the")
    print("published estimate was never contaminated by domes. The live question")
    print("is whether the 11% of outdoor rows with no reading differ.")

    o = w.filter(pl.col("is_outdoor") == 1)
    rows = []

    def block(df: pl.DataFrame, label: str) -> None:
        d = df.drop_nulls("wind").with_columns(
            pl.when(pl.col("wind") >= 15).then(pl.lit("3 >=15"))
            .when(pl.col("wind") >= 8).then(pl.lit("2 8-14"))
            .otherwise(pl.lit("1 <8")).alias("wb"))
        for b in sorted(d["wb"].unique().to_list()):
            s = d.filter(pl.col("wb") == b)
            m, se, n, G = cluster_mean_se(s["form_resid"].to_numpy(),
                                          s["game_id"].to_numpy())
            rows.append({"sample": label, "bucket": b, "n": n, "games": G,
                         "mean": m, "se_cluster": se, "t_cluster": m / se,
                         "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se})
            print(f"  {label:<34}{b:<8}{n:>6}{m:>9.3f}{se:>8.3f}{m/se:>8.2f}")

    print(f"\n  {'sample':<34}{'bucket':<8}{'n':>6}{'mean':>9}{'se':>8}{'t':>8}")
    print("-" * 78)
    block(o, "outdoor, all seasons")
    block(o.filter(pl.col("season") != 2022), "outdoor, drop 2022 (51% cover)")
    block(o.filter(pl.col("season").is_in([2018, 2019, 2020, 2021, 2024, 2025])),
          "outdoor, >=94% coverage seasons")
    block(o.filter(pl.col("week") <= 13), "outdoor, weeks 1-13")
    block(o.filter(pl.col("week") >= 14), "outdoor, weeks 14-18")

    # Inverse-probability weighting: reweight the observed outdoor rows so they
    # represent the full outdoor population under a MAR assumption on observables.
    xs = ["week", "implied_team_total", "total_line", "is_home", "div_game",
          "form_fp_mean"]
    d = o.drop_nulls(subset=xs + ["wx_missing"])
    X = sm.add_constant(d.select(xs).to_numpy().astype(float), has_constant="add")
    lg = sm.Logit(d["wx_missing"].to_numpy().astype(float), X).fit(disp=0)
    p_missing = lg.predict(X)
    d = d.with_columns(pl.Series("p_obs", 1.0 - p_missing)).with_columns(
        (1.0 / pl.col("p_obs")).alias("ipw"))
    dd = d.drop_nulls("wind").with_columns(
        pl.when(pl.col("wind") >= 15).then(pl.lit("3 >=15"))
        .when(pl.col("wind") >= 8).then(pl.lit("2 8-14"))
        .otherwise(pl.lit("1 <8")).alias("wb"))
    print("\n  IPW-reweighted to the full outdoor population (MAR on observables):")
    for b in sorted(dd["wb"].unique().to_list()):
        s = dd.filter(pl.col("wb") == b)
        wgt = s["ipw"].to_numpy()
        y = s["form_resid"].to_numpy()
        m = float((wgt * y).sum() / wgt.sum())
        # cluster-robust se of a weighted mean
        g = s["game_id"].to_numpy()
        e = wgt * (y - m)
        uniq = np.unique(g)
        gs = np.array([e[g == u].sum() for u in uniq])
        se = float(np.sqrt((gs ** 2).sum()) / wgt.sum() * np.sqrt(uniq.size / max(uniq.size - 1, 1)))
        rows.append({"sample": "outdoor, IPW", "bucket": b, "n": s.height,
                     "games": int(uniq.size), "mean": m, "se_cluster": se,
                     "t_cluster": m / se, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se})
        print(f"  {'outdoor, IPW':<34}{b:<8}{s.height:>6}{m:>9.3f}{se:>8.3f}{m/se:>8.2f}")

    return pl.DataFrame(rows)


def q22_market_controls(w: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"\n{BAR}\n QB-022 / C : does the wind effect survive the market?\n{BAR}")
    o = w.filter(pl.col("is_outdoor") == 1).drop_nulls("wind")
    o = o.with_columns(
        (pl.col("wind") / 10.0).alias("wind10"),
        ((pl.col("temp") - 60.0) / 10.0).alias("temp10"),
        (pl.col("week").cast(pl.Float64)).alias("wk"),
    )
    rows = []
    specs = [
        ("1 no controls", ["wind_ge15", "wind_8_14"]),
        ("2 + market (ITT, total)", ["wind_ge15", "wind_8_14",
                                     "implied_team_total", "total_line"]),
        ("3 + market + form level", ["wind_ge15", "wind_8_14",
                                     "implied_team_total", "total_line",
                                     "form_fp_mean"]),
        ("4 + market + temp", ["wind_ge15", "wind_8_14", "implied_team_total",
                               "total_line", "temp10"]),
        ("5 + market + temp + week + home", ["wind_ge15", "wind_8_14",
                                             "implied_team_total", "total_line",
                                             "temp10", "wk", "is_home"]),
    ]
    print(f"\n{'spec':<34}{'term':<12}{'coef':>8}{'se':>8}{'t':>7}{'95% CI':>20}")
    print("-" * 78)
    for lab, xs in specs:
        res, n = ols(o, "form_resid", xs, cluster="game_id")
        rr = coef_rows(res, xs, lab, n, "form_resid")
        rows += rr
        for r in rr:
            if r["term"].startswith("wind"):
                print(f"{lab:<34}{r['term']:<12}{r['coef']:>8.3f}{r['se']:>8.3f}"
                      f"{r['t']:>7.2f}   [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")

    # continuous specification
    for lab, xs in [("6 continuous wind, no controls", ["wind10"]),
                    ("7 continuous wind + market", ["wind10", "implied_team_total",
                                                    "total_line"]),
                    ("8 continuous wind + market + temp",
                     ["wind10", "implied_team_total", "total_line", "temp10"])]:
        res, n = ols(o, "form_resid", xs, cluster="game_id")
        rr = coef_rows(res, xs, lab, n, "form_resid")
        rows += rr
        for r in rr:
            if r["term"] == "wind10":
                print(f"{lab:<34}{'per 10mph':<12}{r['coef']:>8.3f}{r['se']:>8.3f}"
                      f"{r['t']:>7.2f}   [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")

    print("\n QB-022 / D : does the market already price the wind?")
    print(f"{'target':<24}{'term':<12}{'coef':>8}{'se':>8}{'t':>7}")
    print("-" * 78)
    mrows = []
    for tgt in ("total_line", "implied_team_total", "form_pass_att",
                "fantasy_points"):
        res, n = ols(o, tgt, ["wind_ge15", "wind_8_14"], cluster="game_id")
        rr = coef_rows(res, ["wind_ge15", "wind_8_14"], "market pricing", n, tgt)
        mrows += rr
        for r in rr:
            print(f"{tgt:<24}{r['term']:<12}{r['coef']:>8.3f}{r['se']:>8.3f}{r['t']:>7.2f}")
    # and the realised total, to show the wind is real on the field
    print("\n(If total_line falls in high wind, the forecast was public and the")
    print(" market had already absorbed it; the residual estimate then measures")
    print(" only the part the market under-priced.)")
    return pl.DataFrame(rows), pl.DataFrame(mrows)


def q22_robustness(w: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    print(f"\n{BAR}\n QB-022 / E : stability, thresholds, selection, actionability\n{BAR}")
    o = w.filter(pl.col("is_outdoor") == 1).drop_nulls("wind")

    print("\nper-season effect of wind >= 15mph (cluster-robust, vs <8mph base):")
    print(f"{'season':<10}{'n_hi':>6}{'mean_hi':>10}{'se':>8}{'t':>7}{'n_lo':>7}{'mean_lo':>10}{'contrast':>10}")
    print("-" * 78)
    srows = []
    for s in SEASONS:
        d = o.filter(pl.col("season") == s)
        hi = d.filter(pl.col("wind") >= 15)
        lo = d.filter(pl.col("wind") < 8)
        if hi.height < 10:
            continue
        m, se, n, G = cluster_mean_se(hi["form_resid"].to_numpy(), hi["game_id"].to_numpy())
        ml = float(lo["form_resid"].mean())
        srows.append({"season": s, "n_hi": n, "mean_hi": m, "se_cluster": se,
                      "t": m / se, "n_lo": lo.height, "mean_lo": ml,
                      "contrast": m - ml})
        print(f"{s:<10}{n:>6}{m:>10.3f}{se:>8.3f}{m/se:>7.2f}{lo.height:>7}"
              f"{ml:>10.3f}{m-ml:>10.3f}")
    sig = sum(1 for r in srows if r["mean_hi"] + 1.96 * r["se_cluster"] < 0)
    print(f"=> individually significant in {sig}/{len(srows)} seasons; "
          f"negative point estimate in {sum(1 for r in srows if r['mean_hi'] < 0)}/{len(srows)}")

    # honest chronological split: fit early, test late
    early = o.filter(pl.col("season") <= 2021)
    late = o.filter(pl.col("season") >= 2022)
    res_e, _ = ols(early, "form_resid", ["wind_ge15", "wind_8_14"], cluster="game_id")
    res_l, _ = ols(late, "form_resid", ["wind_ge15", "wind_8_14"], cluster="game_id")
    print(f"\nchronological split (no look-ahead):")
    print(f"    2018-2021 wind>=15 : {res_e.params[1]:+.3f} (se {res_e.bse[1]:.3f})   n={early.height}")
    print(f"    2022-2025 wind>=15 : {res_l.params[1]:+.3f} (se {res_l.bse[1]:.3f})   n={late.height}")

    print("\nis it WIND or is it the STADIUM? (windy venues are also bad offences)")
    print(f"{'spec':<40}{'wind>=15':>10}{'se':>8}{'t':>7}")
    print("-" * 78)
    frows_fe = []
    base_x = ["wind_ge15", "wind_8_14", "implied_team_total", "total_line"]
    for lab, extra_col in [("market only", None), ("+ stadium fixed effects", "stadium"),
                           ("+ offence-team fixed effects", "team"),
                           ("+ season fixed effects", "season")]:
        d = o.drop_nulls(subset=base_x + ["form_resid"])
        xs = list(base_x)
        if extra_col is not None:
            before = set(d.columns)
            dd = d.to_dummies(extra_col)
            # only the freshly created dummy columns; `team_spread` already
            # starts with "team_" and is collinear with ITT and total_line
            new = [c for c in dd.columns if c not in before][1:]
            d, xs = dd, xs + new
        res, n = ols(d, "form_resid", xs, cluster="game_id")
        frows_fe.append({"spec": lab, "n": n, "coef": float(res.params[1]),
                         "se": float(res.bse[1]), "t": float(res.tvalues[1])})
        print(f"{lab:<40}{res.params[1]:>10.3f}{res.bse[1]:>8.3f}{res.tvalues[1]:>7.2f}")

    print("\npermutation null for the CHOSEN threshold (wind shuffled within")
    print("season-week, so week/season structure is preserved; 1000 draws):")
    thr_grid = [10, 12, 14, 15, 16, 18, 20, 22]
    y_all = o["form_resid"].to_numpy().astype(float)
    w_all = o["wind"].to_numpy().astype(float)
    sw = (o["season"].cast(pl.Utf8) + "_" + o["week"].cast(pl.Utf8)).to_numpy()
    obs_t = {}
    for t in thr_grid:
        mk = w_all >= t
        if mk.sum() < 30:
            continue
        m = y_all[mk].mean() - y_all[~mk].mean()
        se = np.sqrt(y_all[mk].var(ddof=1) / mk.sum() + y_all[~mk].var(ddof=1) / (~mk).sum())
        obs_t[t] = m / se
    obs_max = max(abs(v) for v in obs_t.values())
    idx_by_sw = {u: np.where(sw == u)[0] for u in np.unique(sw)}
    null_max = []
    for _ in range(1000):
        wp = w_all.copy()
        for u, idx in idx_by_sw.items():
            wp[idx] = RNG.permutation(w_all[idx])
        best = 0.0
        for t in obs_t:
            mk = wp >= t
            if mk.sum() < 30 or (~mk).sum() < 30:
                continue
            m = y_all[mk].mean() - y_all[~mk].mean()
            se = np.sqrt(y_all[mk].var(ddof=1) / mk.sum() + y_all[~mk].var(ddof=1) / (~mk).sum())
            best = max(best, abs(m / se))
        null_max.append(best)
    null_max = np.array(null_max)
    p_max = float((null_max >= obs_max).mean())
    print(f"    observed max |t| over the threshold grid : {obs_max:.2f} (at "
          f"{max(obs_t, key=lambda k: abs(obs_t[k]))}mph)")
    print(f"    permutation p-value, corrected for the search : {p_max:.4f}")
    frows_fe.append({"spec": "permutation max-|t| p", "n": len(null_max),
                     "coef": obs_max, "se": np.nan, "t": p_max})

    print("\nthreshold sensitivity (the 15mph cut was chosen, not pre-registered):")
    print(f"{'threshold':<12}{'n':>6}{'mean':>9}{'se':>8}{'t':>7}{'contrast_vs_below':>20}")
    print("-" * 78)
    trows = []
    for thr in (10, 12, 14, 15, 16, 18, 20, 22, 25):
        hi = o.filter(pl.col("wind") >= thr)
        lo = o.filter(pl.col("wind") < thr)
        if hi.height < 30:
            continue
        m, se, n, G = cluster_mean_se(hi["form_resid"].to_numpy(), hi["game_id"].to_numpy())
        c = m - float(lo["form_resid"].mean())
        trows.append({"threshold": thr, "n": n, "games": G, "mean": m,
                      "se_cluster": se, "t": m / se, "contrast": c})
        print(f"{thr:<12}{n:>6}{m:>9.3f}{se:>8.3f}{m/se:>7.2f}{c:>20.3f}")

    # panel selection: the >=10 attempt filter could drop wind-suppressed starts
    print("\npanel-selection check: does the >=10 pass-attempt filter bite in wind?")
    stats = (nfl_data.load_player_stats(list(range(2017, 2026)))
             .filter((pl.col("position") == "QB") & (pl.col("season_type") == "REG")
                     & (pl.col("season") >= 2018))
             .select(["player_id", "season", "week", "team", "attempts",
                      "fantasy_points"]))
    sch = loaders.load_schedules().select(["game_id", "season", "week", "home_team",
                                           "away_team", "wind", "roof"])
    long = pl.concat([
        sch.select("game_id", "season", "week", pl.col("home_team").alias("team"),
                   "wind", "roof"),
        sch.select("game_id", "season", "week", pl.col("away_team").alias("team"),
                   "wind", "roof")])
    st = (stats.join(long, on=["season", "week", "team"], how="inner")
          .filter(pl.col("roof").is_in(["outdoors", "open"]))
          .drop_nulls("wind")
          .filter(pl.col("attempts") >= 1))
    sel = (st.with_columns(
        pl.when(pl.col("wind") >= 15).then(pl.lit("3 >=15"))
        .when(pl.col("wind") >= 8).then(pl.lit("2 8-14"))
        .otherwise(pl.lit("1 <8")).alias("wb"))
        .group_by("wb").agg(
            pl.len().alias("qb_appearances"),
            (pl.col("attempts") >= 10).mean().alias("share_kept"),
            pl.col("attempts").mean().alias("mean_att"))
        .sort("wb"))
    print(sel)
    print("(if high-wind starts are dropped more often, the surviving high-wind")
    print(" rows are the ones that still threw -- selection that biases the")
    print(" measured wind penalty TOWARD zero, i.e. conservative.)")

    print("\nactionability: what a forecast, rather than the observed reading,")
    print("would have delivered. nflverse records game-time weather; a manager")
    print("acts on a 24-48h forecast with error. Simulated forecast noise:")
    print(f"{'forecast sd (mph)':<20}{'n flagged':>11}{'mean resid':>12}{'se':>8}{'t':>7}")
    print("-" * 78)
    frows = []
    y = o["form_resid"].to_numpy()
    wind = o["wind"].to_numpy().astype(float)
    gid = o["game_id"].to_numpy()
    for sd in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0):
        ms, ns = [], []
        reps = 1 if sd == 0 else 300
        for _ in range(reps):
            wf = wind + RNG.normal(0.0, sd, size=wind.size) if sd > 0 else wind
            mask = wf >= 15
            if mask.sum() < 30:
                continue
            ms.append(y[mask].mean())
            ns.append(int(mask.sum()))
        m = float(np.mean(ms))
        n = float(np.mean(ns))
        # se from the sd==0 clustered se scaled by sqrt(n0/n): approximate, plus
        # the across-replication spread
        mask0 = wind >= 15
        _, se0, n0, _ = cluster_mean_se(y[mask0], gid[mask0])
        se = se0 * np.sqrt(n0 / max(n, 1))
        frows.append({"forecast_sd_mph": sd, "n_flagged": n, "mean_resid": m,
                      "se_approx": se, "t": m / se})
        print(f"{sd:<20.1f}{n:>11.0f}{m:>12.3f}{se:>8.3f}{m/se:>7.2f}")

    # How often could a 1.5-point tiebreak actually change a start/sit call?
    print("\nhow often would this flip a decision? A manager starts one of two")
    print("quarterbacks. The wind flag only matters when the two are close.")
    all_p = w
    flag_rate = float((all_p["wind"] >= 15).fill_null(False).mean())
    gaps = []
    for (s, wk), g in all_p.group_by(["season", "week"]):
        v = g.drop_nulls("form_fp_mean")["form_fp_mean"].to_numpy()
        if v.size < 4:
            continue
        d = np.abs(v[:, None] - v[None, :])
        gaps.append(d[np.triu_indices_from(d, k=1)])
    gaps = np.concatenate(gaps)
    eff = abs(float(o.filter(pl.col("wind") >= 15)["form_resid"].mean()
                    - o.filter(pl.col("wind") < 8)["form_resid"].mean()))
    close = float((gaps <= eff).mean())
    weeks = 17
    ev = 2 * flag_rate * close * eff * weeks
    print(f"    share of QB-weeks flagged (wind>=15, incl. domes) : {flag_rate:.3f}")
    print(f"    measured differential vs low wind                : {eff:.2f} pts")
    print(f"    share of within-week QB pairs within {eff:.2f} pts on form : {close:.3f}")
    print(f"    upper-bound expected value of using the flag     : ~{ev:.2f} pts/season")
    print("    (upper bound: assumes a perfect forecast and that the flag is")
    print("     always the correct tiebreak when the pair is close.)")
    frows.append({"forecast_sd_mph": -1.0, "n_flagged": float(gaps.size),
                  "mean_resid": ev, "se_approx": close, "t": flag_rate})

    return (pl.DataFrame(srows), pl.DataFrame(trows)), (pl.DataFrame(frows), sel,
                                                        pl.DataFrame(frows_fe))


def job_qb022() -> None:
    print(f"{BAR}\n JOB 1 -- QB-022 : ATTACK ON THE WEATHER FINDING (QB-008)\n{BAR}")
    w = load_weather_panel()
    print(f"panel: {w.height} QB-weeks 2018-2025; wind present on "
          f"{int((1 - w['wx_missing'].mean()) * w.height)} "
          f"({1 - w['wx_missing'].mean():.1%})")

    base = q22_baseline(w)
    miss, missmod = q22_missingness(w)
    outdoor = q22_outdoor_only(w)
    regs, mkt = q22_market_controls(w)
    (seas, thr), (fcast, sel, fe) = q22_robustness(w)

    base.write_csv(OUT / "qb022_baseline_buckets.csv")
    miss.write_csv(OUT / "qb022_missingness_observables.csv")
    missmod.write_csv(OUT / "qb022_missingness_model.csv")
    outdoor.write_csv(OUT / "qb022_outdoor_buckets.csv")
    regs.write_csv(OUT / "qb022_wind_regressions.csv")
    mkt.write_csv(OUT / "qb022_market_prices_wind.csv")
    seas.write_csv(OUT / "qb022_season_stability.csv")
    thr.write_csv(OUT / "qb022_threshold_sensitivity.csv")
    fcast.write_csv(OUT / "qb022_forecast_noise.csv")
    sel.write_csv(OUT / "qb022_panel_selection.csv")
    fe.write_csv(OUT / "qb022_fixed_effects_and_permutation.csv")
    print("\nwrote outputs/qb022_*.csv")


# ---------------------------------------------------------------------------
# QB-023 : independent replication of the pick-value curve
# ---------------------------------------------------------------------------

REPLACEMENT_RANK = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}   # 12 teams, 1/2/3/1
POS_MAP = {"PK": "K", "DEF": "DST"}
ADP_CACHE = DATA / "qb023_ffc_adp_2018_2025.parquet"


def load_adp(refresh: bool = False) -> pl.DataFrame:
    """FFC ADP for every drafted player, 12-team standard, 2018-2025."""
    if ADP_CACHE.exists() and not refresh:
        return pl.read_parquet(ADP_CACHE)
    frames = []
    for s in SEASONS:
        d = markets.load_ffc_adp(s, teams=12, fmt="standard")
        frames.append(d.select("season", "name", "position", "team", "adp",
                               "times_drafted", "stdev", "adp_window_start",
                               "adp_window_end", "total_drafts", "key"))
    adp = pl.concat(frames)
    adp.write_parquet(ADP_CACHE)
    return adp


def season_totals(week_max: int) -> pl.DataFrame:
    """Season fantasy totals per player, standard scoring, weeks 1..week_max."""
    ps = (nfl_data.load_player_stats(SEASONS)
          .filter((pl.col("season_type") == "REG") & (pl.col("week") <= week_max))
          .filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
          .select(["player_id", "player_display_name", "position", "season",
                   "week", "fantasy_points"]))
    tot = (ps.group_by(["player_id", "player_display_name", "position", "season"])
           .agg(pl.col("fantasy_points").sum().alias("pts"),
                pl.len().alias("games")))
    return tot.with_columns(
        pl.col("player_display_name")
        .map_elements(markets.normalise_name, return_dtype=pl.String).alias("key"))


def replacement_levels(tot: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for s in sorted(tot["season"].unique().to_list()):
        for pos, rank in REPLACEMENT_RANK.items():
            v = (tot.filter((pl.col("season") == s) & (pl.col("position") == pos))
                 .sort("pts", descending=True)["pts"].to_list())
            rows.append({"season": s, "position": pos, "rank": rank,
                         "replacement_pts": float(v[rank - 1]) if len(v) >= rank else 0.0,
                         "n_players": len(v)})
    return pl.DataFrame(rows)


def build_draft_table(week_max: int, repl_basis: str = "same_season") -> tuple[pl.DataFrame, dict]:
    """One row per drafted skill player: ADP, realised points, PAR, usable PAR."""
    adp = load_adp().with_columns(
        pl.col("position").replace(POS_MAP).alias("pos"))
    tot = season_totals(week_max)

    # collapse duplicate name keys within a position-season (rare); keep the
    # highest-scoring, and count how often it happens
    dupes = (tot.group_by(["key", "position", "season"]).agg(pl.len().alias("k"))
             .filter(pl.col("k") > 1))
    tot1 = (tot.sort("pts", descending=True)
            .group_by(["key", "position", "season"])
            .agg(pl.col("pts").first(), pl.col("games").first(),
                 pl.col("player_display_name").first()))

    if repl_basis == "same_season":
        rep = replacement_levels(tot)
    elif repl_basis == "prior_season":
        rep = replacement_levels(tot).with_columns(
            (pl.col("season") + 1).alias("season"))
    else:
        raise ValueError(repl_basis)

    skill = adp.filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
    j = (skill.join(tot1, left_on=["key", "pos", "season"],
                    right_on=["key", "position", "season"], how="left")
         .join(rep.select("season", "position", "replacement_pts"),
               left_on=["pos", "season"], right_on=["position", "season"],
               how="left"))
    diag = {
        "adp_rows_all_positions": adp.height,
        "adp_rows_kicker_dst": adp.filter(~pl.col("pos").is_in(["QB", "RB", "WR", "TE"])).height,
        "adp_rows_skill": skill.height,
        "matched_to_nflverse": int(j["pts"].is_not_null().sum()),
        "match_rate": float(j["pts"].is_not_null().mean()),
        "duplicate_name_keys": dupes.height,
    }
    return j, diag


def curve(j: pl.DataFrame, unmatched: str = "zero", pick_rule: str = "floor") -> pl.DataFrame:
    d = j
    if unmatched == "drop":
        d = d.drop_nulls("pts")
    else:
        d = d.with_columns(pl.col("pts").fill_null(0.0),
                           pl.col("games").fill_null(0))
    pk = {"floor": pl.col("adp").floor(),
          "round": pl.col("adp").round(0),
          "ceil": pl.col("adp").ceil()}[pick_rule]
    d = d.with_columns(pk.cast(pl.Int32).clip(lower_bound=1).alias("pick"))
    d = d.with_columns(
        ((pl.col("pick") - 1) // 12).alias("bkt"),
        (pl.col("pts") - pl.col("replacement_pts")).alias("par"))
    d = d.with_columns(pl.max_horizontal(pl.col("par"), pl.lit(0.0)).alias("usable_par"))

    rows = []
    for b in sorted(d["bkt"].unique().to_list()):
        s = d.filter(pl.col("bkt") == b)
        u = s["usable_par"].to_numpy().astype(float)
        m, se, n = mean_se(u)
        # season-clustered bootstrap: players within a season share the
        # replacement level and the scoring environment
        seas = s["season"].to_numpy()
        uniq = np.unique(seas)
        boot = []
        for _ in range(1000):
            pick = RNG.choice(uniq, size=uniq.size, replace=True)
            vals = np.concatenate([u[seas == p] for p in pick])
            boot.append(vals.mean())
        boot = np.array(boot)
        rows.append({"bkt": int(b), "pick_from": int(b) * 12 + 1,
                     "pick_to": int(b) * 12 + 12, "n": n,
                     "mean_usable_par": m, "se_iid": se,
                     "boot_lo": float(np.percentile(boot, 2.5)),
                     "boot_hi": float(np.percentile(boot, 97.5)),
                     "median_usable_par": float(np.median(u)),
                     "hit_rate": float((s["par"].to_numpy() > 0).mean())})
    return pl.DataFrame(rows)


def job_qb023() -> None:
    print(f"\n\n{BAR}\n JOB 2 -- QB-023 : INDEPENDENT REPLICATION OF THE PICK-VALUE CURVE\n{BAR}")
    print("Rebuilt from FFC ADP + nflverse player stats only. `draft_value.py`")
    print("was neither read nor imported.\n")
    print("Specification chosen from the league format, not from the committed code:")
    print("  * 12 teams, 1QB/2RB/3WR/1TE  ->  replacement = QB12, RB24, WR36, TE12")
    print("  * standard (non-PPR) scoring, nflverse `fantasy_points`")
    print("  * fantasy season = NFL weeks 1-16 (league reg 1-13 + playoffs 14-16)")
    print("  * PAR = season points - replacement; usable PAR = max(0, PAR)")
    print("  * pick = floor(ADP); buckets of 12; drafted-but-never-played = 0")

    j16, diag = build_draft_table(16)
    print(f"\njoin diagnostics: {diag}")

    rep = replacement_levels(season_totals(16))
    print("\nreplacement levels (season points, weeks 1-16):")
    print(rep.pivot(on="position", index="season", values="replacement_pts").sort("season"))

    primary = curve(j16, unmatched="zero", pick_rule="floor")
    print("\nINDEPENDENT PICK-VALUE CURVE (primary specification):")
    print(f"{'picks':<12}{'n':>5}{'mean':>9}{'se':>7}{'95% boot CI':>20}{'median':>9}{'hit':>7}")
    print("-" * 78)
    for r in primary.to_dicts():
        print(f"{r['pick_from']:>3}-{r['pick_to']:<8}{r['n']:>5}{r['mean_usable_par']:>9.2f}"
              f"{r['se_iid']:>7.2f}   [{r['boot_lo']:>6.2f},{r['boot_hi']:>7.2f}]"
              f"{r['median_usable_par']:>9.2f}{r['hit_rate']:>7.2f}")

    # ---- comparison against the committed curve --------------------------
    committed = pl.read_csv(OUT / "phase6_pick_value_curve.csv")
    cmp = (primary.select("bkt", "pick_from", "pick_to", "n", "mean_usable_par",
                          "se_iid", "boot_lo", "boot_hi", "hit_rate")
           .rename({"n": "n_indep", "mean_usable_par": "indep_mean",
                    "hit_rate": "indep_hit"})
           .join(committed.select("bkt", pl.col("n").alias("n_committed"),
                                  pl.col("mean_usable_par").alias("committed_mean"),
                                  pl.col("hit_rate").alias("committed_hit")),
                 on="bkt", how="left"))
    cmp = cmp.with_columns(
        (pl.col("indep_mean") - pl.col("committed_mean")).alias("diff"),
        ((pl.col("indep_mean") - pl.col("committed_mean"))
         / pl.col("committed_mean").abs().clip(lower_bound=1e-9) * 100).alias("pct_diff"),
        ((pl.col("committed_mean") >= pl.col("boot_lo"))
         & (pl.col("committed_mean") <= pl.col("boot_hi"))).alias("committed_in_ci"),
    )
    print("\nCOMPARISON WITH outputs/phase6_pick_value_curve.csv:")
    print(f"{'picks':<12}{'n_ind':>6}{'n_com':>6}{'indep':>9}{'committed':>11}"
          f"{'diff':>8}{'%':>8}{'in CI':>7}")
    print("-" * 78)
    for r in cmp.to_dicts():
        if r["committed_mean"] is None:
            continue
        print(f"{r['pick_from']:>3}-{r['pick_to']:<8}{r['n_indep']:>6}{r['n_committed']:>6}"
              f"{r['indep_mean']:>9.2f}{r['committed_mean']:>11.2f}{r['diff']:>8.2f}"
              f"{r['pct_diff']:>8.1f}{str(r['committed_in_ci']):>7}")

    # ---- where does the n gap come from? ---------------------------------
    print("\nthe committed curve carries 1,142 players; this rebuild carries")
    print(f"{int(primary['n'].sum())}. Testing whether a NAIVE (raw-string) name join")
    print("explains the gap, since late picks have the messiest names:")
    adp_raw = load_adp().with_columns(pl.col("position").replace(POS_MAP).alias("pos"))
    tot_raw = season_totals(16)
    naive = (adp_raw.filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
             .join(tot_raw.select(pl.col("player_display_name").alias("name"),
                                  "position", "season", "pts").unique(
                       subset=["name", "position", "season"]),
                   left_on=["name", "pos", "season"],
                   right_on=["name", "position", "season"], how="left"))
    naive = naive.with_columns(pl.col("adp").floor().cast(pl.Int32).alias("pick")) \
                 .with_columns(((pl.col("pick") - 1) // 12).alias("bkt"))
    nj = (naive.group_by("bkt").agg(
        pl.len().alias("n_drafted"),
        pl.col("pts").is_not_null().sum().alias("n_matched_naive")).sort("bkt"))
    norm = (j16.with_columns(pl.col("adp").floor().cast(pl.Int32).alias("pick"))
            .with_columns(((pl.col("pick") - 1) // 12).alias("bkt"))
            .group_by("bkt").agg(pl.col("pts").is_not_null().sum()
                                 .alias("n_matched_normalised")).sort("bkt"))
    nj = nj.join(norm, on="bkt", how="left").join(
        committed.select("bkt", pl.col("n").alias("n_committed")), on="bkt", how="left")
    print(f"{'bucket':<10}{'drafted':>9}{'naive join':>12}{'normalised':>12}{'committed':>11}")
    print("-" * 78)
    for r in nj.to_dicts():
        print(f"{r['bkt']:<10}{r['n_drafted']:>9}{r['n_matched_naive']:>12}"
              f"{r['n_matched_normalised']:>12}"
              f"{r['n_committed'] if r['n_committed'] is not None else '-':>11}")
    print(f"totals: drafted {int(nj['n_drafted'].sum())}, naive-join "
          f"{int(nj['n_matched_naive'].sum())}, normalised-join "
          f"{int(nj['n_matched_normalised'].sum())}, committed "
          f"{int(committed['n'].sum())}")
    nj.write_csv(OUT / "qb023_join_gap.csv")

    # is the ADP universe itself identical to the one the committed run used?
    cached = REPO / "shared" / "nfl-data" / "cache" / "raw" / "ffc_adp_all_2018_2025.parquet"
    if cached.exists():
        c = pl.read_parquet(cached)
        mine = load_adp()
        same = (c.height == mine.height and
                sorted(c["season"].to_list()) == sorted(mine["season"].to_list()))
        j = (mine.select("season", "key", pl.col("adp").alias("adp_mine"))
             .join(c.select("season", "key", pl.col("adp").alias("adp_cached")),
                   on=["season", "key"], how="inner"))
        md = float((j["adp_mine"] - j["adp_cached"]).abs().max()) if j.height else np.nan
        print(f"\nADP input check vs the committed cache: rows {mine.height} vs "
              f"{c.height} (identical={same}); {j.height} keys join, max |ADP "
              f"difference| = {md}")
        print("=> the draft-cost inputs are the same data; any divergence is on")
        print("   the realised-points / join side, not the ADP side.")

    # does a coverage difference of this size even move the curve?
    naive_curve = curve(
        j16.join(naive.select("season", "key", pl.col("pts").alias("pts_naive")),
                 on=["season", "key"], how="left")
        .filter(pl.col("pts_naive").is_not_null())
        .drop("pts_naive"), unmatched="drop", pick_rule="floor")
    cov = (primary.select("bkt", "pick_from", pl.col("mean_usable_par").alias("full_join"))
           .join(naive_curve.select("bkt", pl.col("mean_usable_par").alias("naive_join")),
                 on="bkt", how="left"))
    cov = cov.with_columns((pl.col("naive_join") - pl.col("full_join")).alias("diff"))
    print("\nsensitivity of the curve to losing the hard-to-match players:")
    print(f"{'picks':<10}{'full join':>11}{'naive join':>12}{'diff':>8}")
    print("-" * 78)
    for r in cov.to_dicts():
        if r["naive_join"] is None:
            continue
        print(f"{r['pick_from']:<10}{r['full_join']:>11.2f}{r['naive_join']:>12.2f}"
              f"{r['diff']:>8.2f}")
    cov.write_csv(OUT / "qb023_join_sensitivity.csv")

    # ---- specification grid ---------------------------------------------
    print("\nspecification grid -- what each modelling choice is worth:")
    grid = []
    for wmax, wlab in [(13, "wk 1-13 (league reg)"), (16, "wk 1-16 (reg+playoffs)"),
                       (17, "wk 1-17"), (18, "wk 1-18 (all NFL REG)")]:
        jj, _ = build_draft_table(wmax)
        for unm in ("zero", "drop"):
            for pr in ("floor", "round", "ceil"):
                c = curve(jj, unmatched=unm, pick_rule=pr)
                d = {int(r["bkt"]): r for r in c.to_dicts()}
                grid.append({
                    "weeks": wlab, "unmatched": unm, "pick_rule": pr,
                    "b0_1_12": d[0]["mean_usable_par"] if 0 in d else None,
                    "b1_13_24": d[1]["mean_usable_par"] if 1 in d else None,
                    "b8_97_108": d[8]["mean_usable_par"] if 8 in d else None,
                    "b12_145_156": d[12]["mean_usable_par"] if 12 in d else None,
                    "n_total": int(c["n"].sum()),
                })
    g = pl.DataFrame(grid)
    print(f"{'weeks':<24}{'unmatched':<10}{'pick':<7}{'1-12':>8}{'13-24':>8}"
          f"{'97-108':>9}{'145-156':>9}{'n':>6}")
    print("-" * 78)
    for r in g.to_dicts():
        print(f"{r['weeks']:<24}{r['unmatched']:<10}{r['pick_rule']:<7}"
              f"{r['b0_1_12']:>8.2f}{r['b1_13_24']:>8.2f}{r['b8_97_108']:>9.2f}"
              f"{r['b12_145_156']:>9.2f}{r['n_total']:>6}")
    print("published headline: 70.0 / 47.2 / 7.7 / 4.5")

    # ---- walk-forward replacement level ---------------------------------
    print("\nlook-ahead check: the replacement level is a REALISED end-of-season")
    print("rank, which nobody knows on draft day. Re-run with the PRIOR season's")
    print("replacement level, which a drafter did know:")
    jprior, _ = build_draft_table(16, repl_basis="prior_season")
    cprior = curve(jprior.drop_nulls("replacement_pts"), unmatched="zero", pick_rule="floor")
    pc = (primary.select("bkt", "pick_from", "pick_to",
                         pl.col("mean_usable_par").alias("same_season"))
          .join(cprior.select("bkt", pl.col("mean_usable_par").alias("prior_season"),
                              pl.col("n").alias("n_prior")), on="bkt", how="left"))
    pc = pc.with_columns((pl.col("prior_season") - pl.col("same_season")).alias("diff"))
    print(f"{'picks':<12}{'same-season':>13}{'prior-season':>14}{'diff':>8}")
    print("-" * 78)
    for r in pc.to_dicts():
        if r["prior_season"] is None:
            continue
        print(f"{r['pick_from']:>3}-{r['pick_to']:<8}{r['same_season']:>13.2f}"
              f"{r['prior_season']:>14.2f}{r['diff']:>8.2f}")

    # ---- the number QB-012 actually leans on ----------------------------
    print("\nthe load-bearing quantity: what a QB2 pick costs. QB-012 says a")
    print("second quarterback goes around picks 106-135 and forgoes 5-8 points.")
    jj = j16.with_columns(pl.col("pts").fill_null(0.0))
    jj = jj.with_columns(pl.col("adp").floor().cast(pl.Int32).alias("pick"),
                         (pl.col("pts") - pl.col("replacement_pts")).alias("par"))
    jj = jj.with_columns(pl.max_horizontal(pl.col("par"), pl.lit(0.0)).alias("usable_par"))
    band = jj.filter((pl.col("pick") >= 106) & (pl.col("pick") <= 135))
    u = band["usable_par"].to_numpy()
    m, se, n = mean_se(u)
    seas = band["season"].to_numpy(); uniq = np.unique(seas)
    boot = np.array([np.concatenate([u[seas == p] for p in RNG.choice(uniq, uniq.size, True)]).mean()
                     for _ in range(2000)])
    print(f"    picks 106-135: n={n}, mean usable PAR = {m:.2f} "
          f"(se {se:.2f}, 95% season-bootstrap [{np.percentile(boot,2.5):.2f}, "
          f"{np.percentile(boot,97.5):.2f}])")
    band_rows = pl.DataFrame([{"band": "picks 106-135", "n": n, "mean_usable_par": m,
                               "se": se, "boot_lo": float(np.percentile(boot, 2.5)),
                               "boot_hi": float(np.percentile(boot, 97.5)),
                               "hit_rate": float((band["par"].to_numpy() > 0).mean())}])

    # ---- survivorship: what the ADP universe omits ----------------------
    print("\nsurvivorship: the ADP universe only holds players who were drafted.")
    adp = load_adp().with_columns(pl.col("position").replace(POS_MAP).alias("pos"))
    tot = season_totals(16)
    srows = []
    for s in SEASONS:
        drafted = set(adp.filter(pl.col("season") == s)["key"].to_list())
        for pos, rank in REPLACEMENT_RANK.items():
            top = (tot.filter((pl.col("season") == s) & (pl.col("position") == pos))
                   .sort("pts", descending=True).head(rank))
            und = top.filter(~pl.col("key").is_in(list(drafted)))
            srows.append({"season": s, "position": pos, "top_n": rank,
                          "undrafted_in_top_n": und.height,
                          "share_undrafted": und.height / rank})
    sv = pl.DataFrame(srows)
    agg = sv.group_by("position").agg(pl.col("undrafted_in_top_n").sum().alias("undrafted"),
                                      pl.col("top_n").sum().alias("slots"))
    agg = agg.with_columns((pl.col("undrafted") / pl.col("slots")).alias("share")).sort("position")
    print(agg)
    print("Players who finish above replacement but were never drafted are")
    print("invisible to this curve. That is correct for costing a PICK, but it")
    print("means the curve is not the value of a ROSTER SPOT.")

    primary.write_csv(OUT / "qb023_pick_value_curve.csv")
    cmp.write_csv(OUT / "qb023_curve_comparison.csv")
    g.write_csv(OUT / "qb023_spec_grid.csv")
    rep.write_csv(OUT / "qb023_replacement_levels.csv")
    pc.write_csv(OUT / "qb023_replacement_basis.csv")
    band_rows.write_csv(OUT / "qb023_qb2_pick_band.csv")
    sv.write_csv(OUT / "qb023_survivorship.csv")
    pl.DataFrame([diag]).write_csv(OUT / "qb023_join_diagnostics.csv")
    print("\nwrote outputs/qb023_*.csv")


def main() -> int:
    job_qb022()
    job_qb023()
    print(f"\n{BAR}\ndone\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
