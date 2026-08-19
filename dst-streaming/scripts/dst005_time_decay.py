"""DST-005: how fast does knowledge of team quality lose predictive value?

Two decay questions, both answered from data rather than assumed:

1. PRESEASON knowledge. A rating frozen before week 1 of season S, built only
   from season S-1, predicting D/ST outcomes in week 1, week 2, ... week 17.
2. IN-SEASON knowledge. A k-game rating ending g games back, predicting the
   game g games later.

Everything is oriented so that a HIGHER rating should mean MORE D/ST points, so
every curve is expected positive and a decay is a fall toward zero. Uncertainty
comes from a cluster bootstrap over team-seasons, because all of one team's
weeks share one preseason rating and are not independent draws.

The contemporaneous market (implied opponent total, DST-002's only surviving
predictor) is drawn as a flat reference: it is re-priced every week and so has
no decay to measure. Caveat stated in the write-up - nflverse carries closing
lines, not the August lookahead lines a drafter actually sees.
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
from scipy.optimize import curve_fit                # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import dst_data                                     # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_width_chars(150)

SEASONS = list(range(2018, 2026))
MAX_WEEK = 17          # 2018-2020 are 17-week seasons; week 18 has only 5 seasons
GAPS = list(range(1, 11))
ROLL_K = 4             # in-season rating window, ~a month of football
N_BOOT = 400
RNG = np.random.default_rng(20260818)
BAR = "=" * 84


# ----------------------------------------------------------------- utilities
def pair_acc(pred: np.ndarray, act: np.ndarray) -> tuple[int, int]:
    """Correct / comparable within-week pairs. The start-sit decision metric."""
    if len(pred) < 2:
        return 0, 0
    dp = np.subtract.outer(pred, pred)
    da = np.subtract.outer(act, act)
    iu = np.triu_indices(len(pred), 1)
    dp, da = dp[iu], da[iu]
    ok = (dp != 0) & (da != 0)
    return int((np.sign(dp[ok]) == np.sign(da[ok])).sum()), int(ok.sum())


def wcorr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def boot_ci(vals: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if len(v) < 20:
        return np.nan, np.nan
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# ----------------------------------------------------------------- decay fits
def _exp0(w, r0, tau):
    return r0 * np.exp(-(w - 1.0) / tau)


def _expa(w, r0, tau, a):
    return a + (r0 - a) * np.exp(-(w - 1.0) / tau)


def _lin(w, r0, b):
    return r0 + b * (w - 1.0)


def fit_half_life(w: np.ndarray, r: np.ndarray, n: np.ndarray) -> dict:
    """Weighted fit of r(w). Returns half-life of the decaying part, in weeks."""
    m = np.isfinite(r) & np.isfinite(w)
    w, r, n = w[m], r[m], n[m]
    if len(w) < 5:
        return {"half_life": np.nan}
    sig = 1.0 / np.sqrt(np.maximum(n, 1))          # SE of a correlation ~ 1/sqrt(n)
    out = {}
    try:
        p0 = [max(r[0], 0.05), 5.0]
        po, _ = curve_fit(_exp0, w, r, p0=p0, sigma=sig, maxfev=20000,
                          bounds=([-1, 0.05], [1, 500]))
        out["exp0"] = po
        out["sse_exp0"] = float(np.sum(((r - _exp0(w, *po)) / sig) ** 2))
    except Exception:
        out["exp0"], out["sse_exp0"] = None, np.inf
    try:
        po, _ = curve_fit(_expa, w, r, p0=[max(r[0], 0.05), 5.0, 0.0], sigma=sig,
                          maxfev=20000, bounds=([-1, 0.05, -1], [1, 500, 1]))
        out["expa"] = po
        out["sse_expa"] = float(np.sum(((r - _expa(w, *po)) / sig) ** 2))
    except Exception:
        out["expa"], out["sse_expa"] = None, np.inf
    po = np.polyfit(w - 1.0, r, 1, w=1.0 / sig)
    out["lin"] = np.array([po[1], po[0]])
    out["sse_lin"] = float(np.sum(((r - _lin(w, po[1], po[0])) / sig) ** 2))
    out["sse_const"] = float(np.sum(((r - np.average(r, weights=1 / sig ** 2)) / sig) ** 2))

    # AIC on the weighted SSE, k params + 1 for scale
    for key, k in (("exp0", 2), ("expa", 3), ("lin", 2), ("const", 1)):
        out[f"aic_{key}"] = out[f"sse_{key}"] + 2 * k
    out["best"] = min(("exp0", "expa", "lin", "const"), key=lambda k: out[f"aic_{k}"])

    if out["exp0"] is not None:
        out["r0"] = float(out["exp0"][0])
        out["half_life"] = float(out["exp0"][1] * np.log(2.0))
    else:
        out["r0"], out["half_life"] = np.nan, np.nan
    return out


# ------------------------------------------------------------ data assembly
def build(d: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Defense ledger with frozen preseason ratings, plus an offence ledger."""
    d = d.sort(["team", "season", "week"])

    # prior-season team ratings, computed on season S-1 only
    per = d.group_by(["team", "season"]).agg(
        pl.col("dst_points").mean().alias("dst_pg"),
        pl.col("points_allowed").mean().alias("pa_pg"),
        pl.col("points_for").mean().alias("pf_pg"),
        pl.len().alias("gp"))
    prior = per.with_columns((pl.col("season") + 1).alias("season")).select(
        "team", "season",
        pl.col("dst_pg").alias("_p_dst"),
        pl.col("pa_pg").alias("_p_pa"),
        pl.col("pf_pg").alias("_p_pf"),
        pl.col("gp").alias("_p_gp"))

    d = (d.join(prior, on=["team", "season"], how="left")
           .join(prior.rename({"team": "opponent"}), on=["opponent", "season"],
                 how="left", suffix="_o"))

    d = d.with_columns(
        # every rating oriented so HIGHER = expect MORE D/ST points
        pl.col("_p_dst").alias("pre_def_dst"),          # own prior D/ST pts/gm
        (-pl.col("_p_pa")).alias("pre_def_sting"),      # own prior stinginess
        (-pl.col("_p_pf_o")).alias("pre_opp_weak"),     # opponent prior offence
        (-pl.col("implied_opp_total")).alias("mkt"),    # re-priced every week
        (pl.col("season").cast(pl.Utf8) + "_" + pl.col("team")).alias("unit"),
    )

    # per-team game index within a season (bye weeks make week != game number)
    d = d.with_columns(pl.int_range(pl.len()).over(["team", "season"]).alias("gidx"))

    off = (d.select(pl.col("opponent").alias("unit_team"), "season", "week",
                    pl.col("points_allowed").alias("off_pts"),
                    pl.col("dst_points").alias("conceded_dst"))
             .sort(["unit_team", "season", "week"])
             .with_columns(pl.int_range(pl.len())
                           .over(["unit_team", "season"]).alias("gidx"),
                           (pl.col("season").cast(pl.Utf8) + "_"
                            + pl.col("unit_team")).alias("unit")))
    return d, off


# ------------------------------------------------------- preseason decay curve
def preseason_curves(d: pl.DataFrame, ratings: list[str]) -> dict:
    """r(week) for each frozen preseason rating, with a cluster bootstrap."""
    sub = d.filter(pl.col("_p_gp").is_not_null() & (pl.col("week") <= MAX_WEEK))
    sub = sub.filter(pl.col("pre_opp_weak").is_not_null())
    y = sub["dst_points"].to_numpy().astype(float)
    wk = sub["week"].to_numpy()
    units, uidx = np.unique(sub["unit"].to_numpy(), return_inverse=True)
    R = {c: sub[c].to_numpy().astype(float) for c in ratings}

    weeks = np.arange(1, MAX_WEEK + 1)
    wmask = [wk == w for w in weeks]
    n_wk = np.array([m.sum() for m in wmask], float)

    point = {c: np.array([wcorr(R[c][m], y[m]) for m in wmask]) for c in ratings}

    # cluster bootstrap over team-seasons
    order = np.argsort(uidx, kind="stable")
    starts = np.searchsorted(uidx[order], np.arange(len(units)))
    ends = np.append(starts[1:], len(order))
    boots = {c: np.full((N_BOOT, len(weeks)), np.nan) for c in ratings}
    hl = {c: np.full(N_BOOT, np.nan) for c in ratings}
    for b in range(N_BOOT):
        pick = RNG.integers(0, len(units), len(units))
        idx = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        bwk, by = wk[idx], y[idx]
        bm = [bwk == w for w in weeks]
        bn = np.array([m.sum() for m in bm], float)
        for c in ratings:
            rc = R[c][idx]
            row = np.array([wcorr(rc[m], by[m]) for m in bm])
            boots[c][b] = row
            hl[c][b] = fit_half_life(weeks, row, bn)["half_life"]
    return {"weeks": weeks, "n": n_wk, "point": point, "boot": boots,
            "hl_boot": hl, "sub": sub, "ratings": ratings}


def bucket_table(sub: pl.DataFrame, ratings: list[str],
                 buckets: list[tuple[int, int]]) -> list[dict]:
    """Correlation and within-week pair accuracy per week bucket, bootstrapped.

    Correlation resamples team-seasons (one frozen rating per cluster); pair
    accuracy resamples season-weeks, because a week is one whole ranking
    problem and pairs inside it are not independent.
    """
    y = sub["dst_points"].to_numpy().astype(float)
    wk = sub["week"].to_numpy()
    units, uidx = np.unique(sub["unit"].to_numpy(), return_inverse=True)
    R = {c: sub[c].to_numpy().astype(float) for c in ratings}
    sw = (sub["season"].cast(pl.Utf8) + "_" + sub["week"].cast(pl.Utf8)).to_numpy()
    _, swidx = np.unique(sw, return_inverse=True)
    sw_groups = [np.where(swidx == g)[0] for g in range(swidx.max() + 1)]

    order = np.argsort(uidx, kind="stable")
    starts = np.searchsorted(uidx[order], np.arange(len(units)))
    ends = np.append(starts[1:], len(order))

    rows = []
    for lo, hi in buckets:
        m = (wk >= lo) & (wk <= hi)
        keep = [g for g in sw_groups if len(g) > 1 and m[g[0]]]
        for c in ratings:
            counts = np.array([pair_acc(R[c][g], y[g]) for g in keep], float)
            a_pt = counts[:, 0].sum() / counts[:, 1].sum()
            ab = np.empty(400)
            for b in range(400):
                p = RNG.integers(0, len(keep), len(keep))
                ab[b] = counts[p, 0].sum() / counts[p, 1].sum()
            rb = np.empty(400)
            for b in range(400):
                pick = RNG.integers(0, len(units), len(units))
                idx = np.concatenate([order[starts[p]:ends[p]] for p in pick])
                idx = idx[m[idx]]
                rb[b] = wcorr(R[c][idx], y[idx])
            rlo, rhi = boot_ci(rb)
            alo, ahi = boot_ci(ab)
            rows.append({"bucket": f"W{lo}-{hi}", "rating": c, "n": int(m.sum()),
                         "r": wcorr(R[c][m], y[m]), "r_lo": rlo, "r_hi": rhi,
                         "pair_acc": a_pt, "acc_lo": alo, "acc_hi": ahi})
    return rows


# -------------------------------------------------------- in-season decay
def lag_curve(df: pl.DataFrame, rating_col: str, target_col: str,
              k: int, label: str) -> dict:
    """r between a k-game rating ending at game t and the outcome at game t+g."""
    df = df.sort(["unit", "gidx"])
    units = df["unit"].to_numpy()
    gi = df["gidx"].to_numpy()
    rate = df[rating_col].to_numpy().astype(float)
    targ = df[target_col].to_numpy().astype(float)
    uu, uidx = np.unique(units, return_inverse=True)

    # per-unit arrays
    per_unit = [[] for _ in uu]
    for i in range(len(units)):
        per_unit[uidx[i]].append(i)

    pairs = {g: ([], [], []) for g in GAPS}   # rating, target, unit id
    for u, rows in enumerate(per_unit):
        rows = np.array(rows)
        rr, tt = rate[rows], targ[rows]
        n = len(rows)
        csum = np.concatenate([[0.0], np.cumsum(rr)])
        for t in range(k - 1, n):
            rating = (csum[t + 1] - csum[t + 1 - k]) / k
            for g in GAPS:
                if t + g < n:
                    pairs[g][0].append(rating)
                    pairs[g][1].append(tt[t + g])
                    pairs[g][2].append(u)

    r_pt, n_pt, arrs = [], [], {}
    for g in GAPS:
        rr = np.array(pairs[g][0])
        tt = np.array(pairs[g][1])
        uid = np.array(pairs[g][2])
        o = np.argsort(uid, kind="stable")
        st = np.searchsorted(uid[o], np.arange(len(uu)))
        en = np.append(st[1:], len(o))
        arrs[g] = (rr, tt, o, st, en)
        r_pt.append(wcorr(rr, tt))
        n_pt.append(len(rr))
    r_pt, n_pt = np.array(r_pt), np.array(n_pt, float)

    boot = np.full((N_BOOT, len(GAPS)), np.nan)
    hl = np.full(N_BOOT, np.nan)
    gaps = np.array(GAPS, float)
    for b in range(N_BOOT):
        pick = RNG.integers(0, len(uu), len(uu))
        row = np.empty(len(GAPS))
        for j, g in enumerate(GAPS):
            rr, tt, o, st, en = arrs[g]
            sel = np.concatenate([o[st[p]:en[p]] for p in pick])
            row[j] = wcorr(rr[sel], tt[sel])
        boot[b] = row
        hl[b] = fit_half_life(gaps, row, n_pt)["half_life"]
    return {"label": label, "gaps": gaps, "r": r_pt, "n": n_pt,
            "boot": boot, "hl_boot": hl, "fit": fit_half_life(gaps, r_pt, n_pt)}


def contrast(sub: pl.DataFrame, ratings: list[str],
             a: tuple[int, int], b: tuple[int, int]) -> list[dict]:
    """Form-free version of the decay question: r in window A minus window B.

    Does not assume any curve shape, so it survives whatever the exponential
    fit does. Same team-season cluster bootstrap.
    """
    y = sub["dst_points"].to_numpy().astype(float)
    wk = sub["week"].to_numpy()
    units, uidx = np.unique(sub["unit"].to_numpy(), return_inverse=True)
    order = np.argsort(uidx, kind="stable")
    starts = np.searchsorted(uidx[order], np.arange(len(units)))
    ends = np.append(starts[1:], len(order))
    ma = (wk >= a[0]) & (wk <= a[1])
    mb = (wk >= b[0]) & (wk <= b[1])
    rows = []
    for c in ratings:
        x = sub[c].to_numpy().astype(float)
        pt = wcorr(x[ma], y[ma]) - wcorr(x[mb], y[mb])
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = RNG.integers(0, len(units), len(units))
            idx = np.concatenate([order[starts[p]:ends[p]] for p in pick])
            diffs[i] = (wcorr(x[idx[ma[idx]]], y[idx[ma[idx]]])
                        - wcorr(x[idx[mb[idx]]], y[idx[mb[idx]]]))
        lo, hi = boot_ci(diffs)
        rows.append({"rating": c, "r_early": wcorr(x[ma], y[ma]),
                     "r_late": wcorr(x[mb], y[mb]), "diff": pt,
                     "diff_lo": lo, "diff_hi": hi,
                     "p_diff_gt0": float(np.mean(diffs > 0))})
    return rows


def fmt_hl(hl: float, lo: float, hi: float) -> str:
    def one(x):
        if not np.isfinite(x):
            return "n/a"
        return ">50" if x > 50 else f"{x:.1f}"
    return f"{one(hl)} wk  [{one(lo)}, {one(hi)}]"


# ------------------------------------------------------------------- main
def main() -> int:
    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    d_raw = dst_data.build_dst_panel(SEASONS)
    d, off = build(d_raw)
    print(f"panel: {d.height} defense-weeks, {d['season'].n_unique()} seasons")

    ratings = ["pre_def_dst", "pre_def_sting", "pre_opp_weak", "mkt"]
    NICE = {"pre_def_dst": "preseason own D/ST pts/gm (prior season)",
            "pre_def_sting": "preseason own stinginess (-prior pts allowed)",
            "pre_opp_weak": "preseason opponent offence (-prior PPG)",
            "mkt": "market this week (-implied opp total)"}

    print(f"\n{BAR}\n DST-005a: PRESEASON KNOWLEDGE, WEEK BY WEEK\n{BAR}")
    pre = preseason_curves(d, ratings)
    weeks, n_wk = pre["weeks"], pre["n"]
    hdr = (f"{'week':>5}{'n':>6}"
           + "".join(f"{c:>22}" for c in ratings))
    print(hdr)
    print("-" * len(hdr))
    for i, w in enumerate(weeks):
        line = f"{w:>5}{int(n_wk[i]):>6}"
        for c in ratings:
            lo, hi = boot_ci(pre["boot"][c][:, i])
            line += f"{pre['point'][c][i]:>+7.3f} [{lo:+.2f},{hi:+.2f}]".rjust(22)
        print(line)
    print("\n(cell: r between the frozen rating and that week's D/ST points, with a")
    print(" 95% cluster-bootstrap CI over team-seasons; 'mkt' is re-priced weekly)")

    print(f"\n{BAR}\n DST-005a: HALF-LIFE OF PRESEASON KNOWLEDGE\n{BAR}")
    print(f"{'rating':<46}{'r(week1)':>10}{'half-life (95% CI)':>26}"
          f"{'P(HL<=6wk)':>12}   best form")
    print("-" * 108)
    hl_rows = []
    for c in ratings:
        f = fit_half_life(weeks, pre["point"][c], n_wk)
        lo, hi = boot_ci(pre["hl_boot"][c])
        r1lo, r1hi = boot_ci(pre["boot"][c][:, 0])
        hb = pre["hl_boot"][c]
        p6 = float(np.mean(hb[np.isfinite(hb)] <= 6.0))
        print(f"{NICE[c]:<46}{pre['point'][c][0]:>+10.3f}"
              f"{fmt_hl(f['half_life'], lo, hi):>26}{p6:>12.2f}   {f['best']}")
        hl_rows.append({"rating": c, "r_week1": pre["point"][c][0],
                        "r_week1_lo": r1lo, "r_week1_hi": r1hi,
                        "half_life": f["half_life"], "hl_lo": lo, "hl_hi": hi,
                        "p_hl_le_6": p6, "best_form": f["best"],
                        "aic_exp": f["aic_exp0"], "aic_const": f["aic_const"],
                        "aic_lin": f["aic_lin"]})
    print("\nbest form = lowest AIC among {exp-to-zero, exp-to-asymptote, linear,")
    print("constant}. 'constant' winning means no decay was detected at all;")
    print("a half-life quoted next to a near-zero week-1 r is a half-life of nothing.")

    print(f"\n{BAR}\n DST-005a: FORM-FREE CONTRAST, WEEKS 1-6 vs 7-17\n{BAR}")
    con = contrast(pre["sub"], ratings, (1, 6), (7, 17))
    print(f"{'rating':<18}{'r(W1-6)':>10}{'r(W7-17)':>11}{'drop':>9}"
          f"{'95% CI on drop':>22}{'P(drop>0)':>11}")
    print("-" * 84)
    for r in con:
        print(f"{r['rating']:<18}{r['r_early']:>+10.3f}{r['r_late']:>+11.3f}"
              f"{r['diff']:>+9.3f}   [{r['diff_lo']:+.3f},{r['diff_hi']:+.3f}]"
              f"{r['p_diff_gt0']:>11.3f}")
    print("\nNo functional form assumed here. If the drop is not clearly positive,")
    print("the decay estimate is not resolvable at this sample size and should be")
    print("reported as such rather than as a precise half-life.")

    print(f"\n{BAR}\n DST-005a: BUCKETED - IS WEEKS 1-6 THE RIGHT WINDOW?\n{BAR}")
    buckets = [(1, 3), (4, 6), (7, 10), (11, 17)]
    brows = bucket_table(pre["sub"], ratings, buckets)
    print(f"{'bucket':<9}{'rating':<16}{'n':>6}{'r':>9}{'95% CI':>18}"
          f"{'pair acc':>11}{'95% CI':>18}")
    print("-" * 90)
    for r in brows:
        print(f"{r['bucket']:<9}{r['rating']:<16}{r['n']:>6}{r['r']:>+9.3f}"
              f"   [{r['r_lo']:+.3f},{r['r_hi']:+.3f}]"
              f"{r['pair_acc'] * 100:>10.2f}%   [{r['acc_lo'] * 100:.2f},"
              f"{r['acc_hi'] * 100:.2f}]")

    print(f"\n{BAR}\n DST-005b: IN-SEASON KNOWLEDGE, N GAMES AGO\n{BAR}")
    curves = [
        lag_curve(d, "dst_points", "dst_points", ROLL_K,
                  f"own defense: {ROLL_K}-game D/ST rating"),
        lag_curve(d, "dst_points", "dst_points", 1,
                  "own defense: single-game D/ST rating"),
        lag_curve(off, "off_pts", "conceded_dst", ROLL_K,
                  f"opponent offence: {ROLL_K}-game PPG rating"),
    ]
    print(f"{'gap (games)':<14}" + "".join(f"{c['label'][:22]:>24}" for c in curves))
    print("-" * 90)
    for j, g in enumerate(GAPS):
        line = f"{int(g):<14}"
        for c in curves:
            lo, hi = boot_ci(c["boot"][:, j])
            line += f"{c['r'][j]:>+8.3f} [{lo:+.2f},{hi:+.2f}]".rjust(24)
        print(line)
    print(f"\n{'curve':<46}{'r(gap=1)':>12}{'half-life (95% CI)':>28}   best form")
    print("-" * 100)
    for c in curves:
        lo, hi = boot_ci(c["hl_boot"])
        print(f"{c['label']:<46}{c['r'][0]:>+8.3f}     "
              f"{fmt_hl(c['fit']['half_life'], lo, hi):>28}   {c['fit']['best']}")
        hl_rows.append({"rating": "inseason::" + c["label"], "r_week1": c["r"][0],
                        "r_week1_lo": boot_ci(c["boot"][:, 0])[0],
                        "r_week1_hi": boot_ci(c["boot"][:, 0])[1],
                        "half_life": c["fit"]["half_life"], "hl_lo": lo, "hl_hi": hi,
                        "best_form": c["fit"]["best"],
                        "aic_exp": c["fit"]["aic_exp0"],
                        "aic_const": c["fit"]["aic_const"],
                        "aic_lin": c["fit"]["aic_lin"]})
    print("\nNote: opponent-offence conceded-D/ST is the sign-flipped analogue - a")
    print("HIGHER opponent PPG rating means FEWER D/ST points conceded, so a")
    print("negative r there is the expected direction and |r| is the signal.")

    # ------------------------------------------------------------- plot
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    cols = {"pre_def_dst": "#1f77b4", "pre_def_sting": "#2ca02c",
            "pre_opp_weak": "#d62728", "mkt": "#7f7f7f"}
    for c in ratings:
        lo = np.array([boot_ci(pre["boot"][c][:, i])[0] for i in range(len(weeks))])
        hi = np.array([boot_ci(pre["boot"][c][:, i])[1] for i in range(len(weeks))])
        ls = "--" if c == "mkt" else "-"
        ax[0].plot(weeks, pre["point"][c], ls, color=cols[c], lw=2, label=NICE[c])
        if c != "mkt":
            ax[0].fill_between(weeks, lo, hi, color=cols[c], alpha=0.13, lw=0)
    f = fit_half_life(weeks, pre["point"]["pre_def_dst"], n_wk)
    if f["exp0"] is not None:
        ax[0].plot(weeks, _exp0(weeks, *f["exp0"]), ":", color="k", lw=1.4,
                   label=f"exp fit, half-life {f['half_life']:.1f} wk")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axvspan(0.5, 6.5, color="gold", alpha=0.15, zorder=0)
    ax[0].text(3.5, ax[0].get_ylim()[1] * 0.94, "weeks 1-6", ha="center", fontsize=9)
    ax[0].set_xlabel("week of season")
    ax[0].set_ylabel("corr(frozen preseason rating, D/ST points)")
    ax[0].set_title("DST-005a  preseason knowledge decays week by week")
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].set_xticks(range(1, MAX_WEEK + 1, 2))

    for c, col in zip(curves, ["#1f77b4", "#9467bd", "#d62728"]):
        lo = np.array([boot_ci(c["boot"][:, j])[0] for j in range(len(GAPS))])
        hi = np.array([boot_ci(c["boot"][:, j])[1] for j in range(len(GAPS))])
        ax[1].plot(c["gaps"], c["r"], "-o", ms=3.5, color=col, lw=2, label=c["label"])
        ax[1].fill_between(c["gaps"], lo, hi, color=col, alpha=0.13, lw=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("gap between rating and outcome (games)")
    ax[1].set_ylabel("corr(rating, outcome)")
    ax[1].set_title("DST-005b  in-season knowledge, N games ago")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "dst005_time_decay.png", dpi=140)
    plt.close(fig)

    pl.DataFrame(hl_rows).write_csv(out / "dst005_half_lives.csv")
    curve_rows = [{"week": int(w), "n": int(n_wk[i]),
                   **{c: float(pre["point"][c][i]) for c in ratings}}
                  for i, w in enumerate(weeks)]
    pl.DataFrame(curve_rows).write_csv(out / "dst005_preseason_curve.csv")
    pl.DataFrame(brows).write_csv(out / "dst005_buckets.csv")
    pl.DataFrame(con).write_csv(out / "dst005_contrast.csv")
    print(f"\nwrote {out / 'dst005_time_decay.png'}")
    print(f"wrote {out / 'dst005_half_lives.csv'}, dst005_preseason_curve.csv, "
          f"dst005_buckets.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
