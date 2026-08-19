"""RB-002: what carries season to season, and what of that is actually useful.

Two separate questions, deliberately not conflated:

  (a) STABILITY  - does metric_t predict metric_{t+1}?
  (b) USEFULNESS - does metric_t predict FANTASY POINTS_{t+1} beyond what
      last season's fantasy points already told you?

Prior QB work found several highly stable variables that added nothing once
the simple baseline was in the model. The same trap is tested for here, out of
sample and walk-forward, never on the same rows used to pick the variable.
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

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import rb_data as R  # noqa: E402
import rb_features as F  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)
BAR = "=" * 78
RNG = np.random.default_rng(20260818)
MIN_G = 8          # a season must be 8+ games to count as a season of role
BOOT = 2000

METRICS = [
    # role / opportunity
    ("snap_share", "role"), ("carry_share", "role"), ("carries_pg", "role"),
    ("touches_pg", "role"), ("targets_pg", "role"), ("target_share_rb", "role"),
    ("rz_carries_pg", "role"), ("gl_carries_pg", "role"),
    ("rz_carry_share", "role"), ("rz_targets_pg", "role"),
    # efficiency / conversion
    ("ypc", "eff"), ("ypr", "eff"), ("catch_rate", "eff"),
    ("rush_td_rate", "eff"), ("td_per_touch", "eff"), ("gl_td_rate", "eff"),
    ("fp_per_touch", "eff"),
    # outcome / environment
    ("fp_pg", "out"), ("fp_total", "out"), ("g", "out"),
    ("implied_team_total", "env"),
]


def transitions(ss: pl.DataFrame) -> pl.DataFrame:
    """Same player in consecutive seasons, both with >= MIN_G games."""
    a = ss.filter(pl.col("g") >= MIN_G)
    b = a.with_columns((pl.col("season") - 1).alias("season_t"))
    cols = [m for m, _ in METRICS]
    nxt = b.select(["player_id", "season_t"] + [pl.col(c).alias(f"{c}_n") for c in cols]
                   + [pl.col("team").alias("team_n")])
    return (a.join(nxt, left_on=["player_id", "season"],
                   right_on=["player_id", "season_t"], how="inner")
            .with_columns((pl.col("team") != pl.col("team_n")).alias("switched")))


def boot_corr(x: np.ndarray, y: np.ndarray, n: int = BOOT) -> tuple:
    """Pearson r with a bootstrap CI. Small n is the whole problem here."""
    r = float(np.corrcoef(x, y)[0, 1])
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    rs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return r, float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))


def stability_table(tr: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for m, kind in METRICS:
        s = tr.drop_nulls([m, f"{m}_n"])
        x = s[m].to_numpy().astype(float)
        y = s[f"{m}_n"].to_numpy().astype(float)
        if len(x) < 25:
            continue
        r, lo, hi = boot_corr(x, y)
        # Spearman guards against one outlier season driving Pearson
        rs = float(np.corrcoef(np.argsort(np.argsort(x)),
                               np.argsort(np.argsort(y)))[0, 1])
        rows.append({"metric": m, "kind": kind, "n": len(x), "r": round(r, 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "spearman": round(rs, 3),
                     "signif": "yes" if lo > 0 else "no"})
    return pl.DataFrame(rows).sort("r", descending=True)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def walk_forward_value(tr: pl.DataFrame, target: str = "fp_pg_n") -> pl.DataFrame:
    """Does metric_t beat 'last year's points' out of sample, chronologically?

    Fit on every transition whose OUTCOME season is strictly before the test
    season, then predict the test season. Nothing from the future is ever in
    the training set.
    """
    tr = tr.with_columns((pl.col("season") + 1).alias("out_season"))
    test_seasons = sorted(tr["out_season"].unique().to_list())[2:]
    rows = []
    for m, kind in METRICS:
        if m in ("fp_pg", "fp_total"):
            continue
        yb, yf, yy = [], [], []
        for ts in test_seasons:
            trn = tr.filter(pl.col("out_season") < ts).drop_nulls(["fp_pg", m, target])
            tst = tr.filter(pl.col("out_season") == ts).drop_nulls(["fp_pg", m, target])
            if trn.height < 40 or tst.height < 8:
                continue
            Xb_tr = trn.select("fp_pg").to_numpy()
            Xf_tr = trn.select("fp_pg", m).to_numpy()
            y_tr = trn[target].to_numpy()
            bb, bf = _ols(Xb_tr, y_tr), _ols(Xf_tr, y_tr)
            yb.append(_pred(bb, tst.select("fp_pg").to_numpy()))
            yf.append(_pred(bf, tst.select("fp_pg", m).to_numpy()))
            yy.append(tst[target].to_numpy())
        if not yy:
            continue
        yb, yf, yy = np.concatenate(yb), np.concatenate(yf), np.concatenate(yy)
        rmse_b = float(np.sqrt(np.mean((yy - yb) ** 2)))
        rmse_f = float(np.sqrt(np.mean((yy - yf) ** 2)))
        # paired bootstrap on the squared-error difference: is the gain real?
        d = (yy - yb) ** 2 - (yy - yf) ** 2
        bs = RNG.integers(0, len(d), size=(BOOT, len(d)))
        md = np.mean(d[bs], axis=1)
        rows.append({"metric": m, "kind": kind, "n_test": len(yy),
                     "rmse_base": round(rmse_b, 3), "rmse_plus": round(rmse_f, 3),
                     "gain": round(rmse_b - rmse_f, 4),
                     "p_gain_pos": round(float(np.mean(md > 0)), 3)})
    return pl.DataFrame(rows).sort("gain", descending=True)


def td_luck(ss: pl.DataFrame, tr: pl.DataFrame) -> None:
    """Is a touchdown rate a skill you can buy, or noise you rent?"""
    print(BAR, "\n 3. TOUCHDOWN RATE: SKILL OR NOISE\n", BAR)

    for m, label in [("rush_td_rate", "rushing TD per carry"),
                     ("td_per_touch", "total TD per touch"),
                     ("gl_td_rate", "TD per goal-line carry"),
                     ("gl_carries_pg", "goal-line carries per game"),
                     ("carries_pg", "carries per game")]:
        s = tr.drop_nulls([m, f"{m}_n"])
        x, y = s[m].to_numpy().astype(float), s[f"{m}_n"].to_numpy().astype(float)
        r, lo, hi = boot_corr(x, y)
        print(f"  {label:<28} r(t,t+1) = {r:+.3f}  [{lo:+.3f},{hi:+.3f}]  n={len(x)}")

    # Split-half within season: the cleanest noise test there is. If a rate is
    # skill, a player's first half predicts his own second half.
    print("\n  split-half within season (odd vs even weeks, same player-season):")
    panel = F.load_or_build()
    half = (panel.with_columns((pl.col("week") % 2).alias("h"))
            .group_by(["player_id", "season", "h"]).agg(
                pl.col("carries").sum().alias("c"),
                pl.col("touches").sum().alias("t"),
                pl.col("rushing_tds").sum().alias("rtd"),
                pl.col("receiving_tds").sum().alias("rectd"),
                pl.col("rushing_yards").sum().alias("ry"),
                pl.col("gl_carries").sum().alias("gl"),
                pl.len().alias("g"))
            .filter(pl.col("c") >= 40))
    w = (half.filter(pl.col("h") == 0).join(
        half.filter(pl.col("h") == 1), on=["player_id", "season"], suffix="_b"))
    for num, den, label in [("rtd", "c", "rush TD rate"),
                            ("ry", "c", "yards per carry"),
                            ("gl", "g", "goal-line carries/gm"),
                            ("c", "g", "carries per game")]:
        a = (w[num] / w[den]).to_numpy().astype(float)
        b = (w[f"{num}_b"] / w[f"{den}_b"]).to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        r, lo, hi = boot_corr(a[ok], b[ok])
        print(f"    {label:<24} r(half,half) = {r:+.3f} [{lo:+.3f},{hi:+.3f}] n={ok.sum()}")

    # regression to the mean, stated as a number a drafter can use
    print("\n  regression to the mean in rushing TD rate:")
    s = tr.drop_nulls(["rush_td_rate", "rush_td_rate_n", "carries"])
    s = s.filter(pl.col("carries") >= 100)
    med = s["rush_td_rate"].median()
    hi_g = s.filter(pl.col("rush_td_rate") > med)
    lo_g = s.filter(pl.col("rush_td_rate") <= med)
    print(f"    high-TD-rate season t (n={hi_g.height}): "
          f"{hi_g['rush_td_rate'].mean():.4f} -> {hi_g['rush_td_rate_n'].mean():.4f}")
    print(f"    low-TD-rate  season t (n={lo_g.height}): "
          f"{lo_g['rush_td_rate'].mean():.4f} -> {lo_g['rush_td_rate_n'].mean():.4f}")
    print(f"    league mean t+1 = {s['rush_td_rate_n'].mean():.4f}  "
          "-> the gap between the groups mostly closes.")


def main() -> None:
    panel = F.load_or_build()
    ss = F.season_rollup(panel, min_games=1)
    tr = transitions(ss)

    print(BAR, "\n 1. SEASON-TO-SEASON STABILITY (necessary, not sufficient)\n", BAR)
    print(f"transitions: {tr.height} player-season pairs, "
          f"{tr['player_id'].n_unique()} players, "
          f"{tr['season'].min()}->{tr['season'].max()+1}, min {MIN_G} games each side")
    st = stability_table(tr)
    print(st)

    print("\n  same team vs changed team (Montgomery is a mover):")
    for flag, lab in [(False, "stayed"), (True, "switched")]:
        sub = tr.filter(pl.col("switched") == flag)
        for m in ("carries_pg", "snap_share", "fp_pg"):
            s = sub.drop_nulls([m, f"{m}_n"])
            if s.height < 20:
                print(f"    {lab:<9} {m:<12} n={s.height} (too few to report)")
                continue
            r, lo, hi = boot_corr(s[m].to_numpy().astype(float),
                                  s[f"{m}_n"].to_numpy().astype(float))
            print(f"    {lab:<9} {m:<12} r={r:+.3f} [{lo:+.3f},{hi:+.3f}] n={s.height}")

    print("\n", BAR, "\n 2. STABLE IS NOT THE SAME AS USEFUL\n", BAR)
    print("Out-of-sample walk-forward. Baseline = last season's points per game.")
    print("A metric earns its place only by cutting RMSE on unseen seasons.")
    wf = walk_forward_value(tr)
    print(wf)
    k = wf.height
    print(f"\n  {k} metrics tested. A 5% threshold expects {0.05*k:.1f} false")
    print("  winners by chance alone; Benjamini-Hochberg at q=0.10 below.")
    p = wf.with_columns((1 - pl.col("p_gain_pos")).alias("p"))
    p = p.sort("p").with_row_index("i", 1).with_columns(
        (pl.col("i") / k * 0.10).alias("bh_thresh"))
    print(p.select("metric", "kind", "gain", "p", "bh_thresh",
                   (pl.col("p") <= pl.col("bh_thresh")).alias("survives_bh")))

    td_luck(ss, tr)

    st.write_csv(R.OUT / "rb002_stability.csv")
    wf.write_csv(R.OUT / "rb002_walkforward.csv")
    print(f"\nwritten: {R.OUT / 'rb002_stability.csv'}, "
          f"{R.OUT / 'rb002_walkforward.csv'}")


if __name__ == "__main__":
    main()
