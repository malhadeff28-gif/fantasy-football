"""Phase 2b: is schedule difficulty real, and is carrying a second QB worth it?

Phase 2 found opponent difficulty adds almost nothing over recent form, and
nothing at all on top of the betting market. Before accepting that as a verdict
on the thesis, this script runs three harder tests.

A. A fairer schedule metric. "Fantasy points allowed to QBs" is contaminated by
   the quality of quarterbacks a defence happened to face. Replacing it with a
   QB-quality-controlled rating (mean of actual minus form-expected) gives the
   schedule hypothesis its best shot.

B. The draft-day test, which is the one this project actually needs. In August
   there is no market and no current-season defensive data. The only usable
   input is *last* season's defensive quality. So: does a prior-season defence
   rating predict this season's QB fantasy outcomes at all? If it does not, the
   preseason schedule-complementarity premise has no foundation, regardless of
   what in-season data shows.

C. The structural value of a second quarterback, which does not depend on
   predictive skill. Even a coin-flip chooser benefits from holding two lottery
   tickets, because max(A, B) exceeds E[A]. This separates the value that comes
   from *having* the option from the value that comes from *choosing* well.

Usage:  python scripts/phase2b_schedule_test.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                      # noqa: E402
import warnings                                 # noqa: E402
from itertools import combinations              # noqa: E402
from pathlib import Path                        # noqa: E402

import numpy as np                              # noqa: E402
import polars as pl                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.features import panel as panel_mod  # noqa: E402

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
BAR = "=" * 78
DEF_WINDOW = 17


def build() -> pl.DataFrame:
    p = panel_mod.build_panel(SEASONS).drop_nulls(
        subset=["form_fp_mean", "form_pass_att", "rush_fp_share"]
    )
    # Form-expectation residual: how much did the QB beat his own recent level?
    return p.with_columns(
        (pl.col("fantasy_points") - pl.col("form_fp_mean")).alias("form_resid")
    )


def add_quality_adjusted_defense(p: pl.DataFrame) -> pl.DataFrame:
    """Defence rating = mean prior (actual - form-expected) allowed. As-of."""
    tw = (
        p.group_by(["opponent", "season", "week"])
        .agg(pl.col("form_resid").mean().alias("d_resid"))
        .sort(["opponent", "season", "week"])
        .with_columns(
            pl.col("d_resid").shift(1)
            .rolling_mean(window_size=DEF_WINDOW, min_samples=3)
            .over("opponent").alias("def_qadj"),
        )
        .select(["opponent", "season", "week", "def_qadj", "d_resid"])
    )
    return p.join(tw.drop("d_resid"), on=["opponent", "season", "week"], how="left")


def prior_season_defense(p: pl.DataFrame) -> pl.DataFrame:
    """Each team's full-season defence rating, usable the FOLLOWING season."""
    season_rating = (
        p.group_by(["opponent", "season"])
        .agg(pl.col("form_resid").mean().alias("season_def_rating"),
             pl.len().alias("n_games"))
        .with_columns((pl.col("season") + 1).alias("apply_season"))
    )
    return season_rating


def test_a(p: pl.DataFrame) -> None:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    print(f"{BAR}\n A. FAIRER SCHEDULE METRIC (QB-quality-controlled)\n{BAR}")
    q = p.drop_nulls(subset=["def_qadj"])
    print(f"rows with a quality-adjusted rating: {q.height}")

    FORM = ["form_fp_mean", "career_fp_mean", "form_pass_att", "form_rush_att"]
    MARKET = ["implied_team_total", "spread_abs", "total_line"]
    sets = {
        "form": FORM,
        "form + naive schedule": FORM + ["def_adj"],
        "form + quality-adj schedule": FORM + ["def_qadj"],
        "form + market": FORM + MARKET,
        "form + market + quality-adj": FORM + MARKET + ["def_qadj"],
    }
    print(f"\n{'feature set':<32}{'MAE':>8}{'RMSE':>9}")
    print("-" * 78)
    for nm, feats in sets.items():
        maes, rmses = [], []
        for s in TEST_SEASONS:
            tr, te = q.filter(pl.col("season") < s), q.filter(pl.col("season") == s)
            if tr.height < 200:
                continue
            Xtr = tr.select(feats).to_numpy().astype(float)
            Xte = te.select(feats).to_numpy().astype(float)
            ytr, yte = tr["fantasy_points"].to_numpy(), te["fantasy_points"].to_numpy()
            sc = StandardScaler().fit(Xtr)
            m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(Xtr), ytr)
            pr = m.predict(sc.transform(Xte))
            maes.append(np.mean(np.abs(pr - yte)))
            rmses.append(np.sqrt(np.mean((pr - yte) ** 2)))
        print(f"{nm:<32}{np.mean(maes):>8.3f}{np.mean(rmses):>9.3f}")

    c = q.select(pl.corr("def_qadj", "form_resid")).item()
    print(f"\ncorr(quality-adjusted defence rating, form residual) = {c:.4f}")


def test_b(p: pl.DataFrame) -> None:
    print(f"\n{BAR}\n B. DRAFT-DAY TEST: does LAST season's defence predict THIS season?\n{BAR}")
    sr = prior_season_defense(p)

    # Year-over-year stability of the defence rating itself.
    cur = sr.select(["opponent", "season", "season_def_rating"])
    nxt = sr.select(["opponent",
                     pl.col("season").alias("season_prev"),
                     pl.col("season_def_rating").alias("rating_prev")])
    yoy = cur.join(
        nxt, left_on=["opponent", pl.col("season") - 1],
        right_on=["opponent", "season_prev"], how="inner",
    )
    r_yoy = float(np.corrcoef(yoy["rating_prev"].to_numpy(),
                              yoy["season_def_rating"].to_numpy())[0, 1])
    print(f"team defence rating, season N-1 -> season N : r = {r_yoy:.4f}  "
          f"(n = {yoy.height} team-seasons)")
    print(f"    => a defence explains {r_yoy ** 2 * 100:.1f}% of its own next-season rating")

    # Does the prior-season rating predict individual QB-week outcomes?
    joined = p.join(
        sr.select(["opponent",
                   pl.col("apply_season").alias("season"),
                   pl.col("season_def_rating").alias("def_prior_season")]),
        on=["opponent", "season"], how="inner",
    )
    r_pred = float(np.corrcoef(joined["def_prior_season"].to_numpy(),
                               joined["form_resid"].to_numpy())[0, 1])
    print(f"\nprior-season defence rating vs QB form residual : r = {r_pred:.4f} "
          f"(n = {joined.height} QB-weeks)")
    print(f"    => explains {r_pred ** 2 * 100:.2f}% of the variance a drafter "
          f"is trying to exploit")

    # Practical magnitude: split schedule into hardest/easiest thirds.
    # The rating is (actual - expected) ALLOWED, so a LOW value is a stingy
    # defence. qcut labels ascending, hence hardest first.
    t = joined.with_columns(
        pl.col("def_prior_season").qcut(3, labels=["hardest", "middle", "easiest"])
        .alias("tercile")
    )
    print(f"\n{'prior-season matchup tercile':<32}{'mean FP':>10}{'n':>8}")
    print("-" * 78)
    for lab in ["hardest", "middle", "easiest"]:
        g = t.filter(pl.col("tercile") == lab)
        print(f"{lab:<32}{g['fantasy_points'].mean():>10.2f}{g.height:>8}")
    spread = (t.filter(pl.col("tercile") == "easiest")["fantasy_points"].mean()
              - t.filter(pl.col("tercile") == "hardest")["fantasy_points"].mean())
    print(f"\neasiest minus hardest third = {spread:+.2f} fantasy points per game")


def test_c(p: pl.DataFrame) -> None:
    print(f"\n{BAR}\n C. STRUCTURAL VALUE OF A SECOND QB (skill-free)\n{BAR}")
    print("For each season, pairs are formed among QBs who started >= 12 games.")
    print("QB1 is whichever of the pair had the higher prior-season scoring rate,")
    print("so the choice of 'primary' uses no in-season information.\n")

    rows = []
    for s in TEST_SEASONS:
        cur = p.filter(pl.col("season") == s)
        counts = cur.group_by("player_id").agg(pl.len().alias("g"))
        eligible = counts.filter(pl.col("g") >= 12)["player_id"].to_list()
        prev = (p.filter(pl.col("season") == s - 1)
                .group_by("player_id")
                .agg(pl.col("fantasy_points").mean().alias("prev_ppg")))
        cur = cur.filter(pl.col("player_id").is_in(eligible)).join(
            prev, on="player_id", how="inner")
        ids = sorted(cur["player_id"].unique().to_list())
        # polars yields the group key as a tuple
        wk = {pid[0]: dict(zip(g["week"].to_list(), g["fantasy_points"].to_list()))
              for pid, g in cur.group_by("player_id")}
        ppg = {r["player_id"]: r["prev_ppg"]
               for r in cur.select(["player_id", "prev_ppg"]).unique().to_dicts()}

        for a, b in combinations(ids, 2):
            wa, wb = wk[a], wk[b]
            weeks = sorted(set(wa) | set(wb))
            if len(weeks) < 12:
                continue
            qb1, qb2 = (a, b) if ppg[a] >= ppg[b] else (b, a)
            solo = opt = played = 0.0
            for w in weeks:
                pa, pb = wk[qb1].get(w), wk[qb2].get(w)
                if pa is None and pb is None:
                    continue
                played += 1
                # single-QB manager: starts QB1, gets 0 on a bye
                solo += pa if pa is not None else 0.0
                # perfect chooser holding both
                opt += max(v for v in (pa, pb) if v is not None)
            if played:
                rows.append({"season": s, "weeks": played,
                             "solo": solo, "oracle": opt})

    d = pl.DataFrame(rows)
    print(f"pairs evaluated: {d.height:,}")
    print(f"\n{'strategy':<34}{'pts/season':>12}{'vs single QB':>15}")
    print("-" * 78)
    solo_m, orc_m = d["solo"].mean(), d["oracle"].mean()
    print(f"{'start QB1 every week (1 QB)':<34}{solo_m:>12.1f}{'--':>15}")
    print(f"{'perfect weekly choice (2 QBs)':<34}{orc_m:>12.1f}{orc_m - solo_m:>+15.1f}")
    print(f"\nThe gap is the ENTIRE prize the pairing strategy competes for.")
    print(f"A real chooser captures only a fraction of it -- Phase 2 measured")
    print(f"pregame ranking accuracy at ~64%, barely above a coin flip at 50%.")

    # What fraction does a realistic (64%-accurate) chooser capture?
    frac = 0.64
    print(f"\nIf a chooser is right {frac:.0%} of the time, naive expectation is")
    print(f"roughly {solo_m + (orc_m - solo_m) * ((frac - 0.5) / 0.5):.1f} pts/season, "
          f"i.e. about {(orc_m - solo_m) * ((frac - 0.5) / 0.5):+.1f} over one QB.")
    print("Phase 5 will measure this properly with real pregame projections")
    print("rather than this interpolation.")

    out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
    d.write_csv(out / "phase2b_pair_structural_value.csv")
    print(f"\nwrote outputs/phase2b_pair_structural_value.csv")


def main() -> int:
    p = build()
    p = add_quality_adjusted_defense(p)
    test_a(p)
    test_b(p)
    test_c(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
