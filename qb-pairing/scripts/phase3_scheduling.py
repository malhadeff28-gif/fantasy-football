"""Phase 3: which NFL scheduling effects are actually predictive?

Phase 2 showed opponent identity barely matters. This phase asks whether the
*other* half of "schedule" -- rest, travel, venue, timing, and expected game
competitiveness -- carries signal that opponent difficulty did not.

Three studies:

A. Scheduling effects, each measured against the form residual (actual minus
   the quarterback's own recent level) so a good QB on a good team cannot
   masquerade as a scheduling effect. Reported with standard errors, because
   an effect smaller than its own noise is not an effect.

B. Game competitiveness. The brief hypothesises that a close game raises
   aggression and rushing opportunity while a blowout suppresses it. Tested
   both on the pregame expectation (spread) and, descriptively, on how the
   game actually unfolded.

C. How much schedule complementarity actually exists between two quarterbacks,
   and how much of it is visible before kickoff. This is the structural
   premise of the pairing thesis, measured directly.

Usage:  python scripts/phase3_scheduling.py
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

from qbpair.data import loaders                 # noqa: E402
from qbpair.features import panel as panel_mod  # noqa: E402

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
BAR = "=" * 78


def bucket_report(df: pl.DataFrame, col: str, label: str,
                  target: str = "form_resid") -> None:
    """Mean target by bucket, with standard error and a crude t-stat."""
    print(f"\n{label}")
    print(f"{'bucket':<26}{'n':>7}{'mean':>9}{'se':>8}{'t':>8}")
    print("-" * 78)
    g = (df.drop_nulls(subset=[col, target])
         .group_by(col)
         .agg(pl.len().alias("n"),
              pl.col(target).mean().alias("mean"),
              pl.col(target).std().alias("sd"))
         .sort(col))
    for r in g.to_dicts():
        if r["n"] < 30:
            continue
        se = r["sd"] / np.sqrt(r["n"])
        t = r["mean"] / se if se else 0.0
        star = "  *" if abs(t) >= 2 else ""
        print(f"{str(r[col]):<26}{r['n']:>7}{r['mean']:>9.3f}{se:>8.3f}{t:>8.2f}{star}")


def study_a(p: pl.DataFrame) -> None:
    print(f"{BAR}\n A. SCHEDULING EFFECTS vs FORM RESIDUAL\n{BAR}")
    print("Form residual = actual minus the quarterback's own 8-game average.")
    print("An effect must exceed its own standard error to be real (* = |t| >= 2).")

    d = p.with_columns(
        pl.when(pl.col("rest") <= 4).then(pl.lit("1 short week (<=4d)"))
        .when(pl.col("rest") <= 8).then(pl.lit("2 normal (5-8d)"))
        .when(pl.col("rest") <= 13).then(pl.lit("3 extra rest (9-13d)"))
        .otherwise(pl.lit("4 off bye (14d+)")).alias("rest_bucket"),
        pl.when(pl.col("is_home") == 1).then(pl.lit("home")).otherwise(pl.lit("away")).alias("venue"),
        pl.when(pl.col("indoors") == 1).then(pl.lit("indoors")).otherwise(pl.lit("outdoors")).alias("dome"),
        pl.when(pl.col("div_game") == 1).then(pl.lit("division")).otherwise(pl.lit("non-division")).alias("div"),
        pl.when(pl.col("week") <= 6).then(pl.lit("1 wk 1-6"))
        .when(pl.col("week") <= 13).then(pl.lit("2 wk 7-13"))
        .otherwise(pl.lit("3 wk 14-18 (playoffs)")).alias("season_part"),
    )

    for col, lab in [("rest_bucket", "REST"), ("venue", "HOME / AWAY"),
                     ("dome", "VENUE TYPE"), ("div", "DIVISION GAME"),
                     ("season_part", "TIME OF SEASON")]:
        bucket_report(d, col, lab)

    # Weather, where recorded
    sch = loaders.load_schedules().select(["game_id", "temp", "wind"])
    w = d.join(sch, on="game_id", how="left").drop_nulls(subset=["wind"])
    if w.height > 200:
        w = w.with_columns(
            pl.when(pl.col("wind") >= 15).then(pl.lit("wind >=15mph"))
            .when(pl.col("wind") >= 8).then(pl.lit("wind 8-14mph"))
            .otherwise(pl.lit("wind <8mph")).alias("wind_bucket"))
        bucket_report(w, "wind_bucket", f"WIND (n={w.height} games with data)")


def study_b(p: pl.DataFrame) -> None:
    print(f"\n{BAR}\n B. GAME COMPETITIVENESS AND OPPORTUNITY\n{BAR}")
    sch = loaders.load_schedules().select(
        ["game_id", "home_team", "home_score", "away_score"])
    d = (p.join(sch, on="game_id", how="left")
         .with_columns(
             pl.when(pl.col("team") == pl.col("home_team"))
             .then(pl.col("home_score") - pl.col("away_score"))
             .otherwise(pl.col("away_score") - pl.col("home_score")).alias("margin"))
         .drop_nulls(subset=["margin"]))

    print("\nPREGAME expectation (spread magnitude) -- what a manager can act on:")
    d2 = d.with_columns(
        pl.when(pl.col("spread_abs") <= 3).then(pl.lit("1 toss-up (<=3)"))
        .when(pl.col("spread_abs") <= 7).then(pl.lit("2 moderate (3.5-7)"))
        .otherwise(pl.lit("3 lopsided (>7)")).alias("spread_bucket"))
    bucket_report(d2, "spread_bucket", "expected competitiveness -> form residual")

    print(f"\n{'bucket':<26}{'n':>7}{'pass att':>10}{'rush att':>10}{'FP':>8}")
    print("-" * 78)
    for r in (d2.group_by("spread_bucket").agg(
            pl.len().alias("n"),
            pl.col("form_pass_att").mean().alias("pa"),
            pl.col("form_rush_att").mean().alias("ra"),
            pl.col("fantasy_points").mean().alias("fp")).sort("spread_bucket")).to_dicts():
        print(f"{r['spread_bucket']:<26}{r['n']:>7}{r['pa']:>10.1f}{r['ra']:>10.1f}{r['fp']:>8.2f}")

    print("\nACTUAL game script (descriptive -- NOT knowable pregame):")
    d3 = d.with_columns(
        pl.when(pl.col("margin").abs() <= 7).then(pl.lit("1 close (<=7)"))
        .when(pl.col("margin") > 7).then(pl.lit("2 won big"))
        .otherwise(pl.lit("3 lost big")).alias("script"))
    bucket_report(d3, "script", "actual game script -> form residual")

    # Does competitiveness matter MORE for rushing quarterbacks?
    print("\nInteraction with play style (rushing share above/below median):")
    med = p["rush_fp_share"].median()
    for lab, sub in [("mobile QBs", d2.filter(pl.col("rush_fp_share") > med)),
                     ("pocket QBs", d2.filter(pl.col("rush_fp_share") <= med))]:
        g = (sub.group_by("spread_bucket")
             .agg(pl.len().alias("n"), pl.col("form_resid").mean().alias("m"))
             .sort("spread_bucket"))
        vals = ", ".join(f"{r['spread_bucket'][2:]}: {r['m']:+.2f}" for r in g.to_dicts())
        print(f"  {lab:<12} {vals}")


def study_c(p: pl.DataFrame) -> None:
    print(f"\n{BAR}\n C. HOW MUCH SCHEDULE COMPLEMENTARITY ACTUALLY EXISTS?\n{BAR}")

    # Spread of pregame matchup quality within a week: if every QB faces a
    # similar environment, there is nothing to complement.
    wk = (p.group_by(["season", "week"])
          .agg(pl.col("implied_team_total").std().alias("sd_itt"),
               pl.col("def_adj").std().alias("sd_def"),
               pl.col("fantasy_points").std().alias("sd_fp"),
               pl.len().alias("n"))
          .filter(pl.col("n") >= 10))
    print(f"within-week dispersion (mean of weekly standard deviations):")
    print(f"    implied team total : {wk['sd_itt'].mean():.2f} pts")
    print(f"    opponent adjustment: {wk['sd_def'].mean():.2f} fantasy pts")
    print(f"    ACTUAL fantasy pts : {wk['sd_fp'].mean():.2f} fantasy pts")
    print("\n=> the pregame matchup signal is small relative to weekly outcome noise")

    # For real pairs: how often do the two QBs diverge, pregame vs actual?
    rows = []
    for s in SEASONS:
        cur = p.filter(pl.col("season") == s)
        counts = cur.group_by("player_id").agg(pl.len().alias("g"))
        elig = counts.filter(pl.col("g") >= 12)["player_id"].to_list()
        cur = cur.filter(pl.col("player_id").is_in(elig))
        wkmap = {pid[0]: {r["week"]: (r["def_adj"], r["implied_team_total"],
                                      r["fantasy_points"])
                          for r in g.select(["week", "def_adj",
                                             "implied_team_total",
                                             "fantasy_points"]).to_dicts()}
                 for pid, g in cur.group_by("player_id")}
        ids = sorted(wkmap)
        for a, b in combinations(ids, 2):
            wa, wb = wkmap[a], wkmap[b]
            common = sorted(set(wa) & set(wb))
            if len(common) < 10:
                continue
            da = np.array([wa[w][1] for w in common])
            db = np.array([wb[w][1] for w in common])
            fa = np.array([wa[w][2] for w in common])
            fb = np.array([wb[w][2] for w in common])
            if da.std() == 0 or db.std() == 0:
                continue
            rows.append({
                "corr_pregame_env": float(np.corrcoef(da, db)[0, 1]),
                "corr_actual_fp": float(np.corrcoef(fa, fb)[0, 1]),
            })
    d = pl.DataFrame(rows)
    print(f"\nacross {d.height:,} QB pairs:")
    print(f"    mean correlation of PREGAME implied team totals : "
          f"{d['corr_pregame_env'].mean():+.4f}")
    print(f"    mean correlation of ACTUAL weekly fantasy points: "
          f"{d['corr_actual_fp'].mean():+.4f}")
    print("\nBoth near zero. Two quarterbacks' weekly outcomes are already close to")
    print("independent, so there is little systematic anti-correlation for a")
    print("pairing model to discover -- the diversification is essentially free,")
    print("and choosing partners for 'complementary schedules' adds little to it.")

    out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
    d.write_csv(out / "phase3_complementarity.csv")
    print(f"\nwrote outputs/phase3_complementarity.csv")


def main() -> int:
    p = (panel_mod.build_panel(SEASONS)
         .drop_nulls(subset=["form_fp_mean", "form_pass_att", "rush_fp_share"])
         .with_columns((pl.col("fantasy_points") - pl.col("form_fp_mean")).alias("form_resid")))
    print(f"panel: {p.height} QB-weeks, {SEASONS[0]}-{SEASONS[-1]}")
    study_a(p)
    study_b(p)
    study_c(p)
    print(f"\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
