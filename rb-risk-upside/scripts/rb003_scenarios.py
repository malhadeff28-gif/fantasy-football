"""RB-003: workload scenarios for the four backs, from historical comparables.

No single projected workload is assumed. For each back the comparable pool is
built from pre-season information only - prior role, durability, experience,
ADP, team change and the competition in the room - and the empirical
distribution of what those comparables actually did next becomes the scenario
set. Selection never touches an outcome.
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
import rb_simulate as S  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)
BAR = "=" * 78
K = 30


def build_targets(sit: pl.DataFrame) -> list[S.Target]:
    ids = R.target_ids()
    gsis = dict(ids.select("gsis_id", "display_name").iter_rows())
    rows = sit.filter((pl.col("season") == R.TARGET_SEASON)
                      & pl.col("gsis_id").is_in(list(gsis)))
    targets = []
    for r in rows.iter_rows(named=True):
        prior = {c: float(r[c]) for c in S.MATCH}
        prior["switched"] = int(r["switched"])
        targets.append(S.Target(
            name=gsis[r["gsis_id"]], player_id=r["gsis_id"], team=r["team"],
            season=R.TARGET_SEASON, prior=prior, adp=float(r["adp"]),
            adp_rb_rank=int(r["adp_rb_rank"])))
    order = {n: i for i, n in enumerate(
        ["D'Andre Swift", "Tony Pollard", "Quinshon Judkins", "David Montgomery"])}
    return sorted(targets, key=lambda t: order.get(t.name, 99))


def main() -> None:
    panel = F.load_or_build()
    ss = F.season_rollup(panel, min_games=1)
    sit = S.situation_panel(ss, list(range(2018, R.TARGET_SEASON + 1)))
    out = S.outcome_panel(ss)

    print(BAR, "\n 1. PRE-SEASON SITUATION VECTORS (2026)\n", BAR)
    targets = build_targets(sit)
    tv = pl.DataFrame([{"name": t.name, "team": t.team, "adp": t.adp,
                        "adp_rb_rank": t.adp_rb_rank, **t.prior}
                       for t in targets])
    with pl.Config(tbl_cols=14, float_precision=2, tbl_width_chars=200):
        print(tv)
    print("\n comp_draft_capital = best draft slot among the OTHER backs in the")
    print(" room (300 = none drafted). comp_prior_touches_pg = the most any of")
    print(" them handled last season. switched = changed team this offseason.")

    print("\n", BAR, "\n 2. COMPARABLES AND WHAT HAPPENED TO THEM\n", BAR)
    print(f" pool = every RB season 2019-2025 with a prior season; k = {K};")
    print(" matching uses pre-season features only, outcomes attach afterwards.")

    all_comps, all_scen = {}, {}
    for t in targets:
        c = S.comparables(t, sit, out, k=K)
        all_comps[t.name] = c
        print(f"\n--- {t.name} ({t.team}), ADP {t.adp:.1f} / RB{t.adp_rb_rank} ---")
        print(f"  prior season: {t.prior['prior_touches_pg']:.1f} touches/gm, "
              f"{t.prior['prior_games']:.0f} games, snap {t.prior['prior_snap_share']:.2f}"
              f"{', CHANGED TEAM' if t.prior['switched'] else ''}")
        show = c.select(
            pl.col("full_name").str.slice(0, 17).alias("comparable"), "season",
            pl.col("team").alias("tm"),
            pl.col("dist").round(2),
            pl.col("prior_touches_pg").round(1).alias("prior_tch"),
            pl.col("out_touches_pg").round(1).alias("got_tch"),
            pl.col("out_games").alias("gm"),
            pl.col("out_fp_total").round(0).alias("fp"),
            pl.col("out_pos_rank").alias("rank"),
        ).head(12)
        with pl.Config(tbl_rows=14, tbl_width_chars=140):
            print(show)

        rel = (c["out_touches_pg"] / c["prior_touches_pg"]).to_numpy()
        print(f"  role ratio (next/prior touches per game): "
              f"median {np.median(rel):.2f}, "
              f"p10 {np.percentile(rel,10):.2f}, p90 {np.percentile(rel,90):.2f}")
        print(f"  games played next season: median {np.median(c['out_games']):.0f}, "
              f"p10 {np.percentile(c['out_games'],10):.0f}, "
              f"share under 13 games: {float((c['out_games'] < 13).mean()):.2f}")

        sc = S.role_scenarios(c)
        all_scen[t.name] = sc
        print("  role scenarios (Wilson 95% interval on each probability):")
        for row in sc.iter_rows(named=True):
            print(f"    {row['band']:<10} p={row['p']:.2f} "
                  f"[{row['p_lo']:.2f},{row['p_hi']:.2f}]  n={row['n']:>2}  "
                  f"median season pts {row['med_fp']}")

    print("\n", BAR, "\n 3. SCENARIO SUMMARY: DOWNSIDE / BASE / UPSIDE\n", BAR)
    rows = []
    for t in targets:
        c = all_comps[t.name]
        tch = c["out_touches_pg"].to_numpy()
        fp = c["out_fp_total"].to_numpy()
        rows.append({
            "name": t.name, "team": t.team, "adp_rb_rank": t.adp_rb_rank,
            "down_tch": round(float(np.percentile(tch, 20)), 1),
            "base_tch": round(float(np.median(tch)), 1),
            "up_tch": round(float(np.percentile(tch, 80)), 1),
            "down_fp": round(float(np.percentile(fp, 20))),
            "base_fp": round(float(np.median(fp))),
            "up_fp": round(float(np.percentile(fp, 80))),
            "skew": round(float((np.percentile(fp, 80) - np.median(fp))
                                / max(np.median(fp) - np.percentile(fp, 20), 1)), 2),
        })
    summ = pl.DataFrame(rows)
    with pl.Config(tbl_cols=12, tbl_width_chars=160):
        print(summ)
    print("\n skew > 1 means the comparable pool's upside tail is longer than its")
    print(" downside tail: that is the positively skewed workload distribution")
    print(" the project is looking for.")

    print("\n", BAR, "\n 4. SENSITIVITY TO k\n", BAR)
    krows = []
    for k in (15, 20, 30, 45, 60):
        for t in targets:
            c = S.comparables(t, sit, out, k=k)
            krows.append({"k": k, "name": t.name,
                          "median_tch": round(float(c["out_touches_pg"].median()), 1),
                          "median_fp": round(float(c["out_fp_total"].median())),
                          "p_bellcow": round(float(
                              (c["out_touches_pg"] >= 18).mean()), 2)})
    kt = pl.DataFrame(krows).pivot(on="name", index="k", values="median_fp")
    print(" median comparable season points by k:")
    with pl.Config(tbl_width_chars=140):
        print(kt)
    kb = pl.DataFrame(krows).pivot(on="name", index="k", values="p_bellcow")
    print("\n P(bellcow role, 18+ touches/gm) by k:")
    with pl.Config(tbl_width_chars=140):
        print(kb)

    pl.concat([c.with_columns(pl.lit(n).alias("target"))
               for n, c in all_comps.items()]).write_parquet(
        R.DATA / "rb003_comparables.parquet")
    summ.write_csv(R.OUT / "rb003_scenarios.csv")
    print(f"\nwritten: {R.DATA / 'rb003_comparables.parquet'}, "
          f"{R.OUT / 'rb003_scenarios.csv'}")


if __name__ == "__main__":
    main()
