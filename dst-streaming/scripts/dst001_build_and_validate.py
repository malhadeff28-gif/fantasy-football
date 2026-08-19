"""DST-001: reconstruct D/ST scoring historically and validate it.

A scoring implementation nobody checked is a silent source of wrong answers for
every experiment downstream, so this validates before anything else is built.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                        # noqa: E402
from pathlib import Path                          # noqa: E402

import polars as pl                               # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import dst_data                                   # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_width_chars(150)
SEASONS = list(range(2018, 2026))
BAR = "=" * 84


def main() -> int:
    d = dst_data.build_dst_panel(SEASONS).select(dst_data.DST_KEEP)
    print(f"D/ST panel: {d.height} defense-weeks, {SEASONS[0]}-{SEASONS[-1]}")
    print(f"per season: {d.group_by('season').agg(pl.len().alias('n')).sort('season')['n'].to_list()}")

    print(f"\n{BAR}\n SANITY: distribution of weekly D/ST fantasy points\n{BAR}")
    s = d["dst_points"]
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        print(f"  p{int(q*100):<3} {s.quantile(q):>7.2f}")
    print(f"  mean {s.mean():>6.2f}   sd {s.std():>6.2f}   min {s.min():>6.1f}   max {s.max():>6.1f}")

    print(f"\n{BAR}\n SANITY: points-allowed tier is doing the work it should\n{BAR}")
    tier = (d.group_by("_dst_pa").agg(pl.len().alias("n"),
                                      pl.col("points_allowed").min().alias("pa_min"),
                                      pl.col("points_allowed").max().alias("pa_max"),
                                      pl.col("dst_points").mean().alias("mean_pts"))
            .sort("_dst_pa", descending=True))
    print(tier)

    print(f"\n{BAR}\n VALIDATION: top D/ST seasons should be recognisable\n{BAR}")
    tot = (d.group_by(["season", "team"]).agg(pl.col("dst_points").sum().alias("pts"),
                                              pl.len().alias("g"))
           .sort(["season", "pts"], descending=[False, True]))
    for yr in SEASONS:
        top = tot.filter(pl.col("season") == yr).head(3)
        line = ", ".join(f"{r['team']} {r['pts']:.0f}" for r in top.to_dicts())
        print(f"  {yr}: {line}")

    print(f"\n{BAR}\n SCORING SENSITIVITY: CBS/Yahoo tiers vs ESPN tiers\n{BAR}")
    e = dst_data.build_dst_panel(SEASONS, espn_tiers=True)
    j = d.join(e.select(["game_id", "team", pl.col("dst_points").alias("espn_pts")]),
               on=["game_id", "team"], how="inner")
    print(f"  CBS/Yahoo mean {j['dst_points'].mean():.2f} vs ESPN mean {j['espn_pts'].mean():.2f}")
    print(f"  correlation between the two scorings: {j.select(pl.corr('dst_points','espn_pts')).item():.4f}")
    print("  => tier choice shifts the level; check later whether it changes RANKINGS")

    print(f"\n{BAR}\n COMPONENT SHARE: where do D/ST points actually come from?\n{BAR}")
    comp = d.select(
        (pl.col("def_sacks").mean() * 1.0).alias("sacks"),
        (pl.col("def_interceptions").mean() * 2.0).alias("interceptions"),
        (pl.col("fumble_recovery_opp").mean() * 2.0).alias("fumble_rec"),
        (pl.col("def_tds").mean() * 6.0).alias("def_TD"),
        (pl.col("special_teams_tds").mean() * 6.0).alias("ST_TD"),
        (pl.col("def_safeties").mean() * 2.0).alias("safety"),
        pl.col("_dst_pa").mean().alias("points_allowed_tier"),
    )
    for k, v in comp.to_dicts()[0].items():
        print(f"  {k:<22}{v:>7.2f} pts/game")

    out = PROJECT / "outputs"; out.mkdir(exist_ok=True)
    d.write_parquet(out / "dst_panel.parquet")
    d.write_csv(out / "dst001_panel_sample.csv") if d.height < 20000 else None
    print(f"\nwrote {out / 'dst_panel.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
