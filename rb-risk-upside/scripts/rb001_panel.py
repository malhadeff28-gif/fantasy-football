"""RB-001: verify the four 2026 situations, then build the RB-week panel.

Verification first and from data only: team, depth-chart slot, the whole RB
room, and draft cost. Team assignment is NOT taken from memory - one of the
four has moved.
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

import polars as pl  # noqa: E402

import rb_data as R  # noqa: E402
import rb_features as F  # noqa: E402
import markets  # noqa: E402
import nfl_data  # noqa: E402
import nflreadpy as nflr  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
BAR = "=" * 78


def verify_situations() -> dict:
    print(BAR, "\n 1. 2026 SITUATION - VERIFIED FROM DATA, NOT MEMORY\n", BAR)

    ids = R.target_ids()
    print("\n[players] master player table\n",
          ids.select("display_name", "gsis_id", "position", "latest_team",
                     "draft_year", "draft_round", "draft_pick", "status"))

    # rosters_weekly is capped at the last completed season; the 2026 season
    # has not kicked off, so 2026 team membership comes from season rosters.
    try:
        nfl_data.load_rosters_weekly([2026])
        weekly_2026 = True
    except ValueError as e:
        weekly_2026 = False
        print(f"\n[limit] load_rosters_weekly refuses 2026: {e}")
        print("        -> 2026 team from load_rosters (season roster) instead.")

    ros = nflr.load_rosters(seasons=[2024, 2025, 2026]).select(
        "season", "team", "gsis_id", "full_name", "position", "years_exp",
        "entry_year", "draft_number")
    mine = ros.filter(pl.col("gsis_id").is_in(ids["gsis_id"].to_list()))
    print("\n[rosters] team by season\n",
          mine.sort(["full_name", "season"]).select(
              "full_name", "season", "team", "years_exp"))

    teams26 = dict(mine.filter(pl.col("season") == 2026)
                   .select("full_name", "team").iter_rows())

    # depth charts are timestamped, not week-numbered; take the latest snapshot
    dc = nfl_data.load_depth_charts([2025])
    rb = dc.filter(pl.col("pos_abb") == "RB")
    latest = rb["dt"].max()
    print(f"\n[depth_charts] latest snapshot dt = {latest} "
          f"(range {dc['dt'].min()} .. {dc['dt'].max()})")
    print("  NOTE this snapshot predates the 2026 draft; the 2026 season roster"
          "\n  is the more current source for rookie competition.")

    rooms = {}
    for name, team in sorted(teams26.items()):
        chart = (rb.filter((pl.col("team") == team) & (pl.col("dt") == latest))
                 .sort("pos_rank").select("player_name", "pos_rank"))
        room = (ros.filter((pl.col("season") == 2026) & (pl.col("team") == team)
                           & (pl.col("position").is_in(["RB", "FB"])))
                .sort(["years_exp", "draft_number"])
                .select("full_name", "years_exp", "entry_year", "draft_number"))
        rooms[name] = (team, chart, room)
        print(f"\n--- {name}  ({team}) ---")
        print(" depth chart (Mar 2026):",
              ", ".join(f"{r[0]}#{r[1]}" for r in chart.iter_rows()))
        print(room)

    adp = markets.load_ffc_adp(2026)
    rbadp = adp.filter(pl.col("position") == "RB").sort("adp").with_row_index(
        "rb_rank", 1)
    tgt = rbadp.filter(pl.col("key").is_in(list(R.TARGETS)))
    print(f"\n[ADP] FFC 12-team standard, {adp['total_drafts'][0]} drafts, "
          f"{adp['adp_window_start'][0]} .. {adp['adp_window_end'][0]}")
    print(tgt.select("rb_rank", "name", "team", "adp", "stdev", "high", "low",
                     "times_drafted"))

    try:
        cbs = R.cbs_nonppr(2026)
        c = cbs.filter(pl.col("key").is_in(list(R.TARGETS)))
        print("\n[CBS consensus, rescored to this league's non-PPR scoring]")
        with pl.Config(float_precision=1, tbl_cols=14):
            print(c.select("player", "team", "cbs_rb_rank", "games", "rush_att",
                           "rush_yds", "rush_td", "targets", "receptions",
                           "rec_yds", "rec_td", "cbs_std_points", "cbs_std_ppg"))
    except Exception as e:  # network / layout change must not kill the panel
        cbs = None
        print(f"\n[CBS] unavailable: {e}")

    return {"ids": ids, "rosters": mine, "teams26": teams26, "rooms": rooms,
            "adp": rbadp, "cbs": cbs, "weekly_2026": weekly_2026}


def prior_usage(panel: pl.DataFrame) -> pl.DataFrame:
    print(BAR, "\n 2. WHAT EACH BACK ACTUALLY DID, 2023-2025\n", BAR)
    ss = F.season_rollup(panel, min_games=1)
    ids = R.target_ids()["gsis_id"].to_list()
    mine = ss.filter(pl.col("player_id").is_in(ids) & (pl.col("season") >= 2023))
    cols = ["player_display_name", "season", "team", "g", "pos_rank", "fp_total",
            "fp_pg", "carries_pg", "targets_pg", "snap_share", "carry_share",
            "rz_carry_share", "gl_carries_pg", "ypc", "rush_td_rate",
            "implied_team_total"]
    out = (mine.select(cols).sort(["player_display_name", "season"])
           .with_columns(pl.col("player_display_name").str.slice(0, 12).alias("rb"))
           .drop("player_display_name"))
    with pl.Config(tbl_cols=20, float_precision=3, fmt_str_lengths=13,
                   tbl_width_chars=200):
        print(out.select("rb", "season", "team", "g", "pos_rank", "fp_total",
                         "fp_pg", "carries_pg", "targets_pg", "snap_share"))
        print(out.select("rb", "season", "carry_share", "rz_carry_share",
                         "gl_carries_pg", "ypc", "rush_td_rate",
                         "implied_team_total"))
    return out


def main() -> None:
    sit = verify_situations()

    print("\n", BAR, "\n 3. PANEL BUILD\n", BAR)
    panel = F.load_or_build(R.PANEL_SEASONS, refresh=True)
    print(f"rows={panel.height}  players={panel['player_id'].n_unique()}  "
          f"seasons={panel['season'].min()}-{panel['season'].max()}")

    cov = pl.DataFrame({
        "column": panel.columns,
        "null_pct": [round(panel[c].null_count() / panel.height * 100, 2)
                     for c in panel.columns],
    }).filter(pl.col("null_pct") > 0).sort("null_pct", descending=True)
    print("\n[coverage] columns with missing data\n", cov)

    prior_usage(panel)

    print("\n", BAR, "\n 4. AS-OF INTEGRITY CHECK\n", BAR)
    F._demo()
    print(f"\npanel written to {F.PANEL_PATH}")


if __name__ == "__main__":
    main()
