"""D/ST weekly panel: reconstruct defense fantasy scoring from raw components.

Points allowed comes from the schedule (the opponent score), everything else
from nflverse team stats. Scoring itself lives in shared/utilities/scoring.py
so all three labs agree on what a point is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))
import bootstrap                                  # noqa: E402
bootstrap.setup()
import nfl_data                                   # noqa: E402
import scoring                                    # noqa: E402


def build_dst_panel(seasons: list[int], espn_tiers: bool = False) -> pl.DataFrame:
    """One row per defense-week with realised fantasy points."""
    ts = nfl_data.load_team_stats(seasons).filter(pl.col("season_type") == "REG")

    sched = nfl_data.load_schedules().select(
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "total_line",
        "roof", "div_game", "gameday")

    # points allowed = whatever the opponent scored
    pa = pl.concat([
        sched.select("game_id", pl.col("home_team").alias("team"),
                     pl.col("away_score").alias("points_allowed"),
                     pl.col("home_score").alias("points_for")),
        sched.select("game_id", pl.col("away_team").alias("team"),
                     pl.col("home_score").alias("points_allowed"),
                     pl.col("away_score").alias("points_for")),
    ])

    ctx = pl.concat([
        nfl_data.team_game_context(s) for s in seasons
    ]).filter(pl.col("game_type") == "REG").select(
        "game_id", "team", "opponent", "is_home", "team_spread", "total_line",
        "implied_team_total", "implied_opp_total", "indoors", "spread_abs",
        "div_game", "gameday")

    df = (ts.join(pa, on=["game_id", "team"], how="inner")
            .join(ctx, on=["game_id", "team"], how="inner"))
    return scoring.dst_points(df, "points_allowed", espn_tiers=espn_tiers)


DST_KEEP = [
    "season", "week", "game_id", "gameday", "team", "opponent", "is_home",
    "points_allowed", "points_for", "dst_points", "_dst_base", "_dst_pa",
    "def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties",
    "def_tds", "special_teams_tds",
    "team_spread", "total_line", "implied_team_total", "implied_opp_total",
    "indoors", "spread_abs", "div_game",
]
