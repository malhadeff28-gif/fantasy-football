"""RB-specific data access: shared loaders plus the four-player situation check.

Nothing here models anything. It resolves player identity, verifies the 2026
team/role situation from data rather than memory, and hands back tidy frames the
feature and simulation modules consume.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))
import bootstrap  # noqa: E402

bootstrap.setup()

import polars as pl  # noqa: E402

import markets  # noqa: E402
import nfl_data  # noqa: E402
import scoring  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")

OUT = Path(__file__).resolve().parents[1] / "outputs"
DATA = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

TARGET_SEASON = 2026
PANEL_SEASONS = list(range(2018, 2026))

# The four backs under study, keyed by markets.normalise_name.
TARGETS = {
    "dandre swift": "D'Andre Swift",
    "tony pollard": "Tony Pollard",
    "quinshon judkins": "Quinshon Judkins",
    "david montgomery": "David Montgomery",
}


def norm(name: str) -> str:
    return markets.normalise_name(name)


def target_ids() -> pl.DataFrame:
    """Resolve the four backs to gsis ids via the master player table."""
    p = nfl_data.load_players()
    keep = [c for c in ("gsis_id", "display_name", "position", "latest_team",
                        "draft_year", "draft_round", "draft_pick", "birth_date",
                        "status", "entry_year", "rookie_season", "last_season")
            if c in p.columns]
    p = p.select(keep).with_columns(
        pl.col("display_name").map_elements(norm, return_dtype=pl.String).alias("key")
    )
    return p.filter(pl.col("key").is_in(list(TARGETS)) & (pl.col("position") == "RB"))


def situation_2026() -> dict[str, pl.DataFrame]:
    """Everything needed to verify team, role and competition without memory.

    Returns frames rather than a verdict: the verdict belongs in the writeup.
    """
    ids = target_ids()
    gsis = ids["gsis_id"].to_list()

    rw = nfl_data.load_rosters_weekly([2025, TARGET_SEASON])
    ros_cols = [c for c in ("season", "week", "team", "gsis_id", "full_name",
                            "position", "depth_chart_position", "status",
                            "status_description_abbr", "years_exp")
                if c in rw.columns]
    mine = rw.select(ros_cols).filter(pl.col("gsis_id").is_in(gsis))

    dc = nfl_data.load_depth_charts([2025, TARGET_SEASON])

    adp = markets.load_ffc_adp(TARGET_SEASON)

    return {"ids": ids, "rosters": mine, "depth_charts": dc, "adp": adp}


def rb_teammates(team: str, season: int) -> pl.DataFrame:
    """Every RB on a team's roster in a season - the competition check."""
    rw = nfl_data.load_rosters_weekly([season])
    cols = [c for c in ("season", "week", "team", "gsis_id", "full_name",
                        "position", "status", "years_exp")
            if c in rw.columns]
    return (rw.select(cols)
            .filter((pl.col("team") == team) & (pl.col("position") == "RB"))
            .sort("week"))


def rb_weekly(seasons: list[int]) -> pl.DataFrame:
    """RB weekly box scores, regular season, rescored to this league."""
    s = nfl_data.load_player_stats(seasons).filter(
        (pl.col("position") == "RB") & (pl.col("season_type") == "REG")
    )
    return s.with_columns(scoring.skill_points(s).alias("fp"))


def cbs_nonppr(season: int = TARGET_SEASON) -> pl.DataFrame:
    """CBS RB projections rescored to THIS league.

    CBS publishes under a PPR configuration; its `cbs_fantasy_points` is not
    this league's number, so the components are rescored.
    """
    s = scoring.Scoring()
    c = markets.load_cbs_projections(season, "RB")
    return c.with_columns(
        (pl.col("rush_yds") * s.rush_yds + pl.col("rush_td") * s.rush_td
         + pl.col("rec_yds") * s.rec_yds + pl.col("rec_td") * s.rec_td
         + pl.col("fumbles_lost") * s.fumble_lost).alias("cbs_std_points")
    ).with_columns(
        (pl.col("cbs_std_points") / pl.col("games").clip(lower_bound=1)).alias("cbs_std_ppg"),
        pl.col("cbs_std_points").rank("ordinal", descending=True).alias("cbs_rb_rank"),
    )


def pfr_crosswalk() -> pl.DataFrame:
    """snap_counts is keyed by pfr id; everything else by gsis. Bridge them."""
    p = nfl_data.load_players()
    col = "pfr_id" if "pfr_id" in p.columns else "pfr_player_id"
    return (p.select(pl.col(col).alias("pfr_player_id"), "gsis_id")
            .drop_nulls().unique(subset=["pfr_player_id"], keep="first"))


# --- play-by-play red-zone / goal-line, memory-safe ------------------------
# Same construction as qb-pairing/scripts/wave1_advanced_qb_vars.py: pbp is 372
# columns and a whole season will not fit alongside everything else at ~1.7 GB
# free, so each season parquet is read with column pruning, aggregated, freed.

PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
           "play_by_play_{s}.parquet")
PROCESSED = REPO / "shared" / "nfl-data" / "cache" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

_PBP_COLS = ["season", "week", "season_type", "posteam", "play_type",
             "rush_attempt", "pass_attempt", "yardline_100",
             "rusher_player_id", "receiver_player_id", "rush_touchdown"]


def pbp_rb_game(seasons: list[int], refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(player-game, team-game) red-zone and goal-line opportunity counts.

    Red zone is yardline_100 <= 20, goal line <= 5, matching the qb-pairing
    definition so the two labs are comparable.
    """
    ptag = PROCESSED / f"rb001_pbp_player_{min(seasons)}_{max(seasons)}.parquet"
    ttag = PROCESSED / f"rb001_pbp_team_{min(seasons)}_{max(seasons)}.parquet"
    if ptag.exists() and ttag.exists() and not refresh:
        return pl.read_parquet(ptag), pl.read_parquet(ttag)

    import requests

    raw_dir = PROCESSED / "_pbp_tmp"
    raw_dir.mkdir(exist_ok=True)
    fp, ft = [], []

    for s in seasons:
        f = raw_dir / f"pbp_{s}.parquet"
        if not f.exists():
            r = requests.get(PBP_URL.format(s=s), timeout=900,
                             headers={"User-Agent": "rb-risk-upside/1.0"})
            r.raise_for_status()
            f.write_bytes(r.content)
        p = (pl.read_parquet(f, columns=_PBP_COLS)
             .filter((pl.col("season_type") == "REG")
                     & pl.col("posteam").is_not_null()))
        rz = pl.col("yardline_100") <= 20
        gl = pl.col("yardline_100") <= 5
        run = pl.col("rush_attempt") == 1
        pas = pl.col("pass_attempt") == 1

        rush = (p.filter(run & pl.col("rusher_player_id").is_not_null())
                .group_by(["rusher_player_id", "season", "week"]).agg(
                    pl.len().alias("pbp_rush_att"),
                    rz.sum().alias("rz_carries"),
                    gl.sum().alias("gl_carries"),
                    pl.col("rush_touchdown").fill_null(0).sum().alias("pbp_rush_td"))
                .rename({"rusher_player_id": "player_id"}))
        tgt = (p.filter(pas & pl.col("receiver_player_id").is_not_null())
               .group_by(["receiver_player_id", "season", "week"]).agg(
                   pl.len().alias("pbp_targets"),
                   rz.sum().alias("rz_targets"))
               .rename({"receiver_player_id": "player_id"}))
        fp.append(rush.join(tgt, on=["player_id", "season", "week"], how="full",
                            coalesce=True))

        ft.append(p.group_by(["posteam", "season", "week"]).agg(
            run.sum().alias("team_rush_att"),
            pas.sum().alias("team_pass_att"),
            (run & rz).sum().alias("team_rz_carries"),
            (run & gl).sum().alias("team_gl_carries"),
            (pas & rz).sum().alias("team_rz_targets"),
        ).rename({"posteam": "team"}))
        del p

    player = pl.concat(fp).fill_null(0)
    team = pl.concat(ft)
    player.write_parquet(ptag)
    team.write_parquet(ttag)
    return player, team


def _demo() -> None:
    ids = target_ids()
    assert ids.height == 4, f"expected 4 target RBs, got {ids.height}\n{ids}"
    w = rb_weekly([2024])
    # non-PPR: a 100-yard 1-TD game with 5 catches for 40 is 20.0, not 25.0
    assert abs(float(w["fp"].max()) - 40.0) < 20.0, "implausible max RB week"
    print(ids)
    print("rb_data self-check OK")


if __name__ == "__main__":
    _demo()
