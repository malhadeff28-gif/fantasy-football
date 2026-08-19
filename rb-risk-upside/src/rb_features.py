"""RB-week panel with as-of (no look-ahead) features, plus season rollups.

Construction is the shift-then-roll discipline of
`qb-pairing/src/qbpair/features/panel.py`: every rolling statistic is computed
on `.shift(1)` within `.over("player_id")`, so a game can never enter its own
feature. Market context (spread, total, implied team total) is closing-line
data that exists before kickoff and is therefore used unshifted, exactly as the
QB panel does.

Unlike the QB panel there is no "de facto starter" attempt filter. A running
back's committee share IS the object of study, so every RB-game with a recorded
snap or touch stays in and `snap_share` carries the role information.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import polars as pl  # noqa: E402

import rb_data as R  # noqa: E402
import nfl_data  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")

FORM = 8      # rolling form window, games
SHORT = 4     # recent-role window, games
PANEL_PATH = R.DATA / "rb_week_panel.parquet"

# Workload columns whose season-to-season behaviour RB-002 tests.
ROLE_COLS = ["snap_share", "carry_share", "target_share_rb", "rz_carry_share",
             "gl_carries_pg", "targets_pg", "carries_pg", "touches_pg",
             "rz_targets_pg"]
EFF_COLS = ["ypc", "ypr", "catch_rate", "rush_td_rate", "td_per_touch",
            "fp_per_touch", "fp_pg"]


def _raw_rb_games(seasons: list[int]) -> pl.DataFrame:
    """One row per RB-game: box score + snaps + red-zone + team denominators."""
    load = list(range(min(seasons) - 1, max(seasons) + 1))
    s = R.rb_weekly(load).select(
        "player_id", "player_display_name", "season", "week", "game_id", "team",
        "opponent_team", "carries", "rushing_yards", "rushing_tds", "targets",
        "receptions", "receiving_yards", "receiving_tds", "target_share", "fp",
        (pl.col("rushing_fumbles_lost").fill_null(0)
         + pl.col("receiving_fumbles_lost").fill_null(0)).alias("fumbles_lost"),
    )

    snaps = (nfl_data.load_snap_counts(load)
             .filter(pl.col("game_type") == "REG")
             .join(R.pfr_crosswalk(), on="pfr_player_id", how="inner")
             .select(pl.col("gsis_id").alias("player_id"), "season", "week",
                     "offense_snaps", pl.col("offense_pct").alias("snap_share")))

    pl_pbp, tm_pbp = R.pbp_rb_game(load)

    df = (s.join(snaps, on=["player_id", "season", "week"], how="left")
          .join(pl_pbp, on=["player_id", "season", "week"], how="left")
          .join(tm_pbp, left_on=["team", "season", "week"],
                right_on=["team", "season", "week"], how="left"))

    z = pl.col("carries").fill_null(0) + pl.col("targets").fill_null(0)
    return df.with_columns(
        pl.col(["rz_carries", "gl_carries", "rz_targets", "pbp_rush_att",
                "pbp_targets"]).fill_null(0),
        z.alias("touches"),
    ).with_columns(
        # shares against that game's own team denominators
        (pl.col("carries") / pl.col("team_rush_att").clip(lower_bound=1)).alias("carry_share"),
        (pl.col("rz_carries") / pl.col("team_rz_carries").clip(lower_bound=1)).alias("rz_carry_share"),
        (pl.col("gl_carries") / pl.col("team_gl_carries").clip(lower_bound=1)).alias("gl_carry_share"),
        (pl.col("targets") / pl.col("team_pass_att").clip(lower_bound=1)).alias("target_share_rb"),
        (pl.col("rushing_yards") / pl.col("carries").clip(lower_bound=1)).alias("ypc"),
        (pl.col("receiving_yards") / pl.col("receptions").clip(lower_bound=1)).alias("ypr"),
        (pl.col("receptions") / pl.col("targets").clip(lower_bound=1)).alias("catch_rate"),
        (pl.col("rushing_tds") / pl.col("carries").clip(lower_bound=1)).alias("rush_td_rate"),
        ((pl.col("rushing_tds") + pl.col("receiving_tds"))
         / z.clip(lower_bound=1)).alias("td_per_touch"),
        (pl.col("fp") / z.clip(lower_bound=1)).alias("fp_per_touch"),
    ).sort(["player_id", "season", "week"])


def build_panel(seasons: list[int] = tuple(R.PANEL_SEASONS)) -> pl.DataFrame:
    """RB-week panel: as-of features plus the realised fantasy points."""
    seasons = list(seasons)
    g = _raw_rb_games(seasons)

    def roll(col: str, alias: str, w: int = FORM) -> pl.Expr:
        return (pl.col(col).shift(1).rolling_mean(window_size=w, min_samples=1)
                .over("player_id").alias(alias))

    feat = g.with_columns(
        # --- role, long and short window (shift-then-roll: no self-leak) ----
        *[roll(c, f"f_{c}") for c in
          ("snap_share", "carry_share", "rz_carry_share", "gl_carry_share",
           "target_share_rb", "carries", "targets", "touches", "rz_carries",
           "gl_carries", "rz_targets")],
        *[roll(c, f"s_{c}", SHORT) for c in
          ("snap_share", "carry_share", "touches", "gl_carries")],
        # --- efficiency, which RB-002 tests for stability ------------------
        *[roll(c, f"f_{c}") for c in
          ("ypc", "ypr", "catch_rate", "rush_td_rate", "td_per_touch",
           "fp_per_touch", "fp")],
        pl.col("fp").shift(1).rolling_std(window_size=FORM, min_samples=2)
        .over("player_id").alias("f_fp_sd"),
        pl.col("fp").shift(1).is_not_null().cum_sum().over("player_id").alias("games_prior"),
        # --- team scoring environment, prior games only ---------------------
        pl.col("team_rush_att").shift(1).rolling_mean(window_size=FORM, min_samples=1)
        .over("team").alias("f_team_rush_att"),
        pl.col("team_gl_carries").shift(1).rolling_mean(window_size=FORM, min_samples=1)
        .over("team").alias("f_team_gl_carries"),
    )

    ctx = pl.concat([nfl_data.team_game_context(s) for s in
                     range(min(seasons) - 1, max(seasons) + 1)]).select(
        "game_id", "season", "week", "team", "is_home", "team_spread",
        "total_line", "implied_team_total", "spread_abs", "indoors", "rest")

    return (feat.join(ctx, on=["game_id", "season", "week", "team"], how="left")
            .filter(pl.col("season").is_in(seasons))
            .sort(["season", "week", "player_id"]))


def season_rollup(panel: pl.DataFrame, min_games: int = 6) -> pl.DataFrame:
    """Player-season aggregates for stability testing (RB-002).

    Rate columns are recomputed from summed components, not averaged across
    games: a mean of per-game YPC over-weights low-carry games.
    """
    a = (panel.group_by(["player_id", "player_display_name", "season"]).agg(
        pl.len().alias("g"),
        pl.col("team").mode().first().alias("team"),
        pl.col("fp").sum().alias("fp_total"),
        pl.col("fp").mean().alias("fp_pg"),
        pl.col("carries").sum().alias("carries"),
        pl.col("targets").sum().alias("targets"),
        pl.col("receptions").sum().alias("receptions"),
        pl.col("touches").sum().alias("touches"),
        pl.col("rushing_yards").sum().alias("rush_yds"),
        pl.col("receiving_yards").sum().alias("rec_yds"),
        pl.col("rushing_tds").sum().alias("rush_td"),
        pl.col("receiving_tds").sum().alias("rec_td"),
        pl.col("rz_carries").sum().alias("rz_carries"),
        pl.col("gl_carries").sum().alias("gl_carries"),
        pl.col("rz_targets").sum().alias("rz_targets"),
        pl.col("team_rush_att").sum().alias("team_rush_att"),
        pl.col("team_pass_att").sum().alias("team_pass_att"),
        pl.col("team_rz_carries").sum().alias("team_rz_carries"),
        pl.col("team_gl_carries").sum().alias("team_gl_carries"),
        pl.col("snap_share").mean().alias("snap_share"),
        pl.col("implied_team_total").mean().alias("implied_team_total"),
    ).filter(pl.col("g") >= min_games))

    return a.with_columns(
        (pl.col("carries") / pl.col("team_rush_att").clip(lower_bound=1)).alias("carry_share"),
        (pl.col("targets") / pl.col("team_pass_att").clip(lower_bound=1)).alias("target_share_rb"),
        (pl.col("rz_carries") / pl.col("team_rz_carries").clip(lower_bound=1)).alias("rz_carry_share"),
        (pl.col("gl_carries") / pl.col("g")).alias("gl_carries_pg"),
        (pl.col("rz_carries") / pl.col("g")).alias("rz_carries_pg"),
        (pl.col("rz_targets") / pl.col("g")).alias("rz_targets_pg"),
        (pl.col("carries") / pl.col("g")).alias("carries_pg"),
        (pl.col("targets") / pl.col("g")).alias("targets_pg"),
        (pl.col("touches") / pl.col("g")).alias("touches_pg"),
        (pl.col("rush_yds") / pl.col("carries").clip(lower_bound=1)).alias("ypc"),
        (pl.col("rec_yds") / pl.col("receptions").clip(lower_bound=1)).alias("ypr"),
        (pl.col("receptions") / pl.col("targets").clip(lower_bound=1)).alias("catch_rate"),
        (pl.col("rush_td") / pl.col("carries").clip(lower_bound=1)).alias("rush_td_rate"),
        ((pl.col("rush_td") + pl.col("rec_td")) / pl.col("touches").clip(lower_bound=1)
         ).alias("td_per_touch"),
        (pl.col("fp_total") / pl.col("touches").clip(lower_bound=1)).alias("fp_per_touch"),
        (pl.col("rush_td") / pl.col("gl_carries").clip(lower_bound=1)).alias("gl_td_rate"),
    ).sort(["season", "fp_total"], descending=[False, True]).with_columns(
        pl.col("fp_total").rank("ordinal", descending=True).over("season").alias("pos_rank")
    )


def load_or_build(seasons: list[int] = tuple(R.PANEL_SEASONS),
                  refresh: bool = False) -> pl.DataFrame:
    if PANEL_PATH.exists() and not refresh:
        return pl.read_parquet(PANEL_PATH)
    p = build_panel(seasons)
    p.write_parquet(PANEL_PATH)
    return p


def _demo() -> None:
    """The one check that matters: prove no feature sees its own game."""
    p = build_panel([2023, 2024])
    # A player's first panel game must have no rolling history at all.
    first = p.sort(["player_id", "season", "week"]).group_by("player_id").first()
    debut = first.filter(pl.col("games_prior") == 0)
    assert debut.height > 0, "no debut rows to test"
    assert debut["f_snap_share"].null_count() == debut.height, \
        "look-ahead: a first-ever game has a rolling snap share"
    assert debut["f_fp"].null_count() == debut.height, \
        "look-ahead: a first-ever game has rolling fantasy points"

    # And a hand-checked case: game k's f_carries must equal the mean of that
    # player's carries in games 1..k-1 (max 8 back), never including game k.
    # Only players whose career starts inside the panel can be checked this
    # way; build_panel deliberately loads one extra prior season for history,
    # so a veteran's first panel row already carries legitimate 2022 form.
    fresh = debut["player_id"].to_list()
    top = (p.filter(pl.col("player_id").is_in(fresh))
           .group_by("player_id").len().sort("len", descending=True)
           .head(1)["player_id"].item())
    one = p.filter(pl.col("player_id") == top).sort(["season", "week"])
    car = one["carries"].to_list()
    got = one["f_carries"].to_list()
    for k in range(1, len(car)):
        want = sum(car[max(0, k - FORM):k]) / len(car[max(0, k - FORM):k])
        assert abs(got[k] - want) < 1e-9, f"shift-then-roll broken at game {k}"
    print(f"panel rows={p.height} players={p['player_id'].n_unique()}  as-of check OK")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _demo()
