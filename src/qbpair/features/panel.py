"""Vectorised QB-week panel with as-of (no look-ahead) guarantees.

`pregame.build_qb_week` filters history explicitly for a single week, which is
easy to verify but slow across many seasons. This module produces the same
features for every QB-week at once using a shift-then-roll construction: every
rolling statistic is computed on `.shift(1)`, so the current game can never
enter its own feature. `validate_against_pregame()` cross-checks that the two
implementations agree.

De facto starters only: rows require >= 10 pass attempts. This makes the panel
"given he started, how did he score", and deliberately sets aside the separate
problem of identifying the starter pregame (depth charts / injuries).
"""
from __future__ import annotations

import polars as pl

from ..data import loaders

MIN_ATT = 10
FORM_WINDOW = 8      # QB rolling form, games
DEF_WINDOW = 17      # opponent defence, roughly one season
SHRINK_K = 6.0       # games of league-prior weight


def _qb_form_panel(stats: pl.DataFrame) -> pl.DataFrame:
    """Per QB-week rolling form, computed only from that quarterback's earlier games."""
    s = stats.sort(["player_id", "season", "week"])

    def prior_roll(col: str, alias: str, w: int = FORM_WINDOW) -> pl.Expr:
        return (
            pl.col(col).shift(1).rolling_mean(window_size=w, min_samples=1)
            .over("player_id").alias(alias)
        )

    return s.with_columns(
        prior_roll("fantasy_points", "form_fp_mean"),
        pl.col("fantasy_points").shift(1)
        .rolling_std(window_size=FORM_WINDOW, min_samples=2)
        .over("player_id").alias("form_fp_sd"),
        prior_roll("attempts", "form_pass_att"),
        prior_roll("carries", "form_rush_att"),
        prior_roll("rushing_yards", "form_rush_yds"),
        prior_roll("passing_yards", "form_pass_yds"),
        prior_roll("passing_tds", "form_pass_td"),
        prior_roll("rushing_tds", "form_rush_td"),
        # play-style inputs for archetype work (Phase 4)
        prior_roll("passing_air_yards", "form_air_yds"),
        prior_roll("passing_cpoe", "form_cpoe"),
        prior_roll("sacks_suffered", "form_sacks"),
        pl.col("fantasy_points").shift(1).is_not_null().cum_sum()
        .over("player_id").alias("games_prior"),
        pl.col("fantasy_points").shift(1).cum_sum().over("player_id").alias("_csum"),
    ).with_columns(
        (pl.col("_csum") / pl.col("games_prior")).alias("career_fp_mean")
    ).with_columns(
        # play-style axes, all from prior games only
        (
            (pl.col("form_rush_yds") * 0.1 + pl.col("form_rush_td") * 6)
            / pl.col("form_fp_mean").clip(lower_bound=1.0)
        ).alias("rush_fp_share"),
        # average depth of target: how far downfield he throws
        (pl.col("form_air_yds") / pl.col("form_pass_att").clip(lower_bound=1.0))
        .alias("form_adot"),
        # pressure exposure
        (pl.col("form_sacks")
         / (pl.col("form_pass_att") + pl.col("form_sacks")).clip(lower_bound=1.0))
        .alias("form_sack_rate"),
    ).drop("_csum")


def _defense_panel(stats: pl.DataFrame) -> pl.DataFrame:
    """Per team-week QB fantasy points allowed, shrunk toward an as-of league mean."""
    tw = (
        stats.group_by(["opponent_team", "season", "week"])
        .agg(pl.col("fantasy_points").max().alias("qb_fp_allowed"))
        .sort(["opponent_team", "season", "week"])
    )

    # The league mean is itself computed only from earlier weeks, so the
    # shrinkage target carries no future information either.
    lg = (
        tw.group_by(["season", "week"])
        .agg(pl.col("qb_fp_allowed").mean().alias("lg_wk"))
        .sort(["season", "week"])
        .with_columns(
            pl.col("lg_wk").shift(1).cum_sum().alias("_c"),
            pl.col("lg_wk").shift(1).is_not_null().cum_sum().alias("_n"),
        )
        .with_columns((pl.col("_c") / pl.col("_n")).alias("lg_prior"))
        .select(["season", "week", "lg_prior"])
    )

    tw = tw.with_columns(
        pl.col("qb_fp_allowed").shift(1)
        .rolling_mean(window_size=DEF_WINDOW, min_samples=1)
        .over("opponent_team").alias("def_raw"),
        pl.col("qb_fp_allowed").shift(1).is_not_null().cum_sum()
        .over("opponent_team").alias("def_n"),
    ).join(lg, on=["season", "week"], how="left")

    return tw.with_columns(
        (
            (pl.col("def_raw").fill_null(pl.col("lg_prior")) * pl.col("def_n")
             + pl.col("lg_prior") * SHRINK_K)
            / (pl.col("def_n") + SHRINK_K)
        ).alias("def_fp_allowed")
    ).with_columns(
        (pl.col("def_fp_allowed") - pl.col("lg_prior")).alias("def_adj")
    ).select(
        pl.col("opponent_team").alias("opponent"), "season", "week",
        "def_fp_allowed", "def_adj", "def_n", "lg_prior",
    )


def build_panel(seasons: list[int]) -> pl.DataFrame:
    """QB-week panel: pregame features plus the realised fantasy points."""
    # Pull one extra prior season so early weeks have form history.
    load_seasons = list(range(min(seasons) - 1, max(seasons) + 1))
    stats = (
        loaders.load_player_stats(load_seasons)
        .filter(
            (pl.col("position") == "QB")
            & (pl.col("season_type") == "REG")
            & (pl.col("attempts") >= MIN_ATT)
        )
    )
    schedules = loaders.load_schedules()

    form = _qb_form_panel(stats)
    dfn = _defense_panel(stats)

    ctx = pl.concat([
        schedules.select(
            "game_id", "season", "week", "gameday",
            pl.col("home_team").alias("team"),
            pl.lit(1).alias("is_home"),
            pl.col("spread_line").alias("team_spread"),
            "total_line", pl.col("home_rest").alias("rest"),
            "div_game", "roof",
        ),
        schedules.select(
            "game_id", "season", "week", "gameday",
            pl.col("away_team").alias("team"),
            pl.lit(0).alias("is_home"),
            (-pl.col("spread_line")).alias("team_spread"),
            "total_line", pl.col("away_rest").alias("rest"),
            "div_game", "roof",
        ),
    ]).with_columns(
        (pl.col("total_line") / 2 + pl.col("team_spread") / 2).alias("implied_team_total"),
        (pl.col("total_line") / 2 - pl.col("team_spread") / 2).alias("implied_opp_total"),
        pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8).alias("indoors"),
        pl.col("team_spread").abs().alias("spread_abs"),
    ).drop("roof")

    panel = (
        form.rename({"opponent_team": "opponent"})
        .join(ctx, on=["game_id", "season", "week", "team"], how="inner")
        .join(dfn, on=["opponent", "season", "week"], how="left")
        .filter(pl.col("season").is_in(seasons))
    )

    keep = [
        "player_id", "player_display_name", "season", "week", "gameday",
        "game_id", "team", "opponent", "fantasy_points",
        "form_fp_mean", "form_fp_sd", "career_fp_mean", "games_prior",
        "form_pass_att", "form_rush_att", "form_rush_yds", "rush_fp_share",
        "form_adot", "form_cpoe", "form_sack_rate",
        "is_home", "team_spread", "spread_abs", "total_line",
        "implied_team_total", "implied_opp_total", "rest", "div_game", "indoors",
        "def_fp_allowed", "def_adj", "def_n",
    ]
    return panel.select([c for c in keep if c in panel.columns]).sort(
        ["season", "week", "player_id"]
    )


def validate_against_pregame(season: int, week: int, n: int = 6) -> pl.DataFrame:
    """Cross-check the vectorised panel against the explicit-filter builder."""
    from . import pregame

    stats = loaders.load_player_stats(list(range(season - 2, season + 1))).filter(
        pl.col("season_type") == "REG"
    )
    schedules = loaders.load_schedules()
    slow = pregame.build_qb_week(stats, schedules, season, week).select(
        "player_display_name",
        pl.col("implied_team_total").alias("slow_itt"),
        pl.col("team_spread").alias("slow_spread"),
    )
    fast = build_panel([season]).filter(pl.col("week") == week).select(
        "player_display_name",
        pl.col("implied_team_total").alias("fast_itt"),
        pl.col("team_spread").alias("fast_spread"),
    )
    return slow.join(fast, on="player_display_name", how="inner").with_columns(
        (pl.col("slow_itt") - pl.col("fast_itt")).abs().alias("itt_diff"),
        (pl.col("slow_spread") - pl.col("fast_spread")).abs().alias("spread_diff"),
    ).head(n)
