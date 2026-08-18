"""Pregame feature construction with enforced as-of (no look-ahead) discipline.

Every feature in this module is derived only from rows that were observable
*before* the target game kicked off. The single primitive `as_of_expr` encodes
that rule; all feature builders route through it so look-ahead cannot sneak in
via a forgotten filter.
"""
from __future__ import annotations

import polars as pl

LEAGUE_MEAN_PRIOR_GAMES = 6.0  # shrinkage strength for opponent priors


def as_of_expr(season: int, week: int) -> pl.Expr:
    """Rows strictly earlier than (season, week) -- the no-look-ahead rule."""
    return (pl.col("season") < season) | (
        (pl.col("season") == season) & (pl.col("week") < week)
    )


def history(df: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """All rows observable before the given season/week."""
    return df.filter(as_of_expr(season, week))


# --- game context (known at kickoff) ----------------------------------------

def team_game_context(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """One row per team playing in that week, from the team's perspective.

    Betting inputs are decomposed rather than used raw: a team's implied point
    total carries the fantasy signal, while the spread carries game script.
    """
    g = schedules.filter((pl.col("season") == season) & (pl.col("week") == week))

    home = g.select(
        pl.col("game_id"), pl.col("season"), pl.col("week"), pl.col("gameday"),
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
        pl.lit(1).alias("is_home"),
        pl.col("spread_line").alias("team_spread"),      # + = team favoured
        pl.col("total_line"), pl.col("home_rest").alias("rest"),
        pl.col("div_game"), pl.col("roof"), pl.col("surface"),
    )
    away = g.select(
        pl.col("game_id"), pl.col("season"), pl.col("week"), pl.col("gameday"),
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
        pl.lit(0).alias("is_home"),
        (-pl.col("spread_line")).alias("team_spread"),
        pl.col("total_line"), pl.col("away_rest").alias("rest"),
        pl.col("div_game"), pl.col("roof"), pl.col("surface"),
    )

    return pl.concat([home, away]).with_columns(
        # implied points: total/2 + spread/2 from the team's own perspective
        (pl.col("total_line") / 2 + pl.col("team_spread") / 2).alias("implied_team_total"),
        (pl.col("total_line") / 2 - pl.col("team_spread") / 2).alias("implied_opp_total"),
        (pl.col("roof").is_in(["dome", "closed"])).cast(pl.Int8).alias("indoors"),
        pl.col("team_spread").abs().alias("game_closeness"),  # smaller = tighter
    )


# --- quarterback form (prior games only) ------------------------------------

def qb_form(
    stats: pl.DataFrame, season: int, week: int, window: int = 8
) -> pl.DataFrame:
    """Rolling QB form computed only from games before (season, week)."""
    hist = history(stats, season, week).filter(pl.col("position") == "QB")
    hist = hist.sort(["player_id", "season", "week"])

    recent = (
        hist.group_by("player_id")
        .tail(window)
        .group_by("player_id")
        .agg(
            pl.len().alias("form_games"),
            pl.col("fantasy_points").mean().alias("form_fp_mean"),
            pl.col("fantasy_points").std().alias("form_fp_sd"),
            pl.col("attempts").mean().alias("form_pass_att"),
            pl.col("carries").mean().alias("form_rush_att"),
            pl.col("rushing_yards").mean().alias("form_rush_yds"),
            pl.col("passing_yards").mean().alias("form_pass_yds"),
            pl.col("passing_tds").mean().alias("form_pass_td"),
            pl.col("rushing_tds").mean().alias("form_rush_td"),
            pl.col("player_display_name").last().alias("player_display_name"),
        )
    )

    career = hist.group_by("player_id").agg(
        pl.len().alias("career_games"),
        pl.col("fantasy_points").mean().alias("career_fp_mean"),
    )

    out = recent.join(career, on="player_id", how="left")
    # Rushing share of fantasy production = the core "play style" axis.
    return out.with_columns(
        (
            (pl.col("form_rush_yds") * 0.1 + pl.col("form_rush_td") * 6)
            / pl.col("form_fp_mean").clip(lower_bound=1.0)
        ).alias("rush_fp_share")
    )


# --- opponent defence (prior games only) ------------------------------------

def defense_vs_qb(
    stats: pl.DataFrame, season: int, week: int, lookback_seasons: int = 1
) -> pl.DataFrame:
    """Fantasy points allowed to the primary QB, shrunk toward league mean.

    Early in a season a team has few observed games, so a raw average is mostly
    noise; shrinkage keeps Week-2 opponents from looking like elite defences on
    one game of evidence.
    """
    hist = history(stats, season, week).filter(
        (pl.col("position") == "QB")
        & (pl.col("season") >= season - lookback_seasons)
        & (pl.col("attempts") >= 10)  # the game's actual starter
    )

    allowed = (
        hist.group_by(["opponent_team", "season", "week"])
        .agg(pl.col("fantasy_points").max().alias("qb_fp_allowed"))
        .group_by("opponent_team")
        .agg(
            pl.len().alias("def_games"),
            pl.col("qb_fp_allowed").mean().alias("def_fp_allowed_raw"),
        )
        .rename({"opponent_team": "opponent"})
    )

    league_mean = allowed["def_fp_allowed_raw"].mean()
    k = LEAGUE_MEAN_PRIOR_GAMES
    return allowed.with_columns(
        (
            (pl.col("def_fp_allowed_raw") * pl.col("def_games") + league_mean * k)
            / (pl.col("def_games") + k)
        ).alias("def_fp_allowed"),
        pl.lit(league_mean).alias("def_league_mean"),
    ).with_columns(
        (pl.col("def_fp_allowed") - pl.col("def_league_mean")).alias("def_adj")
    )


# --- assembly ---------------------------------------------------------------

def build_qb_week(
    stats: pl.DataFrame, schedules: pl.DataFrame, season: int, week: int
) -> pl.DataFrame:
    """Pregame feature table for every QB with a game in (season, week)."""
    ctx = team_game_context(schedules, season, week)
    form = qb_form(stats, season, week)
    dfn = defense_vs_qb(stats, season, week)

    # Who is on which roster this week is itself pregame information; we take it
    # from the most recent observed game rather than from the target week's
    # box score, so nothing leaks from the game being predicted.
    last_team = (
        history(stats, season, week)
        .filter(pl.col("position") == "QB")
        .sort(["player_id", "season", "week"])
        .group_by("player_id")
        .agg(pl.col("team").last().alias("team"))
    )

    return (
        form.join(last_team, on="player_id", how="inner")
        .join(ctx, on="team", how="inner")
        .join(dfn, on="opponent", how="left")
    )
