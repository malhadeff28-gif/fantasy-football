"""As-of D/ST features. Same shift-then-roll discipline as the QB panel.

Every rolling statistic is computed on .shift(1), so a game can never enter its
own feature. Opponent-offence quality is built the same way from the opposing
side of the ledger.
"""
from __future__ import annotations

import polars as pl

FORM_W = 8      # recent defensive form
OPP_W = 8       # recent offensive generosity of the opponent
SHRINK_K = 4.0


def add_dst_features(d: pl.DataFrame) -> pl.DataFrame:
    """Attach prior-only features to a D/ST panel."""
    d = d.sort(["team", "season", "week"])

    def roll(col: str, alias: str, w: int = FORM_W) -> pl.Expr:
        return (pl.col(col).shift(1).rolling_mean(window_size=w, min_samples=1)
                .over("team").alias(alias))

    d = d.with_columns(
        roll("dst_points", "form_dst_pts"),
        pl.col("dst_points").shift(1).rolling_std(window_size=FORM_W, min_samples=2)
        .over("team").alias("form_dst_sd"),
        roll("def_sacks", "form_sacks"),
        roll("def_interceptions", "form_ints"),
        roll("fumble_recovery_opp", "form_fum"),
        roll("def_tds", "form_def_td"),
        roll("points_allowed", "form_pa"),
        pl.col("dst_points").shift(1).is_not_null().cum_sum().over("team")
        .alias("games_prior"),
    )

    # How generous each OFFENCE has been to opposing defenses, prior only.
    # This is the direct analogue of opponent difficulty on the QB side: a
    # defense scores well largely because of who it is playing.
    gen = (d.select(pl.col("opponent").alias("off_team"), "season", "week",
                    pl.col("dst_points").alias("allowed_dst_pts"),
                    pl.col("points_allowed").alias("off_points_scored"))
           .sort(["off_team", "season", "week"])
           .with_columns(
               pl.col("allowed_dst_pts").shift(1)
               .rolling_mean(window_size=OPP_W, min_samples=1)
               .over("off_team").alias("opp_generosity"),
               pl.col("off_points_scored").shift(1)
               .rolling_mean(window_size=OPP_W, min_samples=1)
               .over("off_team").alias("opp_ppg_prior"),
               pl.col("allowed_dst_pts").shift(1).is_not_null().cum_sum()
               .over("off_team").alias("opp_games_prior"),
           )
           .select(["off_team", "season", "week", "opp_generosity",
                    "opp_ppg_prior", "opp_games_prior"]))

    d = d.join(gen, left_on=["opponent", "season", "week"],
               right_on=["off_team", "season", "week"], how="left")

    # Shrink generosity toward the league mean so a week-2 opponent cannot look
    # extreme on a single game of evidence.
    lg = d["opp_generosity"].mean()
    return d.with_columns(
        ((pl.col("opp_generosity") * pl.col("opp_games_prior") + lg * SHRINK_K)
         / (pl.col("opp_games_prior") + SHRINK_K)).alias("opp_generosity_shrunk")
    )


FEATURES = ["form_dst_pts", "form_sacks", "form_ints", "form_fum", "form_pa",
            "opp_generosity_shrunk", "opp_ppg_prior",
            "implied_opp_total", "implied_team_total", "spread_abs",
            "is_home", "indoors"]
