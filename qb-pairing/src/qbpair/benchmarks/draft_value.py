"""Draft capital and roster-slot value.

A second quarterback is not free. He costs a draft pick and one of seven bench
slots, and the pairing thesis only holds if his marginal contribution clears
what that pick and slot would otherwise return.

The unit of account here is **points above replacement (PAR)**: a player's
season total minus what the league's last startable player at his position
scored. Replacement levels follow the target league exactly -- 12 teams,
1QB/2RB/3WR/1TE, no flex -- so QB replacement is QB12, RB is RB24, WR is WR36,
TE is TE12.

For a bench player the relevant quantity is `max(0, PAR)`: a pick that finishes
below replacement is not a loss of that magnitude, it simply never gets started.
"""
from __future__ import annotations

import re

import polars as pl

from ..config import LEAGUE
from ..data import loaders

# Starters x teams = how deep the league digs at each position.
REPLACEMENT_RANK = {
    "QB": LEAGUE.teams * LEAGUE.starters["QB"],   # 12
    "RB": LEAGUE.teams * LEAGUE.starters["RB"],   # 24
    "WR": LEAGUE.teams * LEAGUE.starters["WR"],   # 36
    "TE": LEAGUE.teams * LEAGUE.starters["TE"],   # 12
}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)


def normalise_name(name: str) -> str:
    """Loose key for joining FFC names to nflverse names."""
    n = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _SUFFIX.sub("", n)
    return " ".join(n.split())


def season_totals(seasons: list[int]) -> pl.DataFrame:
    """Regular-season fantasy totals per player, standard scoring."""
    st = loaders.load_player_stats(seasons).filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(list(REPLACEMENT_RANK))
    )
    return (
        st.group_by(["season", "player_id", "player_display_name", "position"])
        .agg(pl.col("fantasy_points").sum().alias("season_points"),
             pl.len().alias("games"))
        .with_columns(pl.col("player_display_name")
                      .map_elements(normalise_name, return_dtype=pl.String)
                      .alias("key"))
    )


def replacement_levels(totals: pl.DataFrame) -> pl.DataFrame:
    """The season-points total of the last startable player at each position."""
    rows = []
    for (season, pos), g in totals.group_by(["season", "position"]):
        rank = REPLACEMENT_RANK[pos]
        vals = g["season_points"].sort(descending=True).to_list()
        rows.append({
            "season": season, "position": pos,
            "replacement_points": vals[rank - 1] if len(vals) >= rank else (
                vals[-1] if vals else 0.0),
        })
    return pl.DataFrame(rows)


def points_above_replacement(seasons: list[int]) -> pl.DataFrame:
    totals = season_totals(seasons)
    repl = replacement_levels(totals)
    return totals.join(repl, on=["season", "position"], how="left").with_columns(
        (pl.col("season_points") - pl.col("replacement_points")).alias("par")
    ).with_columns(
        pl.col("par").clip(lower_bound=0.0).alias("usable_par")
    )


def load_adp(seasons: list[int], teams: int = 12) -> pl.DataFrame:
    """Historical ADP from real drafts, one row per player-season."""
    frames = []
    for s in seasons:
        try:
            df = loaders.load_ffc_adp(s, teams=teams)
        except Exception:
            continue
        frames.append(
            df.select(
                pl.lit(s).alias("season"),
                pl.col("name").alias("adp_name"),
                pl.col("position").alias("adp_position"),
                pl.col("adp").cast(pl.Float64),
                pl.col("times_drafted").cast(pl.Int64),
                pl.col("stdev").cast(pl.Float64),
            ).with_columns(
                pl.col("adp_name").map_elements(normalise_name,
                                                return_dtype=pl.String).alias("key")
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()


def adp_with_outcomes(seasons: list[int]) -> pl.DataFrame:
    """Join what a player cost on draft day to what he actually returned."""
    adp = load_adp(seasons)
    par = points_above_replacement(seasons)
    return adp.join(par, on=["season", "key"], how="inner")


def pick_value_curve(seasons: list[int], bucket: int = 12) -> pl.DataFrame:
    """Expected usable PAR by draft-pick bucket -- the cost of spending a pick.

    Excludes quarterbacks, since the question is what a pick returns when spent
    on somebody other than a second QB.
    """
    d = adp_with_outcomes(seasons).filter(pl.col("position") != "QB")
    return (
        d.with_columns(((pl.col("adp") - 1) // bucket).cast(pl.Int32).alias("bkt"))
        .group_by("bkt")
        .agg(
            pl.len().alias("n"),
            (pl.col("bkt").first() * bucket + 1).alias("pick_from"),
            (pl.col("bkt").first() * bucket + bucket).alias("pick_to"),
            pl.col("usable_par").mean().alias("mean_usable_par"),
            pl.col("usable_par").median().alias("median_usable_par"),
            (pl.col("par") > 0).mean().alias("hit_rate"),
        )
        .sort("bkt")
    )
