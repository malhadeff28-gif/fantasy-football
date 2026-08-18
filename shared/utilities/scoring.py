"""League format and scoring, shared by every lab.

One definition of the league so three projects cannot silently disagree about
what a point is. The QB implementation is verified: reconstructing nflverse
fantasy_points from component stats matched to 0.0 maximum absolute difference
across 1,389 QB-weeks in 2024-2025.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class Scoring:
    """Standard, 4-point passing touchdowns."""

    pass_yds: float = 0.04
    pass_td: float = 4.0
    interception: float = -2.0
    rush_yds: float = 0.1
    rush_td: float = 6.0
    rec: float = 0.0          # non-PPR
    rec_yds: float = 0.1
    rec_td: float = 6.0
    fumble_lost: float = -2.0
    two_point: float = 2.0


@dataclass(frozen=True)
class DSTScoring:
    """Team defense / special teams.

    Points-allowed tiers follow the CBS and Yahoo convention, which is what this
    league uses. ESPN uses a compressed top tier (5 rather than 10 for a
    shutout); `points_allowed_espn` is provided so sensitivity to that choice can
    be measured rather than assumed.
    """

    sack: float = 1.0
    interception: float = 2.0
    fumble_recovery: float = 2.0
    safety: float = 2.0
    defensive_td: float = 6.0
    special_teams_td: float = 6.0
    blocked_kick: float = 2.0

    @staticmethod
    def points_allowed(pa: float) -> float:
        if pa <= 0: return 10.0
        if pa <= 6: return 7.0
        if pa <= 13: return 4.0
        if pa <= 20: return 1.0
        if pa <= 27: return 0.0
        if pa <= 34: return -1.0
        return -4.0

    @staticmethod
    def points_allowed_espn(pa: float) -> float:
        if pa <= 0: return 5.0
        if pa <= 6: return 4.0
        if pa <= 13: return 3.0
        if pa <= 17: return 1.0
        if pa <= 27: return 0.0
        if pa <= 34: return -1.0
        if pa <= 45: return -3.0
        return -5.0


@dataclass(frozen=True)
class League:
    """12 teams, 1QB/2RB/3WR/1TE/K/DST, no flex, 7 bench."""

    teams: int = 12
    starters: dict = field(default_factory=lambda: {
        "QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1})
    bench_slots: int = 7
    has_flex: bool = False
    scoring: Scoring = field(default_factory=Scoring)
    dst_scoring: DSTScoring = field(default_factory=DSTScoring)
    regular_season_weeks: tuple = tuple(range(1, 14))
    playoff_weeks: tuple = (14, 15, 16)

    @property
    def roster_size(self) -> int:
        return sum(self.starters.values()) + self.bench_slots

    @property
    def replacement_rank(self) -> dict:
        """The last startable player at each position across the league."""
        return {p: self.teams * n for p, n in self.starters.items()}


LEAGUE = League()
TARGET_SEASON = 2026

# Modern rushing-QB era. Line coverage would allow 2007+, retained as EXTENDED
# for robustness checks.
BACKTEST_SEASONS = tuple(range(2018, 2026))
BACKTEST_SEASONS_EXTENDED = tuple(range(2007, 2026))


def skill_points(df: pl.DataFrame, s: Scoring = Scoring()) -> pl.Expr:
    """Fantasy points for a QB/RB/WR/TE row from component stats."""
    fum = (pl.col("sack_fumbles_lost").fill_null(0)
           + pl.col("rushing_fumbles_lost").fill_null(0)
           + pl.col("receiving_fumbles_lost").fill_null(0))
    two = (pl.col("passing_2pt_conversions").fill_null(0)
           + pl.col("rushing_2pt_conversions").fill_null(0)
           + pl.col("receiving_2pt_conversions").fill_null(0))
    return (
        pl.col("passing_yards").fill_null(0) * s.pass_yds
        + pl.col("passing_tds").fill_null(0) * s.pass_td
        + pl.col("passing_interceptions").fill_null(0) * s.interception
        + pl.col("rushing_yards").fill_null(0) * s.rush_yds
        + pl.col("rushing_tds").fill_null(0) * s.rush_td
        + pl.col("receptions").fill_null(0) * s.rec
        + pl.col("receiving_yards").fill_null(0) * s.rec_yds
        + pl.col("receiving_tds").fill_null(0) * s.rec_td
        + fum * s.fumble_lost
        + two * s.two_point
    )


def dst_points(team_week: pl.DataFrame, points_allowed_col: str = "points_allowed",
               d: DSTScoring = DSTScoring(), espn_tiers: bool = False) -> pl.DataFrame:
    """Add a dst_points column to a team-week frame.

    Expects the nflverse team_stats defensive columns plus a points-allowed
    column joined from the schedule.
    """
    tier = d.points_allowed_espn if espn_tiers else d.points_allowed
    blocks = (pl.col("def_punt_blocks").fill_null(0)
              + pl.col("def_pat_blocks").fill_null(0)
              + pl.col("def_fg_blocks").fill_null(0))
    base = (
        pl.col("def_sacks").fill_null(0) * d.sack
        + pl.col("def_interceptions").fill_null(0) * d.interception
        + pl.col("fumble_recovery_opp").fill_null(0) * d.fumble_recovery
        + pl.col("def_safeties").fill_null(0) * d.safety
        + pl.col("def_tds").fill_null(0) * d.defensive_td
        + pl.col("special_teams_tds").fill_null(0) * d.special_teams_td
        + blocks * d.blocked_kick
    )
    return team_week.with_columns(
        base.alias("_dst_base"),
        pl.col(points_allowed_col)
        .map_elements(tier, return_dtype=pl.Float64).alias("_dst_pa"),
    ).with_columns(
        (pl.col("_dst_base") + pl.col("_dst_pa")).alias("dst_points")
    )
