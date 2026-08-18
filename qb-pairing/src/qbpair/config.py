"""Project-wide configuration: league rules, scoring, paths.

Everything the model needs to know about *this* league lives here so the
research code stays reusable for other formats and seasons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]          # qb-pairing/
REPO = PROJECT.parent                                  # fantasy-football/
ROOT = PROJECT
# nflverse pulls are identical for every lab, so the cache is shared
SHARED_CACHE = REPO / "shared" / "nfl-data" / "cache"
DATA_RAW = SHARED_CACHE / "raw"
DATA_PROCESSED = SHARED_CACHE / "processed"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class Scoring:
    """Standard (non-PPR) 4-point passing-TD scoring.

    Verified against nflverse `fantasy_points` on 1,389 QB-weeks (2024-2025)
    with 0.0 maximum absolute difference -- see research/DATA_SOURCES.md.
    """

    pass_yds: float = 0.04
    pass_td: float = 4.0
    interception: float = -2.0
    rush_yds: float = 0.1
    rush_td: float = 6.0
    rec_yds: float = 0.1
    rec_td: float = 6.0
    fumble_lost: float = -2.0
    two_point: float = 2.0


@dataclass(frozen=True)
class League:
    """The target league: 12 managers, 1QB/2RB/3WR/1TE/K/DST, no flex, 7 bench."""

    teams: int = 12
    starters: dict = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1}
    )
    bench_slots: int = 7
    has_flex: bool = False
    scoring: Scoring = field(default_factory=Scoring)

    # Fantasy calendar
    regular_season_weeks: tuple = tuple(range(1, 14))   # weeks 1-13
    playoff_weeks: tuple = (14, 15, 16)                 # user-specified

    @property
    def roster_size(self) -> int:
        return sum(self.starters.values()) + self.bench_slots


LEAGUE = League()

# Season currently being optimised. The pipeline is season-agnostic; this is
# only the default target.
TARGET_SEASON = 2026

# Backtest window: modern rushing-QB era (user decision, 2026-08-17).
# Line coverage would allow 2007+, but QB fantasy scoring changed shape in the
# rushing-QB era, so the primary window is deliberately narrower. 8 seasons is
# thin for interaction terms -- prefer archetype-level effects and regularised
# models over per-QB coefficients, and always report out-of-sample.
BACKTEST_SEASONS = tuple(range(2018, 2026))      # primary: 2018-2025
BACKTEST_SEASONS_EXTENDED = tuple(range(2007, 2026))  # available for robustness
