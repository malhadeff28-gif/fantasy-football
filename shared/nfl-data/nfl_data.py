"""nflverse ingestion, shared by every lab in this repository.

Each pull is cached to parquet under shared/nfl-data/cache/raw so a backtest
reruns identically even if an upstream feed changes. All three projects read
the same NFL foundation, which is the point of keeping it here.

Verified 2026-08-17 against live endpoints. Note nfl_data_py is deprecated;
nflreadpy is its maintained successor.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

CACHE = Path(__file__).resolve().parent / "cache" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)


def _cached(name: str, fetch, refresh: bool = False) -> pl.DataFrame:
    path = CACHE / f"{name}.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = fetch()
    df.write_parquet(path)
    return df


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """Games 1999-2026 with closing betting lines.

    Betting columns are 100 percent populated for 2007-2025: spread_line,
    total_line, away/home_moneyline, spread odds, over/under odds. Also carries
    roof, surface, temp, wind, rest days and both starting quarterbacks.

    spread_line is positive when the HOME team is favoured; verified
    empirically at corr +0.443 with home margin over 2015-2025.
    """
    import nflreadpy as nfl

    return _cached("schedules", lambda: nfl.load_schedules(), refresh)


def load_player_stats(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Weekly player box scores including fantasy_points (standard scoring)."""
    import nflreadpy as nfl

    tag = f"player_stats_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_player_stats(seasons=seasons), refresh)


def load_team_stats(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Weekly team stats, offense and defense. The basis for D/ST scoring."""
    import nflreadpy as nfl

    tag = f"team_stats_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_team_stats(seasons=seasons), refresh)


def load_ff_rankings(refresh: bool = False) -> pl.DataFrame:
    """FantasyPros expert consensus rankings via ffverse. Carries a cbs_id."""
    import nflreadpy as nfl

    return _cached("ff_rankings", lambda: nfl.load_ff_rankings(), refresh)


def load_ff_opportunity(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Expected fantasy points, for separating skill from touchdown luck."""
    import nflreadpy as nfl

    tag = f"ff_opportunity_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_ff_opportunity(seasons=seasons), refresh)


def load_snap_counts(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Snap participation. Core input for running back workload modelling."""
    import nflreadpy as nfl

    tag = f"snap_counts_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_snap_counts(seasons=seasons), refresh)


def load_pfr_advstats(seasons: list[int], stat_type: str = "pass",
                      refresh: bool = False) -> pl.DataFrame:
    """Pro Football Reference advanced stats: pressure, blitz, hurries."""
    import nflreadpy as nfl

    tag = f"pfr_{stat_type}_{min(seasons)}_{max(seasons)}"
    return _cached(
        tag, lambda: nfl.load_pfr_advstats(seasons=seasons, stat_type=stat_type),
        refresh)


def load_nextgen_stats(seasons: list[int], stat_type: str = "passing",
                       refresh: bool = False) -> pl.DataFrame:
    """NFL Next Gen Stats: time to throw, aggressiveness, expected completion."""
    import nflreadpy as nfl

    tag = f"ngs_{stat_type}_{min(seasons)}_{max(seasons)}"
    return _cached(
        tag, lambda: nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type),
        refresh)


def load_injuries(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Weekly injury reports. Needed for pregame starter identification."""
    import nflreadpy as nfl

    tag = f"injuries_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_injuries(seasons=seasons), refresh)


def load_depth_charts(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Weekly depth charts, the other half of starter identification."""
    import nflreadpy as nfl

    tag = f"depth_charts_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_depth_charts(seasons=seasons), refresh)


def bye_weeks(season: int) -> dict[str, int | None]:
    """Team to bye week for a season, derived from the schedule itself."""
    s = load_schedules().filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG"))
    weeks = sorted(s["week"].unique().to_list())
    teams = sorted(set(s["home_team"].to_list()) | set(s["away_team"].to_list()))
    out: dict[str, int | None] = {}
    for t in teams:
        played = set(
            s.filter((pl.col("home_team") == t) | (pl.col("away_team") == t))["week"].to_list())
        missing = [w for w in weeks if w not in played]
        out[t] = missing[0] if missing else None
    return out


def team_game_context(season: int, weeks: list[int] | None = None) -> pl.DataFrame:
    """One row per team-game with market inputs decomposed to team perspective.

    Implied team total is the fantasy-relevant decomposition of spread plus
    total; a favourite is not automatically the better fantasy environment.
    """
    g = load_schedules().filter(pl.col("season") == season)
    if weeks is not None:
        g = g.filter(pl.col("week").is_in(weeks))

    def side(team_col: str, opp_col: str, rest_col: str, sign: int, home: int):
        return g.select(
            "game_id", "season", "week", "gameday", "game_type",
            pl.col(team_col).alias("team"),
            pl.col(opp_col).alias("opponent"),
            pl.lit(home).alias("is_home"),
            (sign * pl.col("spread_line")).alias("team_spread"),
            "total_line",
            pl.col(rest_col).alias("rest"),
            "div_game", "roof", "surface",
        )

    both = pl.concat([
        side("home_team", "away_team", "home_rest", 1, 1),
        side("away_team", "home_team", "away_rest", -1, 0),
    ])
    return both.with_columns(
        (pl.col("total_line") / 2 + pl.col("team_spread") / 2).alias("implied_team_total"),
        (pl.col("total_line") / 2 - pl.col("team_spread") / 2).alias("implied_opp_total"),
        pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8).alias("indoors"),
        pl.col("team_spread").abs().alias("spread_abs"),
    )
