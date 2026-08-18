"""Data-source adapters.

Every external provider is wrapped in a small function here so a source can be
swapped without touching modelling code. Results are cached to
`data/raw/*.parquet` so a backtest is reproducible even if an upstream feed
changes or goes away.
"""
from __future__ import annotations

import polars as pl

from ..config import DATA_RAW

DATA_RAW.mkdir(parents=True, exist_ok=True)


def _cached(name: str, fetch, refresh: bool = False) -> pl.DataFrame:
    """Return a cached parquet if present, otherwise fetch and persist it."""
    path = DATA_RAW / f"{name}.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = fetch()
    df.write_parquet(path)
    return df


# --- nflverse ---------------------------------------------------------------

def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """Game schedule + closing betting lines, 1999-present.

    Key columns: game_id, season, week, gameday, gametime, away_team,
    home_team, spread_line, total_line, away/home_moneyline, roof, surface,
    div_game, away_rest, home_rest, result.

    `spread_line` is positive when the HOME team is favoured (verified
    empirically: corr with home margin = +0.443 over 2015-2025).
    """
    import nflreadpy as nfl

    return _cached("schedules", lambda: nfl.load_schedules(), refresh)


def load_player_stats(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Weekly player box-score stats including `fantasy_points` (standard)."""
    import nflreadpy as nfl

    tag = f"player_stats_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_player_stats(seasons=seasons), refresh)


def load_ff_rankings(refresh: bool = False) -> pl.DataFrame:
    """FantasyPros expert consensus rankings (ECR) via ffverse."""
    import nflreadpy as nfl

    return _cached("ff_rankings", lambda: nfl.load_ff_rankings(), refresh)


# --- draft cost -------------------------------------------------------------

def load_ffc_adp(season: int, teams: int = 12, fmt: str = "standard") -> pl.DataFrame:
    """Fantasy Football Calculator ADP -- free public API, real drafts.

    Returns one row per player with adp, times_drafted, high, low, stdev, bye.
    """
    import requests

    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"
    r = requests.get(url, params={"teams": teams, "year": season}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    df = pl.DataFrame(payload["players"])
    return df.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(payload["meta"]["start_date"]).alias("adp_window_start"),
        pl.lit(payload["meta"]["end_date"]).alias("adp_window_end"),
        pl.lit(payload["meta"]["total_drafts"]).alias("total_drafts"),
    )


# --- sportsbook props (The Odds API, paid tier) -----------------------------
#
# Player props are the one input with no free historical archive. The Odds API
# Business tier ($99/mo) carries NFL props with history from 2023-05-03 at
# 5-minute snapshots. Adopted by user decision 2026-08-17.
#
# The key is read from the ODDS_API_KEY environment variable and is never
# logged, printed, or written to disk. Set it in a local `.env` (gitignored) or
# the shell; do not pass it as a literal.

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NFL_KEY = "americanfootball_nfl"

# Markets worth pulling for a QB. `player_pass_tds` is the single most
# fantasy-relevant prop under 4pt-passing-TD scoring.
QB_PROP_MARKETS = (
    "player_pass_tds",
    "player_pass_yds",
    "player_pass_attempts",
    "player_pass_completions",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_anytime_td",
)


def odds_api_key() -> str | None:
    """Return the configured Odds API key, or None if unset."""
    import os

    return os.environ.get("ODDS_API_KEY") or None


def odds_api_status() -> dict:
    """Cheap connectivity/quota probe. Never echoes the key."""
    import requests

    key = odds_api_key()
    if not key:
        return {"configured": False, "reason": "ODDS_API_KEY not set"}
    r = requests.get(
        f"{ODDS_API_BASE}/sports", params={"apiKey": key}, timeout=30
    )
    return {
        "configured": True,
        "http_status": r.status_code,
        "requests_remaining": r.headers.get("x-requests-remaining"),
        "requests_used": r.headers.get("x-requests-used"),
        "nfl_available": NFL_KEY in r.text if r.ok else None,
        "error": None if r.ok else r.text[:200],
    }


def _require_key() -> str:
    key = odds_api_key()
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Add it to your environment or a local "
            ".env file (gitignored). See research/DATA_SOURCES.md section 6."
        )
    return key


def list_events(date_iso: str | None = None) -> list[dict]:
    """NFL events. With `date_iso` (e.g. '2024-11-17T17:00:00Z') uses the
    historical snapshot endpoint, which is what a backtest needs."""
    import requests

    key = _require_key()
    if date_iso:
        url = f"{ODDS_API_BASE}/historical/sports/{NFL_KEY}/events"
        params = {"apiKey": key, "date": date_iso}
    else:
        url = f"{ODDS_API_BASE}/sports/{NFL_KEY}/events"
        params = {"apiKey": key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return payload.get("data", payload)


def event_props(
    event_id: str,
    markets: tuple = QB_PROP_MARKETS,
    date_iso: str | None = None,
    regions: str = "us",
) -> dict:
    """Player props for one event.

    Pass `date_iso` to retrieve the pregame snapshot for a historical game --
    essential for backtesting, since only prices published *before* kickoff are
    admissible.
    """
    import requests

    key = _require_key()
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    if date_iso:
        url = f"{ODDS_API_BASE}/historical/sports/{NFL_KEY}/events/{event_id}/odds"
        params["date"] = date_iso
    else:
        url = f"{ODDS_API_BASE}/sports/{NFL_KEY}/events/{event_id}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# --- prediction markets (Kalshi, free, forward-only) ------------------------

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_QB_SERIES = ("KXNFLPASSTDS", "KXNFLPASSATT", "KXNFLSEASONPASSYDS")
KALSHI_GAME_SERIES = ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL")


def kalshi_markets(series_ticker: str, status: str = "open") -> list[dict]:
    """Open markets for a Kalshi series. Public endpoint, no key required.

    Kalshi has no free price history, so these must be logged prospectively to
    build an archive -- see scripts/log_markets.py.
    """
    import requests

    r = requests.get(
        f"{KALSHI_BASE}/markets",
        params={"series_ticker": series_ticker, "status": status, "limit": 1000},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("markets", [])
