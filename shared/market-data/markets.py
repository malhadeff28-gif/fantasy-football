"""Betting-market and fantasy-consensus ingestion, shared by every lab.

Sources verified live on 2026-08-17. Nothing here costs money except the Odds
API, which is env-var gated and inert without a key.

Redundancy warning for callers: spread, total, implied team total and player
props all encode overlapping market belief. Do not award a player or defense
several independent bonuses that are really the same opinion counted four
times. Use `market_redundancy` to measure the overlap before combining them.
"""
from __future__ import annotations

import os
import re

import polars as pl

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)


def normalise_name(name: str) -> str:
    """Loose join key across providers with different name conventions."""
    n = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _SUFFIX.sub("", n)
    return " ".join(n.split())


# Provider team codes differ from nflverse; join failures are silent otherwise.
TEAM_FIXUP = {"JAC": "JAX", "LAR": "LA", "STL": "LA", "WSH": "WAS",
              "SD": "LAC", "OAK": "LV", "NEP": "NE", "TAM": "TB",
              "KAN": "KC", "SFO": "SF", "GNB": "GB", "NOR": "NO"}


# --- ADP from real drafts (Fantasy Football Calculator, free) ----------------

def load_ffc_adp(season: int, teams: int = 12, fmt: str = "standard") -> pl.DataFrame:
    """Average draft position from real drafts, with dispersion.

    The stdev column matters as much as the mean: it quantifies how reliably a
    target can actually be waited on, which turns a theoretical plan into a
    draftable one.
    """
    import requests

    r = requests.get(f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}",
                     params={"teams": teams, "year": season}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return pl.DataFrame(payload["players"]).with_columns(
        pl.lit(season).alias("season"),
        pl.lit(payload["meta"]["start_date"]).alias("adp_window_start"),
        pl.lit(payload["meta"]["end_date"]).alias("adp_window_end"),
        pl.lit(payload["meta"]["total_drafts"]).alias("total_drafts"),
        pl.col("name").map_elements(normalise_name, return_dtype=pl.String).alias("key"),
    )


# --- CBS consensus projections (free, server-rendered) ----------------------

CBS_URL = ("https://www.cbssports.com/fantasy/football/stats/"
           "{pos}/{season}/season/projections/ppr/")

_CBS_SKILL = ["player", "games", "pass_att", "completions", "pass_yds",
              "pass_ypg", "pass_td", "interceptions", "rating", "rush_att",
              "rush_yds", "rush_avg", "rush_td", "fumbles_lost",
              "cbs_fantasy_points", "cbs_fppg"]

# CBS publishes a different column set per position; QB is not the template.
_CBS_RB = ["player", "games", "rush_att", "rush_yds", "rush_avg", "rush_td",
           "targets", "receptions", "rec_yds", "rec_ypg", "rec_avg", "rec_td",
           "fumbles_lost", "cbs_fantasy_points", "cbs_fppg"]
_CBS_WRTE = ["player", "games", "targets", "receptions", "rec_yds", "rec_ypg",
             "rec_avg", "rec_td", "rush_att", "rush_yds", "rush_avg", "rush_td",
             "fumbles_lost", "cbs_fantasy_points", "cbs_fppg"]
_CBS_SCHEMA = {"QB": _CBS_SKILL, "RB": _CBS_RB, "WR": _CBS_WRTE, "TE": _CBS_WRTE}


def load_cbs_projections(season: int, pos: str = "QB") -> pl.DataFrame:
    """CBS season projections with component stats retained.

    CBS publishes its own fantasy total under a scoring configuration that does
    not match this league, so callers should rescore from components rather
    than trusting cbs_fantasy_points.
    """
    import requests

    r = requests.get(CBS_URL.format(pos=pos, season=season),
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                     timeout=45)
    r.raise_for_status()
    schema = _CBS_SCHEMA.get(pos.upper(), _CBS_SKILL)
    rows = re.findall(r'<tr class="TableBase-bodyTr">(.*?)</tr>', r.text, re.S)
    recs = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < len(schema):
            continue
        vals = [" ".join(re.sub(r"<[^>]+>", " ", c).split()) for c in cells]
        parts = vals[0].split()
        team = parts[-1]

        def num(i):
            try:
                return float(vals[i].replace(",", ""))
            except Exception:
                return 0.0

        rec = {c: num(i) for i, c in enumerate(schema) if c != "player"}
        # cell repeats as "J. Allen QB BUF Josh Allen QB BUF"; the full name is
        # the segment between the first and second "<POS> <TEAM>" marker
        marker = f"{pos} {team}"
        seg = vals[0].split(marker)
        rec["player"] = seg[1].strip() if len(seg) > 2 and seg[1].strip() else vals[0]
        rec["team"] = team
        recs.append(rec)

    df = pl.DataFrame(recs)
    return df.with_columns(
        pl.when(pl.col("player").str.starts_with(pl.col("team") + " "))
        .then(pl.col("player").str.slice(pl.col("team").str.len_chars() + 1))
        .otherwise(pl.col("player")).str.strip_chars().alias("player"),
        pl.col("team").replace(TEAM_FIXUP).alias("team"),
    ).with_columns(
        pl.col("player").map_elements(normalise_name, return_dtype=pl.String).alias("key")
    )


# --- Kalshi prediction markets (free, forward-only) -------------------------

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_GAME_SERIES = ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL")
KALSHI_QB_SERIES = ("KXNFLPASSTDS", "KXNFLPASSATT", "KXNFLSEASONPASSYDS")


def kalshi_markets(series_ticker: str, status: str = "open") -> list[dict]:
    """Open markets for a Kalshi series. Public, no key.

    Kalshi publishes no free price history, so these must be logged forward to
    build an archive; nothing historical can be backtested from them.
    """
    import requests

    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": series_ticker, "status": status,
                             "limit": 1000}, timeout=30)
    r.raise_for_status()
    return r.json().get("markets", [])


# --- The Odds API (paid tier, env-var gated) --------------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4"
NFL_KEY = "americanfootball_nfl"
QB_PROP_MARKETS = ("player_pass_tds", "player_pass_yds", "player_pass_attempts",
                   "player_pass_completions", "player_pass_interceptions",
                   "player_rush_yds", "player_anytime_td")


def odds_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY") or None


def odds_api_status() -> dict:
    """Connectivity and quota probe. Never echoes the key."""
    import requests

    key = odds_api_key()
    if not key:
        return {"configured": False, "reason": "ODDS_API_KEY not set"}
    r = requests.get(f"{ODDS_BASE}/sports", params={"apiKey": key}, timeout=30)
    return {"configured": True, "http_status": r.status_code,
            "requests_remaining": r.headers.get("x-requests-remaining"),
            "nfl_available": NFL_KEY in r.text if r.ok else None}


# --- redundancy diagnostics -------------------------------------------------

def market_redundancy(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Pairwise correlation between market variables.

    Guards against counting one market opinion several times. A defense facing
    a low implied total, a large spread and a low opposing passing-TD prop has
    not received three pieces of evidence.
    """
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sub = df.drop_nulls(subset=[a, b])
            if sub.height < 30:
                continue
            rows.append({"a": a, "b": b, "n": sub.height,
                         "r": float(sub.select(pl.corr(a, b)).item())})
    return pl.DataFrame(rows).sort("r", descending=True) if rows else pl.DataFrame()
