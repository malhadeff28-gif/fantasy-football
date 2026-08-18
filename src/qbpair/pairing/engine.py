"""Pairing engine: score any two quarterbacks week by week, without hindsight.

The weekly decision rule is deliberately simple and fixed before the outcome is
seen: start whichever quarterback the pregame projection rates higher. Every
number this module produces is therefore reproducible from information that
existed before kickoff.

Terminology note: a quarterback with no row in a given week is recorded as
`unavailable` rather than `on bye`. In a backtest the two are indistinguishable
from the panel alone -- he may have been on bye, injured, or benched -- and
conflating them would overstate how cleanly byes can be planned around. For a
forward season byes come from the schedule and are known exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import polars as pl

from ..config import LEAGUE


@dataclass
class WeekDecision:
    week: int
    proj_a: float | None
    proj_b: float | None
    started: str            # "A", "B", or "replacement"
    points: float
    coverage_fail: bool


@dataclass
class PairResult:
    qb_a: str
    qb_b: str
    season: int
    regular_points: float = 0.0
    playoff_points: float = 0.0
    weeks_played: int = 0
    choice_weeks: int = 0        # both available: a real decision existed
    solo_weeks: int = 0          # exactly one available
    empty_weeks: int = 0         # neither available
    coverage_failures: int = 0
    started_a: int = 0
    started_b: int = 0
    divergence: float = 0.0      # mean |proj_a - proj_b| on choice weeks
    decisions: list[WeekDecision] = field(default_factory=list)

    @property
    def total_points(self) -> float:
        return self.regular_points + self.playoff_points

    def weighted_score(self, playoff_weight: float = 1.0) -> float:
        """Regular season plus a weighted playoff component.

        The brief requires the playoff weeks to add value without letting three
        good matchups outweigh a poor regular season, so the playoff term is an
        additive bonus on top of, not a replacement for, regular-season output.
        """
        return self.regular_points + playoff_weight * self.playoff_points


def _slate(panel: pl.DataFrame, season: int, proj_col: str, outcome_col: str) -> dict:
    """player_id -> {week: (projection, outcome)} for one season."""
    cur = panel.filter(pl.col("season") == season)
    out = {}
    for pid, g in cur.group_by("player_id"):
        out[pid[0]] = {
            r["week"]: (r[proj_col], r[outcome_col])
            for r in g.select(["week", proj_col, outcome_col]).to_dicts()
        }
    return out


def score_pair(
    slate: dict,
    names: dict,
    qb_a: str,
    qb_b: str,
    season: int,
    weeks: list[int],
    replacement: dict[int, float] | float = 12.0,
    coverage_threshold: float = 14.0,
    stream_floor: dict[int, float] | float | None = None,
) -> PairResult:
    """Run the weekly start/sit rule across a season for one pair.

    `stream_floor` gives the pair manager the same waiver access a single-QB
    manager has: if the better of his two quarterbacks projects below the
    streamable level, he starts a streamer instead. Without it, holding two
    quarterbacks can score *worse* than holding one, which is an artefact of
    the simulation rather than a real property of the strategy -- a manager is
    never forced to start a bad QB2 simply because he owns him.

    The comparison uses projections, never outcomes, so it stays pregame.
    """
    wa, wb = slate.get(qb_a, {}), slate.get(qb_b, {})
    res = PairResult(qb_a=names.get(qb_a, qb_a), qb_b=names.get(qb_b, qb_b),
                     season=season)
    div_total = 0.0

    for w in weeks:
        ra, rb = wa.get(w), wb.get(w)
        repl = replacement[w] if isinstance(replacement, dict) else replacement

        if ra is None and rb is None:
            res.empty_weeks += 1
            pts, started, best_proj = repl, "replacement", None
        elif rb is None:
            res.solo_weeks += 1
            pts, started, best_proj = ra[1], "A", ra[0]
            res.started_a += 1
        elif ra is None:
            res.solo_weeks += 1
            pts, started, best_proj = rb[1], "B", rb[0]
            res.started_b += 1
        else:
            res.choice_weeks += 1
            div_total += abs(ra[0] - rb[0])
            if ra[0] >= rb[0]:
                pts, started, best_proj = ra[1], "A", ra[0]
                res.started_a += 1
            else:
                pts, started, best_proj = rb[1], "B", rb[0]
                res.started_b += 1

        # Same waiver access a single-QB manager has: if the better of the two
        # projects below streamable level, stream instead. Decided on
        # projections, so no outcome information is used.
        if stream_floor is not None and best_proj is not None:
            floor = (stream_floor[w] if isinstance(stream_floor, dict)
                     else stream_floor)
            if best_proj < floor:
                if started == "A":
                    res.started_a -= 1
                elif started == "B":
                    res.started_b -= 1
                pts, started, best_proj = repl, "replacement", None

        fail = best_proj is not None and best_proj < coverage_threshold
        if best_proj is None:
            fail = True
        if fail:
            res.coverage_failures += 1

        res.weeks_played += 1
        if w in LEAGUE.playoff_weeks:
            res.playoff_points += pts
        else:
            res.regular_points += pts
        res.decisions.append(WeekDecision(w, ra[0] if ra else None,
                                          rb[0] if rb else None,
                                          started, pts, fail))

    res.divergence = div_total / res.choice_weeks if res.choice_weeks else 0.0
    return res


def rank_pairs(
    panel: pl.DataFrame,
    season: int,
    candidates: list[str] | None = None,
    proj_col: str = "proj",
    outcome_col: str = "fantasy_points",
    min_weeks: int = 10,
    replacement: dict[int, float] | float = 12.0,
    playoff_weight: float = 1.0,
) -> pl.DataFrame:
    """Score every candidate pair for a season and return them ranked."""
    slate = _slate(panel, season, proj_col, outcome_col)
    names = {
        r["player_id"]: r["player_display_name"]
        for r in panel.filter(pl.col("season") == season)
        .select(["player_id", "player_display_name"]).unique().to_dicts()
    }
    if candidates is None:
        counts = (panel.filter(pl.col("season") == season)
                  .group_by("player_id").agg(pl.len().alias("n")))
        candidates = counts.filter(pl.col("n") >= min_weeks)["player_id"].to_list()

    weeks = sorted(
        panel.filter(pl.col("season") == season)["week"].unique().to_list())

    rows = []
    for a, b in combinations(sorted(candidates), 2):
        r = score_pair(slate, names, a, b, season, weeks, replacement)
        if r.weeks_played < min_weeks:
            continue
        rows.append({
            "season": season, "qb_a": r.qb_a, "qb_b": r.qb_b,
            "regular_points": round(r.regular_points, 2),
            "playoff_points": round(r.playoff_points, 2),
            "total_points": round(r.total_points, 2),
            "weighted": round(r.weighted_score(playoff_weight), 2),
            "choice_weeks": r.choice_weeks,
            "solo_weeks": r.solo_weeks,
            "empty_weeks": r.empty_weeks,
            "coverage_failures": r.coverage_failures,
            "divergence": round(r.divergence, 2),
            "started_a": r.started_a, "started_b": r.started_b,
        })
    return pl.DataFrame(rows).sort("weighted", descending=True)


def weekly_replacement(panel: pl.DataFrame, season: int,
                       quantile: float = 0.5) -> dict[int, float]:
    """Per-week replacement level: what a streamer realistically scores.

    Defaults to the median starter, which is generous to the single-QB
    alternative and therefore a conservative benchmark for the pairing thesis.
    """
    g = (panel.filter(pl.col("season") == season)
         .group_by("week")
         .agg(pl.col("fantasy_points").quantile(quantile).alias("r")))
    return {r["week"]: r["r"] for r in g.to_dicts()}
