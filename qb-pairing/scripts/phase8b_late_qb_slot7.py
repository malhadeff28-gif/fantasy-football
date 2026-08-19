"""Late-QB strategy from the 7th pick of a 12-team draft.

Two questions, both narrower and more answerable than the general pairing
thesis:

  A. Does deliberately taking no quarterback until round 7 hold up historically?
  B. From the 7/12 slot specifically, which quarterback should be targeted, at
     which pick, and how likely is he to be there?

Availability matters as much as value here. A quarterback who ranks well but is
gone before your pick is not a plan. Each quarterback's draft position is
treated as roughly normal around his ADP with the observed standard deviation,
so P(available at pick N) = P(draft position >= N). FFC publishes the standard
deviation directly, so this is measured spread rather than a guess -- though the
normal shape is an approximation. [inference]

RERUN 2026-08-18 (QB-024). Four corrections landed under this script; see
research/wave2/QB-024_slot7_rerun.md. The two that changed the answer:

  * `benchmarks/draft_value.py::season_totals` now stops at week 16, so the
    pick-value curve and the QB12 replacement level are on the league calendar.
    This script was still SUMMING QB POINTS THROUGH WEEK 18 while subtracting a
    16-week replacement level, which flattered every plan by roughly 45 points.
    Fixed here: `LAST_WEEK` truncates the backtest to the weeks this league
    plays.
  * the QB2 marginal is negative once it is measured rank-matched, so the
    insurance premium this script used to add is gone. Section C now prices a
    partner from measured components instead, and the headline output is a
    single quarterback rather than a pair.

Usage:  python scripts/phase8b_late_qb_slot7.py [SLOT] [TEAMS]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys                                       # noqa: E402
import warnings                                  # noqa: E402
from itertools import combinations               # noqa: E402
from math import erf, sqrt                       # noqa: E402
from pathlib import Path                         # noqa: E402

import polars as pl                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.benchmarks import draft_value as dv  # noqa: E402
from qbpair.config import LEAGUE                 # noqa: E402
from qbpair.data import loaders                  # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")

_ARG = [a for a in sys.argv[1:] if a.isdigit()]
SLOT = int(_ARG[0]) if _ARG else 7
TEAMS = int(_ARG[1]) if len(_ARG) > 1 else 12
ROSTER = 16
SEASON = 2026
FIRST_QB_ROUND = 7          # "no QB before round 7"
SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
BAR = "=" * 92

# The fantasy calendar. This league finishes in NFL week 16, so weeks 17-18 are
# irrelevant on BOTH sides of the ledger: the pick-value curve stops there
# (draft_value.season_totals) and so must the quarterback points measured here.
LAST_WEEK = max(LEAGUE.playoff_weeks)                 # 16
GAMES_PLAYED = LAST_WEEK - 1                          # 16 league weeks, 1 bye

# Constants carried over from earlier phases, all re-derived after QB-020.
#
#   CHOICE_EDGE  the weekly start/sit option a second quarterback buys. Was 4.9
#                (QB-013, measured on a universe that was not ADP-selectable).
#                QB-020 measured it rank-matched at +1.7 pts/season, invariant
#                to the streamer level, and that is the figure used here.
#   the insurance premium is GONE. It interpolated between +15 and +35 points,
#                anchored on the QB-012 marginals of +9.7 / +30.1, and both were
#                retired by QB-020: rank-matched, holding QB1 fixed, a second
#                quarterback is worth -10.6 pts/season, negative at 7 of 7 QB1
#                ranks. A premium cannot be re-derived from a negative marginal,
#                so a partner is now priced from its two measurable parts --
#                bye-week coverage and CHOICE_EDGE -- against its pick cost.
CHOICE_EDGE = 1.7
REPLACEMENT_PPG = 16.0      # streamable QB, ~= QB12 realised / 15 games


def snake_picks(slot: int, teams: int, rounds: int) -> list[int]:
    """Overall pick numbers for a snake draft slot."""
    out = []
    for r in range(1, rounds + 1):
        out.append((r - 1) * teams + (slot if r % 2 == 1 else teams - slot + 1))
    return out


def p_available(adp: float, sd: float, pick: int) -> float:
    """P(still on the board at `pick`), normal approximation around ADP."""
    sd = max(sd, 1.0)
    z = (pick - adp) / sd
    return 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))


def cost_lookup(curve: pl.DataFrame):
    """Expected non-QB usable PAR at a given overall pick -- the cost of a pick.

    AUDIT (QB-024): `phase6_draft_cost.py::pick_cost_lookup` had a bug where a
    float ADP between two integer bucket edges (48.1 is in neither 37-48 nor
    49-60) matched nothing and fell through to the deepest bucket at 0.0 cost,
    which made 8% of picks free. This script never had that bug -- it has always
    rounded and clamped before the scan -- and `_check_cost_lookup` below now
    holds that line. One lookup serves both the backtest and the 2026 board;
    there used to be two copies of it, which is how the sibling bug survived.
    """
    rows = curve.to_dicts()
    lo, hi = rows[0]["pick_from"], rows[-1]["pick_to"]

    def cost(pick_or_adp: float) -> float:
        p = int(min(max(round(pick_or_adp), lo), hi))
        for r in rows:
            if r["pick_from"] <= p <= r["pick_to"]:
                return float(r["mean_usable_par"])
        return float(rows[-1]["mean_usable_par"])

    return cost


def _check_cost_lookup(cost, curve: pl.DataFrame) -> None:
    """Regression guard for the bucket-gap bug class. Costs microseconds."""
    rows = curve.to_dicts()
    lo, hi = rows[0]["pick_from"], rows[-1]["pick_to"]
    for tenth in range(lo * 10, hi * 10 + 1):
        p = tenth / 10.0
        want = next(float(r["mean_usable_par"]) for r in rows
                    if r["pick_from"] <= round(p) <= r["pick_to"])
        assert cost(p) == want, (
            f"pick {p} charged {cost(p)}, its bucket is worth {want}")
    # out of range clamps rather than falling through to the cheapest bucket
    assert cost(0.4) == cost(lo), "below-range pick did not clamp"
    assert cost(hi + 99.0) == cost(hi), "above-range pick did not clamp"


def _season_prep(proj: pl.DataFrame, adp_hist: pl.DataFrame, season: int,
                 last_week: int) -> dict:
    """Everything one season of the backtest needs, built once."""
    from qbpair.pairing import engine

    cur = proj.filter(pl.col("season") == season)
    names = {r["player_id"]: r["player_display_name"]
             for r in cur.select(["player_id", "player_display_name"])
             .unique().to_dicts()}
    keymap: dict[str, str] = {}
    for pid, nm in names.items():
        keymap.setdefault(dv.normalise_name(nm), pid)
    return {
        "season": season,
        "weeks": [w for w in sorted(cur["week"].unique().to_list())
                  if w <= last_week],
        "slate": engine._slate(cur, season, "proj", "fantasy_points"),
        "repl": engine.weekly_replacement(cur, season),
        # streamable level, in projection units, so the pair manager has the
        # same waiver access the single-QB manager already gets on byes
        "floor": {r["week"]: r["f"] for r in cur.group_by("week").agg(
            pl.col("proj").median().alias("f")).to_dicts()},
        "names": names,
        "board": sorted(adp_hist.filter(pl.col("season") == season).to_dicts(),
                        key=lambda r: r["adp"]),
        "keymap": keymap,
    }


def _plan_points(prep: dict, cost, p1: int | None, p2: int | None):
    """Score one plan for one season. Picks come from ADP only, never outcome."""
    from qbpair.pairing import engine

    def best_at(pick: int | None, exclude=()):
        """Expected best QB still available at `pick` (or overall if None)."""
        for r in prep["board"]:
            pid = prep["keymap"].get(r["key"])
            if not pid or pid in exclude:
                continue
            if pick is None or r["adp"] >= pick:
                return pid, r["adp"]
        return None, None

    a, adp_a = best_at(p1)
    if a is None:
        return None
    spend = cost(adp_a)
    slate, weeks, repl = prep["slate"], prep["weeks"], prep["repl"]
    if p2 is None:
        pts = sum(slate[a][w][1] if w in slate.get(a, {}) else repl.get(w, 12.0)
                  for w in weeks)
    else:
        b, adp_b = best_at(p2, exclude=(a,))
        if b is None:
            return None
        spend += cost(adp_b)
        pts = engine.score_pair(slate, prep["names"], a, b, prep["season"], weeks,
                                replacement=repl,
                                stream_floor=prep["floor"]).total_points
    return pts, spend


def backtest_late_qb(first_qb_pick: int, *, cost, qb_repl: dict,
                     features: list[str] | None = None,
                     last_week: int = LAST_WEEK, sweep_picks=(),
                     verbose: bool = True):
    """Historical test of the constraint itself, 2020-2025.

    Each strategy picks by ADP only -- the expected best quarterback still on
    the board at a given pick -- so nothing is chosen with hindsight. Points are
    the actual weekly outcomes under the pregame start rule, over the weeks this
    league plays (1-`last_week`).
    """
    from qbpair.features import panel as pm
    from qbpair.models import projection

    feats = features or projection.FEATURES
    panel = pm.build_panel(SEASONS).drop_nulls(subset=feats)
    proj = projection.fit_walk_forward(panel, TEST_SEASONS, features=feats)
    adp_hist = dv.load_adp(SEASONS).filter(pl.col("adp_position") == "QB")
    prep = {s: _season_prep(proj, adp_hist, s, last_week) for s in TEST_SEASONS}

    if verbose:
        print(f"\n{BAR}\n A2. DOES WAITING UNTIL ROUND {FIRST_QB_ROUND} ACTUALLY "
              f"WORK?  (2020-2025, NFL weeks 1-{last_week})\n{BAR}")

    plans = {
        "premium QB1 only": (None, None),
        "premium QB1 + late QB2 (~127)": (None, 127),
        f"late single QB (at {first_qb_pick}, no QB2)": (first_qb_pick, None),
        f"late pair (QB at {first_qb_pick}, then ~127)": (first_qb_pick, 127),
    }
    res = []
    for s in TEST_SEASONS:
        for label, (p1, p2) in plans.items():
            got = _plan_points(prep[s], cost, p1, p2)
            if got is None:
                continue
            pts, spend = got
            res.append({"season": s, "plan": label, "pts": round(pts, 1),
                        "cost": round(spend, 1),
                        "net": round(pts - qb_repl[s] - spend, 1)})

    d = pl.DataFrame(res)
    common = (d.group_by("season").agg(pl.len().alias("k"))
              .filter(pl.col("k") == d["plan"].n_unique())["season"].to_list())
    f = d.filter(pl.col("season").is_in(common))
    n_seasons = len(common)
    agg = (f.group_by("plan").agg(pl.col("pts").mean().alias("pts"),
                                  pl.col("cost").mean().alias("cost"),
                                  pl.col("net").mean().alias("net"),
                                  pl.col("net").std().alias("sd"),
                                  pl.col("net").min().alias("worst"))
           .with_columns((pl.col("sd") / (n_seasons ** 0.5)).alias("se"))
           .sort("net", descending=True))

    # Paired differences against the late single QB. Plans share seasons and
    # often share quarterbacks, so their errors are strongly correlated and the
    # unpaired SEs above overstate how uncertain the DIFFERENCES are.
    base_label = f"late single QB (at {first_qb_pick}, no QB2)"
    by_plan = {lab: {r["season"]: r["net"] for r in
                     f.filter(pl.col("plan") == lab).to_dicts()}
               for lab in plans}
    paired = []
    for lab in plans:
        if lab == base_label:
            continue
        diff = [by_plan[lab][s] - by_plan[base_label][s] for s in common]
        m = sum(diff) / len(diff)
        sd = (sum((x - m) ** 2 for x in diff) / max(len(diff) - 1, 1)) ** 0.5
        se = sd / len(diff) ** 0.5
        paired.append({"plan": lab, "vs": base_label, "mean_diff": m,
                       "se": se, "t": m / se if se else float("nan"),
                       "n": len(diff)})

    if verbose:
        print(f"seasons compared: {sorted(common)}  (n = {n_seasons})")
        print(f"\n{'plan':<40}{'QB pts':>9}{'cost':>8}{'NET':>8}{'+/- se':>9}{'worst':>8}")
        print("-" * 92)
        for r in agg.to_dicts():
            print(f"{r['plan']:<40}{r['pts']:>9.1f}{r['cost']:>8.1f}"
                  f"{r['net']:>8.1f}{r['se']:>9.1f}{r['worst']:>8.1f}")
        rows_ = agg.to_dicts()
        gap = rows_[0]["net"] - rows_[-1]["net"]
        pooled = (sum(r["se"] ** 2 for r in (rows_[0], rows_[-1]))) ** 0.5
        print(f"\nBest minus worst plan: {gap:.1f} points, unpaired combined "
              f"standard error {pooled:.1f}.")
        print(f"\n{'paired difference vs late single QB':<40}{'diff':>9}"
              f"{'+/- se':>9}{'t':>7}{'n':>5}")
        print("-" * 92)
        for r in paired:
            print(f"{r['plan']:<40}{r['mean_diff']:>+9.1f}{r['se']:>9.1f}"
                  f"{r['t']:>7.2f}{r['n']:>5}")
        print("\nNET = QB points minus QB12 replacement minus the measured value")
        print("of the picks spent, all over NFL weeks 1-%d." % last_week)

    sweep = []
    if sweep_picks:
        if verbose:
            print(f"\n{BAR}\n A3. IF YOU WAIT ON QB1, WHEN IS QB2 WORTH IT?\n{BAR}")
            print(f"QB1 fixed at pick {first_qb_pick}; QB2 taken at varying picks.")
            print("A closer-priced QB2 is a real alternative each week; a very "
                  "late one\nis rarely the right start, so he adds points only "
                  "through byes.\n")
        for p2 in sweep_picks:
            tot = []
            for s in TEST_SEASONS:
                got = _plan_points(prep[s], cost, first_qb_pick, p2)
                if got is None:
                    continue
                pts, spend = got
                tot.append(pts - qb_repl[s] - spend)
            if tot:
                m = sum(tot) / len(tot)
                sd = (sum((x - m) ** 2 for x in tot) / max(len(tot) - 1, 1)) ** 0.5
                sweep.append({"qb2_pick": "none" if p2 is None else str(p2),
                              "net": m, "se": sd / len(tot) ** 0.5,
                              "worst": min(tot), "n": len(tot)})
        if verbose:
            print(f"{'QB2 taken at':<16}{'NET':>9}{'+/- se':>9}{'worst season':>15}"
                  f"{'seasons':>9}")
            print("-" * 92)
            for r in sweep:
                # n < 6 means the ADP board ran out before that pick in some
                # season, so that cell is not comparable with the others. The
                # published sweep averaged over different seasons per cell
                # without saying so.
                flag = "" if r["n"] == len(TEST_SEASONS) else "  <- unbalanced"
                print(f"{r['qb2_pick']:<16}{r['net']:>9.1f}{r['se']:>9.1f}"
                      f"{r['worst']:>15.1f}{r['n']:>9}{flag}")
            bal = [r for r in sweep if r["n"] == len(TEST_SEASONS)]
            spread = max(r["net"] for r in bal) - min(r["net"] for r in bal)
            med_se = sorted(r["se"] for r in bal)[len(bal) // 2]
            print(f"\nCAUTION: this sweep is not monotonic, and the spread across "
                  f"picks\n({spread:.0f} points) is comparable to the typical "
                  f"standard error\n({med_se:.0f}). Each cell is one specific "
                  f"quarterback per season over\nsix seasons, so the ups and downs "
                  f"reflect which players happened to\nsit at that ADP, not a real "
                  f"optimum. Do not read a best pick off this.")

    return agg, pl.DataFrame(sweep) if sweep else pl.DataFrame(), f, paired


def load_board_2026(season: int = SEASON, teams: int = TEAMS) -> pl.DataFrame:
    """QB board: live FFC ADP joined to CBS projections, with bye weeks."""
    sched = loaders.load_schedules().filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG"))
    teams_all = sorted(set(sched["home_team"].to_list())
                       | set(sched["away_team"].to_list()))
    weeks = sorted(sched["week"].unique().to_list())
    bye = {}
    for t in teams_all:
        played = set(sched.filter((pl.col("home_team") == t)
                                  | (pl.col("away_team") == t))["week"].to_list())
        miss = [w for w in weeks if w not in played]
        bye[t] = miss[0] if miss else None

    adp = loaders.load_ffc_adp(season, teams=teams).filter(
        pl.col("position") == "QB").with_columns(
        pl.col("name").map_elements(dv.normalise_name,
                                    return_dtype=pl.String).alias("key"))
    cbs = loaders.load_cbs_projections(season, "QB").with_columns(
        pl.col("player").map_elements(dv.normalise_name,
                                      return_dtype=pl.String).alias("key"))
    q = (adp.join(cbs.select(["key", "proj_points_league", "proj_ppg_league"]),
                  on="key", how="inner")
         .with_columns(pl.col("team").replace(
             {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "SD": "LAC",
              "OAK": "LV"}).alias("team")))
    return q.with_columns(
        pl.col("team").replace_strict(bye, default=None).alias("bye_week"),
        # CBS projects 17 games; this league plays 16 weeks, of which the
        # quarterback is on bye for one.
        (pl.col("proj_ppg_league") * GAMES_PLAYED).alias("proj_league_season"),
    )


def single_qb_table(q: pl.DataFrame, cost, repl_season: float,
                    picks: list[int], min_avail: float = 0.55) -> pl.DataFrame:
    """Which single quarterback to target, at which pick, and how likely.

    `take_at` is the LATEST of your picks at which he is still `min_avail`
    likely to be there: taking him earlier than that spends a better pick on a
    player who would have lasted.
    """
    rows = []
    for r in q.to_dicts():
        take, av = None, None
        for p in picks:
            pa = p_available(r["adp"], r["stdev"], p)
            if pa >= min_avail:
                take, av = p, pa
        if take is None:                       # gone before your first QB pick
            take = picks[0]
            av = p_available(r["adp"], r["stdev"], picks[0])
        season_pts = r["proj_ppg_league"] * GAMES_PLAYED + REPLACEMENT_PPG
        rows.append({
            "qb": r["name"], "team": r["team"], "adp": r["adp"],
            "sd": r["stdev"], "bye": r["bye_week"],
            # a quarterback who is gone before your first QB pick is not a
            # target however well he ranks, so the recommendation is filtered
            # on this rather than on projected points
            "gettable": av >= 0.35,
            "proj_ppg": round(r["proj_ppg_league"], 2),
            "proj_16wk": round(season_pts, 1),
            "take_at": take, "p_available": round(av, 3),
            "pick_cost": round(cost(take), 1),
            "net": round(season_pts - repl_season - cost(take), 1),
            **{f"p{p}": round(p_available(r["adp"], r["stdev"], p), 3)
               for p in picks},
        })
    return pl.DataFrame(rows).sort("net", descending=True)


def pair_table(q: pl.DataFrame, cost, repl_season: float, p1: int,
               p2_opts: list[int]) -> pl.DataFrame:
    """Every gettable pair, with the partner priced from measured components.

    marginal(QB2) = bye-week coverage + CHOICE_EDGE - the pick he costs.
    There is no insurance term: rank-matched, a second quarterback does not have
    one (QB-020). This is the OPTIMISTIC end of the range, because CBS
    projections do not forecast the QB2 busting.
    """
    rows = []
    for a, b in combinations(q.to_dicts(), 2):
        qb1, qb2 = (a, b) if a["adp"] <= b["adp"] else (b, a)
        av1 = p_available(qb1["adp"], qb1["stdev"], p1)
        if av1 < 0.35:                       # QB1 must plausibly be there
            continue
        # Take QB2 as LATE as you can still reasonably get him.
        best = None
        for p in p2_opts:
            av2 = p_available(qb2["adp"], qb2["stdev"], p)
            if av2 >= 0.65:
                best = (p, av2)
        if best is None:
            continue
        p2, av2 = best
        same_bye = qb1["bye_week"] is not None and qb1["bye_week"] == qb2["bye_week"]
        solo = qb1["proj_ppg_league"] * GAMES_PLAYED + REPLACEMENT_PPG
        bye_gain = 0.0 if same_bye else max(
            0.0, qb2["proj_ppg_league"] - REPLACEMENT_PPG)
        c2 = cost(p2)
        rows.append({
            "qb1": qb1["name"], "qb2": qb2["name"],
            "adp1": qb1["adp"], "adp2": qb2["adp"],
            "take1_at": p1, "take2_at": p2,
            "avail1": round(av1 * 100), "avail2": round(av2 * 100),
            "bye1": qb1["bye_week"], "bye2": qb2["bye_week"],
            "bye_clash": same_bye,
            "solo_net": round(solo - repl_season - cost(p1), 1),
            "bye_cover": round(bye_gain, 1),
            "choice_edge": CHOICE_EDGE,
            "qb2_cost": round(c2, 1),
            "qb2_marginal": round(bye_gain + CHOICE_EDGE - c2, 1),
            "net": round(solo - repl_season - cost(p1)
                         + bye_gain + CHOICE_EDGE - c2, 1),
        })
    return pl.DataFrame(rows)


def main() -> int:
    picks = snake_picks(SLOT, TEAMS, ROSTER)
    qb_picks = [p for i, p in enumerate(picks, start=1) if i >= FIRST_QB_ROUND]

    print(BAR)
    print(f" DRAFT SLOT {SLOT} of {TEAMS} -- overall picks")
    print(BAR)
    print("  " + "  ".join(f"R{i}:{p}" for i, p in enumerate(picks, 1)))
    print(f"\nConstraint: no quarterback before round {FIRST_QB_ROUND}, "
          f"so the first QB comes at pick {qb_picks[0]} or later.")
    print(f"Rounds 1-{FIRST_QB_ROUND - 1} (picks "
          f"{', '.join(str(p) for p in picks[:FIRST_QB_ROUND - 1])}) go to "
          f"skill positions.")

    q = load_board_2026()
    curve = dv.pick_value_curve(SEASONS).filter(pl.col("n") >= 20)
    cost = cost_lookup(curve)
    _check_cost_lookup(cost, curve)

    qb_repl = {r["season"]: r["replacement_points"]
               for r in dv.replacement_levels(dv.season_totals(SEASONS))
               .filter(pl.col("position") == "QB").to_dicts()}
    realised_repl = sum(qb_repl[s] for s in TEST_SEASONS[-5:]) / 5

    # The 2026 board is CBS projections; realised seasons are not. Projections
    # do not forecast injuries, so the projected QB12 sits well above the
    # realised one and NET on this board is a RANKING, not a forecast of
    # surplus. The replacement used here is therefore the 12th quarterback on
    # the same projection scale.
    proj_repl = (sorted(q["proj_ppg_league"].to_list(), reverse=True)[11]
                 * GAMES_PLAYED + REPLACEMENT_PPG)

    # ---------- A. availability at your picks ----------
    print(f"\n{BAR}\n A. QB AVAILABILITY AT YOUR PICKS  (% still on the board)\n{BAR}")
    print(f"ADP refreshed live: FFC {TEAMS}-team standard, drafts from "
          f"{q['adp_window_start'][0]} to {q['adp_window_end'][0]} "
          f"(n = {q['total_drafts'][0]}). ADP moves daily; rerun before drafting.")
    show = q.filter(pl.col("adp") <= 200).sort("adp")
    cols = qb_picks[:7]
    print(f"{'QB':<20}{'tm':>4}{'ADP':>7}{'sd':>6}{'bye':>5}{'CBSppg':>8}" +
          "".join(f"{('R' + str(i + FIRST_QB_ROUND)):>7}" for i in range(len(cols))))
    print(f"{'':<42}" + "".join(f"{('p' + str(p)):>7}" for p in cols))
    print("-" * 92)
    for r in show.head(22).to_dicts():
        probs = "".join(f"{p_available(r['adp'], r['stdev'], p) * 100:>6.0f}%"
                        for p in cols)
        print(f"{r['name']:<20}{r['team']:>4}{r['adp']:>7.1f}{r['stdev']:>6.1f}"
              f"{str(r['bye_week']):>5}{r['proj_ppg_league']:>8.1f}{probs}")

    # ---------- A2/A3. does waiting actually work? ----------
    agg, sweep, per_season, paired = backtest_late_qb(
        qb_picks[0], cost=cost, qb_repl=qb_repl,
        sweep_picks=[None, 90, 103, 114, 127, 138, 151, 162])

    # ---------- B. the single quarterback to target ----------
    print(f"\n{BAR}\n B. WHICH SINGLE QB TO TARGET FROM {SLOT}/{TEAMS}, "
          f"NO QB BEFORE ROUND {FIRST_QB_ROUND}\n{BAR}")
    print(f"Replacement here is the 12th-best PROJECTED QB "
          f"({proj_repl:.0f} pts over {LAST_WEEK} weeks); the realised QB12 has "
          f"averaged\n{realised_repl:.0f} over 2021-2025, and the gap is "
          f"projection optimism, not surplus. Read NET as a ranking.")
    print(f"take_at = latest of your picks where he is still >=55% likely to be "
          f"there.\n")
    single = single_qb_table(q, cost, proj_repl, qb_picks[:6])
    print(f"{'QB':<20}{'tm':>4}{'ADP':>7}{'ppg':>7}{'take at':>9}{'P(there)':>10}"
          f"{'cost':>7}{'NET':>8}{'bye':>5}")
    print("-" * 92)
    for r in single.filter(pl.col("gettable")).head(12).to_dicts():
        print(f"{r['qb']:<20}{r['team']:>4}{r['adp']:>7.1f}{r['proj_ppg']:>7.1f}"
              f"{r['take_at']:>9}{r['p_available'] * 100:>9.0f}%"
              f"{r['pick_cost']:>7.1f}{r['net']:>8.1f}{str(r['bye']):>5}")
    gone = single.filter(~pl.col("gettable")).sort("adp")
    print("\nGone before pick %d, whatever they project: %s"
          % (qb_picks[0], ", ".join(f"{r['qb']} ({r['adp']:.0f})"
                                    for r in gone.head(8).to_dicts())))

    # ---------- C. is a second quarterback worth it? ----------
    print(f"\n{BAR}\n C. WHAT A SECOND QUARTERBACK IS WORTH IN 2026\n{BAR}")
    pairs = pair_table(q, cost, proj_repl, qb_picks[0], qb_picks[1:6])
    ok = pairs.filter(~pl.col("bye_clash")).sort("net", descending=True)
    print("The insurance premium is retired (QB-020: rank-matched, a partner is")
    print("worth -10.6 pts/season). A partner is priced from what can still be")
    print("measured: bye-week coverage + a +1.7 weekly choice edge - his pick.\n")
    # The marginal depends on the PARTNER, not on which QB1 he sits behind
    # (bye clashes aside), so it is reported per partner rather than per pair.
    per_qb2 = (ok.group_by("qb2").agg(pl.col("take2_at").first(),
                                      pl.col("bye_cover").first(),
                                      pl.col("qb2_cost").first(),
                                      pl.col("qb2_marginal").first(),
                                      pl.len().alias("pairs"))
               .sort("qb2_marginal", descending=True))
    print(f"{'best partner (any QB1)':<28}{'take at':>9}{'cover':>7}{'edge':>6}"
          f"{'cost':>7}{'QB2 adds':>10}")
    print("-" * 92)
    for r in per_qb2.head(8).to_dicts():
        print(f"{r['qb2']:<28}{r['take2_at']:>9}{r['bye_cover']:>7.1f}"
              f"{CHOICE_EDGE:>6.1f}{r['qb2_cost']:>7.1f}"
              f"{r['qb2_marginal']:>+10.1f}")
    best = ok["qb2_marginal"].max() if ok.height else float("nan")
    npos = ok.filter(pl.col("qb2_marginal") > 0).height if ok.height else 0
    print(f"\nBest partner marginal available from this slot: {best:+.1f} points.")
    print(f"Pairs with a positive marginal: {npos} of {ok.height}. This is the")
    print("optimistic bound -- CBS does not project the QB2 busting, and the")
    print("historical rank-matched estimate is -10.6.")

    # ---------- D. does the single-QB ranking survive its assumptions? ----------
    print(f"\n{BAR}\n D. DOES THE TOP OF THE BOARD REORDER UNDER ITS ASSUMPTIONS?"
          f"\n{BAR}")
    # Only two things can reorder this board: how much a pick is worth, and how
    # much of CBS's spread in projected points per game is real. A shift in the
    # replacement level is a constant and cannot reorder anything, so it is not
    # tested here. `shrink` pulls every projection edge toward the field.
    settings = [("cost x0.5", 0.5, 1.0), ("cost x1.0", 1.0, 1.0),
                ("cost x1.5", 1.5, 1.0), ("shrink 0.5", 1.0, 0.5),
                ("shrink 0.25", 1.0, 0.25)]
    tally: dict[str, int] = {}
    TOPN = 5
    gettable = single.filter(pl.col("gettable"))     # ranking men you can have
    for _lab, cm, lam in settings:
        e = (gettable.with_columns(
            ((pl.col("net") + pl.col("pick_cost")) * lam
             - pl.col("pick_cost") * cm).alias("nv"))
            .sort("nv", descending=True).head(TOPN))
        for r in e.to_dicts():
            tally[r["qb"]] = tally.get(r["qb"], 0) + 1
    print(f"{'QB':<22}{'top-5 in':>10}  verdict")
    print("-" * 92)
    for nm, n in sorted(tally.items(), key=lambda t: -t[1])[:10]:
        print(f"{nm:<22}{str(n) + '/' + str(len(settings)):>10}  "
              f"{'robust' if n == len(settings) else 'assumption-dependent'}")
    print("\nThe pick-cost spread across rounds 7-12 is only "
          f"{max(cost(p) for p in qb_picks[:6]) - min(cost(p) for p in qb_picks[:6]):.1f} "
          "points, so the ranking is\ndriven by projected points per game, not by "
          "which of your late picks you spend.")

    # ---------- E. the practical plan ----------
    print(f"\n{BAR}\n E. PRACTICAL DRAFT PLAN FROM {SLOT}/{TEAMS}\n{BAR}")
    for i, p in enumerate(qb_picks[:6]):
        rnd = FIRST_QB_ROUND + i
        avail = [(r["name"], p_available(r["adp"], r["stdev"], p),
                  r["proj_ppg_league"], r["bye_week"]) for r in q.to_dicts()]
        # "live" = plausibly there but not a lock; a 99% QB can wait a round
        live = [x for x in avail if 0.40 <= x[1] <= 0.95]
        live.sort(key=lambda t: -t[2])
        if not live:
            continue
        nm = ", ".join(f"{n} ({pr * 100:.0f}%, bye {b})" for n, pr, _s, b in live[:4])
        print(f"  R{rnd} (pick {p:>3}): {nm}")
    # P(the plan survives contact) -- at least one of the top targets is there
    # at your first QB pick. Treats draft positions as independent, which
    # understates the risk slightly: when one quarterback slides, the others
    # tend to slide with him. [inference]
    top3 = (single.filter(pl.col("gettable")
                          & (pl.col("take_at") == qb_picks[0]))
            .head(3).to_dicts())
    miss = 1.0
    for r in top3:
        miss *= 1.0 - p_available(r["adp"], r["sd"], qb_picks[0])
    print(f"\nP(at least one of {', '.join(r['qb'] for r in top3)} is there at "
          f"pick {qb_picks[0]}): {(1 - miss) * 100:.0f}%")
    print(f"Shared-bye pairs to avoid: {pairs.filter(pl.col('bye_clash')).height} "
          f"of {pairs.height} feasible pairs.")

    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    single.write_csv(out / f"qb024_single_qb_slot{SLOT}_2026.csv")
    ok.write_csv(out / f"qb024_pairs_slot{SLOT}_2026.csv")
    agg.write_csv(out / f"qb024_plan_comparison_slot{SLOT}.csv")
    per_season.write_csv(out / f"qb024_plan_by_season_slot{SLOT}.csv")
    pl.DataFrame(paired).write_csv(out / f"qb024_plan_paired_diffs_slot{SLOT}.csv")
    if sweep.height:
        sweep.write_csv(out / f"qb024_qb2_timing_sweep_slot{SLOT}.csv")
    print(f"\nwrote outputs/qb024_*_slot{SLOT}*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
