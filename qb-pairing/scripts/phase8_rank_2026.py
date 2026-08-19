"""Phase 8: 2026 QB pair rankings, built only on what the research supports.

Phases 2-4 removed schedule difficulty and QB-specific matchup effects from the
model, so this ranking does not pretend to find "complementary schedules".
What it does use is what survived testing:

  * a season-points projection per quarterback (CBS, rescored to league rules,
    blended with a market-consensus rank from real 2026 ADP);
  * real 2026 bye weeks, because bye coverage is a genuine, knowable benefit;
  * the measured weekly start/sit edge, +4.9 pts/season (Phase 7);
  * an insurance premium that scales with how expensive QB1 is, interpolated
    from the Phase 6 measurements (+9.7 net alongside an elite QB1, +30.1
    alongside a mid-priced one);
  * the true cost of both picks, from the measured pick-value curve.

Every component is reported separately so the ranking can be argued with rather
than taken on faith.

Usage:  python scripts/phase8_rank_2026.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                       # noqa: E402
import warnings                                  # noqa: E402
from itertools import combinations               # noqa: E402
from pathlib import Path                         # noqa: E402

import polars as pl                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.benchmarks import draft_value as dv  # noqa: E402
from qbpair.config import LEAGUE                 # noqa: E402
from qbpair.data import loaders                  # noqa: E402

warnings.filterwarnings("ignore")

SEASON = 2026
BAR = "=" * 88

# Measured constants from earlier phases.
WEEKLY_EDGE = 4.9            # Phase 7: model vs always-QB1, pts/season
QB_REPLACEMENT = 272.0       # QB12 season points, 2018-2025 mean
REPLACEMENT_PPG = 16.0       # weekly streamer level

# Phase 6 anchors for the insurance premium (gross of QB2 pick cost).
INS_ELITE_ADP, INS_ELITE = 16.0, 15.0
INS_MID_ADP, INS_MID = 84.0, 35.0


def insurance_premium(qb1_adp: float) -> float:
    """How much QB1 fragility is worth covering, by QB1 price. [inference]

    Anchored on the two regimes Phase 6 measured and linear between them; a
    cheaper QB1 fails more often, so his backup is worth more.
    """
    lo, hi = INS_ELITE_ADP, INS_MID_ADP
    t = (min(max(qb1_adp, lo), hi) - lo) / (hi - lo)
    return INS_ELITE + t * (INS_MID - INS_ELITE)


def main() -> int:
    sched = loaders.load_schedules().filter(
        (pl.col("season") == SEASON) & (pl.col("game_type") == "REG"))
    teams = sorted(set(sched["home_team"].to_list()) | set(sched["away_team"].to_list()))
    all_weeks = sorted(sched["week"].unique().to_list())
    bye = {}
    for t in teams:
        played = set(sched.filter((pl.col("home_team") == t) | (pl.col("away_team") == t))["week"].to_list())
        missing = [w for w in all_weeks if w not in played]
        bye[t] = missing[0] if missing else None

    adp = loaders.load_ffc_adp(SEASON, teams=LEAGUE.teams).filter(
        pl.col("position") == "QB").with_columns(
        pl.col("name").map_elements(dv.normalise_name, return_dtype=pl.String).alias("key"))
    cbs = loaders.load_cbs_projections(SEASON, "QB").with_columns(
        pl.col("player").map_elements(dv.normalise_name, return_dtype=pl.String).alias("key"))

    # FFC team codes -> nflverse codes
    q = (adp.join(cbs.select(["key", "proj_points_league", "proj_ppg_league", "games"]),
                  on="key", how="inner")
         .with_columns(pl.col("team")
                       .replace({"JAC": "JAX", "LAR": "LA", "STL": "LA",
                                 "WSH": "WAS", "SD": "LAC", "OAK": "LV"})
                       .alias("team")))
    q = q.with_columns(
        pl.col("team").replace_strict(bye, default=None).alias("bye_week"))
    unmapped = q.filter(pl.col("bye_week").is_null())["team"].unique().to_list()
    if unmapped:
        print(f"WARNING: teams with no bye resolved: {unmapped}")

    # Drop thin tail buckets, then look up by rounded pick so a fractional ADP
    # cannot fall between two integer bucket edges and silently return zero.
    curve = dv.pick_value_curve(list(range(2018, 2026))).filter(pl.col("n") >= 20)
    rows = curve.to_dicts()
    _lo, _hi = rows[0]["pick_from"], rows[-1]["pick_to"]

    def pick_cost(a: float) -> float:
        pick = int(min(max(round(a), _lo), _hi))
        for r in rows:
            if r["pick_from"] <= pick <= r["pick_to"]:
                return float(r["mean_usable_par"])
        return float(rows[-1]["mean_usable_par"])

    print(BAR)
    print(f" 2026 QB BOARD  (ADP from {adp['total_drafts'][0]} real 12-team drafts, "
          f"{adp['adp_window_start'][0]} to {adp['adp_window_end'][0]})")
    print(BAR)
    print(f"{'QB':<22}{'tm':>4}{'ADP':>7}{'bye':>5}{'CBS pts':>9}{'ppg':>7}{'pick cost':>11}")
    print("-" * 88)
    for r in q.sort("adp").head(24).to_dicts():
        print(f"{r['name']:<22}{r['team']:>4}{r['adp']:>7.1f}{str(r['bye_week']):>5}"
              f"{r['proj_points_league']:>9.1f}{r['proj_ppg_league']:>7.2f}"
              f"{pick_cost(r['adp']):>11.1f}")

    cands = q.filter(pl.col("adp") <= 200).to_dicts()
    pairs = []
    for a, b in combinations(cands, 2):
        qb1, qb2 = (a, b) if a["adp"] <= b["adp"] else (b, a)
        same_bye = qb1["bye_week"] is not None and qb1["bye_week"] == qb2["bye_week"]

        # QB1 carries the season; QB2 covers his bye.
        base = qb1["proj_points_league"]
        bye_gain = 0.0 if same_bye else max(0.0, qb2["proj_ppg_league"] - REPLACEMENT_PPG)
        ins = insurance_premium(qb1["adp"]) * (qb2["proj_ppg_league"] / 18.0)
        cost = pick_cost(qb1["adp"]) + pick_cost(qb2["adp"])
        gross = base + bye_gain + WEEKLY_EDGE + ins
        pairs.append({
            "qb1": qb1["name"], "qb2": qb2["name"],
            "adp1": qb1["adp"], "adp2": qb2["adp"],
            "bye1": qb1["bye_week"], "bye2": qb2["bye_week"],
            "bye_clash": same_bye,
            "cbs_qb1": round(base, 1),
            "bye_gain": round(bye_gain, 1),
            "sel_edge": WEEKLY_EDGE,
            "insurance": round(ins, 1),
            "pick_cost": round(cost, 1),
            "net_value": round(gross - QB_REPLACEMENT - cost, 1),
        })

    d = pl.DataFrame(pairs).sort("net_value", descending=True)
    # The brief requires a shared bye to be penalised heavily unless waiver
    # strategy is modelled, which it is not. Such a pair leaves the manager with
    # no startable quarterback that week -- exactly the position a single-QB
    # manager is in -- so it is excluded from recommendations rather than
    # docked an invented number, and listed separately below.
    ok = d.filter(~pl.col("bye_clash"))

    print(f"\n{BAR}\n TOP 2026 PAIRS BY NET VALUE  (shared-bye pairs excluded)\n{BAR}")
    print(f"{'QB1':<20}{'QB2':<20}{'ADP':>11}{'byes':>8}{'ins':>6}{'cost':>7}{'NET':>8}")
    print("-" * 88)
    for r in ok.head(18).to_dicts():
        adps = f"{r['adp1']:.0f}/{r['adp2']:.0f}"
        byes = f"{r['bye1']}/{r['bye2']}"
        print(f"{r['qb1']:<20}{r['qb2']:<20}{adps:>11}{byes:>8}"
              f"{r['insurance']:>6.1f}{r['pick_cost']:>7.1f}{r['net_value']:>8.1f}")

    print(f"\n{BAR}\n BEST VALUE PAIRS WITH NO EARLY PICK (QB1 ADP > 60)\n{BAR}")
    late = ok.filter(pl.col("adp1") > 60)
    print(f"{'QB1':<20}{'QB2':<20}{'ADP':>11}{'byes':>8}{'cost':>7}{'NET':>8}")
    print("-" * 88)
    for r in late.head(10).to_dicts():
        adps = f"{r['adp1']:.0f}/{r['adp2']:.0f}"
        byes = f"{r['bye1']}/{r['bye2']}"
        print(f"{r['qb1']:<20}{r['qb2']:<20}{adps:>11}{byes:>8}"
              f"{r['pick_cost']:>7.1f}{r['net_value']:>8.1f}")

    clash = d.filter(pl.col("bye_clash"))
    print(f"\npairs sharing a bye week (avoid): {clash.height} of {d.height}")
    if clash.height:
        ex = clash.head(4).to_dicts()
        for r in ex:
            print(f"    {r['qb1']} + {r['qb2']}  (both bye {r['bye1']})")

    print(f"\n{BAR}\n SENSITIVITY: IS THIS RANKING ROBUST?\n{BAR}")
    print("Two inputs are estimates rather than measurements: the insurance")
    print("premium and how heavily a forgone draft pick is charged. If the order")
    print("flips when they are varied over a reasonable range, then the gaps")
    print("between pairs are smaller than the uncertainty, and the ranking should")
    print("not be read as a precise ordering.\n")

    def allen_rank(e: pl.DataFrame) -> int:
        a = (e.with_row_index("rk")
             .filter((pl.col("qb1") == "Josh Allen") | (pl.col("qb2") == "Josh Allen")))
        return int(a["rk"][0]) + 1 if a.height else -1

    for label, col in [("insurance premium", "insurance"), ("pick cost", "pick_cost")]:
        print(f"{label} multiplier:")
        print(f"{'  x':<8}{'top pair':<42}{'net':>8}{'best Allen pair rank':>22}")
        for m in (0.0, 0.5, 1.0, 1.5):
            sign = -1 if col == "pick_cost" else 1
            e = ok.with_columns(
                (pl.col("net_value") - sign * pl.col(col) + sign * pl.col(col) * m)
                .alias("nv")).sort("nv", descending=True)
            t = e.head(1).to_dicts()[0]
            print(f"  {m:<6.1f}{t['qb1'] + ' + ' + t['qb2']:<42}{t['nv']:>8.1f}"
                  f"{allen_rank(e):>22}")
        print()
    print("Read this as the headline result of Phase 8, not a footnote.")

    print(f"\n{BAR}\n WEEKS 14-16 NOTE\n{BAR}")
    p16 = sched.filter(pl.col("week").is_between(14, 16))
    have = p16.filter(pl.col("spread_line").is_not_null()).height
    print(f"2026 playoff-week games: {p16.height}, of which {have} have betting lines.")
    print("Phase 3 found no significant scoring effect for weeks 14-18, and Phase 2")
    print("found prior-season defensive ratings explain 0.39% of QB variance, so no")
    print("playoff-schedule bonus is applied. Doing so would be decoration.")

    out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
    d.write_csv(out / "phase8_pairs_2026.csv")
    q.sort("adp").write_csv(out / "phase8_qb_board_2026.csv")
    print(f"\nwrote outputs/phase8_pairs_2026.csv and phase8_qb_board_2026.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
