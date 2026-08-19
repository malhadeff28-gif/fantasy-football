"""QB-024: what each correction did to the slot-7 late-QB analysis.

`phase8b_late_qb_slot7.py` was published before four corrections landed under
it. This script reruns it with those corrections applied CUMULATIVELY, so each
row of the attribution table is the movement caused by exactly one change:

  V0   as published (baseline)
  V1   + C1a  season_totals stops at week 16 -- as the correction literally
              landed, i.e. the replacement level and pick curve move to the
              league calendar while the script still SUMS QB POINTS to week 18
  V1b  + C1b  the same calendar applied to the QB points too (the consistency
              fix this rerun adds; C1 is not safe to apply by halves)
  V2   + C2   pick-cost lookup rounds and clamps before the bucket scan
  V3   + C3   spread_abs dropped from projection.FEATURES
  V4   + C4   the insurance premium retired (2026 board only; the backtest
              scores realised points and never used it)

The legacy formulas live HERE, not in the production script, because they are
retired and should not be reachable from it.

Usage:  python scripts/qb024_slot7_attribution.py
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
from pathlib import Path                         # noqa: E402

import polars as pl                              # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import phase8b_late_qb_slot7 as p8b              # noqa: E402
from qbpair.benchmarks import draft_value as dv  # noqa: E402
from qbpair.data import loaders                  # noqa: E402
from qbpair.models import projection             # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")

SEASONS = list(range(2018, 2026))
TEST = list(range(2020, 2026))
SLOT, TEAMS = 7, 12
BAR = "=" * 92
SWEEP = [None, 90, 103, 114, 127, 138, 151, 162]

# --- pre-correction values -------------------------------------------------
OLD_FEATURES = ["form_fp_mean", "career_fp_mean", "form_pass_att",
                "form_rush_att", "implied_team_total", "total_line",
                "spread_abs", "is_home", "indoors"]
OLD_QB_REPLACEMENT = 272.0      # hardcoded constant on the 2026 board
OLD_WEEKLY_EDGE = 4.9           # QB-013, superseded by the rank-matched +1.7
OLD_INS_ELITE_ADP, OLD_INS_ELITE = 16.0, 15.0
OLD_INS_MID_ADP, OLD_INS_MID = 84.0, 35.0


def old_insurance_premium(adp: float) -> float:
    """RETIRED. Interpolates the +9.7 / +30.1 marginals that QB-020 broke."""
    t = ((min(max(adp, OLD_INS_ELITE_ADP), OLD_INS_MID_ADP) - OLD_INS_ELITE_ADP)
         / (OLD_INS_MID_ADP - OLD_INS_ELITE_ADP))
    return OLD_INS_ELITE + t * (OLD_INS_MID - OLD_INS_ELITE)


def totals_through(last_week: int):
    """`draft_value.season_totals` with the week cut-off as a parameter."""
    from qbpair.benchmarks.draft_value import REPLACEMENT_RANK, normalise_name

    def fn(seasons):
        st = loaders.load_player_stats(seasons).filter(
            (pl.col("season_type") == "REG")
            & (pl.col("week") <= last_week)
            & pl.col("position").is_in(list(REPLACEMENT_RANK)))
        return (st.group_by(["season", "player_id", "player_display_name",
                             "position"])
                .agg(pl.col("fantasy_points").sum().alias("season_points"),
                     pl.len().alias("games"))
                .with_columns(pl.col("player_display_name")
                              .map_elements(normalise_name,
                                            return_dtype=pl.String).alias("key")))
    return fn


def curve_and_replacement(last_week: int):
    """Pick-value curve and QB12 levels as of a given fantasy calendar.

    Monkeypatches `season_totals` so the rest of the code path -- PAR, usable
    PAR, the ADP join, the buckets -- is the committed one, not a copy of it.
    """
    real = dv.season_totals
    dv.season_totals = totals_through(last_week)
    try:
        curve = dv.pick_value_curve(SEASONS).filter(pl.col("n") >= 20)
        repl = {r["season"]: r["replacement_points"]
                for r in dv.replacement_levels(dv.season_totals(SEASONS))
                .filter(pl.col("position") == "QB").to_dicts()}
    finally:
        dv.season_totals = real
    return curve, repl


def buggy_cost_lookup(curve: pl.DataFrame):
    """The pre-C2 lookup: a float ADP between integer edges matches nothing."""
    rows = curve.to_dicts()

    def cost(adp: float) -> float:
        for r in rows:
            if r["pick_from"] <= adp <= r["pick_to"]:
                return float(r["mean_usable_par"])
        return float(rows[-1]["mean_usable_par"])
    return cost


def legacy_pair_table(q: pl.DataFrame, cost, qb_replacement: float,
                      p1: int, p2_opts: list[int],
                      calendar_16: bool = False) -> pl.DataFrame:
    """The published 2026 pair valuation, insurance premium and all.

    `calendar_16` rebases QB1's projected points onto the 16-week league season
    (CBS projects 17 games) while leaving the retired insurance premium in
    place, which isolates the calendar half of the correction from C4.
    """
    rows = []
    for a, b in combinations(q.to_dicts(), 2):
        qb1, qb2 = (a, b) if a["adp"] <= b["adp"] else (b, a)
        if p8b.p_available(qb1["adp"], qb1["stdev"], p1) < 0.35:
            continue
        best = None
        for p in p2_opts:
            av2 = p8b.p_available(qb2["adp"], qb2["stdev"], p)
            if av2 >= 0.65:
                best = (p, av2)
        if best is None:
            continue
        p2, _av2 = best
        same_bye = (qb1["bye_week"] is not None
                    and qb1["bye_week"] == qb2["bye_week"])
        if same_bye:
            continue
        base = (qb1["proj_ppg_league"] * p8b.GAMES_PLAYED + p8b.REPLACEMENT_PPG
                if calendar_16 else qb1["proj_points_league"])  # CBS 17-game
        bye_gain = max(0.0, qb2["proj_ppg_league"] - p8b.REPLACEMENT_PPG)
        ins = old_insurance_premium(qb1["adp"]) * (qb2["proj_ppg_league"] / 18.0)
        rows.append({
            "qb1": qb1["name"], "qb2": qb2["name"],
            "take1_at": p1, "take2_at": p2,
            "insurance": round(ins, 1),
            "pick_cost": round(cost(p1) + cost(p2), 1),
            "net": round(base + bye_gain + OLD_WEEKLY_EDGE + ins
                         - qb_replacement - cost(p1) - cost(p2), 1),
        })
    return pl.DataFrame(rows).sort("net", descending=True)


def main() -> int:
    picks = p8b.snake_picks(SLOT, TEAMS, 16)
    qb_picks = [p for i, p in enumerate(picks, 1) if i >= p8b.FIRST_QB_ROUND]
    p1 = qb_picks[0]

    curve18, repl18 = curve_and_replacement(18)
    curve16, repl16 = curve_and_replacement(16)
    q = p8b.load_board_2026()

    # C2 AUDIT RESULT: phase8b never had the phase6 bucket-gap bug -- both of
    # its cost functions already rounded and clamped -- so every config below
    # uses the fixed lookup and C2 contributes exactly zero. `Xbug` is the
    # counterfactual: what the bug WOULD have cost had this script shared it.
    cost18 = p8b.cost_lookup(curve18)
    cost16 = p8b.cost_lookup(curve16)
    buggy16 = buggy_cost_lookup(curve16)

    configs = [
        ("V0  as published", dict(cost=cost18, qb_repl=repl18,
                                  features=OLD_FEATURES, last_week=18)),
        ("V1  +C1a week16 curve/repl", dict(cost=cost16, qb_repl=repl16,
                                            features=OLD_FEATURES, last_week=18)),
        ("V1b +C1b week16 QB points", dict(cost=cost16, qb_repl=repl16,
                                           features=OLD_FEATURES, last_week=16)),
        ("V2  +C2 cost lookup fix", dict(cost=cost16, qb_repl=repl16,
                                         features=OLD_FEATURES, last_week=16)),
        ("V3  +C3 drop spread_abs", dict(cost=cost16, qb_repl=repl16,
                                         features=projection.FEATURES,
                                         last_week=16)),
        ("V4  +C4 retire insurance", dict(cost=cost16, qb_repl=repl16,
                                          features=projection.FEATURES,
                                          last_week=16)),
        ("Xbug counterfactual C2", dict(cost=buggy16, qb_repl=repl16,
                                        features=projection.FEATURES,
                                        last_week=16)),
    ]

    plan_rows, sweep_rows = [], []
    for lab, kw in configs:
        agg, sweep, _per_season, paired = p8b.backtest_late_qb(
            p1, sweep_picks=SWEEP, verbose=False, **kw)
        for r in agg.to_dicts():
            plan_rows.append({"config": lab, **r})
        for r in sweep.to_dicts():
            sweep_rows.append({"config": lab, **r})
        for r in paired:
            plan_rows.append({"config": lab, "plan": "PAIRED " + r["plan"],
                              "pts": None, "cost": None, "net": r["mean_diff"],
                              "sd": None, "worst": None, "se": r["se"]})

    plans = pl.DataFrame(plan_rows)
    sweeps = pl.DataFrame(sweep_rows)

    print(BAR)
    print(" QB-024 ATTRIBUTION -- HISTORICAL PLAN COMPARISON (NET, 2020-2025)")
    print(BAR)
    piv = (plans.filter(~pl.col("plan").str.starts_with("PAIRED"))
           .pivot(values="net", index="plan", on="config"))
    cols = [c for c in piv.columns if c != "plan"]
    print(f"{'plan':<40}" + "".join(f"{c.split()[0]:>10}" for c in cols))
    print("-" * 92)
    for r in piv.to_dicts():
        print(f"{r['plan']:<40}" +
              "".join(f"{(r[c] if r[c] is not None else float('nan')):>10.1f}"
                      for c in cols))
    print("\nlegend: " + " | ".join(c for c in cols))

    print(f"\n{BAR}\n SAME, BUT QB POINTS AND PICK COST\n{BAR}")
    for val in ("pts", "cost", "se"):
        pv = (plans.filter(~pl.col("plan").str.starts_with("PAIRED"))
              .pivot(values=val, index="plan", on="config"))
        print(f"\n[{val}]")
        for r in pv.to_dicts():
            print(f"{r['plan']:<40}" +
                  "".join(f"{(r[c] if r[c] is not None else float('nan')):>10.1f}"
                          for c in cols))

    print(f"\n{BAR}\n QB2 TIMING SWEEP (NET by QB2 pick)\n{BAR}")
    sp = sweeps.pivot(values="net", index="qb2_pick", on="config")
    print(f"{'QB2 at':<10}" + "".join(f"{c.split()[0]:>10}" for c in cols))
    print("-" * 92)
    for r in sp.to_dicts():
        print(f"{r['qb2_pick']:<10}" +
              "".join(f"{(r[c] if r[c] is not None else float('nan')):>10.1f}"
                      for c in cols))

    # ---- 2026 board ----
    print(f"\n{BAR}\n 2026 BOARD: TOP OF THE RANKING UNDER EACH CORRECTION\n{BAR}")
    proj_repl = (sorted(q["proj_ppg_league"].to_list(), reverse=True)[11]
                 * p8b.GAMES_PLAYED + p8b.REPLACEMENT_PPG)
    board_rows = []
    legacy_v0 = legacy_pair_table(q, cost18, OLD_QB_REPLACEMENT, p1, qb_picks[1:6])
    legacy_v1 = legacy_pair_table(q, cost16, OLD_QB_REPLACEMENT, p1, qb_picks[1:6])
    legacy_v1b = legacy_pair_table(q, cost16, proj_repl, p1, qb_picks[1:6],
                                   calendar_16=True)
    new_pairs = p8b.pair_table(q, cost16, proj_repl, p1, qb_picks[1:6]).filter(
        ~pl.col("bye_clash")).sort("net", descending=True)
    for lab, t in (("V0  as published", legacy_v0),
                   ("V1  +C1a week16 curve", legacy_v1),
                   ("V1b +C1b league calendar", legacy_v1b),
                   ("V4  +C4 insurance retired", new_pairs)):
        for r in t.head(3).to_dicts():
            board_rows.append({
                "config": lab, "qb1": r["qb1"], "qb2": r["qb2"],
                "take": f"{r['take1_at']}/{r['take2_at']}",
                "insurance": r.get("insurance"),
                "qb2_marginal": r.get("qb2_marginal"), "net": r["net"]})
    bt = pl.DataFrame(board_rows)
    print(f"{'config':<28}{'QB1':<18}{'QB2':<18}{'take':>9}{'ins':>7}"
          f"{'QB2+':>7}{'NET':>8}")
    print("-" * 92)
    for r in bt.to_dicts():
        ins = "-" if r["insurance"] is None else f"{r['insurance']:.1f}"
        mg = "-" if r["qb2_marginal"] is None else f"{r['qb2_marginal']:+.1f}"
        print(f"{r['config']:<28}{r['qb1']:<18}{r['qb2']:<18}{r['take']:>9}"
              f"{ins:>7}{mg:>7}{r['net']:>8.1f}")

    single = p8b.single_qb_table(q, cost16, proj_repl, qb_picks[:6])
    print(f"\n{BAR}\n 2026 SINGLE-QB RECOMMENDATION (the corrected output)\n{BAR}")
    print(f"{'QB':<20}{'ADP':>7}{'take at':>9}{'P(there)':>10}{'NET':>8}")
    print("-" * 92)
    for r in single.filter(pl.col("gettable")).head(8).to_dicts():
        print(f"{r['qb']:<20}{r['adp']:>7.1f}{r['take_at']:>9}"
              f"{r['p_available'] * 100:>9.0f}%{r['net']:>8.1f}")

    # ---- C2 diagnostic: how many picks would the buggy lookup have freed? ----
    adp_hist = dv.load_adp(SEASONS).filter(pl.col("adp_position") == "QB")
    gaps = [r for r in adp_hist.to_dicts()
            if buggy16(r["adp"]) != cost16(r["adp"])]
    used = qb_picks[:6]                 # p1 is qb_picks[0]
    print(f"\n{BAR}\n C2 AUDIT: BUCKET-GAP EXPOSURE\n{BAR}")
    print(f"QB ADP values 2018-2025 charged differently by the buggy lookup: "
          f"{len(gaps)} of {adp_hist.height}")
    print("picks this script spends: " + ", ".join(str(p) for p in sorted(set(used))))
    print("all integers, so the gap never bites here: "
          f"{all(buggy16(p) == cost16(p) for p in used)}")

    out = HERE.parent / "outputs"
    out.mkdir(exist_ok=True)
    plans.write_csv(out / "qb024_attribution_plans.csv")
    sweeps.write_csv(out / "qb024_attribution_sweep.csv")
    bt.write_csv(out / "qb024_attribution_2026_board.csv")
    pl.DataFrame([{"season": s, "repl_week18": repl18[s],
                   "repl_week16": repl16[s]} for s in SEASONS]).write_csv(
        out / "qb024_replacement_levels.csv")
    curve18.join(curve16, on="bkt", suffix="_w16").select(
        ["bkt", "pick_from", "pick_to", "mean_usable_par",
         "mean_usable_par_w16"]).write_csv(out / "qb024_pick_curve_shift.csv")
    print("\nwrote outputs/qb024_attribution_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
