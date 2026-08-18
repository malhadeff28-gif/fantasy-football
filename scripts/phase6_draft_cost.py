"""Phase 6: does a second quarterback earn its draft pick and bench slot?

Everything is expressed in points above replacement so that quarterback points
and the forgone non-QB pick are in the same units:

    net value = QB points delivered to the lineup
              - QB replacement level (QB12)
              - expected return of the picks spent on quarterbacks

The last term is the honest cost. A pick spent on a second quarterback is a
pick not spent on a running back or receiver, and the historical value of picks
in each range is measured directly rather than assumed.

Strategies compared, each defined by ADP rank so no hindsight enters the
selection:

    premium              the QB1 off the board
    premium + late       elite QB plus a cheap partner
    strong value + comp  a mid-priced starter plus a partner
    late pair            two cheap quarterbacks
    single value QB      one mid-priced starter, no QB2
    streaming            no draft capital, best projection off the pool

Usage:  python scripts/phase6_draft_cost.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                       # noqa: E402
import warnings                                  # noqa: E402
from pathlib import Path                         # noqa: E402

import numpy as np                               # noqa: E402
import polars as pl                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.benchmarks import draft_value as dv  # noqa: E402
from qbpair.features import panel as pm          # noqa: E402
from qbpair.models import projection             # noqa: E402
from qbpair.pairing import engine                # noqa: E402

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
BAR = "=" * 78

# QB ADP rank -> strategy. Ranks, not names, so selection uses only market
# consensus available on draft day.
STRATEGIES = {
    "premium (QB1 only)":            [1],
    "premium + late partner":        [1, 16],
    "strong value + complement":     [4, 16],
    "late pair":                     [10, 18],
    "single value QB":               [8],
    "streaming (no QB drafted)":     [],
}
STREAM_MIN_RANK = 20   # pool for the streaming strategy


def pick_cost_lookup(curve: pl.DataFrame):
    """Expected non-QB usable PAR at a given overall pick."""
    rows = curve.to_dicts()

    def cost(adp: float) -> float:
        for r in rows:
            if r["pick_from"] <= adp <= r["pick_to"]:
                return float(r["mean_usable_par"])
        return float(rows[-1]["mean_usable_par"]) if rows else 0.0

    return cost


def main() -> int:
    panel = pm.build_panel(SEASONS).drop_nulls(subset=projection.FEATURES)
    proj = projection.fit_walk_forward(panel, TEST_SEASONS)

    curve = dv.pick_value_curve(SEASONS)
    cost_of = pick_cost_lookup(curve)
    adp = dv.load_adp(SEASONS).filter(pl.col("adp_position") == "QB")
    par = dv.points_above_replacement(SEASONS)
    qb_repl = {r["season"]: r["replacement_points"]
               for r in dv.replacement_levels(dv.season_totals(SEASONS))
               .filter(pl.col("position") == "QB").to_dicts()}

    print(BAR)
    print(" QB DRAFT COST BY SEASON (ADP rank -> overall pick)")
    print(BAR)
    print(f"{'season':<9}{'QB1':>8}{'QB4':>8}{'QB8':>8}{'QB10':>8}{'QB16':>8}{'QB18':>8}")
    print("-" * 78)
    ranked = {}
    for s in TEST_SEASONS:
        a = adp.filter(pl.col("season") == s).sort("adp")
        ranked[s] = a
        picks = a["adp"].to_list()
        row = [f"{picks[i - 1]:.0f}" if len(picks) >= i else "-"
               for i in (1, 4, 8, 10, 16, 18)]
        print(f"{s:<9}" + "".join(f"{v:>8}" for v in row))

    # name key -> player_id, for the season's panel
    results = []
    for s in TEST_SEASONS:
        cur = proj.filter(pl.col("season") == s)
        keymap = {}
        for r in cur.select(["player_id", "player_display_name"]).unique().to_dicts():
            keymap.setdefault(dv.normalise_name(r["player_display_name"]),
                              r["player_id"])
        names = {r["player_id"]: r["player_display_name"]
                 for r in cur.select(["player_id", "player_display_name"])
                 .unique().to_dicts()}
        slate = engine._slate(cur, s, "proj", "fantasy_points")
        weeks = sorted(cur["week"].unique().to_list())
        repl = engine.weekly_replacement(cur, s)
        a = ranked[s]
        qb_rows = a.to_dicts()

        def qb_at(rank: int):
            if len(qb_rows) < rank:
                return None
            r = qb_rows[rank - 1]
            pid = keymap.get(r["key"])
            return (pid, r["adp"], r["adp_name"]) if pid else None

        for label, ranks in STRATEGIES.items():
            if not ranks:   # streaming
                pool = [keymap.get(r["key"]) for r in qb_rows[STREAM_MIN_RANK - 1:]]
                pool = [p for p in pool if p]
                pool += [pid for pid in slate
                         if pid not in {keymap.get(r["key"]) for r in qb_rows}]
                pts = 0.0
                for w in weeks:
                    opts = [(slate[p][w][0], slate[p][w][1])
                            for p in pool if p in slate and w in slate[p]]
                    pts += max(opts)[1] if opts else repl.get(w, 12.0)
                draft_cost = 0.0
                picked = "waiver pool"
            else:
                sel = [qb_at(r) for r in ranks]
                if any(x is None for x in sel):
                    continue
                draft_cost = sum(cost_of(x[1]) for x in sel)
                picked = " + ".join(f"{x[2]} ({x[1]:.0f})" for x in sel)
                if len(sel) == 1:
                    pid = sel[0][0]
                    pts = sum(
                        slate[pid][w][1] if pid in slate and w in slate[pid]
                        else repl.get(w, 12.0) for w in weeks)
                else:
                    pr = engine.score_pair(slate, names, sel[0][0], sel[1][0],
                                           s, weeks, replacement=repl)
                    pts = pr.total_points

            results.append({
                "season": s, "strategy": label, "qbs": picked,
                "qb_points": round(pts, 1),
                "above_repl": round(pts - qb_repl.get(s, 270.0), 1),
                "draft_cost": round(draft_cost, 1),
                "net_value": round(pts - qb_repl.get(s, 270.0) - draft_cost, 1),
            })

    d = pl.DataFrame(results)

    print(f"\n{BAR}\n STRATEGY COMPARISON, AVERAGED OVER {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}\n{BAR}")
    agg = (d.group_by("strategy")
           .agg(pl.col("qb_points").mean().alias("qb_pts"),
                pl.col("above_repl").mean().alias("above_repl"),
                pl.col("draft_cost").mean().alias("cost"),
                pl.col("net_value").mean().alias("net"),
                pl.len().alias("n"))
           .sort("net", descending=True))
    print(f"{'strategy':<30}{'QB pts':>9}{'vs QB12':>10}{'pick cost':>11}{'NET':>9}")
    print("-" * 78)
    for r in agg.to_dicts():
        print(f"{r['strategy']:<30}{r['qb_pts']:>9.1f}{r['above_repl']:>10.1f}"
              f"{r['cost']:>11.1f}{r['net']:>9.1f}")
    print("-" * 78)
    print("NET = QB points delivered, minus QB12 replacement, minus the expected")
    print("return of the picks spent on quarterbacks. Higher is better.")

    print(f"\n{BAR}\n SEASON-BY-SEASON NET VALUE\n{BAR}")
    piv = d.pivot(values="net_value", index="strategy", on="season")
    cols = [c for c in piv.columns if c != "strategy"]
    print(f"{'strategy':<30}" + "".join(f"{c:>8}" for c in cols))
    print("-" * 78)
    for r in piv.to_dicts():
        print(f"{r['strategy']:<30}" +
              "".join(f"{(r[c] if r[c] is not None else float('nan')):>8.1f}" for c in cols))

    # the decisive marginal comparison
    print(f"\n{BAR}\n MARGINAL VALUE OF ADDING QB2\n{BAR}")
    solo = agg.filter(pl.col("strategy") == "premium (QB1 only)")["net"].item()
    pair = agg.filter(pl.col("strategy") == "premium + late partner")["net"].item()
    print(f"premium QB alone        net {solo:+.1f}")
    print(f"premium QB + partner    net {pair:+.1f}   ->  QB2 adds {pair - solo:+.1f}")
    single = agg.filter(pl.col("strategy") == "single value QB")["net"].item()
    latep = agg.filter(pl.col("strategy") == "late pair")["net"].item()
    print(f"single value QB         net {single:+.1f}")
    print(f"late pair               net {latep:+.1f}   ->  second cheap QB adds {latep - single:+.1f}")

    out = Path("outputs"); out.mkdir(exist_ok=True)
    d.write_csv(out / "phase6_strategies.csv")
    curve.write_csv(out / "phase6_pick_value_curve.csv")
    print(f"\nwrote outputs/phase6_strategies.csv and phase6_pick_value_curve.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
