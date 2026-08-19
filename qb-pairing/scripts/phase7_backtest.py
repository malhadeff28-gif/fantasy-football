"""Phase 7: backtest the complete strategy against realistic benchmarks.

The question a manager actually faces is not "what will he score" but "which of
my two quarterbacks do I start". This phase locks every weekly decision using
only pregame information, then scores it, for six decision rules of increasing
sophistication:

    always QB1     never look at the matchup; start the better preseason QB
    ADP            same, stated explicitly as market consensus
    recent form    start whoever has been better over his last 8 games
    market         start whoever has the higher implied team total
    model          the Phase 5 projection
    oracle         perfect foresight; unattainable, defines the ceiling

If the sophisticated rules cannot separate from "always QB1", then the weekly
start/sit decision -- the operational core of the pairing thesis -- carries no
edge, and the value of a second quarterback is purely availability insurance.

Benchmark note: CBS and FantasyPros publish current rankings but no accessible
historical archive, so they cannot be scored over past seasons. Preseason ADP
is used as the historical consensus benchmark, and CBS is compared directly on
the 2026 board in Phase 8.

Usage:  python scripts/phase7_backtest.py
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

RULES = ["always QB1", "ADP", "recent form", "market", "model", "oracle"]


def main() -> int:
    panel = pm.build_panel(SEASONS).drop_nulls(subset=projection.FEATURES)
    proj = projection.fit_walk_forward(panel, TEST_SEASONS)

    adp = dv.load_adp(SEASONS).filter(pl.col("adp_position") == "QB")

    totals = {r: [] for r in RULES}
    hits = {r: [0, 0] for r in RULES}      # correct, total
    per_season = []

    for s in TEST_SEASONS:
        cur = proj.filter(pl.col("season") == s)
        repl = engine.weekly_replacement(cur, s)
        weeks = sorted(cur["week"].unique().to_list())

        adp_s = {r["key"]: r["adp"] for r in adp.filter(pl.col("season") == s).to_dicts()}
        rows = {}
        for pid, g in cur.group_by("player_id"):
            rows[pid[0]] = {
                r["week"]: r for r in g.select(
                    ["week", "fantasy_points", "proj", "form_fp_mean",
                     "implied_team_total", "player_display_name"]).to_dicts()
            }
        names = {p: next(iter(v.values()))["player_display_name"]
                 for p, v in rows.items()}
        akey = {p: adp_s.get(dv.normalise_name(n), 999.0) for p, n in names.items()}

        elig = [p for p, v in rows.items() if len(v) >= 12]
        season_acc = {r: [0, 0] for r in RULES}

        for a, b in combinations(sorted(elig), 2):
            # QB1 = better preseason ADP; uses no in-season information
            qb1, qb2 = (a, b) if akey[a] <= akey[b] else (b, a)
            tot = {r: 0.0 for r in RULES}
            for w in weeks:
                ra, rb = rows[qb1].get(w), rows[qb2].get(w)
                if ra is None and rb is None:
                    for r in RULES:
                        tot[r] += repl.get(w, 12.0)
                    continue
                if ra is None or rb is None:
                    only = ra or rb
                    for r in RULES:
                        tot[r] += only["fantasy_points"]
                    continue

                picks = {
                    "always QB1": ra,
                    "ADP": ra,   # identical by construction; kept for clarity
                    "recent form": ra if ra["form_fp_mean"] >= rb["form_fp_mean"] else rb,
                    "market": ra if ra["implied_team_total"] >= rb["implied_team_total"] else rb,
                    "model": ra if ra["proj"] >= rb["proj"] else rb,
                    "oracle": ra if ra["fantasy_points"] >= rb["fantasy_points"] else rb,
                }
                for r, pick in picks.items():
                    tot[r] += pick["fantasy_points"]
                    if ra["fantasy_points"] != rb["fantasy_points"]:
                        better = ra if ra["fantasy_points"] > rb["fantasy_points"] else rb
                        hits[r][1] += 1
                        season_acc[r][1] += 1
                        if pick is better:
                            hits[r][0] += 1
                            season_acc[r][0] += 1
            for r in RULES:
                totals[r].append(tot[r])
        per_season.append({
            "season": s,
            **{r: (season_acc[r][0] / season_acc[r][1] * 100
                   if season_acc[r][1] else float("nan")) for r in RULES},
        })

    print(BAR)
    print(" WEEKLY START/SIT RULES, ALL DECISIONS LOCKED PREGAME")
    print(BAR)
    print(f"pairs x seasons evaluated: {len(totals['model']):,}")
    print(f"\n{'decision rule':<18}{'pts/season':>12}{'vs always-QB1':>15}{'accuracy':>11}")
    print("-" * 78)
    base = float(np.mean(totals["always QB1"]))
    for r in RULES:
        m = float(np.mean(totals[r]))
        acc = hits[r][0] / hits[r][1] * 100 if hits[r][1] else float("nan")
        d = "--" if r == "always QB1" else f"{m - base:+.1f}"
        print(f"{r:<18}{m:>12.1f}{d:>15}{acc:>10.2f}%")
    print("-" * 78)
    orc = float(np.mean(totals["oracle"]))
    mod = float(np.mean(totals["model"]))
    print(f"model captures {(mod - base) / (orc - base) * 100:.1f}% of the "
          f"available upside over always-QB1")

    print(f"\n{BAR}\n START/SIT ACCURACY BY SEASON (%)\n{BAR}")
    print(f"{'season':<9}" + "".join(f"{r:>14}" for r in RULES if r != "ADP"))
    print("-" * 78)
    for row in per_season:
        print(f"{row['season']:<9}" +
              "".join(f"{row[r]:>14.2f}" for r in RULES if r != "ADP"))

    print(f"\n{BAR}\n WHERE THE THESIS FAILS\n{BAR}")
    form_gain = float(np.mean(totals["recent form"])) - base
    mkt_gain = float(np.mean(totals["market"])) - base
    print(f"recent form over always-QB1 : {form_gain:+.1f} pts/season")
    print(f"market over always-QB1      : {mkt_gain:+.1f} pts/season")
    print(f"model over always-QB1       : {mod - base:+.1f} pts/season")
    print(f"perfect foresight ceiling   : {orc - base:+.1f} pts/season")
    print("\nThe weekly decision is where the pairing thesis claimed its edge.")
    print("Whatever the model adds here is the entire operational payoff; the")
    print("rest of a pair's value is availability insurance that any second")
    print("quarterback would supply.")

    out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
    pl.DataFrame([{"rule": r, "pts_per_season": float(np.mean(totals[r])),
                   "accuracy_pct": hits[r][0] / hits[r][1] * 100 if hits[r][1] else None}
                  for r in RULES]).write_csv(out / "phase7_decision_rules.csv")
    pl.DataFrame(per_season).write_csv(out / "phase7_accuracy_by_season.csv")
    print(f"\nwrote outputs/phase7_decision_rules.csv and phase7_accuracy_by_season.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
