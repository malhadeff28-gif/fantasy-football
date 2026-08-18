"""Phase 2c: what does a REAL chooser capture from a second quarterback?

Phase 2b measured the ceiling (perfect weekly choice) but used an unfair
single-QB baseline: it scored a bye week as zero, when any real manager would
stream a replacement. That inflated the pairing benefit by roughly a full
replacement week.

This script fixes both halves:

* the single-QB manager streams a median starter on QB1's bye, which is
  generous to the strategy the thesis is trying to beat;
* the two-QB manager starts whichever quarterback his own pregame model
  projects higher -- no hindsight, walk-forward projections only.

The gap between model-selected and oracle is the cost of imperfect foresight.
The gap between model-selected and single-QB is the actual prize.

Usage:  python scripts/phase2c_pair_value.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                      # noqa: E402
import warnings                                 # noqa: E402
from itertools import combinations              # noqa: E402
from pathlib import Path                        # noqa: E402

import numpy as np                              # noqa: E402
import polars as pl                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.features import panel as panel_mod  # noqa: E402

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
BAR = "=" * 78

FEATS = ["form_fp_mean", "career_fp_mean", "form_pass_att", "form_rush_att",
         "implied_team_total", "spread_abs", "total_line",
         "def_adj", "def_fp_allowed", "is_home", "indoors", "rest", "div_game"]


def projected_panel() -> pl.DataFrame:
    """Walk-forward pregame projections for every QB-week in the test seasons."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    p = panel_mod.build_panel(SEASONS).drop_nulls(
        subset=["form_fp_mean", "form_pass_att", "rush_fp_share"])
    frames = []
    for s in TEST_SEASONS:
        tr, te = p.filter(pl.col("season") < s), p.filter(pl.col("season") == s)
        Xtr = tr.select(FEATS).to_numpy().astype(float)
        Xte = te.select(FEATS).to_numpy().astype(float)
        sc = StandardScaler().fit(Xtr)
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(
            sc.transform(Xtr), tr["fantasy_points"].to_numpy())
        frames.append(te.with_columns(
            pl.Series("proj", m.predict(sc.transform(Xte)))))
    return pl.concat(frames)


def main() -> int:
    p = projected_panel()

    # Weekly median starter score = what a single-QB manager streams on a bye.
    # Generous to the single-QB strategy on purpose.
    repl = (p.group_by(["season", "week"])
            .agg(pl.col("fantasy_points").median().alias("repl"))
            .sort(["season", "week"]))
    repl_map = {(r["season"], r["week"]): r["repl"] for r in repl.to_dicts()}

    rows = []
    for s in TEST_SEASONS:
        cur = p.filter(pl.col("season") == s)
        counts = cur.group_by("player_id").agg(pl.len().alias("g"))
        eligible = counts.filter(pl.col("g") >= 12)["player_id"].to_list()
        prev = (p.filter(pl.col("season") == s - 1).group_by("player_id")
                .agg(pl.col("fantasy_points").mean().alias("prev_ppg")))
        cur = cur.filter(pl.col("player_id").is_in(eligible)).join(
            prev, on="player_id", how="inner")
        if cur.is_empty():
            continue

        ids = sorted(cur["player_id"].unique().to_list())
        wk = {pid[0]: {r["week"]: (r["fantasy_points"], r["proj"])
                       for r in g.select(["week", "fantasy_points", "proj"]).to_dicts()}
              for pid, g in cur.group_by("player_id")}
        ppg = {r["player_id"]: r["prev_ppg"]
               for r in cur.select(["player_id", "prev_ppg"]).unique().to_dicts()}
        names = {r["player_id"]: r["player_display_name"]
                 for r in cur.select(["player_id", "player_display_name"]).unique().to_dicts()}

        all_weeks = sorted(cur["week"].unique().to_list())
        for a, b in combinations(ids, 2):
            qb1, qb2 = (a, b) if ppg[a] >= ppg[b] else (b, a)
            w1, w2 = wk[qb1], wk[qb2]
            if len(set(w1) | set(w2)) < 12:
                continue
            solo = model = oracle = 0.0
            n = right = 0
            for w in all_weeks:
                r1, r2 = w1.get(w), w2.get(w)
                if r1 is None and r2 is None:
                    continue
                n += 1
                rp = repl_map.get((s, w), 15.0)
                # single-QB manager: QB1, or a streamed median starter on bye
                solo += r1[0] if r1 is not None else rp
                # two-QB manager: start the higher pregame projection
                if r1 is None:
                    model += r2[0]
                elif r2 is None:
                    model += r1[0]
                else:
                    pick = r1 if r1[1] >= r2[1] else r2
                    other = r2 if r1[1] >= r2[1] else r1
                    model += pick[0]
                    if pick[0] != other[0]:
                        right += pick[0] > other[0]
                # perfect foresight
                oracle += max(v[0] for v in (r1, r2) if v is not None)
            rows.append({"season": s, "qb1": names[qb1], "qb2": names[qb2],
                         "weeks": n, "solo": solo, "model": model,
                         "oracle": oracle, "hits": right})

    d = pl.DataFrame(rows)
    solo_m, mod_m, orc_m = d["solo"].mean(), d["model"].mean(), d["oracle"].mean()

    print(BAR)
    print(" REALISTIC VALUE OF CARRYING A SECOND QUARTERBACK")
    print(BAR)
    print(f"pairs evaluated: {d.height:,}   seasons: {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}")
    print(f"\n{'strategy':<40}{'pts/season':>12}{'vs 1 QB':>12}")
    print("-" * 78)
    print(f"{'single QB1 + streamed bye replacement':<40}{solo_m:>12.1f}{'--':>12}")
    print(f"{'two QBs, model-selected start':<40}{mod_m:>12.1f}{mod_m - solo_m:>+12.1f}")
    print(f"{'two QBs, perfect foresight (ceiling)':<40}{orc_m:>12.1f}{orc_m - solo_m:>+12.1f}")
    print("-" * 78)
    capture = (mod_m - solo_m) / (orc_m - solo_m) * 100
    print(f"share of the available prize actually captured: {capture:.1f}%")
    print(f"cost of imperfect foresight: {orc_m - mod_m:.1f} pts/season")

    beat = (d["model"] > d["solo"]).mean() * 100
    print(f"\npairs where the 2-QB strategy beat the 1-QB baseline: {beat:.1f}%")
    print(f"median gain: {(d['model'] - d['solo']).median():+.1f} pts/season")
    print(f"25th / 75th percentile gain: "
          f"{(d['model'] - d['solo']).quantile(0.25):+.1f} / "
          f"{(d['model'] - d['solo']).quantile(0.75):+.1f}")

    print(f"\n{'season':<10}{'1 QB':>10}{'2 QB model':>13}{'oracle':>10}{'gain':>9}")
    print("-" * 78)
    for s in TEST_SEASONS:
        g = d.filter(pl.col("season") == s)
        if g.is_empty():
            continue
        print(f"{s:<10}{g['solo'].mean():>10.1f}{g['model'].mean():>13.1f}"
              f"{g['oracle'].mean():>10.1f}{g['model'].mean() - g['solo'].mean():>+9.1f}")

    print(f"\n{BAR}")
    print(" Context: one bench slot in a 12-team league. Phase 6 must weigh this")
    print(" gain against the best alternative use of that slot and the draft")
    print(" capital spent on QB2.")
    print(BAR)

    out = Path("outputs"); out.mkdir(exist_ok=True)
    d.write_csv(out / "phase2c_pair_value.csv")
    top = d.with_columns((pl.col("model") - pl.col("solo")).alias("gain")) \
           .sort("gain", descending=True).head(12)
    print(f"\nBest observed pairs (hindsight -- illustrative only):")
    print(f"{'season':<8}{'QB1':<22}{'QB2':<22}{'gain':>8}")
    for r in top.to_dicts():
        print(f"{r['season']:<8}{r['qb1']:<22}{r['qb2']:<22}{r['gain']:>+8.1f}")
    print(f"\nwrote outputs/phase2c_pair_value.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
