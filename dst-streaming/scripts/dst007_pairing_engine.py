"""DST-007: score every two-defense combination across weeks 1-6.

The weekly starter is chosen PREGAME, by expected points, before any outcome is
seen. Because the choice is made on expectation but the outcome is drawn from
the calibrated component simulator, the pair inherits a real distribution rather
than a point estimate - which is the whole reason DST-006 exists.

Two things this deliberately does NOT do:
  * it does not rank defenses on last season, because DST-005 rejected that
    (preseason ratings lose more than half their value by week 6, and lose to
    the market even in week 1);
  * it does not add defensive form or opponent-offence features, because
    DST-002 and DST-004 showed the opponent implied total already carries the
    entire signal.

It also answers, on D/ST evidence alone, whether a SECOND defense pays for its
pick against one defense plus waiver-wire streaming.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                        # noqa: E402
from itertools import combinations                # noqa: E402
from pathlib import Path                          # noqa: E402

import numpy as np                                # noqa: E402
import polars as pl                               # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))
import bootstrap                                  # noqa: E402
bootstrap.setup()
import markets                                    # noqa: E402
import nfl_data                                   # noqa: E402
import dst_data                                   # noqa: E402
import dst_distributions as dd                    # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_width_chars(160)

TRAIN = list(range(2018, 2026))
SEASON = 2026
WEEKS = list(range(1, 7))
N_SIMS = 3000
UNDRAFTED = 200.0
BAR = "=" * 92


def build_2026_frame() -> pl.DataFrame:
    """Defense-weeks for 2026 weeks 1-6, with the one predictor that matters."""
    ctx = nfl_data.team_game_context(SEASON, WEEKS).filter(
        pl.col("game_type") == "REG")
    return ctx.select("season", "week", "team", "opponent",
                      "implied_opp_total", "implied_team_total",
                      "is_home", "indoors").drop_nulls("implied_opp_total")


def pair_totals(sims: np.ndarray, rows: pl.DataFrame, idx: dict, exp: np.ndarray,
                a: str, b: str, weeks: list[int]) -> np.ndarray | None:
    """Simulated season totals for a pair, starter chosen pregame each week."""
    total = np.zeros(sims.shape[0])
    used = 0
    for w in weeks:
        ia, ib = idx.get((a, w)), idx.get((b, w))
        if ia is None and ib is None:
            continue                       # both on bye
        if ia is None:
            pick = ib
        elif ib is None:
            pick = ia
        else:
            pick = ia if exp[ia] >= exp[ib] else ib
        total += sims[:, pick]
        used += 1
    return total if used else None


def main() -> int:
    print(BAR)
    print(" DST-007: PAIRING ENGINE, 2026 WEEKS 1-6")
    print(BAR)

    hist = dst_data.build_dst_panel(TRAIN)
    sim = dd.fit(hist)
    fwd = build_2026_frame()
    print(f"2026 weeks 1-6 defense-games with a market line: {fwd.height}")

    dist, sims = dd.distributions(sim, fwd, n_sims=N_SIMS, keep_sims=True)
    rows = dist
    idx = {(r["team"], r["week"]): i
           for i, r in enumerate(rows.iter_rows(named=True))}
    exp = rows["exp_pts"].to_numpy()
    print(f"simulated {sims.shape[0]} draws x {sims.shape[1]} defense-weeks")

    print(f"\n{BAR}\n BEST SINGLE MATCHUPS, WEEKS 1-6 (the trivial rule, made explicit)\n{BAR}")
    print(f"{'wk':>3}  {'D':<5}{'vs':<6}{'opp imp':>9}{'exp':>7}{'p50':>6}{'p90':>6}{'top12':>8}{'bust':>7}")
    print("-" * 92)
    for r in rows.sort("exp_pts", descending=True).head(12).iter_rows(named=True):
        t12 = r["p_top12"] if r["p_top12"] is not None else float("nan")
        print(f"{r['week']:>3}  {r['team']:<5}{r['opponent']:<6}"
              f"{r['implied_opp_total']:>9.1f}{r['exp_pts']:>7.2f}{r['p50']:>6.1f}"
              f"{r['p90']:>6.1f}{t12:>7.0%}{r['p_bust']:>7.0%}")

    adp = markets.load_ffc_adp(SEASON).filter(pl.col("position") == "DEF")
    cost = {r["team"]: r["adp"] for r in adp.iter_rows(named=True)}
    teams = sorted(rows["team"].unique().to_list())
    print(f"\ndefenses with a 2026 ADP: {len(cost)} of {len(teams)}; "
          f"the rest are waiver-wire streamers")

    print(f"\n{BAR}\n EVERY TWO-DEFENSE COMBINATION\n{BAR}")
    recs = []
    for a, b in combinations(teams, 2):
        t6 = pair_totals(sims, rows, idx, exp, a, b, WEEKS)
        t4 = pair_totals(sims, rows, idx, exp, a, b, WEEKS[:4])
        if t6 is None or t4 is None:
            continue
        sub_all = rows.filter(pl.col("team").is_in([a, b]))
        both_bad = 0
        t12w = t6w = 0.0
        for w in WEEKS:
            sub = sub_all.filter(pl.col("week") == w)
            if not sub.height:
                continue
            if sub["exp_pts"].max() < 5.5:
                both_bad += 1
            p12 = sub["p_top12"].fill_null(0.0).to_numpy()
            p6 = sub["p_top6"].fill_null(0.0).to_numpy()
            t12w += 1 - float(np.prod(1 - p12))
            t6w += 1 - float(np.prod(1 - p6))
        ca, cb = cost.get(a, UNDRAFTED), cost.get(b, UNDRAFTED)
        recs.append({
            "dst_a": a, "dst_b": b, "adp_a": round(ca, 1), "adp_b": round(cb, 1),
            "w1_4_exp": round(float(t4.mean()), 1),
            "w1_6_exp": round(float(t6.mean()), 1),
            "w1_6_p10": round(float(np.percentile(t6, 10)), 1),
            "w1_6_p50": round(float(np.percentile(t6, 50)), 1),
            "w1_6_p90": round(float(np.percentile(t6, 90)), 1),
            "exp_top12_weeks": round(t12w, 2),
            "exp_top6_weeks": round(t6w, 2),
            "both_bad_weeks": both_bad,
            "cost_sum": round(ca + cb, 1),
        })

    d = pl.DataFrame(recs)
    print(f"pairs scored: {d.height}")

    def show(df: pl.DataFrame, title: str, n: int = 10) -> None:
        print(f"\n{title}")
        print(f"{'A':<6}{'B':<7}{'ADP':>13}{'W1-4':>8}{'W1-6':>8}{'p10':>7}"
              f"{'p90':>7}{'top12wk':>10}{'bad':>5}")
        print("-" * 92)
        for r in df.head(n).iter_rows(named=True):
            adps = f"{r['adp_a']:.0f}/{r['adp_b']:.0f}"
            print(f"{r['dst_a']:<6}{r['dst_b']:<7}{adps:>13}{r['w1_4_exp']:>8.1f}"
                  f"{r['w1_6_exp']:>8.1f}{r['w1_6_p10']:>7.1f}{r['w1_6_p90']:>7.1f}"
                  f"{r['exp_top12_weeks']:>10.2f}{r['both_bad_weeks']:>5}")

    show(d.sort("w1_6_exp", descending=True), "BEST OVERALL, WEEKS 1-6")
    show(d.sort("w1_4_exp", descending=True), "BEST FIRST FOUR WEEKS")
    show(d.filter(pl.col("cost_sum") >= 260).sort("w1_6_exp", descending=True),
         "BEST INEXPENSIVE (both late or undrafted)")
    show(d.sort("w1_6_p10", descending=True), "HIGHEST FLOOR (10th percentile)")
    show(d.sort("w1_6_p90", descending=True), "HIGHEST CEILING (90th percentile)")

    print(f"\n{BAR}\n DOES A SECOND DEFENSE PAY FOR ITS PICK? (D/ST evidence only)\n{BAR}")
    print("Defense A is held fixed. Compare the best DRAFTED partner against")
    print("simply streaming the best undrafted defense each week.\n")
    drafted = [t for t in teams if t in cost]
    pool = [t for t in teams if t not in cost]
    gains = []
    for a in drafted:
        bs = [pair_totals(sims, rows, idx, exp, a, b, WEEKS) for b in pool]
        bs = [x for x in bs if x is not None]
        bd = [pair_totals(sims, rows, idx, exp, a, b, WEEKS)
              for b in drafted if b != a]
        bd = [x for x in bd if x is not None]
        if not bs or not bd:
            continue
        s = max(float(x.mean()) for x in bs)
        dr = max(float(x.mean()) for x in bd)
        gains.append((a, s, dr, dr - s))

    print(f"{'D/ST':<8}{'A + streamer':>15}{'A + drafted D2':>17}{'gain':>10}")
    print("-" * 92)
    for a, s, dr, g in sorted(gains, key=lambda t: -t[3])[:12]:
        print(f"{a:<8}{s:>15.1f}{dr:>17.1f}{g:>+10.1f}")
    if gains:
        g = [x[3] for x in gains]
        m, sd = float(np.mean(g)), float(np.std(g, ddof=1))
        print("-" * 92)
        print(f"mean gain from drafting a second defense: {m:+.2f} points over "
              f"weeks 1-6 (se {sd / len(g) ** 0.5:.2f}), positive for "
              f"{sum(1 for x in g if x > 0)} of {len(g)}")
        print("\nJudged on D/ST evidence alone, per the isolation instruction.")

    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    d.write_csv(out / "dst007_pairs_2026.csv")
    rows.write_csv(out / "dst007_weekly_2026.csv")
    pl.DataFrame([{"dst": a, "with_streamer": s, "with_drafted": dr, "gain": gg}
                  for a, s, dr, gg in gains]).write_csv(out / "dst007_marginal.csv")
    print("\nwrote dst007_pairs_2026.csv, dst007_weekly_2026.csv, dst007_marginal.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
