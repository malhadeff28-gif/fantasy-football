"""Late-QB strategy from the 7th pick of a 12-team draft.

Two questions, both narrower and more answerable than the general pairing
thesis:

  A. Does deliberately taking no quarterback until round 7 hold up historically?
  B. From the 7/12 slot specifically, which QB pairs are actually gettable, and
     which of them survive the Phase 8 sensitivity problem?

Availability matters as much as value here. A pair that ranks well but is gone
before your pick is not a plan. Each quarterback's draft position is treated as
roughly normal around his ADP with the observed standard deviation, so
P(available at pick N) = P(draft position >= N). FFC publishes the standard
deviation directly, so this is measured spread rather than a guess -- though
the normal shape is an approximation. [inference]

Usage:  python scripts/phase8b_late_qb_slot7.py [SLOT] [TEAMS]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                       # noqa: E402
import warnings                                  # noqa: E402
from itertools import combinations               # noqa: E402
from math import erf, sqrt                       # noqa: E402
from pathlib import Path                         # noqa: E402

import polars as pl                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.benchmarks import draft_value as dv  # noqa: E402
from qbpair.data import loaders                  # noqa: E402

warnings.filterwarnings("ignore")

SLOT = int(sys.argv[1]) if len(sys.argv) > 1 else 7
TEAMS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ROSTER = 16
SEASON = 2026
FIRST_QB_ROUND = 7          # "no QB before round 7"
BAR = "=" * 92

# Constants carried over from earlier phases.
WEEKLY_EDGE = 4.9
QB_REPLACEMENT = 272.0
REPLACEMENT_PPG = 16.0
INS_ELITE_ADP, INS_ELITE = 16.0, 15.0
INS_MID_ADP, INS_MID = 84.0, 35.0


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


def insurance_premium(adp: float) -> float:
    t = (min(max(adp, INS_ELITE_ADP), INS_MID_ADP) - INS_ELITE_ADP) / (
        INS_MID_ADP - INS_ELITE_ADP)
    return INS_ELITE + t * (INS_MID - INS_ELITE)


def backtest_late_qb(first_qb_pick: int) -> None:
    """Historical test of the constraint itself, 2020-2025.

    Each strategy picks by ADP only -- the expected best quarterback still on
    the board at a given pick -- so nothing is chosen with hindsight. Points
    are the actual weekly outcomes under the pregame start rule.
    """
    from qbpair.features import panel as pm
    from qbpair.models import projection
    from qbpair.pairing import engine

    print(f"\n{BAR}\n A2. DOES WAITING UNTIL ROUND {FIRST_QB_ROUND} ACTUALLY WORK? "
          f"(2020-2025)\n{BAR}")

    seasons = list(range(2018, 2026))
    test = list(range(2020, 2026))
    panel = pm.build_panel(seasons).drop_nulls(subset=projection.FEATURES)
    proj = projection.fit_walk_forward(panel, test)
    adp_hist = dv.load_adp(seasons).filter(pl.col("adp_position") == "QB")

    curve = dv.pick_value_curve(seasons).filter(pl.col("n") >= 20)
    crows = curve.to_dicts()
    lo, hi = crows[0]["pick_from"], crows[-1]["pick_to"]

    def cost(a: float) -> float:
        p = int(min(max(round(a), lo), hi))
        for r in crows:
            if r["pick_from"] <= p <= r["pick_to"]:
                return float(r["mean_usable_par"])
        return float(crows[-1]["mean_usable_par"])

    qb12 = {r["season"]: r["replacement_points"]
            for r in dv.replacement_levels(dv.season_totals(seasons))
            .filter(pl.col("position") == "QB").to_dicts()}

    plans = {
        f"late pair (QB at {first_qb_pick}, then ~127)": ("late", first_qb_pick, 127),
        f"late single QB (at {first_qb_pick}, no QB2)": ("late1", first_qb_pick, None),
        "premium QB1 only": ("prem", None, None),
        "premium QB1 + late QB2 (~127)": ("prem2", None, 127),
    }
    res = []
    for s in test:
        cur = proj.filter(pl.col("season") == s)
        repl = engine.weekly_replacement(cur, s)
        weeks = sorted(cur["week"].unique().to_list())
        slate = engine._slate(cur, s, "proj", "fantasy_points")
        # streamable level, in projection units, so the pair manager has the
        # same waiver access the single-QB manager already gets on byes
        floor = {r["week"]: r["f"] for r in
                 cur.group_by("week").agg(
                     pl.col("proj").median().alias("f")).to_dicts()}
        names = {r["player_id"]: r["player_display_name"]
                 for r in cur.select(["player_id", "player_display_name"]).unique().to_dicts()}
        keymap = {}
        for pid, nm in names.items():
            keymap.setdefault(dv.normalise_name(nm), pid)
        board = sorted(adp_hist.filter(pl.col("season") == s).to_dicts(),
                       key=lambda r: r["adp"])

        def best_at(pick: int | None, exclude=()):
            """Expected best QB still available at `pick` (or overall if None)."""
            for r in board:
                pid = keymap.get(r["key"])
                if not pid or pid in exclude:
                    continue
                if pick is None or r["adp"] >= pick:
                    return pid, r["adp"]
            return None, None

        for label, (kind, p1, p2) in plans.items():
            a, adp_a = best_at(p1)
            if a is None:
                continue
            spend = cost(adp_a)
            if p2 is None:
                pts = sum(slate[a][w][1] if w in slate.get(a, {}) else repl.get(w, 12.0)
                          for w in weeks)
            else:
                b, adp_b = best_at(p2, exclude=(a,))
                if b is None:
                    continue
                spend += cost(adp_b)
                pts = engine.score_pair(slate, names, a, b, s, weeks,
                                        replacement=repl,
                                        stream_floor=floor).total_points
            res.append({"season": s, "plan": label, "pts": round(pts, 1),
                        "cost": round(spend, 1),
                        "net": round(pts - qb12.get(s, 272.0) - spend, 1)})

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
    print(f"seasons compared: {sorted(common)}  (n = {n_seasons})")
    print(f"\n{'plan':<40}{'QB pts':>9}{'cost':>8}{'NET':>8}{'+/- se':>9}{'worst':>8}")
    print("-" * 92)
    for r in agg.to_dicts():
        print(f"{r['plan']:<40}{r['pts']:>9.1f}{r['cost']:>8.1f}"
              f"{r['net']:>8.1f}{r['se']:>9.1f}{r['worst']:>8.1f}")
    rows_ = agg.to_dicts()
    gap = rows_[0]["net"] - rows_[-1]["net"]
    pooled = (sum(r["se"] ** 2 for r in (rows_[0], rows_[-1]))) ** 0.5
    print(f"\nBest minus worst plan: {gap:.1f} points, with a combined standard")
    print(f"error of {pooled:.1f}. Six seasons is a small sample; treat differences")
    print(f"smaller than roughly {2 * pooled:.0f} points as not established.")
    print("\nNET = QB points minus QB12 replacement minus the measured value of")
    print("the picks spent. Waiting is judged on what the saved picks are worth.")

    # ---- when should QB2 be taken, if at all? ----
    print(f"\n{BAR}\n A3. IF YOU WAIT ON QB1, WHEN IS QB2 WORTH IT?\n{BAR}")
    print(f"QB1 fixed at pick {first_qb_pick}; QB2 taken at varying picks.")
    print("A closer-priced QB2 is a real alternative each week; a very late one")
    print("is rarely the right start, so he adds points only through byes.\n")

    sweep = []
    for p2 in [None, 90, 103, 114, 127, 138, 151, 162]:
        tot = []
        for s in test:
            cur = proj.filter(pl.col("season") == s)
            repl = engine.weekly_replacement(cur, s)
            weeks = sorted(cur["week"].unique().to_list())
            slate = engine._slate(cur, s, "proj", "fantasy_points")
            floor = {r["week"]: r["f"] for r in cur.group_by("week").agg(
                pl.col("proj").median().alias("f")).to_dicts()}
            names = {r["player_id"]: r["player_display_name"] for r in
                     cur.select(["player_id", "player_display_name"]).unique().to_dicts()}
            keymap = {}
            for pid, nm in names.items():
                keymap.setdefault(dv.normalise_name(nm), pid)
            board = sorted(adp_hist.filter(pl.col("season") == s).to_dicts(),
                           key=lambda r: r["adp"])

            def pick_qb(minpick, exclude=()):
                for r in board:
                    pid = keymap.get(r["key"])
                    if pid and pid not in exclude and r["adp"] >= minpick:
                        return pid, r["adp"]
                return None, None

            a, adp_a = pick_qb(first_qb_pick)
            if a is None:
                continue
            spend = cost(adp_a)
            if p2 is None:
                pts = sum(slate[a][w][1] if w in slate.get(a, {}) else repl.get(w, 12.0)
                          for w in weeks)
            else:
                b, adp_b = pick_qb(p2, exclude=(a,))
                if b is None:
                    continue
                spend += cost(adp_b)
                pts = engine.score_pair(slate, names, a, b, s, weeks,
                                        replacement=repl, stream_floor=floor).total_points
            tot.append(pts - qb12.get(s, 272.0) - spend)
        if tot:
            m = sum(tot) / len(tot)
            sd = (sum((x - m) ** 2 for x in tot) / max(len(tot) - 1, 1)) ** 0.5
            sweep.append({"qb2_pick": "none" if p2 is None else str(p2),
                          "net": m, "se": sd / len(tot) ** 0.5,
                          "worst": min(tot), "n": len(tot)})
    print(f"{'QB2 taken at':<16}{'NET':>9}{'+/- se':>9}{'worst season':>15}")
    print("-" * 92)
    for r in sweep:
        print(f"{r['qb2_pick']:<16}{r['net']:>9.1f}{r['se']:>9.1f}{r['worst']:>15.1f}")
    spread = max(r["net"] for r in sweep) - min(r["net"] for r in sweep)
    med_se = sorted(r["se"] for r in sweep)[len(sweep) // 2]
    print(f"\nCAUTION: this sweep is not monotonic, and the spread across picks")
    print(f"({spread:.0f} points) is comparable to the typical standard error")
    print(f"({med_se:.0f}). Each cell is one specific quarterback per season over")
    print(f"six seasons, so the ups and downs reflect which players happened to")
    print(f"sit at that ADP, not a real optimum. Do not read a best pick off this.")


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
          f"{', '.join(str(p) for p in picks[:FIRST_QB_ROUND - 1])}) go to skill positions.")

    # ---------- data ----------
    sched = loaders.load_schedules().filter(
        (pl.col("season") == SEASON) & (pl.col("game_type") == "REG"))
    teams_all = sorted(set(sched["home_team"].to_list()) | set(sched["away_team"].to_list()))
    weeks = sorted(sched["week"].unique().to_list())
    bye = {}
    for t in teams_all:
        played = set(sched.filter((pl.col("home_team") == t) | (pl.col("away_team") == t))["week"].to_list())
        miss = [w for w in weeks if w not in played]
        bye[t] = miss[0] if miss else None

    adp = loaders.load_ffc_adp(SEASON, teams=TEAMS).filter(
        pl.col("position") == "QB").with_columns(
        pl.col("name").map_elements(dv.normalise_name, return_dtype=pl.String).alias("key"))
    cbs = loaders.load_cbs_projections(SEASON, "QB").with_columns(
        pl.col("player").map_elements(dv.normalise_name, return_dtype=pl.String).alias("key"))
    q = (adp.join(cbs.select(["key", "proj_points_league", "proj_ppg_league"]),
                  on="key", how="inner")
         .with_columns(pl.col("team").replace(
             {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "SD": "LAC", "OAK": "LV"}).alias("team")))
    q = q.with_columns(pl.col("team").replace_strict(bye, default=None).alias("bye_week"))

    curve = dv.pick_value_curve(list(range(2018, 2026))).filter(pl.col("n") >= 20)
    crows = curve.to_dicts()
    lo, hi = crows[0]["pick_from"], crows[-1]["pick_to"]

    def pick_cost(a: float) -> float:
        p = int(min(max(round(a), lo), hi))
        for r in crows:
            if r["pick_from"] <= p <= r["pick_to"]:
                return float(r["mean_usable_par"])
        return float(crows[-1]["mean_usable_par"])

    # ---------- A. availability at your picks ----------
    print(f"\n{BAR}\n A. QB AVAILABILITY AT YOUR PICKS  (% still on the board)\n{BAR}")
    show = q.filter(pl.col("adp") <= 200).sort("adp")
    cols = qb_picks[:7]
    print(f"{'QB':<20}{'tm':>4}{'ADP':>7}{'sd':>6}{'bye':>5}{'CBSpts':>8}" +
          "".join(f"{('R' + str(i + FIRST_QB_ROUND)):>7}" for i in range(len(cols))))
    print(f"{'':<42}" + "".join(f"{('p' + str(p)):>7}" for p in cols))
    print("-" * 92)
    for r in show.head(22).to_dicts():
        probs = "".join(f"{p_available(r['adp'], r['stdev'], p) * 100:>6.0f}%" for p in cols)
        print(f"{r['name']:<20}{r['team']:>4}{r['adp']:>7.1f}{r['stdev']:>6.1f}"
              f"{str(r['bye_week']):>5}{r['proj_points_league']:>8.1f}{probs}")

    # ---------- A2. does waiting actually work? ----------
    backtest_late_qb(qb_picks[0])

    # ---------- B. feasible pairs ----------
    p1, p2_opts = qb_picks[0], qb_picks[1:6]
    cands = q.filter(pl.col("adp") <= 200).to_dicts()
    rows = []
    for a, b in combinations(cands, 2):
        qb1, qb2 = (a, b) if a["adp"] <= b["adp"] else (b, a)
        # QB1 must be plausibly there at your first QB pick
        av1 = p_available(qb1["adp"], qb1["stdev"], p1)
        if av1 < 0.35:
            continue
        # Take QB2 as LATE as you can still reasonably get him. Grabbing him
        # early wastes a pick on a player who would have lasted, which is the
        # whole point of waiting at the position.
        best = None
        for p in p2_opts:
            av2 = p_available(qb2["adp"], qb2["stdev"], p)
            if av2 >= 0.65:
                best = (p, av2)
        if best is None:
            continue
        p2, av2 = best
        same_bye = qb1["bye_week"] is not None and qb1["bye_week"] == qb2["bye_week"]
        base = qb1["proj_points_league"]
        bye_gain = 0.0 if same_bye else max(0.0, qb2["proj_ppg_league"] - REPLACEMENT_PPG)
        ins = insurance_premium(qb1["adp"]) * (qb2["proj_ppg_league"] / 18.0)
        cost = pick_cost(p1) + pick_cost(p2)
        rows.append({
            "qb1": qb1["name"], "qb2": qb2["name"],
            "adp1": qb1["adp"], "adp2": qb2["adp"],
            "take1_at": p1, "take2_at": p2,
            "avail1": round(av1 * 100), "avail2": round(av2 * 100),
            "bye1": qb1["bye_week"], "bye2": qb2["bye_week"],
            "bye_clash": same_bye,
            "cbs_qb1": round(base, 1), "insurance": round(ins, 1),
            "pick_cost": round(cost, 1),
            "net": round(base + bye_gain + WEEKLY_EDGE + ins - QB_REPLACEMENT - cost, 1),
        })

    d = pl.DataFrame(rows)
    ok = d.filter(~pl.col("bye_clash")).sort("net", descending=True)

    print(f"\n{BAR}\n B. BEST GETTABLE PAIRS, NO QB BEFORE ROUND {FIRST_QB_ROUND}\n{BAR}")
    print(f"{'QB1':<19}{'QB2':<19}{'take at':>12}{'avail':>10}{'byes':>8}{'cost':>7}{'NET':>8}")
    print("-" * 92)
    for r in ok.head(15).to_dicts():
        print(f"{r['qb1']:<19}{r['qb2']:<19}"
              f"{str(r['take1_at']) + '/' + str(r['take2_at']):>12}"
              f"{str(r['avail1']) + '%/' + str(r['avail2']) + '%':>10}"
              f"{str(r['bye1']) + '/' + str(r['bye2']):>8}"
              f"{r['pick_cost']:>7.1f}{r['net']:>8.1f}")

    # ---------- C. robustness across the Phase 8 sensitivity range ----------
    print(f"\n{BAR}\n C. WHICH PAIRS SURVIVE THE SENSITIVITY PROBLEM?\n{BAR}")
    print("Phase 8 showed the ranking reorders when the insurance premium and")
    print("pick-cost weight are varied. A pair only deserves confidence if it")
    print("stays near the top across all of those settings.\n")
    settings = [("ins x0.5", 0.5, 1.0), ("ins x1.0", 1.0, 1.0),
                ("ins x1.5", 1.5, 1.0), ("cost x0.5", 1.0, 0.5),
                ("cost x1.5", 1.0, 1.5)]
    TOPN = 10
    tally: dict[tuple[str, str], int] = {}
    for _lab, im, cm in settings:
        e = ok.with_columns(
            (pl.col("net") - pl.col("insurance") + pl.col("insurance") * im
             + pl.col("pick_cost") - pl.col("pick_cost") * cm).alias("nv")
        ).sort("nv", descending=True).head(TOPN)
        for r in e.to_dicts():
            tally[(r["qb1"], r["qb2"])] = tally.get((r["qb1"], r["qb2"]), 0) + 1

    robust = sorted(tally.items(), key=lambda t: -t[1])
    print(f"{'QB1':<19}{'QB2':<19}{'top-10 in':>12}  verdict")
    print("-" * 92)
    for (a, b), n in robust[:12]:
        verdict = ("robust" if n == len(settings)
                   else "assumption-dependent" if n <= 2 else "mostly stable")
        print(f"{a:<19}{b:<19}{str(n) + '/' + str(len(settings)):>12}  {verdict}")

    # ---------- D. the practical plan ----------
    print(f"\n{BAR}\n D. PRACTICAL DRAFT PLAN FROM {SLOT}/{TEAMS}\n{BAR}")
    for i, p in enumerate(qb_picks[:6]):
        rnd = FIRST_QB_ROUND + i
        avail = [(r["name"], p_available(r["adp"], r["stdev"], p),
                  r["proj_points_league"], r["bye_week"])
                 for r in q.to_dicts()]
        # "live" = plausibly there but not a lock; a 99% QB can wait a round
        live = [x for x in avail if 0.40 <= x[1] <= 0.95]
        live.sort(key=lambda t: -t[2])
        if not live:
            continue
        names = ", ".join(f"{n} ({pr * 100:.0f}%, bye {b})" for n, pr, _s, b in live[:4])
        print(f"  R{rnd} (pick {p:>3}): {names}")

    print(f"\nShared-bye pairs to avoid: {d.filter(pl.col('bye_clash')).height} "
          f"of {d.height} feasible pairs.")
    for r in d.filter(pl.col("bye_clash")).head(5).to_dicts():
        print(f"    {r['qb1']} + {r['qb2']}  (both bye {r['bye1']})")

    out = Path("outputs"); out.mkdir(exist_ok=True)
    ok.write_csv(out / f"late_qb_slot{SLOT}_pairs_2026.csv")
    print(f"\nwrote outputs/late_qb_slot{SLOT}_pairs_2026.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
