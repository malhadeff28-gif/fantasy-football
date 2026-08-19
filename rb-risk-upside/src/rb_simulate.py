"""Comparable-based workload scenarios (RB-003) and season Monte Carlo (RB-004).

The design follows directly from RB-002. Volume is what persists; per-touch
efficiency and touchdown rate barely do. So the simulator predicts ROLE from
comparable historical situations and then draws PRODUCTION from the pool of
historical running-back games at that role level, which regresses efficiency
and touchdown rate to the pool exactly as the stability evidence says it should.

Two things are strictly forbidden here and are enforced by construction:
  - a comparable is never selected using its own t+1 fantasy outcome;
  - the comparable pool for a target season contains only transitions whose
    outcome season is earlier than the target's.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import rb_data as R  # noqa: E402
import rb_features as F  # noqa: E402
import nflreadpy as nflr  # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")

# Matching features. Fixed in advance, not searched: prior role, prior
# durability, experience, market price, team change and the competition in the
# room. Weights say what a drafter actually cares about, they are not tuned.
MATCH = {
    "prior_touches_pg": 2.0,
    "prior_carries_pg": 1.5,
    "prior_snap_share": 1.0,
    "prior_target_share": 0.5,
    "prior_games": 0.5,
    "years_exp": 0.5,
    "adp_rb_rank": 1.5,
    "comp_draft_capital": 1.0,
    "comp_prior_touches_pg": 1.0,
}
SWITCH_PENALTY = 0.6   # extra distance for a mismatched team-change status

# Role bands, in touches per game. Chosen from what the roles are called in a
# draft room, not from the outcome distribution.
BANDS = [("bellcow", 18.0, 99.0), ("lead", 13.0, 18.0),
         ("committee", 9.0, 13.0), ("backup", 0.0, 9.0)]

FULL_SEASON = 17


@dataclass
class Target:
    name: str
    player_id: str
    team: str
    season: int
    prior: dict          # prior-season situation vector
    adp: float
    adp_rb_rank: int


# --- situation features ----------------------------------------------------

def _adp_history() -> pl.DataFrame:
    """FFC ADP for every season, with a positional rank. Pre-season by design."""
    hist = pl.read_parquet(
        R.REPO / "shared/nfl-data/cache/raw/ffc_adp_all_2018_2025.parquet"
    ).select("season", "key", pl.col("adp_position").alias("pos"), "adp")
    import markets

    cur = markets.load_ffc_adp(R.TARGET_SEASON).select(
        "season", "key", pl.col("position").alias("pos"), "adp")
    both = pl.concat([hist, cur]).filter(pl.col("pos") == "RB")
    return both.with_columns(
        pl.col("adp").rank("ordinal").over("season").alias("adp_rb_rank"))


def _rb_rooms(seasons: list[int]) -> pl.DataFrame:
    """Every RB on every roster, with draft capital. The competition input."""
    r = nflr.load_rosters(seasons=seasons)
    return r.filter(pl.col("position").is_in(["RB", "FB"])).select(
        "season", "team", "gsis_id", "full_name",
        pl.col("draft_number").cast(pl.Float64).fill_null(300.0),
        pl.col("entry_year").cast(pl.Int32), pl.col("years_exp").cast(pl.Int32))


def situation_panel(ss: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """One row per (player, season Y): everything knowable BEFORE Y kicks off.

    Prior-season role comes from Y-1, competition from the Y roster, price from
    the Y pre-season ADP. Nothing from Y itself.
    """
    rooms = _rb_rooms(seasons)
    adp = _adp_history()

    prior = ss.select(
        "player_id", "player_display_name",
        (pl.col("season") + 1).alias("season"),
        pl.col("touches_pg").alias("prior_touches_pg"),
        pl.col("carries_pg").alias("prior_carries_pg"),
        pl.col("targets_pg").alias("prior_targets_pg"),
        pl.col("snap_share").alias("prior_snap_share"),
        pl.col("target_share_rb").alias("prior_target_share"),
        pl.col("carry_share").alias("prior_carry_share"),
        pl.col("gl_carries_pg").alias("prior_gl_carries_pg"),
        pl.col("g").alias("prior_games"),
        pl.col("fp_pg").alias("prior_fp_pg"),
        pl.col("fp_total").alias("prior_fp_total"),
        pl.col("team").alias("prior_team"),
        pl.col("ypc").alias("prior_ypc"),
        pl.col("fp_per_touch").alias("prior_fp_per_touch"),
    )

    me = rooms.join(prior, left_on=["gsis_id", "season"],
                    right_on=["player_id", "season"], how="inner")

    # competition = the best other back in the room, by draft capital and by
    # what he actually did last year
    others = rooms.join(
        prior.select("player_id", "season", "prior_touches_pg"),
        left_on=["gsis_id", "season"], right_on=["player_id", "season"],
        how="left").with_columns(pl.col("prior_touches_pg").fill_null(0.0))
    comp = (others.join(me.select("season", "team", "gsis_id")
                        .rename({"gsis_id": "me_id"}),
                        on=["season", "team"], how="inner")
            .filter(pl.col("gsis_id") != pl.col("me_id"))
            .group_by(["season", "team", "me_id"]).agg(
                pl.col("draft_number").min().alias("comp_draft_capital"),
                pl.col("prior_touches_pg").max().alias("comp_prior_touches_pg"),
                pl.len().alias("comp_n_backs"),
                # a rookie or second-year high pick is the specific threat
                pl.when(pl.col("entry_year") >= pl.col("season") - 1)
                .then(pl.col("draft_number")).otherwise(300.0).min()
                .alias("comp_young_draft")))

    out = (me.join(comp, left_on=["season", "team", "gsis_id"],
                   right_on=["season", "team", "me_id"], how="left")
           .with_columns(pl.col("comp_draft_capital").fill_null(300.0),
                         pl.col("comp_young_draft").fill_null(300.0),
                         pl.col("comp_prior_touches_pg").fill_null(0.0),
                         pl.col("comp_n_backs").fill_null(0))
           .join(adp.select("season", "key", "adp", "adp_rb_rank"),
                 left_on=[pl.col("full_name").map_elements(R.norm,
                                                           return_dtype=pl.String),
                          "season"],
                 right_on=["key", "season"], how="left"))

    return out.with_columns(
        (pl.col("team") != pl.col("prior_team")).cast(pl.Int8).alias("switched"),
        # undrafted in a 12-team league: treat as pick 200 so distance is finite
        pl.col("adp").fill_null(200.0),
        pl.col("adp_rb_rank").fill_null(60).cast(pl.Float64),
    )


def outcome_panel(ss: pl.DataFrame) -> pl.DataFrame:
    """Season Y realised role and points. Joined to situations only AFTER
    comparables are chosen."""
    return ss.select(
        "player_id", "season",
        pl.col("g").alias("out_games"),
        pl.col("touches_pg").alias("out_touches_pg"),
        pl.col("carries_pg").alias("out_carries_pg"),
        pl.col("targets_pg").alias("out_targets_pg"),
        pl.col("snap_share").alias("out_snap_share"),
        pl.col("fp_total").alias("out_fp_total"),
        pl.col("fp_pg").alias("out_fp_pg"),
        pl.col("pos_rank").alias("out_pos_rank"),
    )


# --- comparables -----------------------------------------------------------

def comparables(target: Target, sit: pl.DataFrame, out: pl.DataFrame,
                k: int = 30) -> pl.DataFrame:
    """k nearest historical situations, chosen on pre-season features only.

    The pool is restricted to outcome seasons strictly earlier than the
    target's, so this is usable walk-forward and never peeks.
    """
    pool = sit.filter((pl.col("season") < target.season)
                      & pl.col("prior_touches_pg").is_not_null())
    if pool.height == 0:
        raise ValueError("empty comparable pool")

    cols = list(MATCH)
    P = pool.select(cols).to_numpy().astype(float)
    mu, sd = np.nanmean(P, axis=0), np.nanstd(P, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    w = np.array([MATCH[c] for c in cols])

    t = np.array([target.prior.get(c, np.nan) for c in cols], dtype=float)
    zt = (t - mu) / sd
    zP = (P - mu) / sd
    d = np.sqrt(np.nansum(w * (zP - zt) ** 2, axis=1) / w.sum())
    d = d + SWITCH_PENALTY * (
        pool["switched"].to_numpy() != target.prior["switched"]).astype(float)

    ranked = pool.with_columns(pl.Series("dist", d)).sort("dist").head(k)
    # outcomes attach only now, after selection is complete
    return (ranked.join(out, left_on=["gsis_id", "season"],
                        right_on=["player_id", "season"], how="inner")
            .sort("dist"))


def role_scenarios(comps: pl.DataFrame) -> pl.DataFrame:
    """Empirical probability of each role band, plus the missed-time rate."""
    n = comps.height
    rows = []
    for name, lo, hi in BANDS:
        s = comps.filter((pl.col("out_touches_pg") >= lo)
                         & (pl.col("out_touches_pg") < hi))
        p = s.height / n
        # Wilson interval: n is 30, a naive proportion CI would lie
        z, ph = 1.96, p
        den = 1 + z * z / n
        cen = (ph + z * z / (2 * n)) / den
        halfw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        rows.append({"band": name, "touches_lo": lo, "n": s.height,
                     "p": round(p, 3), "p_lo": round(max(0, cen - halfw), 3),
                     "p_hi": round(min(1, cen + halfw), 3),
                     "med_fp": round(float(s["out_fp_total"].median()), 1)
                     if s.height else None})
    return pl.DataFrame(rows)


# --- Monte Carlo -----------------------------------------------------------

def game_pool(panel: pl.DataFrame, ss: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Historical RB-games tagged with their player-season touches-per-game.

    Sampling production from games at a matched role level is what makes
    efficiency and touchdown rate regress to the league, which is exactly what
    RB-002 says they do.
    """
    tagged = panel.join(
        ss.select("player_id", "season", pl.col("touches_pg").alias("season_role")),
        on=["player_id", "season"], how="inner")
    return (tagged["season_role"].to_numpy().astype(float),
            tagged["fp"].to_numpy().astype(float))


def rank_thresholds(ss: pl.DataFrame) -> dict[int, np.ndarray]:
    """Season point totals of the RBn finisher, per season, on a 17-game basis.

    2018-2020 were 16-game seasons; their thresholds are scaled by 17/16 so a
    2026 projection is compared against a like-for-like bar.
    """
    out: dict[int, list[float]] = {3: [], 5: [], 12: [], 24: [], 36: []}
    for s in sorted(ss["season"].unique().to_list()):
        yr = ss.filter(pl.col("season") == s).sort("fp_total", descending=True)
        scale = 17 / 16 if s <= 2020 else 1.0
        for n in out:
            if yr.height >= n:
                out[n].append(float(yr["fp_total"][n - 1]) * scale)
    return {n: np.array(v) for n, v in out.items()}


def adp_baseline(sit: pl.DataFrame, out: pl.DataFrame) -> tuple:
    """What a back at ADP rank r has historically returned, 2018-2025.

    The bar for 'was this pick worth it' is not the same for RB20 and RB26.
    Fitted on log rank because the drop from RB2 to RB8 is not the drop from
    RB22 to RB28. Returns (predict_fn, r2, n).
    """
    d = (sit.join(out, left_on=["gsis_id", "season"],
                  right_on=["player_id", "season"], how="inner")
         .filter(pl.col("adp_rb_rank") <= 60))
    x = np.log(d["adp_rb_rank"].to_numpy().astype(float))
    y = d["out_fp_total"].to_numpy().astype(float)
    b = np.polyfit(x, y, 1)
    r2 = 1 - np.var(y - np.polyval(b, x)) / np.var(y)
    return (lambda r: float(np.polyval(b, np.log(r))), float(r2), len(y))


def simulate(target: Target, comps: pl.DataFrame, pool: tuple, thresholds: dict,
             n_sims: int = 50_000, seed: int = 20260818,
             eff_tilt: float = 0.0, role_mode: str = "ratio") -> dict:
    """Season outcome distribution for one back.

    Each draw: pick a comparable (inverse-distance weighted), take its realised
    role and games played, then draw that many games from the pool of
    historical RB-games at the implied role level.

    `role_mode` is a genuine modelling fork, not a knob:
      "ratio"    - the comparable's role CHANGE is transferred and applied to
                   this player's own prior role. Respects where he actually
                   was. Shrinks a target whose prior sits at the low end of a
                   matched set.
      "absolute" - the comparable's realised role level is used directly.
                   Says the SITUATION determines the workload, not the man.
      "blend"    - the geometric mean of the two, for readers who believe both.
    All three are reported; if the answer only holds under one of them, that is
    a finding about the metric rather than about the players.
    `eff_tilt` optionally scales every game by a player-specific efficiency
    factor; the baseline is 0.0 because RB-002 found per-touch efficiency
    barely persists.
    """
    rng = np.random.default_rng(seed)
    roles, fps = pool

    w = 1.0 / (0.25 + comps["dist"].to_numpy().astype(float))
    w = w / w.sum()
    c_abs = comps["out_touches_pg"].to_numpy().astype(float)
    c_ratio = c_abs / np.maximum(comps["prior_touches_pg"].to_numpy(), 1e-6)
    c_games = comps["out_games"].to_numpy().astype(float)

    j = rng.choice(len(w), size=n_sims, p=w)
    # comparables played 16- and 17-game seasons; express games as a rate
    avail = np.clip(c_games[j] / np.where(comps["season"].to_numpy()[j] <= 2020,
                                          16.0, 17.0), 0.05, 1.0)
    games = np.clip(np.round(avail * FULL_SEASON), 1, FULL_SEASON).astype(int)
    r_ratio = target.prior["prior_touches_pg"] * c_ratio[j]
    if role_mode == "ratio":
        role = r_ratio
    elif role_mode == "absolute":
        role = c_abs[j]
    elif role_mode == "blend":
        role = np.sqrt(np.maximum(r_ratio, 0.1) * np.maximum(c_abs[j], 0.1))
    else:
        raise ValueError(f"unknown role_mode {role_mode}")
    role = np.clip(role, 1.0, 30.0)

    # pre-bucket the game pool by role so each draw is an O(1) index lookup
    edges = np.arange(0, 31, 1.0)
    bucket = np.clip(np.digitize(roles, edges) - 1, 0, len(edges) - 2)
    order = np.argsort(bucket, kind="stable")
    b_sorted, fp_sorted = bucket[order], fps[order]
    starts = np.searchsorted(b_sorted, np.arange(len(edges) - 1), side="left")
    ends = np.searchsorted(b_sorted, np.arange(len(edges) - 1), side="right")

    rb = np.clip(np.digitize(role, edges) - 1, 0, len(edges) - 2)
    # widen empty or thin buckets outward until there is something to sample
    lo_i, hi_i = starts[rb].copy(), ends[rb].copy()
    thin = (hi_i - lo_i) < 40
    for step in (1, 2, 3, 5):
        if not thin.any():
            break
        b2l = np.clip(rb[thin] - step, 0, len(edges) - 2)
        b2h = np.clip(rb[thin] + step, 0, len(edges) - 2)
        lo_i[thin] = starts[b2l]
        hi_i[thin] = ends[b2h]
        thin = (hi_i - lo_i) < 40

    gmax = games.max()
    u = rng.random((n_sims, gmax))
    idx = (lo_i[:, None] + (u * (hi_i - lo_i)[:, None]).astype(int))
    draws = fp_sorted[idx]
    mask = np.arange(gmax)[None, :] < games[:, None]
    totals = (draws * mask).sum(axis=1)
    if eff_tilt:
        totals = totals * (1.0 + eff_tilt)

    # threshold uncertainty is real: the RB12 bar moves 20+ points year to year
    th = {n: rng.choice(v, size=n_sims) for n, v in thresholds.items()}

    q = np.percentile(totals, [10, 25, 50, 75, 90])
    # consensus projects a healthy 17 games; this is the like-for-like number
    full = float((totals / games * FULL_SEASON).mean())
    return {
        "name": target.name, "n_sims": n_sims, "role_mode": role_mode,
        "mean": float(totals.mean()), "sd": float(totals.std()),
        "mean_if_17_games": full,
        "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]),
        "p_top3": float((totals >= th[3]).mean()),
        "p_top5": float((totals >= th[5]).mean()),
        "p_top12": float((totals >= th[12]).mean()),
        "p_top24": float((totals >= th[24]).mean()),
        "p_bust24": float((totals < th[24]).mean()),
        "p_bust36": float((totals < th[36]).mean()),
        "mean_games": float(games.mean()),
        "mean_role": float(role.mean()),
        "totals": totals,
    }


def convergence(target: Target, comps: pl.DataFrame, pool: tuple, thresholds: dict,
                sizes=(2_000, 5_000, 10_000, 25_000, 50_000, 100_000),
                reps: int = 5) -> pl.DataFrame:
    """How many draws are actually needed. Repeat each size on new seeds and
    report the spread of the estimate, not a single run."""
    rows = []
    for n in sizes:
        med, top12, p90 = [], [], []
        for r in range(reps):
            s = simulate(target, comps, pool, thresholds, n_sims=n,
                         seed=20260818 + 1000 * r)
            med.append(s["median"]); top12.append(s["p_top12"]); p90.append(s["p90"])
        rows.append({"n_sims": n,
                     "median_sd": round(float(np.std(med)), 3),
                     "p_top12_sd": round(float(np.std(top12)), 5),
                     "p90_sd": round(float(np.std(p90)), 3)})
    return pl.DataFrame(rows)


def _demo() -> None:
    """Check the two things that would silently invalidate everything."""
    panel = F.load_or_build()
    ss = F.season_rollup(panel, min_games=1)
    sit = situation_panel(ss, list(range(2018, 2027)))
    out = outcome_panel(ss)

    t = Target(name="test", player_id="x", team="CHI", season=2024,
               prior={"prior_touches_pg": 18.0, "prior_carries_pg": 15.0,
                      "prior_snap_share": 0.6, "prior_target_share": 0.08,
                      "prior_games": 16.0, "years_exp": 4.0,
                      "adp_rb_rank": 20.0, "comp_draft_capital": 150.0,
                      "comp_prior_touches_pg": 5.0, "switched": 0},
               adp=42.0, adp_rb_rank=20)
    c = comparables(t, sit, out, k=30)
    assert c.height > 0, "no comparables found"
    assert c["season"].max() < t.season, "LOOK-AHEAD: a comparable is from the future"

    th = rank_thresholds(ss)
    p = game_pool(panel, ss)
    s = simulate(t, c, p, th, n_sims=5_000)
    assert 0 < s["median"] < 450, f"implausible median {s['median']}"
    assert s["p10"] <= s["median"] <= s["p90"], "percentiles out of order"
    assert 0 <= s["p_top12"] <= 1
    print(f"comps={c.height} median={s['median']:.1f} p_top12={s['p_top12']:.3f} "
          "-- rb_simulate self-check OK")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _demo()
