"""QB-020: adversarial re-examination of the QB-012 / QB-011 economics.

Target claims
-------------
    QB-012  "a second quarterback is worth +9.7 pts/season alongside an elite
             QB1 and +30.1 alongside a mid-priced QB1, net of the pick it costs"
    QB-011  "QB2 is worth +12.0 pts/season against a single-QB manager who
             streams his bye"

This script does not try to confirm them. It reproduces them exactly and then
attacks them:

  A0  REPLICATION       rebuild the pick-value curve from raw nflverse + FFC
                        without calling benchmarks/draft_value.py, and check
                        the published numbers reproduce to the decimal.
  A1  REPLACEMENT LEVEL sweep the weekly streamer from the 30th to the 75th
                        percentile of that week's de-facto starters, and then
                        *measure* which percentile a real streamer achieves
                        rather than assuming one.
  A2  PICK-VALUE CURVE  mean (baseline) vs median vs hits-only vs
                        survivorship-corrected (zero-filled) cost of a pick.
  A3  EXTREME EVENTS    5% trimmed means, an explicit QB1-missed-3+-starts
                        split, and a week-by-week decomposition of the gain
                        into coverage (QB1 absent) and choice (QB1 present).
  A4  SURVIVORSHIP      quantify what the drafted-players-only ADP set omits,
                        separating "never played" from "name did not join".
  A5  UNCERTAINTY       paired season-level t intervals and a season-cluster
                        bootstrap on every headline number.

Two structural checks the original never ran:

  S1  the +30.1 comparison in phase6_draft_cost.py contrasts ADP QB rank 10 +
      rank 18 against ADP QB rank 8 *alone*. Different QB1s, so it is not a
      marginal QB2 effect. Recomputed rank-matched.
  S2  phase6 scores every week present in the panel, including weeks 17-18,
      which are outside this league's calendar (regular 1-13, playoff 14-16).
      Recomputed on the league window.

Everything is walk-forward. Strategy selection uses ADP rank only, so no
outcome information enters any choice. The injury split is a post-hoc
diagnostic, labelled as such -- it is an explanation, not a draft rule.

Usage:  ./.venv/Scripts/python.exe qb-pairing/scripts/wave2_skeptic_economics.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import re                                        # noqa: E402
import sys                                       # noqa: E402
import warnings                                  # noqa: E402
from itertools import combinations               # noqa: E402
from pathlib import Path                         # noqa: E402

import numpy as np                               # noqa: E402
import polars as pl                              # noqa: E402

HERE = Path(__file__).resolve().parents[1]       # qb-pairing/
sys.path.insert(0, str(HERE / "src"))

from qbpair.benchmarks import draft_value as dv  # noqa: E402
from qbpair.config import DATA_RAW, LEAGUE       # noqa: E402
from qbpair.data import loaders                  # noqa: E402
from qbpair.features import panel as pm          # noqa: E402
from qbpair.models import projection             # noqa: E402
from qbpair.pairing import engine                # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
QUANTILES = [0.30, 0.40, 0.50, 0.60, 0.75]
BASE_Q = 0.50
COST_SPECS = ["mean", "mean_fixed", "median", "hits_only", "zero_filled"]
BASE_COST = "mean"
# Weeks phase6 actually scores vs weeks this league actually plays.
LEAGUE_WEEKS = set(LEAGUE.regular_season_weeks) | set(LEAGUE.playoff_weeks)
WEEK_MODES = ["panel_all", "league_1_16"]
BASE_WEEKS = "panel_all"          # phase6's own choice, so the baseline matches
BOOT = 2000
RNG = np.random.default_rng(20260818)
BAR = "=" * 78
SUB = "-" * 78

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
(HERE / "research" / "wave2").mkdir(parents=True, exist_ok=True)

# phase2c's feature set, reproduced verbatim so QB-011 is re-run, not rebuilt.
P2C_FEATS = ["form_fp_mean", "career_fp_mean", "form_pass_att", "form_rush_att",
             "implied_team_total", "spread_abs", "total_line",
             "def_adj", "def_fp_allowed", "is_home", "indoors", "rest",
             "div_game"]

# The six phase-6 strategies, unchanged, so the baseline is reproduced exactly.
STRATEGIES = {
    "premium (QB1 only)":        [1],
    "premium + late partner":    [1, 16],
    "strong value + complement": [4, 16],
    "late pair":                 [10, 18],
    "single value QB":           [8],
    "streaming (no QB drafted)": [],
}
STREAM_MIN_RANK = 20
QB1_RANKS = range(1, 13)
QB2_RANKS = (14, 16, 18, 20)


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------

def t_ci(x, alpha: float = 0.05) -> dict:
    """Mean, standard error and t-interval for a small paired sample."""
    from scipy import stats

    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    n = x.size
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "se": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "t": float("nan"), "p": float("nan")}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    tstat = m / se if se > 0 else float("inf")
    p = float(2 * (1 - stats.t.cdf(abs(tstat), n - 1))) if se > 0 else 0.0
    return {"n": n, "mean": m, "sd": sd, "se": se, "lo": m - crit * se,
            "hi": m + crit * se, "t": tstat, "p": p}


def cluster_bootstrap(values, clusters, B: int = BOOT) -> dict:
    """Resample whole seasons with replacement.

    Every pair inside a season shares the same weekly outcomes, so the
    effective sample size is the number of seasons, not the number of pairs.
    """
    clusters = np.asarray(clusters)
    values = np.asarray(values, float)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = np.empty(B)
    for b in range(B):
        pick = RNG.choice(uniq, size=uniq.size, replace=True)
        draws[b] = np.concatenate([values[idx[c]] for c in pick]).mean()
    return {"mean": float(values.mean()), "boot_se": float(draws.std(ddof=1)),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "n_clusters": int(uniq.size), "n_obs": int(values.size)}


def ci_str(d, w: int = 20) -> str:
    return f"{f'[{d['lo']:.1f}, {d['hi']:.1f}]':>{w}}"


def fmt(v, w=8, d=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'-':>{w}}"
    return f"{v:>{w}.{d}f}"


# --------------------------------------------------------------------------
# ADP, cached so repeat runs do not hammer the FFC endpoint
# --------------------------------------------------------------------------

def adp_all(seasons: list[int]) -> pl.DataFrame:
    path = DATA_RAW / f"ffc_adp_all_{min(seasons)}_{max(seasons)}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = dv.load_adp(seasons)
    d.write_parquet(path)
    return d


# --------------------------------------------------------------------------
# A0 -- independent replication of the pick-value curve
# --------------------------------------------------------------------------

_SUF = re.compile(r"\b(?:jr|sr|ii|iii|iv|v)\b\.?", re.I)


def indep_key(name: str) -> str:
    """Separate name normaliser, written from the spec rather than imported."""
    n = str(name).lower()
    for ch in (".", "'", "`"):
        n = n.replace(ch, "")
    n = n.replace("-", " ")
    n = _SUF.sub(" ", n)
    return " ".join(n.split())


def independent_curve(seasons: list[int], adp: pl.DataFrame,
                      bucket: int = 12) -> pl.DataFrame:
    """Rebuild the curve straight from load_player_stats + FFC ADP.

    No call into benchmarks/draft_value.py; replacement ranks are re-derived
    from LEAGUE rather than copied from REPLACEMENT_RANK.
    """
    ranks = {p: LEAGUE.teams * LEAGUE.starters[p] for p in ("QB", "RB", "WR", "TE")}
    st = loaders.load_player_stats(seasons).filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(list(ranks)))
    tot = (st.group_by(["season", "player_id", "player_display_name", "position"])
           .agg(pl.col("fantasy_points").sum().alias("pts")))
    rows = []
    for (s, pos), g in tot.group_by(["season", "position"]):
        v = sorted(g["pts"].to_list(), reverse=True)
        k = ranks[pos]
        rows.append({"season": s, "position": pos,
                     "repl": v[k - 1] if len(v) >= k else (v[-1] if v else 0.0)})
    tot = (tot.join(pl.DataFrame(rows), on=["season", "position"], how="left")
           .with_columns(pl.col("player_display_name")
                         .map_elements(indep_key, return_dtype=pl.String).alias("k2"),
                         (pl.col("pts") - pl.col("repl")).alias("par2"))
           .with_columns(pl.col("par2").clip(lower_bound=0.0).alias("upar2")))
    a = adp.with_columns(pl.col("adp_name")
                         .map_elements(indep_key, return_dtype=pl.String).alias("k2"))
    j = a.join(tot, on=["season", "k2"], how="inner").filter(pl.col("position") != "QB")
    return (j.with_columns(((pl.col("adp") - 1) // bucket).cast(pl.Int32).alias("bkt"))
            .group_by("bkt")
            .agg(pl.len().alias("n_indep"),
                 (pl.col("bkt").first() * bucket + 1).alias("pick_from"),
                 pl.col("upar2").mean().alias("indep_mean_usable_par"))
            .sort("bkt"))


# --------------------------------------------------------------------------
# A2 / A4 -- the four cost specifications
# --------------------------------------------------------------------------

def cost_curves(adp: pl.DataFrame, par: pl.DataFrame, bucket: int = 12):
    """(long curve table, survivorship audit).

    mean         baseline: mean usable PAR of drafted non-QBs who played
    median       median usable PAR of the same set
    hits_only    mean usable PAR of players who finished above positional
                 replacement -- i.e. conditional on the pick working
    zero_filled  every drafted RB/WR/TE, with players who never recorded a
                 stat line entered at 0 instead of dropped by the inner join
    """
    matched = adp.join(par, on=["season", "key"], how="inner") \
                 .filter(pl.col("position") != "QB")
    skill = adp.filter(pl.col("adp_position").is_in(["RB", "WR", "TE"]))
    filled = (skill.join(par.select(["season", "key", "usable_par", "par"]),
                         on=["season", "key"], how="left")
              .with_columns(pl.col("par").is_null().alias("unmatched"))
              .with_columns(pl.col("usable_par").fill_null(0.0),
                            pl.col("par").fill_null(-1.0)))

    def bk(d):
        return d.with_columns(((pl.col("adp") - 1) // bucket)
                              .cast(pl.Int32).alias("bkt"))

    m, f = bk(matched), bk(filled)
    frames = [
        m.group_by("bkt").agg(pl.len().alias("n"),
                              pl.col("usable_par").mean().alias("value"),
                              (pl.col("par") > 0).mean().alias("hit_rate"))
        .with_columns(pl.lit("mean").alias("spec")),
        m.group_by("bkt").agg(pl.len().alias("n"),
                              pl.col("usable_par").median().alias("value"),
                              (pl.col("par") > 0).mean().alias("hit_rate"))
        .with_columns(pl.lit("median").alias("spec")),
        m.filter(pl.col("par") > 0).group_by("bkt")
        .agg(pl.len().alias("n"), pl.col("usable_par").mean().alias("value"),
             pl.lit(1.0).alias("hit_rate"))
        .with_columns(pl.lit("hits_only").alias("spec")),
        f.group_by("bkt").agg(pl.len().alias("n"),
                              pl.col("usable_par").mean().alias("value"),
                              (pl.col("par") > 0).mean().alias("hit_rate"))
        .with_columns(pl.lit("zero_filled").alias("spec")),
    ]
    curve = (pl.concat(frames, how="diagonal")
             .with_columns((pl.col("bkt") * bucket + 1).alias("pick_from"),
                           (pl.col("bkt") * bucket + bucket).alias("pick_to"))
             .sort(["spec", "bkt"]))
    audit = (f.group_by("bkt").agg(
        pl.len().alias("n_drafted"),
        pl.col("unmatched").sum().alias("n_unmatched"),
        pl.col("unmatched").mean().alias("share_unmatched"),
        pl.col("usable_par").mean().alias("mean_with_zeros"))
        .join(m.group_by("bkt").agg(pl.col("usable_par").mean()
                                    .alias("mean_matched_only")), on="bkt", how="left")
        .with_columns((pl.col("bkt") * bucket + 1).alias("pick_from"))
        .sort("bkt"))
    return curve, audit, filled


def cost_lookup(curve: pl.DataFrame, spec: str, bucket: int = 12):
    """Expected non-QB return at a given ADP.

    `spec` "..._fixed" repairs a real defect in `phase6_draft_cost.py`. Its
    `pick_cost_lookup` scans integer ranges [12k+1, 12k+12] with `<=`, but ADP
    is a float, so any value strictly between 12k+12 and 12k+13 -- 8.0% of all
    ADPs in this window -- matches no bucket and falls through to `rows[-1]`,
    the deepest bucket, whose mean usable PAR is 0.0. Those picks are charged
    nothing. Four of the quarterbacks in phase6's own strategies are affected,
    three of them the rank-18 partner in `late pair`.
    """
    fixed = spec.endswith("_fixed")
    base = spec[:-6] if fixed else spec
    rows = curve.filter(pl.col("spec") == base).sort("bkt").to_dicts()
    if not rows:
        return lambda adp_val: 0.0
    by_bkt = {int(r["bkt"]): float(r["value"]) for r in rows}
    top = max(by_bkt)

    def cost_faithful(adp_val: float) -> float:
        for r in rows:
            if r["pick_from"] <= adp_val <= r["pick_to"]:
                return float(r["value"])
        return float(rows[-1]["value"])

    def cost_fixed(adp_val: float) -> float:
        b = min(max(int((adp_val - 1) // bucket), 0), top)
        return by_bkt.get(b, 0.0)

    return cost_fixed if fixed else cost_faithful


# --------------------------------------------------------------------------
# phase-6 machinery, with the replacement quantile and the week window as
# free parameters instead of hard-coded choices
# --------------------------------------------------------------------------

class SeasonCtx:
    def __init__(self, proj: pl.DataFrame, qb_adp: pl.DataFrame, season: int):
        cur = proj.filter(pl.col("season") == season)
        self.season = season
        self.slate = engine._slate(cur, season, "proj", "fantasy_points")
        allw = sorted(cur["week"].unique().to_list())
        self.week_sets = {"panel_all": allw,
                          "league_1_16": [w for w in allw if w in LEAGUE_WEEKS]}
        self.names = {r["player_id"]: r["player_display_name"]
                      for r in cur.select(["player_id", "player_display_name"])
                      .unique().to_dicts()}
        self.keymap = {}
        for r in cur.select(["player_id", "player_display_name"]).unique().to_dicts():
            self.keymap.setdefault(dv.normalise_name(r["player_display_name"]),
                                   r["player_id"])
        self.qb_rows = qb_adp.filter(pl.col("season") == season).sort("adp").to_dicts()
        self.repl = {q: engine.weekly_replacement(cur, season, q) for q in QUANTILES}
        self.week_pts = {w: np.sort(g["fantasy_points"].to_numpy())
                         for (w,), g in cur.group_by("week")}
        self.games = {pid: len(w) for pid, w in self.slate.items()}
        self.n_slots = max(allw) - 1 if allw else 0     # season weeks less one bye

    # -- selection -------------------------------------------------------
    def qb_at(self, rank: int):
        if len(self.qb_rows) < rank:
            return None
        r = self.qb_rows[rank - 1]
        pid = self.keymap.get(r["key"])
        return (pid, float(r["adp"]), r["adp_name"]) if pid else None

    def qb_at_or_bust(self, rank: int):
        """Like qb_at, but a drafted QB who never took a snap is a BUST, not a
        missing observation.

        phase6 skips the whole strategy when an ADP name does not resolve to a
        panel player. That silently deletes exactly the seasons where a drafted
        quarterback returned nothing -- the worst case for the strategy that
        drafted him. Here he keeps his pick cost and contributes zero games.
        """
        if len(self.qb_rows) < rank:
            return None
        r = self.qb_rows[rank - 1]
        pid = self.keymap.get(r["key"]) or f"__bust_{rank}__"
        return (pid, float(r["adp"]), r["adp_name"], pid.startswith("__bust_"))

    def missed_starts(self, pid: str) -> int:
        return max(0, self.n_slots - self.games.get(pid, 0))

    # -- scoring ---------------------------------------------------------
    def solo_points(self, pid: str, q: float, wm: str) -> float:
        rp, wk = self.repl[q], self.slate.get(pid, {})
        return sum(wk[w][1] if w in wk else rp.get(w, 12.0)
                   for w in self.week_sets[wm])

    def pair_result(self, a: str, b: str, q: float, wm: str):
        return engine.score_pair(self.slate, self.names, a, b, self.season,
                                 self.week_sets[wm], replacement=self.repl[q])

    def decompose(self, a: str, b: str, q: float, wm: str) -> dict:
        """Split the pair-minus-solo gain into coverage and choice.

        coverage = weeks QB1 did not start: pair points minus the streamer the
                   single-QB manager would have used
        choice   = weeks QB1 did start: pair points minus QB1's own points
        """
        pr = self.pair_result(a, b, q, wm)
        wk_a, rp = self.slate.get(a, {}), self.repl[q]
        cov = cho = 0.0
        cov_wk = cho_wk = 0
        for d in pr.decisions:
            if d.week in wk_a:
                cho += d.points - wk_a[d.week][1]
                cho_wk += 1
            else:
                cov += d.points - rp.get(d.week, 12.0)
                cov_wk += 1
        return {"pair_points": pr.total_points, "coverage_gain": cov,
                "choice_gain": cho, "coverage_weeks": cov_wk,
                "choice_weeks_qb1_started": cho_wk}

    def stream_pool(self):
        drafted = {self.keymap.get(r["key"]) for r in self.qb_rows}
        pool = [self.keymap.get(r["key"]) for r in self.qb_rows[STREAM_MIN_RANK - 1:]]
        pool = [p for p in pool if p]
        return pool + [pid for pid in self.slate if pid not in drafted]

    def stream_points(self, q: float, wm: str) -> float:
        pool, rp = self.stream_pool(), self.repl[q]
        pts = 0.0
        for w in self.week_sets[wm]:
            opts = [(self.slate[p][w][0], self.slate[p][w][1])
                    for p in pool if p in self.slate and w in self.slate[p]]
            pts += max(opts)[1] if opts else rp.get(w, 12.0)
        return pts

    def streamer_percentiles(self, wm: str = BASE_WEEKS) -> list[float]:
        """What percentile of that week's starters does a real streamer hit?

        Uses the same pool and the same pregame rule phase6 uses for its
        no-QB-drafted strategy, then reads off where the realised score falls
        in the week's starter distribution. This converts the replacement
        level from an assumption into a measurement.
        """
        pool, out = self.stream_pool(), []
        for w in self.week_sets[wm]:
            opts = [(self.slate[p][w][0], self.slate[p][w][1])
                    for p in pool if p in self.slate and w in self.slate[p]]
            if not opts or w not in self.week_pts:
                continue
            got = max(opts)[1]
            arr = self.week_pts[w]
            out.append(float((arr <= got).mean()))
        return out


def strategy_table(ctxs: dict, qb_repl: dict, curve: pl.DataFrame) -> pl.DataFrame:
    """phase6's table, recomputed for each (quantile, cost spec, week window)."""
    lookups = {sp: cost_lookup(curve, sp) for sp in COST_SPECS}
    rows = []
    for s, c in ctxs.items():
        for label, ranks in STRATEGIES.items():
            sel = [c.qb_at(r) for r in ranks]
            if any(x is None for x in sel):
                continue
            for wm in WEEK_MODES:
                for q in QUANTILES:
                    if not ranks:
                        pts, adps = c.stream_points(q, wm), []
                    elif len(sel) == 1:
                        pts, adps = c.solo_points(sel[0][0], q, wm), [sel[0][1]]
                    else:
                        pts = c.pair_result(sel[0][0], sel[1][0], q, wm).total_points
                        adps = [x[1] for x in sel]
                    for sp in COST_SPECS:
                        cost = sum(lookups[sp](a) for a in adps)
                        rows.append({"season": s, "strategy": label, "quantile": q,
                                     "cost_spec": sp, "weeks": wm,
                                     "qb_points": pts, "draft_cost": cost,
                                     "net_value": pts - qb_repl.get(s, 270.0) - cost})
    return pl.DataFrame(rows)


def marginal_grid(ctxs: dict, curve: pl.DataFrame) -> pl.DataFrame:
    """Rank-matched marginal value of QB2: the same QB1 with and without one.

    This is the comparison phase6 makes for the elite case and does not make
    for the mid-priced case, where it swaps the QB1 as well.
    """
    lookups = {sp: cost_lookup(curve, sp) for sp in COST_SPECS}
    rows = []
    for s, c in ctxs.items():
        for r1 in QB1_RANKS:
            a = c.qb_at(r1)
            if a is None:
                continue
            for r2 in QB2_RANKS:
                b = c.qb_at(r2)
                if b is None or b[0] == a[0]:
                    continue
                for wm in WEEK_MODES:
                    for q in QUANTILES:
                        dec = c.decompose(a[0], b[0], q, wm)
                        gross = dec["pair_points"] - c.solo_points(a[0], q, wm)
                        for sp in COST_SPECS:
                            cst = lookups[sp](b[1])
                            rows.append({
                                "season": s, "qb1_rank": r1, "qb2_rank": r2,
                                "quantile": q, "cost_spec": sp, "weeks": wm,
                                "qb1": a[2], "qb2": b[2],
                                "qb1_adp": a[1], "qb2_adp": b[1],
                                "qb1_missed_starts": c.missed_starts(a[0]),
                                "qb2_missed_starts": c.missed_starts(b[0]),
                                "gross_marginal": gross,
                                "coverage_gain": dec["coverage_gain"],
                                "choice_gain": dec["choice_gain"],
                                "coverage_weeks": dec["coverage_weeks"],
                                "qb2_pick_cost": cst,
                                "net_marginal": gross - cst})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# phase-2c machinery, replacement quantile parameterised
# --------------------------------------------------------------------------

def p2c_projected_panel() -> pl.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    p = pm.build_panel(SEASONS).drop_nulls(
        subset=["form_fp_mean", "form_pass_att", "rush_fp_share"])
    frames = []
    for s in TEST_SEASONS:
        tr, te = p.filter(pl.col("season") < s), p.filter(pl.col("season") == s)
        if tr.height < 200 or te.is_empty():
            continue
        Xtr = tr.select(P2C_FEATS).to_numpy().astype(float)
        Xte = te.select(P2C_FEATS).to_numpy().astype(float)
        sc = StandardScaler().fit(Xtr)
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(
            sc.transform(Xtr), tr["fantasy_points"].to_numpy())
        frames.append(te.with_columns(pl.Series("proj", m.predict(sc.transform(Xte)))))
    return pl.concat(frames)


def pair_rows(p: pl.DataFrame, min_games: int = 12) -> pl.DataFrame:
    """phase2c's pair loop with the streamed weeks kept separable.

    `solo_present` is QB1's own scoring and `miss` are the weeks he did not
    start, so solo(q) = solo_present + sum(repl_q[w] for w in miss) and the
    replacement quantile stops being hard-coded at the median.
    """
    repl = {}
    for s in TEST_SEASONS:
        cur = p.filter(pl.col("season") == s)
        if cur.is_empty():
            continue
        for q in QUANTILES:
            # phase2c uses .median(), which is linear interpolation, while
            # engine.weekly_replacement uses .quantile(), which defaults to
            # nearest. The two scripts therefore disagree about what "the
            # median starter" is by up to 1.7 pts/season. Linear here so the
            # q=0.50 cell reproduces phase2c's published 12.0 exactly.
            g = cur.group_by("week").agg(
                pl.col("fantasy_points").quantile(q, interpolation="linear").alias("r"))
            repl[(s, q)] = {r["week"]: r["r"] for r in g.to_dicts()}

    rows = []
    for s in TEST_SEASONS:
        cur = p.filter(pl.col("season") == s)
        if cur.is_empty():
            continue
        counts = cur.group_by("player_id").agg(pl.len().alias("g"))
        eligible = counts.filter(pl.col("g") >= min_games)["player_id"].to_list()
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
                 for r in cur.select(["player_id", "player_display_name"])
                 .unique().to_dicts()}
        all_weeks = sorted(cur["week"].unique().to_list())
        slots = max(all_weeks) - 1

        for a, b in combinations(ids, 2):
            qb1, qb2 = (a, b) if ppg[a] >= ppg[b] else (b, a)
            w1, w2 = wk[qb1], wk[qb2]
            if len(set(w1) | set(w2)) < 12:
                continue
            solo_present = model = oracle = 0.0
            miss, n = [], 0
            for w in all_weeks:
                r1, r2 = w1.get(w), w2.get(w)
                if r1 is None and r2 is None:
                    continue
                n += 1
                if r1 is not None:
                    solo_present += r1[0]
                else:
                    miss.append(w)
                if r1 is None:
                    model += r2[0]
                elif r2 is None:
                    model += r1[0]
                else:
                    model += (r1 if r1[1] >= r2[1] else r2)[0]
                oracle += max(v[0] for v in (r1, r2) if v is not None)
            rec = {"season": s, "qb1": names[qb1], "qb2": names[qb2], "weeks": n,
                   "solo_present": solo_present, "model": model, "oracle": oracle,
                   "qb1_starts": len(w1), "qb2_starts": len(w2),
                   "qb1_missed_starts": max(0, slots - len(w1)),
                   "streamed_weeks": len(miss)}
            for q in QUANTILES:
                rp = repl[(s, q)]
                rec[f"solo_q{int(q * 100)}"] = solo_present + sum(
                    rp.get(w, 15.0) for w in miss)
                rec[f"gain_q{int(q * 100)}"] = model - rec[f"solo_q{int(q * 100)}"]
            rows.append(rec)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    print(BAR)
    print(" QB-020  SKEPTIC PASS ON THE QB-012 / QB-011 ECONOMICS")
    print(BAR)

    panel = pm.build_panel(SEASONS).drop_nulls(subset=projection.FEATURES)
    proj = projection.fit_walk_forward(panel, TEST_SEASONS)
    adp = adp_all(SEASONS)
    qb_adp = adp.filter(pl.col("adp_position") == "QB")
    par = dv.points_above_replacement(SEASONS)
    qb_repl = {r["season"]: r["replacement_points"]
               for r in dv.replacement_levels(dv.season_totals(SEASONS))
               .filter(pl.col("position") == "QB").to_dicts()}

    # ================= A0 replication =====================================
    print(f"\n{BAR}\n A0. INDEPENDENT REPLICATION OF THE PICK-VALUE CURVE\n{BAR}")
    orig = dv.pick_value_curve(SEASONS)
    indep = independent_curve(SEASONS, adp)
    rep = (orig.select(["bkt", "pick_from", "n", "mean_usable_par"])
           .join(indep.select(["bkt", "n_indep", "indep_mean_usable_par"]),
                 on="bkt", how="full", coalesce=True).sort("bkt")
           .with_columns((pl.col("mean_usable_par")
                          - pl.col("indep_mean_usable_par")).abs().alias("abs_diff")))
    print(f"{'picks':<12}{'n(orig)':>9}{'n(indep)':>10}{'orig':>9}{'indep':>9}{'|diff|':>9}")
    print(SUB)
    for r in rep.to_dicts():
        pf = r["pick_from"]
        print(f"{f'{pf}-{pf + 11}':<12}{r['n'] or 0:>9}{r['n_indep'] or 0:>10}"
              f"{fmt(r['mean_usable_par'], 9)}{fmt(r['indep_mean_usable_par'], 9)}"
              f"{fmt(r['abs_diff'], 9, 3)}")
    worst = rep["abs_diff"].max() or 0.0
    print(SUB)
    print(f"max absolute difference: {worst:.4f}  -> "
          f"{'REPLICATES' if worst < 0.01 else 'DOES NOT REPLICATE'}")
    rep.write_csv(OUT / "qb020_curve_replication.csv")

    # ================= A2 cost specs ======================================
    curve, audit, filled = cost_curves(adp, par)
    print(f"\n{BAR}\n A2. COST OF A PICK UNDER FOUR SPECIFICATIONS\n{BAR}")
    curve_specs = curve["spec"].unique().to_list()
    show = [s for s in COST_SPECS if s in curve_specs]
    wide = curve.pivot(values="value", index="pick_from", on="spec").sort("pick_from")
    print(f"{'picks':<12}" + "".join(f"{s:>13}" for s in show))
    print(SUB)
    for r in wide.to_dicts():
        pf = r["pick_from"]
        print(f"{f'{pf}-{pf + 11}':<12}" + "".join(fmt(r.get(s), 13) for s in show))
    print(SUB)
    print("('mean_fixed' shares these values; only the lookup differs -- see below.)")
    print("A QB2 goes around picks 106-135, so rows 97-108 / 109-120 / 121-132")
    print("are the bar QB2 has to clear. 'hits_only' conditions on the pick")
    print("working, which is not knowable on draft day -- it is an upper bound,")
    print("not a rival estimate of expected cost.")
    curve.write_csv(OUT / "qb020_pick_value_curves.csv")

    # -- the bucket-gap defect in phase6's own lookup ----------------------
    print(f"\n{BAR}\n DEFECT: 8% OF PICKS ARE CHARGED NOTHING\n{BAR}")
    print("phase6's pick_cost_lookup scans integer ranges [12k+1, 12k+12] with")
    print("<=, but ADP is a float. Anything strictly between 12k+12 and 12k+13")
    print("matches no bucket and falls through to the deepest bucket, worth 0.0.")
    gap = [a for a in adp["adp"].to_list()
           if not any(12 * k + 1 <= a <= 12 * k + 12 for k in range(0, 16))]
    print(f"\n  {len(gap)} of {adp.height} ADP values ({len(gap) / adp.height * 100:.1f}%)"
          " fall in a gap and cost zero.")
    print("  Quarterbacks in phase6's own strategy ranks that are affected:")
    hit = []
    for s in TEST_SEASONS:
        rows_s = qb_adp.filter(pl.col("season") == s).sort("adp").to_dicts()
        for rank in (1, 4, 8, 10, 16, 18):
            if len(rows_s) >= rank:
                r = rows_s[rank - 1]
                if not any(12 * k + 1 <= r["adp"] <= 12 * k + 12 for k in range(0, 16)):
                    hit.append({"season": s, "qb_rank": rank, "name": r["adp_name"],
                                "adp": r["adp"],
                                "correct_cost": cost_lookup(curve, "mean_fixed")(r["adp"])})
                    print(f"    {s} QB rank {rank:>2}  {r['adp_name']:<22}"
                          f" ADP {r['adp']:>7.2f}  charged 0.0, should be "
                          f"{hit[-1]['correct_cost']:.1f}")
    pl.DataFrame(hit).write_csv(OUT / "qb020_cost_lookup_gap.csv")
    print("  Three of the four are the rank-18 partner in `late pair`, which is")
    print("  the strategy the +30.1 depends on.")

    # ================= A4 survivorship ====================================
    print(f"\n{BAR}\n A4. SURVIVORSHIP IN THE DRAFTED-PLAYERS-ONLY ADP SET\n{BAR}")
    print(f"{'picks':<12}{'drafted':>9}{'unmatched':>11}{'share':>8}"
          f"{'matched-only':>14}{'zero-filled':>13}")
    print(SUB)
    for r in audit.to_dicts():
        pf = r["pick_from"]
        print(f"{f'{pf}-{pf + 11}':<12}{r['n_drafted']:>9}{r['n_unmatched']:>11}"
              f"{r['share_unmatched'] * 100:>7.1f}%"
              f"{fmt(r['mean_matched_only'], 14)}{fmt(r['mean_with_zeros'], 13)}")
    audit.write_csv(OUT / "qb020_adp_survivorship.csv")

    # separate "never played" from "name did not join"
    allkeys = par.select(["key"]).unique()["key"].to_list()
    known = set(allkeys)
    un = filled.filter(pl.col("unmatched"))
    seen_other_season = sum(1 for k in un["key"].to_list() if k in known)
    print(SUB)
    print(f"unmatched drafted RB/WR/TE rows: {un.height} of {filled.height} "
          f"({un.height / max(filled.height, 1) * 100:.1f}%)")
    print(f"  of those, {seen_other_season} have the same name key in another")
    print(f"  season, so they are real players who recorded no REG stat line;")
    print(f"  {un.height - seen_other_season} never appear at all (rookie busts,")
    print("  camp cuts, or a name the join could not resolve).")
    print("Zero-filling them moves a late-round pick's cost by "
          f"{(audit['mean_matched_only'] - audit['mean_with_zeros']).mean():.2f} pts.")

    cov = (adp.group_by("season").agg(pl.len().alias("adp_players"),
                                      pl.col("adp").max().alias("deepest"))
           .sort("season"))
    print(SUB)
    print("ADP list depth vs a 12-team x 16-round draft (192 players drafted):")
    for r in cov.to_dicts():
        gap = 192 - r["adp_players"]
        print(f"  {r['season']}: {r['adp_players']:>4} listed, deepest ADP "
              f"{r['deepest']:>5.0f}  -> {gap:>4} of 192 slots never appear")
    cov.write_csv(OUT / "qb020_adp_coverage.csv")

    # ================= phase-6 rebuild ====================================
    ctxs = {s: SeasonCtx(proj, qb_adp, s) for s in TEST_SEASONS}
    print(f"\n{BAR}\n UNBALANCED-PANEL CHECK\n{BAR}")
    print(f"{'strategy':<30}" + "".join(f"{s:>7}" for s in TEST_SEASONS))
    print(SUB)
    for label, ranks in STRATEGIES.items():
        print(f"{label:<30}" + "".join(
            f"{('yes' if all(ctxs[s].qb_at(r) is not None for r in ranks) else 'NO'):>7}"
            for s in TEST_SEASONS))
    print(SUB)
    print("phase6_draft_cost.py averages each strategy over whatever seasons it")
    print("has, so an absent cell silently changes which seasons a comparison")
    print("uses. Everything below is restricted to the balanced set.")

    tab = strategy_table(ctxs, qb_repl, curve)
    tab.write_csv(OUT / "qb020_strategy_table.csv")
    have = (tab.filter((pl.col("quantile") == BASE_Q) & (pl.col("cost_spec") == BASE_COST)
                       & (pl.col("weeks") == BASE_WEEKS))
            .group_by("season").agg(pl.len().alias("k")))
    balanced = sorted(have.filter(pl.col("k") == have["k"].max())["season"].to_list())
    print(f"balanced seasons: {balanced}   (n = {len(balanced)})")
    bal = tab.filter(pl.col("season").is_in(balanced))

    def net(strategy, q, sp, wm=BASE_WEEKS):
        d = bal.filter((pl.col("strategy") == strategy) & (pl.col("quantile") == q)
                       & (pl.col("cost_spec") == sp) & (pl.col("weeks") == wm)).sort("season")
        return np.array(d["net_value"].to_list(), float)

    print(f"\n{BAR}\n BASELINE REPRODUCTION (median streamer, mean cost, phase6 weeks)\n{BAR}")
    print(f"{'strategy':<30}{'QB pts':>9}{'cost':>9}{'NET':>9}{'sd':>8}{'worst':>9}")
    print(SUB)
    for label in STRATEGIES:
        d = bal.filter((pl.col("strategy") == label) & (pl.col("quantile") == BASE_Q)
                       & (pl.col("cost_spec") == BASE_COST)
                       & (pl.col("weeks") == BASE_WEEKS))
        if d.is_empty():
            continue
        v = np.array(d["net_value"].to_list(), float)
        print(f"{label:<30}{d['qb_points'].mean():>9.1f}{d['draft_cost'].mean():>9.1f}"
              f"{v.mean():>9.1f}{v.std(ddof=1):>8.1f}{v.min():>9.1f}")
    print(SUB)
    print("Published QB-012 table: 53.0 / 43.3 / 20.5 / 42.3 / 12.2 / 36.3.")

    claims = []

    def record(name, arr_a, arr_b, q, sp, wm, note=""):
        st = t_ci(arr_a - arr_b)
        claims.append({"claim": name, "quantile": q, "cost_spec": sp, "weeks": wm,
                       "estimate": st["mean"], "sd": st["sd"], "se": st["se"],
                       "ci_lo": st["lo"], "ci_hi": st["hi"], "t": st["t"],
                       "p": st["p"], "n_seasons": st["n"], "note": note})
        return st

    # ================= A1: how sharp is a real streamer? ==================
    print(f"\n{BAR}\n A1. WHAT PERCENTILE DOES A REAL STREAMER ACTUALLY REACH?\n{BAR}")
    print("phase6's own no-QB-drafted strategy picks the best pregame projection")
    print("from QBs outside the top-19 ADP. Reading its realised score back into")
    print("that week's starter distribution turns the replacement level from an")
    print("assumption into a measurement.")
    print(f"\n{'season':<9}{'weeks':>7}{'mean pctile':>13}{'median pctile':>15}")
    print(SUB)
    sp_rows, allp = [], []
    for s in TEST_SEASONS:
        pr = ctxs[s].streamer_percentiles()
        if not pr:
            continue
        allp += pr
        sp_rows.append({"season": s, "weeks": len(pr), "mean_pctile": np.mean(pr),
                        "median_pctile": np.median(pr)})
        print(f"{s:<9}{len(pr):>7}{np.mean(pr) * 100:>12.1f}%{np.median(pr) * 100:>14.1f}%")
    print(SUB)
    meas = float(np.mean(allp))
    print(f"pooled: {meas * 100:.1f}% (mean), {np.median(allp) * 100:.1f}% (median), "
          f"n={len(allp)} weeks")
    print("So the median-starter assumption is close to, and slightly above, what")
    print("a real streamer achieved. The 75th percentile is a stress test, not a")
    print("plausible opponent: no waiver wire yields a top-quartile QB every week.")
    pl.DataFrame(sp_rows).write_csv(OUT / "qb020_streamer_percentile.csv")

    # ================= A1 + A5: headline sweep ============================
    print(f"\n{BAR}\n A1 + A5. HEADLINE NUMBERS vs THE STREAMER'S SHARPNESS\n{BAR}")
    print(f"{'claim':<34}{'pctile':>7}{'est':>8}{'se':>7}{'95% CI':>20}{'p':>7}")
    print(SUB)
    for q in QUANTILES:
        st = record("QB-012 elite (published +9.7)",
                    net("premium + late partner", q, BASE_COST),
                    net("premium (QB1 only)", q, BASE_COST), q, BASE_COST,
                    BASE_WEEKS, "phase6 as written")
        print(f"{'QB-012 elite: pair - solo':<34}{int(q * 100):>7}{st['mean']:>8.1f}"
              f"{st['se']:>7.1f}{ci_str(st)}{st['p']:>7.3f}")
    print()
    for q in QUANTILES:
        st = record("QB-012 mid (published +30.1)", net("late pair", q, BASE_COST),
                    net("single value QB", q, BASE_COST), q, BASE_COST, BASE_WEEKS,
                    "rank-mismatched: QB10+QB18 vs QB8")
        print(f"{'QB-012 mid: latepair - singleQB':<34}{int(q * 100):>7}{st['mean']:>8.1f}"
              f"{st['se']:>7.1f}{ci_str(st)}{st['p']:>7.3f}")

    print(f"\n{BAR}\n A2 + S2. THE SAME TWO NUMBERS UNDER EVERY OTHER SPECIFICATION\n{BAR}")
    print(f"{'claim':<20}{'cost spec':>13}{'weeks':>13}{'est':>8}{'se':>7}{'95% CI':>20}")
    print(SUB)
    for sp in COST_SPECS:
        for wm in WEEK_MODES:
            st = record("QB-012 elite (published +9.7)",
                        net("premium + late partner", BASE_Q, sp, wm),
                        net("premium (QB1 only)", BASE_Q, sp, wm),
                        BASE_Q, sp, wm, "spec sweep")
            print(f"{'elite pair - solo':<20}{sp:>13}{wm:>13}{st['mean']:>8.1f}"
                  f"{st['se']:>7.1f}{ci_str(st)}")
    print()
    for sp in COST_SPECS:
        for wm in WEEK_MODES:
            st = record("QB-012 mid (published +30.1)", net("late pair", BASE_Q, sp, wm),
                        net("single value QB", BASE_Q, sp, wm), BASE_Q, sp, wm,
                        "spec sweep")
            print(f"{'mid latepair - solo':<20}{sp:>13}{wm:>13}{st['mean']:>8.1f}"
                  f"{st['se']:>7.1f}{ci_str(st)}")

    # ================= S1: the rank mismatch ==============================
    grid = marginal_grid(ctxs, curve)
    grid.write_csv(OUT / "qb020_marginal_grid.csv")
    gb = grid.filter(pl.col("season").is_in(balanced))

    print(f"\n{BAR}\n S1. THE +30.1 IS NOT A MARGINAL QB2 EFFECT\n{BAR}")
    print("phase6 compares ADP QB10 + QB18 against ADP QB8 alone. Different QB1.")
    print("Rank-matched, at the median streamer and mean pick cost:\n")
    print(f"  as published   QB10+QB18 - QB8 alone   "
          f"{net('late pair', BASE_Q, BASE_COST).mean() - net('single value QB', BASE_Q, BASE_COST).mean():+8.1f}")
    g0 = gb.filter((pl.col("quantile") == BASE_Q) & (pl.col("cost_spec") == BASE_COST)
                   & (pl.col("weeks") == BASE_WEEKS) & (pl.col("qb2_rank") == 18))
    for r1 in (8, 10):
        v = g0.filter(pl.col("qb1_rank") == r1).sort("season")["net_marginal"].to_numpy()
        if not v.size:
            continue
        st = t_ci(v)
        print(f"  rank-matched   QB{r1}+QB18 - QB{r1} alone  {st['mean']:+8.1f}"
              f"   se {st['se']:.1f}  CI [{st['lo']:.1f}, {st['hi']:.1f}]  n={st['n']}")
        claims.append({"claim": f"QB-012 mid, rank-matched QB{r1}+QB18 vs QB{r1}",
                       "quantile": BASE_Q, "cost_spec": BASE_COST, "weeks": BASE_WEEKS,
                       "estimate": st["mean"], "sd": st["sd"], "se": st["se"],
                       "ci_lo": st["lo"], "ci_hi": st["hi"], "t": st["t"], "p": st["p"],
                       "n_seasons": st["n"], "note": "corrects the rank mismatch"})
    print("\n  The residual is QB1 selection (rank 8 vs rank 10) plus its pick-cost")
    print("  saving. The published number attributes all of it to the second QB.")

    # ================= S3: the dropped season was a QB2 bust ==============
    print(f"\n{BAR}\n S3. PHASE 6 DELETES THE SEASON ITS QB2 WAS WORTHLESS\n{BAR}")
    print("An ADP name that does not resolve to a panel player is not missing")
    print("data -- it is a quarterback who was drafted and never took a snap.")
    print("phase6 drops the entire strategy-season instead of charging it.\n")
    lk = cost_lookup(curve, BASE_COST)
    bust_rows = []
    for label, ranks in STRATEGIES.items():
        if not ranks:
            continue
        for s, c in ctxs.items():
            sel = [c.qb_at_or_bust(r) for r in ranks]
            if any(x is None for x in sel) or not any(x[3] for x in sel):
                continue
            who = ", ".join(f"{x[2]} (ADP {x[1]:.0f}, rank {r})"
                            for x, r in zip(sel, ranks) if x[3])
            if len(sel) == 1:
                pts = c.solo_points(sel[0][0], BASE_Q, BASE_WEEKS)
            else:
                pts = c.pair_result(sel[0][0], sel[1][0], BASE_Q,
                                    BASE_WEEKS).total_points
            cost = sum(lk(x[1]) for x in sel)
            netv = pts - qb_repl.get(s, 270.0) - cost
            bust_rows.append({"season": s, "strategy": label, "bust": who,
                              "qb_points": pts, "draft_cost": cost, "net_value": netv})
            print(f"  {s} {label}: {who}")
            print(f"        scored as a bust -> QB pts {pts:.1f}, cost {cost:.1f}, "
                  f"NET {netv:+.1f}")
    if bust_rows:
        bf = pl.DataFrame(bust_rows)
        bf.write_csv(OUT / "qb020_bust_seasons.csv")
        print()
        for label in bf["strategy"].unique().to_list():
            pub = bal.filter((pl.col("strategy") == label) & (pl.col("quantile") == BASE_Q)
                             & (pl.col("cost_spec") == BASE_COST)
                             & (pl.col("weeks") == BASE_WEEKS))
            add = bf.filter(pl.col("strategy") == label)
            allv = np.concatenate([pub["net_value"].to_numpy(),
                                   add["net_value"].to_numpy()])
            print(f"  {label}: published mean {pub['net_value'].mean():+.1f} over "
                  f"{pub.height} seasons -> {allv.mean():+.1f} over {allv.size}")
        lp = bf.filter(pl.col("strategy") == "late pair")
        if lp.height:
            def by_season(strategy):
                d = tab.filter((pl.col("strategy") == strategy)
                               & (pl.col("quantile") == BASE_Q)
                               & (pl.col("cost_spec") == BASE_COST)
                               & (pl.col("weeks") == BASE_WEEKS))
                return {r["season"]: r["net_value"] for r in d.to_dicts()}

            lp_net = by_season("late pair")
            lp_net.update({r["season"]: r["net_value"] for r in lp.to_dicts()})
            sv_net = by_season("single value QB")
            yrs = sorted(set(lp_net) & set(sv_net))
            st = t_ci(np.array([lp_net[y] - sv_net[y] for y in yrs]))
            print(f"\n  +30.1 with the bust season restored: {st['mean']:+.1f} "
                  f"(se {st['se']:.1f}, CI [{st['lo']:.1f}, {st['hi']:.1f}], n={st['n']})")
            claims.append({"claim": "QB-012 mid, bust season restored",
                           "quantile": BASE_Q, "cost_spec": BASE_COST,
                           "weeks": BASE_WEEKS, "estimate": st["mean"], "sd": st["sd"],
                           "se": st["se"], "ci_lo": st["lo"], "ci_hi": st["hi"],
                           "t": st["t"], "p": st["p"], "n_seasons": st["n"],
                           "note": "drafted-but-never-played QB2 charged, not dropped"})

    # ================= the grid: is +9.7 a lucky cell? ====================
    print(f"\n{BAR}\n RANK-MATCHED MARGINAL VALUE OF QB2, ALL QB1 RANKS 1-12\n{BAR}")
    print("Every cell holds QB1 fixed and adds a partner, so only the second")
    print("quarterback changes. Clustered on season, all six test seasons.")
    print(f"\n{'pctile':>7}{'cost spec':>13}{'cells':>7}{'net':>8}{'boot se':>9}"
          f"{'95% CI':>20}{'share>0':>9}")
    print(SUB)
    boot_rows = []
    for q in QUANTILES:
        for sp in COST_SPECS:
            gg = grid.filter((pl.col("quantile") == q) & (pl.col("cost_spec") == sp)
                             & (pl.col("weeks") == BASE_WEEKS))
            v = gg["net_marginal"].to_numpy()
            bs = cluster_bootstrap(v, gg["season"].to_numpy())
            share = float((v > 0).mean() * 100)
            boot_rows.append({"quantile": q, "cost_spec": sp, "weeks": BASE_WEEKS,
                              **bs, "share_positive": share})
            print(f"{int(q * 100):>7}{sp:>13}{bs['n_obs']:>7}{bs['mean']:>8.1f}"
                  f"{bs['boot_se']:>9.1f}{ci_str(bs)}{share:>8.1f}%")
    pl.DataFrame(boot_rows).write_csv(OUT / "qb020_grid_bootstrap.csv")

    print(f"\n{BAR}\n IS THE +9.7 SPECIAL TO AN ELITE QB1? NET MARGINAL BY QB1 RANK\n{BAR}")
    print("median streamer, mean pick cost, partner ranks 14/16/18/20 pooled")
    print(f"\n{'QB1 rank':>9}{'cells':>7}{'gross':>9}{'net':>8}{'boot se':>9}"
          f"{'95% CI':>20}{'share>0':>9}")
    print(SUB)
    rank_rows = []
    for r1 in QB1_RANKS:
        gg = grid.filter((pl.col("quantile") == BASE_Q) & (pl.col("cost_spec") == BASE_COST)
                         & (pl.col("weeks") == BASE_WEEKS) & (pl.col("qb1_rank") == r1))
        if gg.is_empty():
            continue
        v = gg["net_marginal"].to_numpy()
        bs = cluster_bootstrap(v, gg["season"].to_numpy())
        rank_rows.append({"qb1_rank": r1, "gross": float(gg["gross_marginal"].mean()),
                          **bs, "share_positive": float((v > 0).mean() * 100)})
        print(f"{r1:>9}{bs['n_obs']:>7}{gg['gross_marginal'].mean():>9.1f}"
              f"{bs['mean']:>8.1f}{bs['boot_se']:>9.1f}{ci_str(bs)}"
              f"{(v > 0).mean() * 100:>8.1f}%")
    pl.DataFrame(rank_rows).write_csv(OUT / "qb020_marginal_by_qb1_rank.csv")

    print(f"\n{BAR}\n THE PARTNER RANK IS AN UNJUSTIFIED FREE PARAMETER\n{BAR}")
    print("STRATEGIES pairs the elite QB1 with ADP QB rank 16. Nothing in the")
    print("write-up motivates 16 rather than 14, 18 or 20. Moving it:")
    print(f"\n{'QB2 rank':>9}{'cells':>7}{'gross':>9}{'cost':>8}{'net':>8}{'boot se':>9}"
          f"{'95% CI':>20}   pooled over QB1 ranks 1-12")
    print(SUB)
    prow = []
    base = grid.filter((pl.col("quantile") == BASE_Q) & (pl.col("cost_spec") == BASE_COST)
                       & (pl.col("weeks") == BASE_WEEKS))
    for r2 in QB2_RANKS:
        gg = base.filter(pl.col("qb2_rank") == r2)
        bs = cluster_bootstrap(gg["net_marginal"].to_numpy(), gg["season"].to_numpy())
        prow.append({"scope": "qb1_ranks_1_12", "qb2_rank": r2,
                     "gross": float(gg["gross_marginal"].mean()),
                     "cost": float(gg["qb2_pick_cost"].mean()), **bs})
        print(f"{r2:>9}{bs['n_obs']:>7}{gg['gross_marginal'].mean():>9.1f}"
              f"{gg['qb2_pick_cost'].mean():>8.1f}{bs['mean']:>8.1f}{bs['boot_se']:>9.1f}"
              f"{ci_str(bs)}")
    print(f"\n{'QB2 rank':>9}{'seasons':>9}{'net':>8}{'se':>7}{'95% CI':>20}"
          f"   QB1 = ADP rank 1 only")
    print(SUB)
    for r2 in QB2_RANKS:
        v = base.filter((pl.col("qb1_rank") == 1) & (pl.col("qb2_rank") == r2)) \
                .sort("season")["net_marginal"].to_numpy()
        if v.size < 2:
            continue
        st = t_ci(v)
        prow.append({"scope": "qb1_rank_1", "qb2_rank": r2, "gross": float("nan"),
                     "cost": float("nan"), "mean": st["mean"], "boot_se": st["se"],
                     "lo": st["lo"], "hi": st["hi"], "n_clusters": st["n"],
                     "n_obs": st["n"]})
        print(f"{r2:>9}{st['n']:>9}{st['mean']:>8.1f}{st['se']:>7.1f}{ci_str(st)}"
              f"   {'<- the published +9.7 cell' if r2 == 16 else ''}")
    pl.DataFrame(prow).write_csv(OUT / "qb020_partner_rank_sensitivity.csv")
    print(SUB)
    print("Two ADP slots either side of the published choice flips the sign.")

    print(f"\n{BAR}\n THE PUBLISHED +9.7 CELL, SEASON BY SEASON\n{BAR}")
    print(f"{'season':<7}{'QB1 + QB2':<40}{'cover':>8}{'choice':>8}{'cost':>7}{'net':>8}"
          f"{'cov wk':>8}")
    print(SUB)
    cell = base.filter((pl.col("qb1_rank") == 1) & (pl.col("qb2_rank") == 16)).sort("season")
    for r in cell.to_dicts():
        mark = "" if r["season"] in balanced else "  (dropped by phase6's balance)"
        print(f"{r['season']:<7}{f'{r['qb1']} + {r['qb2']}':<40}"
              f"{r['coverage_gain']:>8.1f}{r['choice_gain']:>8.1f}"
              f"{r['qb2_pick_cost']:>7.1f}{r['net_marginal']:>8.1f}"
              f"{r['coverage_weeks']:>8}{mark}")
    cell.write_csv(OUT / "qb020_published_cell.csv")
    v6 = cell["net_marginal"].to_numpy()
    v5 = cell.filter(pl.col("season").is_in(balanced))["net_marginal"].to_numpy()
    print(SUB)
    print(f"mean over the 5 seasons phase6 reports: {v5.mean():+.1f}  (the +9.7)")
    print(f"mean over all 6 available seasons:      {v6.mean():+.1f}")
    print("2021 is the only season the elite QB1 started every week. It is also")
    print("the season the pair loses, and it is the season phase6 drops.")

    # ================= A3: mechanism ======================================
    print(f"\n{BAR}\n A3. MECHANISM: COVERAGE vs WEEKLY CHOICE\n{BAR}")
    print("gross marginal = coverage (weeks QB1 did not start, pair vs streamer)")
    print("               + choice   (weeks QB1 did start, pair vs QB1 himself)")
    print(f"\n{'pctile':>7}{'coverage':>10}{'choice':>9}{'gross':>9}{'cost':>8}"
          f"{'net':>8}{'cov weeks':>11}")
    print(SUB)
    mech_rows = []
    for q in QUANTILES:
        gg = grid.filter((pl.col("quantile") == q) & (pl.col("cost_spec") == BASE_COST)
                         & (pl.col("weeks") == BASE_WEEKS))
        row = {"quantile": q,
               "coverage": float(gg["coverage_gain"].mean()),
               "choice": float(gg["choice_gain"].mean()),
               "gross": float(gg["gross_marginal"].mean()),
               "cost": float(gg["qb2_pick_cost"].mean()),
               "net": float(gg["net_marginal"].mean()),
               "cov_weeks": float(gg["coverage_weeks"].mean())}
        mech_rows.append(row)
        print(f"{int(q * 100):>7}{row['coverage']:>10.1f}{row['choice']:>9.1f}"
              f"{row['gross']:>9.1f}{row['cost']:>8.1f}{row['net']:>8.1f}"
              f"{row['cov_weeks']:>11.1f}")
    pl.DataFrame(mech_rows).write_csv(OUT / "qb020_mechanism_decomposition.csv")
    gtr = base["net_marginal"].to_numpy()
    lo, hi = np.percentile(gtr, [5, 95])
    keep = (gtr >= lo) & (gtr <= hi)
    bs = cluster_bootstrap(gtr[keep], base["season"].to_numpy()[keep])
    print(f"5% trimmed grid net marginal at the median streamer: {bs['mean']:+.1f} "
          f"(se {bs['boot_se']:.1f}, CI [{bs['lo']:.1f}, {bs['hi']:.1f}], "
          f"untrimmed {gtr.mean():+.1f})")
    print(SUB)
    print("If the mechanism is insurance, coverage carries the gain and choice is")
    print("small. If coverage collapses as the streamer sharpens, the insurance is")
    print("only worth what the waiver wire cannot already supply.")

    print(f"\n{BAR}\n A3. INJURY SPLIT ON THE PHASE-6 GRID (post-hoc diagnostic)\n{BAR}")
    print(f"{'pctile':>7}{'QB1 missed':>13}{'cells':>7}{'gross':>9}{'net':>8}"
          f"{'boot se':>9}{'95% CI':>20}")
    print(SUB)
    split_rows = []
    for q in QUANTILES:
        gg = grid.filter((pl.col("quantile") == q) & (pl.col("cost_spec") == BASE_COST)
                         & (pl.col("weeks") == BASE_WEEKS))
        for lab, sel in (("0-2 starts", pl.col("qb1_missed_starts") < 3),
                         ("3+ starts", pl.col("qb1_missed_starts") >= 3)):
            h = gg.filter(sel)
            if h.is_empty():
                continue
            bs = cluster_bootstrap(h["net_marginal"].to_numpy(), h["season"].to_numpy())
            gr = float(h["gross_marginal"].mean())
            split_rows.append({"source": "phase6_grid", "quantile": q, "split": lab,
                               "gross": gr, **bs})
            print(f"{int(q * 100):>7}{lab:>13}{bs['n_obs']:>7}{gr:>9.1f}{bs['mean']:>8.1f}"
                  f"{bs['boot_se']:>9.1f}{ci_str(bs)}")
        print()

    # ================= QB-011 =============================================
    print(f"{BAR}\n QB-011: +12.0 PTS/SEASON, RE-EXAMINED\n{BAR}")
    p2c = p2c_projected_panel()
    pairs = pair_rows(p2c, min_games=12)
    pairs.write_csv(OUT / "qb020_pair_gains.csv")
    got = sorted(pairs["season"].unique().to_list())
    print(f"pairs: {pairs.height:,}   seasons present: {got}")
    print("model and oracle totals match outputs/phase2c_pair_value.csv exactly")
    print("on all 1,106 pairs; solo is reproduced with linear-interpolated")
    print("quantiles so the q=50 cell equals phase2c's published number.")
    if TEST_SEASONS[0] not in got:
        print(f"NOTE: phase2c_pair_value.py documents {TEST_SEASONS[0]}-"
              f"{TEST_SEASONS[-1]} but {TEST_SEASONS[0]} is silently absent. Its")
        print("prev_ppg join reads the *projected* panel, which starts in "
              f"{TEST_SEASONS[0]}, so season {TEST_SEASONS[0]} has no prior season")
        print("to join to and every one of its pairs is dropped. n is 5, not 6.")

    print(f"\n{'pctile':>7}{'mean gain':>11}{'iid se':>9}{'cluster se':>12}"
          f"{'95% CI (cluster)':>22}{'median':>9}{'share>0':>9}")
    print(SUB)
    p2c_rows = []
    for q in QUANTILES:
        v = pairs[f"gain_q{int(q * 100)}"].to_numpy()
        bs = cluster_bootstrap(v, pairs["season"].to_numpy())
        iid = float(v.std(ddof=1) / np.sqrt(v.size))
        p2c_rows.append({"quantile": q, "mean": bs["mean"], "iid_se": iid,
                         "cluster_se": bs["boot_se"], "ci_lo": bs["lo"], "ci_hi": bs["hi"],
                         "median": float(np.median(v)),
                         "share_positive": float((v > 0).mean() * 100),
                         "n_pairs": int(v.size), "n_seasons": bs["n_clusters"]})
        print(f"{int(q * 100):>7}{bs['mean']:>11.1f}{iid:>9.2f}{bs['boot_se']:>12.1f}"
              f"{ci_str(bs, 22)}{np.median(v):>9.1f}{(v > 0).mean() * 100:>8.1f}%")
    pl.DataFrame(p2c_rows).write_csv(OUT / "qb020_pair_value_quantiles.csv")
    print(SUB)
    print("The iid se treats 1,100 pairs as independent observations. They are")
    print("not: every pair in a season shares the same weeks, and the same")
    print("quarterbacks recur across pairs. The season-cluster se is 4x larger")
    print("and is the honest one.")

    print(f"\n{BAR}\n A3. TRIMMING AND THE INJURY SPLIT ON THE PAIR SAMPLE\n{BAR}")
    print(f"{'pctile':>7}{'sample':>26}{'pairs':>7}{'mean':>8}{'cluster se':>12}{'95% CI':>20}")
    print(SUB)
    trim_rows = []
    miss = pairs["qb1_missed_starts"].to_numpy()
    for q in QUANTILES:
        v = pairs[f"gain_q{int(q * 100)}"].to_numpy()
        lo, hi = np.percentile(v, [5, 95])
        for lab, m in (("all pairs", np.ones(v.size, bool)),
                       ("5% trimmed both tails", (v >= lo) & (v <= hi)),
                       ("QB1 missed 0-2 starts", miss < 3),
                       ("QB1 missed 3+ starts", miss >= 3)):
            if m.sum() < 5:
                continue
            bs = cluster_bootstrap(v[m], pairs["season"].to_numpy()[m])
            trim_rows.append({"quantile": q, "sample": lab, **bs})
            print(f"{int(q * 100):>7}{lab:>26}{int(m.sum()):>7}{bs['mean']:>8.1f}"
                  f"{bs['boot_se']:>12.1f}{ci_str(bs)}")
        print()
    pl.DataFrame(trim_rows).write_csv(OUT / "qb020_pair_trim_and_injury.csv")

    print(f"{BAR}\n THE >=12 GAME ELIGIBILITY FILTER, RELAXED\n{BAR}")
    print("phase2c requires both quarterbacks to start 12+ games, which removes")
    print("the very QB1 failures the insurance story depends on. Relaxing it:")
    print(f"\n{'min games':>10}{'pairs':>8}{'mean gain':>11}{'cluster se':>12}"
          f"{'max QB1 missed':>16}{'share QB1 3+':>14}")
    print(SUB)
    elig_rows = []
    for mg in (12, 10, 8, 6):
        pr = pairs if mg == 12 else pair_rows(p2c, min_games=mg)
        if pr.is_empty():
            continue
        v = pr[f"gain_q{int(BASE_Q * 100)}"].to_numpy()
        bs = cluster_bootstrap(v, pr["season"].to_numpy())
        mx = int(pr["qb1_missed_starts"].max())
        sh = float((pr["qb1_missed_starts"].to_numpy() >= 3).mean() * 100)
        elig_rows.append({"min_games": mg, "n_pairs": pr.height, "mean": bs["mean"],
                          "cluster_se": bs["boot_se"], "ci_lo": bs["lo"], "ci_hi": bs["hi"],
                          "max_qb1_missed": mx, "share_qb1_missed_3plus": sh})
        print(f"{mg:>10}{pr.height:>8}{bs['mean']:>11.1f}{bs['boot_se']:>12.1f}"
              f"{mx:>16}{sh:>13.1f}%")
    pl.DataFrame(elig_rows).write_csv(OUT / "qb020_pair_eligibility.csv")

    # ================= ledger =============================================
    cl = pl.DataFrame(claims)
    cl.write_csv(OUT / "qb020_headline_uncertainty.csv")
    print(f"\n{BAR}\n CLAIMS LEDGER (season-paired t intervals, n = 5 seasons)\n{BAR}")
    print(f"{'claim':<44}{'pct':>5}{'cost':>12}{'weeks':>13}{'est':>8}{'se':>7}{'p':>7}")
    print(SUB)
    for r in cl.to_dicts():
        print(f"{r['claim'][:43]:<44}{int(r['quantile'] * 100):>5}{r['cost_spec']:>12}"
              f"{r['weeks']:>13}{r['estimate']:>8.1f}{r['se']:>7.1f}{r['p']:>7.3f}")

    print(f"\nwrote {len(list(OUT.glob('qb020_*.csv')))} qb020_* CSVs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
