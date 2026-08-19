"""QB-021: an attack on the QB-016 market increment.

QB-016 claims the betting market adds +1.85 pp of within-week start/sit ranking
accuracy on top of a public fantasy projection, rising to +4.63 pp on close
calls, and that the projection itself is the redundant input (-0.16 pp given
form and the market).

The obvious weakness is that the "public projection" is a PROXY. CBS publishes
no historical archive, so QB-016 built a stand-in from prior-season points per
game, games played and preseason ADP, validated at r = 0.867 against the real
CBS 2026 board. A proxy systematically WEAKER than a real projection would
inflate the measured increment: the market would be credited with information a
better projection already contains.

This script tries to break the finding six ways.

1. STRONGER PROXY LADDER. Six honest proxies of increasing strength, from
   "prior-season PPG alone" to one carrying prior-season efficiency (YPA, TD
   rate, INT rate, CPOE, EPA/play, sack rate, rushing volume), roster
   continuity, an age/experience/draft-capital curve and a three-season
   games-weighted average. Each is validated against the real CBS 2026 board
   and against realised season outcomes out of sample. The market increment is
   then re-measured on top of each. The deliverable is the RELATIONSHIP between
   proxy strength and the increment, not a single number.

2. CEILING TEST (the decisive one). A deliberately near-oracle season
   projection, fitted WITH HINDSIGHT on the season being tested. Three flavours,
   all clearly labelled as cheating and used only as an upper bound on how good
   any public projection could possibly be:
     hindsight_fit  strongest feature set, ridge refitted in-sample on season S
     oracle_loo     the player's realised season mean EXCLUDING the graded week
     oracle_full    the player's realised season mean INCLUDING the graded week
   No real preseason projection can beat oracle_full; none can beat oracle_loo
   without look-ahead. If the market still adds on top of these, the finding is
   robust in the strongest sense available.

3. PRE-REGISTERED WEEK 10-18 CLAIM. QB-016 reported +3.26 pp in weeks 10-18 vs
   +0.74 pp early and flagged it exploratory. The hypothesis is fixed here in
   advance and evaluated on SEASONS THAT DID NOT EXIST IN QB-016: the panel is
   extended back to 2012 and the claim is graded on test seasons 2014-2019,
   which QB-016 never touched. 2020-2025 (the generating set) is reported
   alongside for contrast only.

4. REDUNDANCY. The market variable correlation matrix is recomputed from raw
   panel columns rather than trusting markets.market_redundancy, the algebraic
   identity implied_team_total == (total_line + team_spread)/2 is checked to
   machine precision, and the increment is re-measured using implied team total
   ALONE against every other single-variable and subset specification.

5. UNCERTAINTY. The +/-0.35 SE came from six test seasons (5 df). Recomputed
   with cluster bootstraps over (season, week) blocks, over seasons, and over
   players (multiway weights m_i * m_j on each pair), plus a season-level
   jackknife. Reported as whether the interval still excludes zero.

6. PLACEBO (added by the skeptic, not requested). The proxy is constant within
   a player-season; the market varies by week. ANY week-varying variable would
   therefore add something. The market is raced against a non-market weekly
   context set (home, indoors, rest, division, opponent defence) to check how
   much of the increment is "market information" rather than "any weekly
   signal at all".

No look-ahead anywhere except the deliberately labelled hindsight proxies in
attack 2, which exist only to bound the answer from above.

Usage:  ./.venv/Scripts/python.exe qb-pairing/scripts/wave2_skeptic_market.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import importlib.util                              # noqa: E402
import sys                                         # noqa: E402
import warnings                                    # noqa: E402
from pathlib import Path                           # noqa: E402

import numpy as np                                 # noqa: E402
import polars as pl                                # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]      # qb-pairing/
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))

import bootstrap                                   # noqa: E402

bootstrap.setup()
import markets                                     # noqa: E402

from qbpair.config import DATA_RAW                 # noqa: E402
from qbpair.data import loaders                    # noqa: E402
from qbpair.features import panel as pm            # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

# The experiment under attack, imported unmodified so the panel construction,
# the ridge and the pair rule are literally the same code, not a rebuild.
_SPEC = importlib.util.spec_from_file_location(
    "wave1_mvc", PROJECT / "scripts" / "wave1_market_vs_consensus.py")
w1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(w1)

BAR = "=" * 78
SUB = "-" * 78
OUT = PROJECT / "outputs"

# QB-016's window, reproduced exactly.
SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
# Extended window for attack 3. 2014-2019 is held out: QB-016 never saw it.
EXT_SEASONS = list(range(2012, 2026))
EXT_TEST = list(range(2014, 2026))
HELD_OUT = list(range(2014, 2020))
GENERATING = list(range(2020, 2026))
FORWARD_SEASON = 2026

MARKET = w1.MARKET                      # implied_team_total, total_line, spread_abs
FORM = w1.FORM
CONTEXT = ["is_home", "indoors", "rest", "div_game", "def_adj"]
CLOSE_PTS = w1.CLOSE_CALL_PTS           # 2.0
ORACLE_K = 4.0                          # games of league prior in the shrunk oracle
MANY_STARTS = 10                        # "the oracle is well estimated" threshold
N_BOOT = 2000
RNG = np.random.default_rng(20260819)

ADP_CACHE = DATA_RAW / "ffc_adp_qb_2010_2026.parquet"
ROSTER_CACHE = DATA_RAW / "rosters_2010_2026.parquet"
STATS_CACHE = DATA_RAW / "player_stats_2010_2025.parquet"
CBS_CACHE = DATA_RAW / "cbs_qb_2026.parquet"


# =========================================================================== #
# Data: preseason-only player-season inputs, richer than QB-016's             #
# =========================================================================== #

STAT_COLS = ["player_id", "player_display_name", "position", "season", "week",
             "season_type", "team", "attempts", "completions", "passing_yards",
             "passing_tds", "passing_interceptions", "sacks_suffered",
             "passing_epa", "passing_cpoe", "carries", "rushing_yards",
             "rushing_tds", "fantasy_points"]


def qb_stats(lo: int, hi: int) -> pl.DataFrame:
    """QB regular-season starts (>= MIN_ATT), lean columns, memory-friendly."""
    if not STATS_CACHE.exists():
        loaders.load_player_stats(list(range(lo, hi + 1)))
        src = DATA_RAW / f"player_stats_{lo}_{hi}.parquet"
    else:
        src = STATS_CACHE
    return (
        pl.scan_parquet(src)
        .select(STAT_COLS)
        .filter((pl.col("position") == "QB")
                & (pl.col("season_type") == "REG")
                & (pl.col("attempts") >= pm.MIN_ATT))
        .collect()
    )


def season_agg(st: pl.DataFrame) -> pl.DataFrame:
    """Per player-season production and efficiency, from finished seasons only.

    Sorted with `maintain_order` because float summation is not associative:
    an unordered group-by makes the ninth decimal of `prev_ypa` wobble between
    runs, which is enough to flip a ridge alpha and move a headline by 0.001 pp.
    """
    return st.sort(["player_id", "season", "week"]).group_by(
        ["player_id", "season"], maintain_order=True).agg(
        pl.col("fantasy_points").mean().alias("ppg"),
        pl.len().alias("g"),
        pl.col("attempts").mean().alias("att_pg"),
        pl.col("attempts").sum().alias("_att"),
        pl.col("passing_yards").sum().alias("_pyd"),
        pl.col("passing_tds").sum().alias("_ptd"),
        pl.col("passing_interceptions").sum().alias("_int"),
        pl.col("sacks_suffered").sum().alias("_sk"),
        pl.col("passing_epa").sum().alias("_epa"),
        (pl.col("passing_cpoe") * pl.col("attempts")).sum().alias("_cpoe_w"),
        pl.col("carries").mean().alias("rush_att_pg"),
        pl.col("rushing_yards").mean().alias("rush_yds_pg"),
        pl.col("rushing_tds").mean().alias("rush_td_pg"),
    ).with_columns(
        (pl.col("_pyd") / pl.col("_att")).alias("ypa"),
        (pl.col("_ptd") / pl.col("_att") * 100).alias("td_rate"),
        (pl.col("_int") / pl.col("_att") * 100).alias("int_rate"),
        (pl.col("_cpoe_w") / pl.col("_att")).alias("cpoe"),
        (pl.col("_epa") / (pl.col("_att") + pl.col("_sk"))).alias("epa_play"),
        (pl.col("_sk") / (pl.col("_att") + pl.col("_sk")) * 100).alias("sack_rate"),
    ).drop("_att", "_pyd", "_ptd", "_int", "_sk", "_epa", "_cpoe_w")


AGG_COLS = ["ppg", "g", "att_pg", "rush_att_pg", "rush_yds_pg", "rush_td_pg",
            "ypa", "td_rate", "int_rate", "cpoe", "epa_play", "sack_rate"]


def team_qb_agg(st: pl.DataFrame) -> pl.DataFrame:
    """Per team-season QB fantasy production: the offensive environment."""
    return st.sort(["team", "season", "week", "player_id"]).group_by(
        ["team", "season"], maintain_order=True).agg(
        pl.col("fantasy_points").sum().alias("_fp"),
        pl.col("week").n_unique().alias("_wk"),
    ).with_columns(
        (pl.col("_fp") / pl.col("_wk")).alias("team_qb_ppg")
    ).select("team", "season", "team_qb_ppg")


def rosters() -> pl.DataFrame:
    """Preseason roster: team, birth date, experience, draft capital."""
    if ROSTER_CACHE.exists():
        r = pl.read_parquet(ROSTER_CACHE)
    else:
        import nflreadpy as nfl
        r = nfl.load_rosters(seasons=list(range(2010, FORWARD_SEASON + 1))).select(
            "season", "team", "position", "gsis_id", "birth_date", "years_exp",
            "entry_year", "rookie_year", "draft_number", "full_name", "week")
        r.write_parquet(ROSTER_CACHE)
    r = r.filter((pl.col("position") == "QB") & pl.col("gsis_id").is_not_null())
    if r.schema["birth_date"] == pl.String:
        r = r.with_columns(pl.col("birth_date").str.to_date(strict=False))
    return r.with_columns(
        pl.col("draft_number").cast(pl.Float64, strict=False).alias("draft_pick_raw"),
        pl.col("gsis_id").alias("player_id"),
    ).unique(subset=["season", "player_id"], keep="first").select(
        "player_id", "season", pl.col("team").alias("team_pre"), "birth_date",
        "years_exp", "entry_year", "draft_pick_raw", "full_name")


def attach_priors(base: pl.DataFrame, sagg: pl.DataFrame, tagg: pl.DataFrame,
                  ros: pl.DataFrame) -> pl.DataFrame:
    """Add every preseason-only input to a frame keyed on (player_id, season).

    Nothing here could have been unknown in August: prior-season aggregates come
    from finished seasons, the roster snapshot is preseason, and birth date,
    entry year and draft pick are fixed before the career starts.
    """
    b = base
    for lag, pre in ((1, "prev"), (2, "prev2"), (3, "prev3")):
        cols = sagg.select(
            "player_id",
            (pl.col("season") + lag).alias("season"),
            *[pl.col(c).alias(f"{pre}_{c}") for c in AGG_COLS],
        )
        b = b.join(cols, on=["player_id", "season"], how="left")

    b = b.join(ros.drop("full_name"), on=["player_id", "season"], how="left")
    # previous season's roster team -> continuity
    prev_team = ros.select("player_id", (pl.col("season") + 1).alias("season"),
                           pl.col("team_pre").alias("team_prev"))
    b = b.join(prev_team, on=["player_id", "season"], how="left")
    # the offensive environment the player is walking into
    tp = tagg.select(pl.col("team").alias("team_pre"),
                     (pl.col("season") + 1).alias("season"),
                     pl.col("team_qb_ppg").alias("team_prev_qb_ppg"))
    b = b.join(tp, on=["team_pre", "season"], how="left")

    w1_, w2_, w3_ = 3.0, 2.0, 1.0

    def wsum(col: str) -> pl.Expr:
        return (
            pl.col(f"prev_{col}").fill_null(0.0) * pl.col("prev_g").fill_null(0.0) * w1_
            + pl.col(f"prev2_{col}").fill_null(0.0) * pl.col("prev2_g").fill_null(0.0) * w2_
            + pl.col(f"prev3_{col}").fill_null(0.0) * pl.col("prev3_g").fill_null(0.0) * w3_
        )

    b = b.with_columns(
        (pl.col("prev_g").fill_null(0.0) * w1_ + pl.col("prev2_g").fill_null(0.0) * w2_
         + pl.col("prev3_g").fill_null(0.0) * w3_).alias("w_games_total")
    ).with_columns(
        *[(wsum(c) / pl.col("w_games_total").clip(lower_bound=1e-9)).alias(f"wavg_{c}")
          for c in ("ppg", "ypa", "td_rate", "epa_play", "rush_yds_pg", "cpoe")]
    )

    return b.with_columns(
        # continuity
        (pl.col("team_pre") == pl.col("team_prev")).cast(pl.Float64)
        .fill_null(0.0).alias("same_team"),
        pl.col("team_prev").is_not_null().cast(pl.Float64).alias("has_prev_team"),
        pl.col("team_prev_qb_ppg").fill_null(0.0).alias("team_prev_qb_ppg_f"),
        pl.col("team_prev_qb_ppg").is_not_null().cast(pl.Float64).alias("has_team_prev"),
        # age / experience / draft capital
        ((pl.date(pl.col("season"), 9, 1) - pl.col("birth_date"))
         .dt.total_days() / 365.25).alias("age_raw"),
        (pl.col("season") - pl.col("entry_year")).cast(pl.Float64).alias("exp_raw"),
        pl.col("draft_pick_raw").fill_null(280.0).alias("draft_pick_f"),
        pl.col("draft_pick_raw").is_null().cast(pl.Float64).alias("undrafted"),
        # presence flags for the multi-season block
        pl.col("prev3_ppg").is_not_null().cast(pl.Float64).alias("has_prev3"),
        pl.col("prev2_ppg").is_not_null().cast(pl.Float64).alias("has_prev2"),
    ).with_columns(
        pl.col("age_raw").fill_null(27.0).alias("age"),
        pl.col("exp_raw").fill_null(0.0).clip(lower_bound=0.0).alias("exp"),
    ).with_columns(
        (pl.col("age") ** 2 / 100).alias("age_sq"),
        (pl.col("exp") ** 2 / 10).alias("exp_sq"),
        (pl.col("exp") < 1).cast(pl.Float64).alias("is_rookie"),
        pl.col("draft_pick_f").log().alias("draft_pick_log"),
        (pl.col("draft_pick_f") <= 32).cast(pl.Float64).alias("first_round"),
        pl.col("age_raw").is_not_null().cast(pl.Float64).alias("has_age"),
    ).with_columns(
        # null-fill every prior-season numeric; the *_has flags carry missingness
        *[pl.col(f"{pre}_{c}").fill_null(0.0).alias(f"{pre}_{c}")
          for pre in ("prev", "prev2", "prev3") for c in AGG_COLS]
    )


# --- proxy ladder ---------------------------------------------------------- #

F_QB016 = w1.CONSENSUS_FEATURES          # the spec under attack, verbatim
EFF = ["prev_ypa", "prev_td_rate", "prev_int_rate", "prev_cpoe", "prev_epa_play",
       "prev_sack_rate", "prev_att_pg", "prev_rush_att_pg", "prev_rush_yds_pg",
       "prev_rush_td_pg", "prev2_ypa", "prev2_td_rate", "prev2_epa_play",
       "prev2_att_pg", "prev2_rush_yds_pg", "has_prev2"]
CONT = ["same_team", "has_prev_team", "team_prev_qb_ppg_f", "has_team_prev"]
AGEX = ["age", "age_sq", "exp", "exp_sq", "is_rookie", "draft_pick_log",
        "first_round", "undrafted", "has_age"]
MULTI = ["wavg_ppg", "wavg_ypa", "wavg_td_rate", "wavg_epa_play",
         "wavg_rush_yds_pg", "wavg_cpoe", "w_games_total", "prev3_ppg",
         "prev3_g", "has_prev3"]

LADDER: dict[str, dict] = {
    "P0_prev_ppg_only": {"feats": ["prev_ppg", "has_prev"], "kind": "fit"},
    "P1_adp_only": {"feats": ["log_adp", "adp_qb_rank", "is_drafted"], "kind": "fit"},
    "P2_qb016": {"feats": F_QB016, "kind": "fit"},
    "P3_plus_efficiency": {"feats": F_QB016 + EFF, "kind": "fit"},
    "P4_plus_continuity_age": {"feats": F_QB016 + EFF + CONT + AGEX, "kind": "fit"},
    "P5_plus_multiseason": {"feats": F_QB016 + EFF + CONT + AGEX + MULTI, "kind": "fit"},
    # deliberately-labelled hindsight ceilings, attack 2
    "H1_hindsight_fit": {"feats": F_QB016 + EFF + CONT + AGEX + MULTI,
                         "kind": "hindsight_fit"},
    "H2_oracle_loo": {"feats": None, "kind": "oracle_loo"},
    "H2b_oracle_loo_shrunk": {"feats": None, "kind": "oracle_loo_shrunk"},
    "H3_oracle_full": {"feats": None, "kind": "oracle_full"},
}
HINDSIGHT = {"H1_hindsight_fit", "H2_oracle_loo", "H2b_oracle_loo_shrunk",
             "H3_oracle_full"}


def build_panel(seasons: list[int], test_seasons: list[int]) -> pl.DataFrame:
    """QB-016's panel, plus the richer preseason inputs and the oracle columns."""
    w1.SEASONS = seasons
    w1.TEST_SEASONS = test_seasons
    w1.load_adp = _load_adp_ext
    p, _ = w1.build_preseason_panel()

    lo = min(seasons) - 4
    st = qb_stats(2010, 2025)
    sagg = season_agg(st.filter(pl.col("season") >= lo))
    tagg = team_qb_agg(st.filter(pl.col("season") >= lo))
    ros = rosters()

    keys = p.select("player_id", "season").unique()
    extra = attach_priors(keys, sagg, tagg, ros)
    keep = ["player_id", "season"] + sorted(
        set(EFF + CONT + AGEX + MULTI) - set(p.columns))
    p = p.join(extra.select([c for c in keep if c in extra.columns]),
               on=["player_id", "season"], how="left")
    # a handful of panel rows have no roster entry; flags already encode that
    p = p.with_columns([pl.col(c).fill_null(0.0) for c in keep[2:]])

    # hindsight oracles: realised season mean, with and without the graded week
    p = p.with_columns(
        pl.col("fantasy_points").mean().over(["player_id", "season"]).alias("_sm"),
        pl.len().over(["player_id", "season"]).cast(pl.Float64).alias("_sn"),
        pl.col("fantasy_points").sum().over(["player_id", "season"]).alias("_ss"),
        pl.col("fantasy_points").mean().over("season").alias("_lg"),
    ).with_columns(
        pl.col("_sm").alias("oracle_full"),
        pl.col("_sn").alias("season_starts"),
        pl.when(pl.col("_sn") > 1)
        .then((pl.col("_ss") - pl.col("fantasy_points")) / (pl.col("_sn") - 1))
        .otherwise(pl.col("_sm")).alias("oracle_loo"),
    ).with_columns(
        # a shrunk oracle is a BETTER season projection than the raw LOO mean
        # when a quarterback started few games: it removes estimation noise, so
        # it raises the ceiling the market has to clear.
        (((pl.col("_sn") - 1).clip(lower_bound=0) * pl.col("oracle_loo")
          + ORACLE_K * pl.col("_lg"))
         / ((pl.col("_sn") - 1).clip(lower_bound=0) + ORACLE_K))
        .alias("oracle_loo_shrunk")
    ).drop("_sm", "_sn", "_ss", "_lg")
    # Joins do not guarantee row order, and least-squares solutions differ in
    # the last bits when the design matrix rows are permuted. Pin the order so
    # every run is bit-reproducible.
    return p.sort(["season", "week", "player_id"])


def _load_adp_ext(seasons: list[int]) -> pl.DataFrame:
    """ADP back to 2010 so the pre-registration holdout has real draft prices."""
    if ADP_CACHE.exists():
        return pl.read_parquet(ADP_CACHE)
    frames = []
    for s in range(2010, FORWARD_SEASON + 1):
        d = markets.load_ffc_adp(s)
        frames.append(d.select(
            "season", "key", "position", pl.col("name").alias("adp_name"),
            pl.col("adp").cast(pl.Float64),
            pl.col("stdev").cast(pl.Float64).alias("adp_stdev"),
            pl.col("times_drafted").cast(pl.Int64),
            "adp_window_start", "adp_window_end"))
    out = pl.concat(frames)
    out.write_parquet(ADP_CACHE)
    return out


# =========================================================================== #
# Engine: proxies, second-stage models, pairs                                 #
# =========================================================================== #

def fit_proxy(p: pl.DataFrame, spec: dict, s: int) -> np.ndarray | None:
    """Proxy value for every row of `p` with season <= s.

    kind 'fit'            : ridge trained on seasons < s (walk-forward, honest)
    kind 'hindsight_fit'  : ridge REFIT on each season in-sample  -- CHEATING
    kind 'oracle_*'       : realised season mean                  -- CHEATING
    """
    sub = p.filter(pl.col("season") <= s)
    kind = spec["kind"]
    if kind.startswith("oracle"):
        return sub[kind].to_numpy().astype(float)
    if kind == "hindsight_fit":
        out = np.empty(sub.height, float)
        seas = sub["season"].to_numpy()
        for t in np.unique(seas):
            m = seas == t
            tr = sub.filter(pl.col("season") == t)
            pred, _ = w1._ridge(tr, tr, spec["feats"])
            out[m] = pred
        return out
    tr = p.filter(pl.col("season") < s)
    if tr.height < 300:
        return None
    pred, _ = w1._ridge(tr, sub, spec["feats"])
    return pred


def pairs_for_week(y, preds, gaps, flags, idx):
    """Concordance for every decided within-week pair. QB-016's rule, verbatim."""
    n = len(idx)
    if n < 2:
        return None
    i, j = np.triu_indices(n, k=1)
    ii, jj = idx[i], idx[j]
    dy = y[ii] - y[jj]
    live = dy != 0
    if not live.any():
        return None
    ii, jj, dy = ii[live], jj[live], dy[live]
    rec = {"i": ii, "j": jj,
           "flag": {k: (f[ii] & f[jj]) for k, f in flags.items()},
           "agree": {}, "gap": {}}
    for name, pr in preds.items():
        dp = pr[ii] - pr[jj]
        rec["agree"][name] = np.where(
            dp == 0, 0.5, (np.sign(dp) == np.sign(dy)).astype(float))
    for name, g in gaps.items():
        rec["gap"][name] = np.abs(g[ii] - g[jj])
    return rec


class PairStore:
    """Pair-level records for every test season, kept flat for fast bootstraps."""

    def __init__(self):
        self.season, self.week, self.pi, self.pj = [], [], [], []
        self.flag: dict[str, list] = {}
        self.agree: dict[str, list] = {}
        self.gap: dict[str, list] = {}

    def add(self, season: int, week: int, rec: dict, pid: np.ndarray):
        n = len(rec["i"])
        self.season.append(np.full(n, season, np.int32))
        self.week.append(np.full(n, week, np.int32))
        self.pi.append(pid[rec["i"]])
        self.pj.append(pid[rec["j"]])
        for k, v in rec["flag"].items():
            self.flag.setdefault(k, []).append(v)
        for k, v in rec["agree"].items():
            self.agree.setdefault(k, []).append(v)
        for k, v in rec["gap"].items():
            self.gap.setdefault(k, []).append(v)

    def finalize(self) -> "PairStore":
        self.season = np.concatenate(self.season)
        self.week = np.concatenate(self.week)
        self.pi = np.concatenate(self.pi)
        self.pj = np.concatenate(self.pj)
        self.flag = {k: np.concatenate(v) for k, v in self.flag.items()}
        self.agree = {k: np.concatenate(v) for k, v in self.agree.items()}
        self.gap = {k: np.concatenate(v) for k, v in self.gap.items()}
        self.block = self.season.astype(np.int64) * 100 + self.week
        return self

    # --- slices ---
    def mask(self, sl: str = "all", gap_key: str | None = None,
             seasons=None, weeks=None) -> np.ndarray:
        m = np.ones(len(self.season), bool)
        if sl == "close":
            m &= self.gap[gap_key] < CLOSE_PTS
        elif sl in self.flag:
            m &= self.flag[sl]
        elif sl != "all":
            raise KeyError(f"unknown slice {sl}")
        if seasons is not None:
            m &= np.isin(self.season, list(seasons))
        if weeks is not None:
            m &= np.isin(self.week, list(weeks))
        return m

    def acc(self, model: str, m: np.ndarray) -> float:
        return float(self.agree[model][m].mean() * 100) if m.any() else float("nan")

    def delta(self, base: str, aug: str, m: np.ndarray) -> float:
        if not m.any():
            return float("nan")
        return float((self.agree[aug][m] - self.agree[base][m]).mean() * 100)

    def per_season_delta(self, base: str, aug: str, m: np.ndarray,
                         seasons: list[int]) -> np.ndarray:
        out = []
        for s in seasons:
            ms = m & (self.season == s)
            out.append(self.delta(base, aug, ms))
        return np.array(out, float)


def se_t(vals: np.ndarray) -> tuple[float, float, float]:
    """Mean, SE and t across seasons -- QB-016's uncertainty, reproduced."""
    v = vals[~np.isnan(vals)]
    if len(v) < 2:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return m, se, (m / se if se else float("nan"))


def boot_blocks(P: PairStore, base: str, aug: str, m: np.ndarray,
                key: np.ndarray, b: int = N_BOOT) -> tuple[float, float, float]:
    """Cluster bootstrap over an arbitrary cluster key (season*100+week, season)."""
    d = P.agree[aug][m] - P.agree[base][m]
    k = key[m]
    uk, inv = np.unique(k, return_inverse=True)
    nk = len(uk)
    sums = np.bincount(inv, weights=d, minlength=nk)
    cnts = np.bincount(inv, minlength=nk).astype(float)
    if cnts.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.integers(0, nk, size=(b, nk))
    num = sums[idx].sum(axis=1)
    den = cnts[idx].sum(axis=1)
    reps = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan) * 100
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(np.nanstd(reps, ddof=1)), float(lo), float(hi)


def boot_players(P: PairStore, base: str, aug: str, m: np.ndarray,
                 b: int = 1000) -> tuple[float, float, float]:
    """Cluster bootstrap over PLAYERS.

    A pair belongs to two clusters at once, so the multiway (pigeonhole)
    bootstrap is used: resample players with replacement and weight each pair by
    m_i * m_j, the product of its two players' multiplicities.
    """
    d = P.agree[aug][m] - P.agree[base][m]
    pi, pj = P.pi[m], P.pj[m]
    ids = np.unique(np.concatenate([pi, pj]))
    remap = np.full(ids.max() + 1, -1, np.int64)
    remap[ids] = np.arange(len(ids))
    a, c = remap[pi], remap[pj]
    n = len(ids)
    reps = np.empty(b)
    for r in range(b):
        mult = RNG.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        w = mult[a] * mult[c]
        sw = w.sum()
        reps[r] = (w @ d) / sw * 100 if sw > 0 else np.nan
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(np.nanstd(reps, ddof=1)), float(lo), float(hi)


def jackknife_seasons(P: PairStore, base: str, aug: str, m: np.ndarray,
                      seasons: list[int]) -> tuple[float, float]:
    """Leave-one-season-out: how much does any single season carry the result?"""
    vals = []
    for s in seasons:
        ms = m & (P.season != s)
        vals.append(P.delta(base, aug, ms))
    v = np.array(vals)
    return float(v.min()), float(v.max())


# =========================================================================== #
# Attack 1 + 2: the proxy-strength ladder, honest rungs and hindsight ceiling  #
# =========================================================================== #

def models_for(proxy: str, extended: bool) -> dict[str, list[str]]:
    """Second-stage feature sets. `_p` is the proxy column for this rung."""
    m = {
        f"{proxy}|base": ["_p"],
        f"{proxy}|market": ["_p"] + MARKET,
        f"{proxy}|form": ["_p"] + FORM,
        f"{proxy}|form+market": ["_p"] + FORM + MARKET,
    }
    if extended:
        m.update({
            f"{proxy}|itt": ["_p", "implied_team_total"],
            f"{proxy}|total": ["_p", "total_line"],
            f"{proxy}|spread_abs": ["_p", "spread_abs"],
            f"{proxy}|team_spread": ["_p", "team_spread"],
            f"{proxy}|opp_total": ["_p", "implied_opp_total"],
            f"{proxy}|itt+total": ["_p", "implied_team_total", "total_line"],
            f"{proxy}|ctx": ["_p"] + CONTEXT,
            f"{proxy}|ctx+market": ["_p"] + CONTEXT + MARKET,
            f"{proxy}|form+itt": ["_p"] + FORM + ["implied_team_total"],
            f"{proxy}|market_only": list(MARKET),
            f"{proxy}|form_only": ["form_fp_mean", "career_fp_mean"] + FORM[1:],
            f"{proxy}|form_only+market": (["form_fp_mean", "career_fp_mean"]
                                         + FORM[1:] + MARKET),
        })
    return m


def run_ladder(p_all: pl.DataFrame, test_seasons: list[int],
               ladder: dict) -> tuple[PairStore, pl.DataFrame, dict]:
    """One walk-forward pass producing every proxy and every second-stage model."""
    P = PairStore()
    err_rows, proxy_rows, coef_rows = [], [], []
    pid_map: dict[str, int] = {}

    for s in test_seasons:
        sub = p_all.filter(pl.col("season") <= s)
        proxies: dict[str, np.ndarray] = {}
        for name, spec in ladder.items():
            v = fit_proxy(p_all, spec, s)
            if v is None:
                break
            proxies[name] = v
        if len(proxies) != len(ladder):
            continue

        seas = sub["season"].to_numpy()
        is_test = seas == s
        test = sub.filter(pl.col("season") == s)
        train_mask = seas < s
        y = test["fantasy_points"].to_numpy().astype(float)

        preds, gaps = {}, {}
        for name in ladder:
            v = proxies[name]
            tr = sub.filter(pl.Series(train_mask)).with_columns(
                pl.Series("_p", v[train_mask]))
            te = test.with_columns(pl.Series("_p", v[is_test]))
            # quality of the proxy itself, on the test season
            proxy_rows.append({
                "season": s, "proxy": name,
                "proxy_mae": float(np.abs(v[is_test] - y).mean()),
                "proxy_rmse": float(np.sqrt(((v[is_test] - y) ** 2).mean())),
                "proxy_corr": float(np.corrcoef(v[is_test], y)[0, 1]),
            })
            ext = name == "P2_qb016"
            for mname, feats in models_for(name, ext).items():
                pr, coef = w1._ridge(tr, te, feats)
                preds[mname] = pr
                e = pr - y
                err_rows.append({"season": s, "model": mname, "n": len(y),
                                 "mae": float(np.abs(e).mean()),
                                 "rmse": float(np.sqrt((e ** 2).mean()))})
                if mname.endswith("|market"):
                    coef_rows.append({"season": s, "proxy": name,
                                      **{k: float(v) for k, v in coef.items()}})
            gaps[name] = preds[f"{name}|base"]

        flags = {
            "drafted": test["is_drafted"].to_numpy() > 0.5,
            "starts10": test["season_starts"].to_numpy() >= MANY_STARTS,
        }
        weeks = test["week"].to_numpy()
        raw_pid = test["player_id"].to_list()
        pid = np.array([pid_map.setdefault(q, len(pid_map)) for q in raw_pid],
                       np.int64)
        for w in np.unique(weeks):
            idx = np.flatnonzero(weeks == w)
            rec = pairs_for_week(y, preds, gaps, flags, idx)
            if rec:
                P.add(s, int(w), rec, pid)

    return (P.finalize(), pl.DataFrame(err_rows),
            {"proxy": pl.DataFrame(proxy_rows),
             "coefs": pl.DataFrame(coef_rows) if coef_rows else pl.DataFrame()})


def proxy_oos_quality(p_all: pl.DataFrame, test_seasons: list[int],
                      ladder: dict) -> pl.DataFrame:
    """Player-season rank agreement with realised outcomes, out of sample."""
    from scipy.stats import spearmanr
    rows = []
    for s in test_seasons:
        sub = p_all.filter(pl.col("season") <= s)
        seas = sub["season"].to_numpy()
        is_test = seas == s
        test = sub.filter(pl.col("season") == s)
        for name, spec in ladder.items():
            v = fit_proxy(p_all, spec, s)
            if v is None:
                continue
            t = test.with_columns(pl.Series("_p", v[is_test]))
            ps = t.sort(["player_id", "week"]).group_by(
                "player_id", maintain_order=True).agg(
                pl.col("fantasy_points").mean().alias("actual_ppg"),
                pl.col("_p").mean().alias("proj"),
                pl.len().alias("games"),
            ).filter(pl.col("games") >= 4)
            a = ps["actual_ppg"].to_numpy().astype(float)
            q = ps["proj"].to_numpy().astype(float)
            rows.append({"season": s, "proxy": name, "n_qbs": ps.height,
                         "pearson_r": float(np.corrcoef(q, a)[0, 1]),
                         "spearman_r": float(spearmanr(q, a).statistic)})
    return pl.DataFrame(rows)


# =========================================================================== #
# CBS 2026 validation of every honest rung                                    #
# =========================================================================== #

def cbs_board() -> pl.DataFrame:
    if CBS_CACHE.exists():
        return pl.read_parquet(CBS_CACHE)
    d = w1.cbs_rescored(FORWARD_SEASON).with_columns(
        pl.col("player").map_elements(markets.normalise_name,
                                      return_dtype=pl.String).alias("key"))
    d.write_parquet(CBS_CACHE)
    return d


def validate_against_cbs(p_all: pl.DataFrame, ladder: dict) -> pl.DataFrame:
    """The single direct check available: proxy vs the real CBS 2026 board."""
    from scipy.stats import spearmanr

    cbs = cbs_board()
    adp = _load_adp_ext([]).filter(
        (pl.col("position") == "QB") & (pl.col("season") == FORWARD_SEASON)
    ).with_columns(pl.col("adp").rank("min").alias("adp_qb_rank_raw"))

    st = qb_stats(2010, 2025)
    sagg = season_agg(st)
    tagg = team_qb_agg(st)
    ros = rosters()

    ids = (p_all.select("player_id", "key").unique(subset=["key"], keep="last"))
    uni = (cbs.select("key", "player", "team", "cbs_ppg_league")
           .join(adp.select("key", "adp", "adp_qb_rank_raw"), on="key", how="left")
           .join(ids, on="key", how="left")
           .with_columns(pl.lit(FORWARD_SEASON).alias("season")))
    # anyone the panel never saw is matched through the 2026 roster by name
    r26 = ros.filter(pl.col("season") == FORWARD_SEASON).with_columns(
        pl.col("full_name").map_elements(markets.normalise_name,
                                         return_dtype=pl.String).alias("key"))
    uni = uni.join(r26.select("key", pl.col("player_id").alias("pid2")).unique(
        subset=["key"], keep="first"), on="key", how="left")
    uni = uni.with_columns(
        pl.coalesce("player_id", "pid2").alias("player_id")).drop("pid2")

    uni = attach_priors(uni, sagg, tagg, ros)
    uni = w1._finish_features(
        uni.with_columns(
            pl.when(pl.col("prev_g") > 0).then(pl.col("prev_ppg"))
            .otherwise(None).alias("prev_ppg_raw"),
            pl.when(pl.col("prev_g") > 0).then(pl.col("prev_g"))
            .otherwise(None).alias("prev_games_raw"),
            pl.when(pl.col("prev2_g") > 0).then(pl.col("prev2_ppg"))
            .otherwise(None).alias("prev2_ppg_raw"),
            pl.when(pl.col("prev2_g") > 0).then(pl.col("prev2_g"))
            .otherwise(None).alias("prev2_games_raw"),
        ))
    uni = uni.with_columns(
        [pl.col(c).fill_null(0.0) for c in set(EFF + CONT + AGEX + MULTI)
         if c in uni.columns])

    train = p_all.filter(pl.col("season") < FORWARD_SEASON)
    board = uni.filter(pl.col("cbs_ppg_league") > 0)
    rows = []
    # Is the r = 0.867 validation really about the projection, or is it just
    # ADP agreement? Correlate the CBS board with its raw ingredients.
    d0 = board.drop_nulls(subset=["adp"])
    cbd = d0["cbs_ppg_league"].to_numpy().astype(float)
    for lab, v in (("-log(ADP)", -np.log(d0["adp"].to_numpy().astype(float))),
                   ("-ADP QB rank",
                    -d0["adp_qb_rank_raw"].to_numpy().astype(float)),
                   ("prior-season PPG", d0["prev_ppg"].to_numpy().astype(float))):
        rows.append({"proxy": f"[raw ingredient] {lab}", "n_features": 1,
                     "n_board": 0, "cbs_pearson": float("nan"),
                     "cbs_spearman": float("nan"), "n_drafted": d0.height,
                     "cbs_pearson_drafted": float(np.corrcoef(v, cbd)[0, 1]),
                     "cbs_spearman_drafted": float(spearmanr(v, cbd).statistic),
                     "proxy_sd": float("nan"), "cbs_sd": float(cbd.std(ddof=1))})
    for name, spec in ladder.items():
        if spec["kind"] != "fit":
            continue
        pred, _ = w1._ridge(train, board, spec["feats"])
        b = board.with_columns(pl.Series("proxy", pred))
        cb = b["cbs_ppg_league"].to_numpy().astype(float)
        px = b["proxy"].to_numpy().astype(float)
        d = b.drop_nulls(subset=["adp"])
        pd_, cd = (d["proxy"].to_numpy().astype(float),
                   d["cbs_ppg_league"].to_numpy().astype(float))
        rows.append({
            "proxy": name, "n_features": len(spec["feats"]),
            "n_board": b.height,
            "cbs_pearson": float(np.corrcoef(px, cb)[0, 1]),
            "cbs_spearman": float(spearmanr(px, cb).statistic),
            "n_drafted": d.height,
            "cbs_pearson_drafted": float(np.corrcoef(pd_, cd)[0, 1]),
            "cbs_spearman_drafted": float(spearmanr(pd_, cd).statistic),
            "proxy_sd": float(px.std(ddof=1)), "cbs_sd": float(cb.std(ddof=1)),
        })
    return pl.DataFrame(rows)


# =========================================================================== #
# main                                                                        #
# =========================================================================== #

def main() -> int:
    OUT.mkdir(exist_ok=True)
    print(BAR)
    print(" QB-021  SKEPTIC PASS ON THE QB-016 MARKET INCREMENT")
    print(BAR)

    p_all = build_panel(SEASONS, TEST_SEASONS)
    print(f"panel rows {p_all.height:,}  seasons {SEASONS[0]}-{SEASONS[-1]}  "
          f"test {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}")

    # ---------------------------------------------------------------- ladder
    print(f"\n{BAR}\n ATTACK 1 + 2  PROXY-STRENGTH LADDER AND HINDSIGHT CEILING\n{BAR}")
    P, err, aux = run_ladder(p_all, TEST_SEASONS, LADDER)
    print(f"decided within-week pairs: {len(P.season):,}")
    err.write_csv(OUT / "qb021_ladder_error_metrics.csv")

    oos = proxy_oos_quality(p_all, TEST_SEASONS, LADDER)
    oos.write_csv(OUT / "qb021_proxy_oos_quality.csv")
    oos_m = oos.group_by("proxy").agg(
        pl.col("pearson_r").mean().alias("oos_pearson"),
        pl.col("spearman_r").mean().alias("oos_spearman")).to_dicts()
    oos_map = {r["proxy"]: r for r in oos_m}

    try:
        cbs = validate_against_cbs(p_all, LADDER)
        cbs.write_csv(OUT / "qb021_cbs_proxy_validation.csv")
        cbs_map = {r["proxy"]: r for r in cbs.to_dicts()}
        print("\nProxy vs the REAL CBS 2026 board (the one direct check available):")
        print(cbs.select("proxy", "n_features", "cbs_pearson", "cbs_spearman",
                         "cbs_pearson_drafted", "cbs_spearman_drafted", "proxy_sd"))
    except Exception as e:
        print(f"CBS validation unavailable: {type(e).__name__}: {e}")
        cbs_map = {}

    mae_map = {r["proxy"]: r["proxy_mae"] for r in
               aux["proxy"].group_by("proxy").agg(
                   pl.col("proxy_mae").mean()).to_dicts()}

    rows = []
    for name in LADDER:
        base, aug = f"{name}|base", f"{name}|market"
        m_all = P.mask("all")
        m_close = P.mask("close", gap_key=name)
        m_close_fix = P.mask("close", gap_key="P2_qb016")
        m_dr = P.mask("drafted")
        d_all = P.delta(base, aug, m_all)
        ps = P.per_season_delta(base, aug, m_all, TEST_SEASONS)
        _, se, t = se_t(ps)
        bse, blo, bhi = boot_blocks(P, base, aug, m_all, P.block)
        r = {
            "proxy": name,
            "hindsight": name in HINDSIGHT,
            "n_features": (len(LADDER[name]["feats"])
                           if LADDER[name]["feats"] else 1),
            "proxy_mae": mae_map.get(name, float("nan")),
            "oos_pearson": oos_map.get(name, {}).get("oos_pearson", float("nan")),
            "oos_spearman": oos_map.get(name, {}).get("oos_spearman", float("nan")),
            "cbs_pearson": cbs_map.get(name, {}).get("cbs_pearson", float("nan")),
            "cbs_spearman": cbs_map.get(name, {}).get("cbs_spearman", float("nan")),
            "base_acc_all": P.acc(base, m_all),
            "market_acc_all": P.acc(aug, m_all),
            "delta_all_pp": d_all,
            "se_seasons": se, "t_seasons": t,
            "boot_block_se": bse, "boot_lo": blo, "boot_hi": bhi,
            "seasons_positive": int((ps > 0).sum()),
            "base_acc_close_own": P.acc(base, m_close),
            "delta_close_own_pp": P.delta(base, aug, m_close),
            "delta_close_fixedset_pp": P.delta(base, aug, m_close_fix),
            "base_acc_drafted": P.acc(base, m_dr),
            "delta_drafted_pp": P.delta(base, aug, m_dr),
            # market on top of proxy + form, the harder baseline
            "base_form_acc": P.acc(f"{name}|form", m_all),
            "delta_over_form_pp": P.delta(f"{name}|form", f"{name}|form+market",
                                          m_all),
        }
        pf = P.per_season_delta(f"{name}|form", f"{name}|form+market", m_all,
                                TEST_SEASONS)
        _, se2, t2 = se_t(pf)
        r["se_over_form"], r["t_over_form"] = se2, t2
        rows.append(r)
    ladder_tab = pl.DataFrame(rows)
    ladder_tab.write_csv(OUT / "qb021_proxy_ladder.csv")

    print("\nIncrement as a function of proxy strength "
          "(H* rungs are DELIBERATE HINDSIGHT, upper bounds only):")
    print(ladder_tab.select(
        "proxy", "hindsight", "proxy_mae", "oos_spearman", "cbs_spearman",
        "base_acc_all", "delta_all_pp", "se_seasons", "t_seasons",
        "seasons_positive"))
    print("\nSame rungs, close calls and the projection+form baseline:")
    print(ladder_tab.select(
        "proxy", "base_acc_close_own", "delta_close_own_pp",
        "delta_close_fixedset_pp", "base_form_acc", "delta_over_form_pp",
        "t_over_form"))

    # relationship between baseline strength and the increment
    hon = ladder_tab.filter(~pl.col("hindsight"))
    x = hon["base_acc_all"].to_numpy()
    yv = hon["delta_all_pp"].to_numpy()
    slope, intercept = np.polyfit(x, yv, 1)
    print(f"\nOLS over the {len(x)} honest rungs: "
          f"delta = {intercept:.2f} {slope:+.3f} * baseline_acc  "
          f"(r = {np.corrcoef(x, yv)[0, 1]:+.3f})")
    ora = ladder_tab.filter(pl.col("proxy") == "H2_oracle_loo").to_dicts()[0]
    print(f"extrapolated to the oracle_loo baseline "
          f"({ora['base_acc_all']:.2f}%): {intercept + slope * ora['base_acc_all']:+.2f} pp; "
          f"actually measured there: {ora['delta_all_pp']:+.2f} pp")

    per_season = []
    for name in LADDER:
        ps = P.per_season_delta(f"{name}|base", f"{name}|market", P.mask("all"),
                                TEST_SEASONS)
        for s, v in zip(TEST_SEASONS, ps):
            per_season.append({"proxy": name, "season": s, "delta_all_pp": v})
    pl.DataFrame(per_season).write_csv(OUT / "qb021_ladder_by_season.csv")
    if aux["coefs"].height:
        aux["coefs"].write_csv(OUT / "qb021_market_coefficients.csv")

    # --- attack 2 detail: is the ceiling result an artefact of oracle noise? --
    print(f"\n{SUB}\n ATTACK 2 DETAIL: the hindsight ceilings, and whether the")
    print(" market only beats them because a leave-one-out season mean is noisy")
    print(SUB)
    ceil_rows = []
    for name in ("P2_qb016", "P5_plus_multiseason", "H1_hindsight_fit",
                 "H2_oracle_loo", "H2b_oracle_loo_shrunk", "H3_oracle_full"):
        base, aug = f"{name}|base", f"{name}|market"
        for sl, lab in (("all", "every pair"),
                        ("starts10", f"both QBs made >= {MANY_STARTS} starts")):
            m = P.mask(sl)
            ps = P.per_season_delta(base, aug, m, TEST_SEASONS)
            _, se, t = se_t(ps)
            ceil_rows.append({
                "proxy": name, "slice": lab, "n_pairs": int(m.sum()),
                "base_acc": P.acc(base, m), "market_acc": P.acc(aug, m),
                "delta_pp": P.delta(base, aug, m), "se_seasons": se, "t": t,
                "seasons_positive": int((ps > 0).sum()),
                "delta_over_form_pp": P.delta(f"{name}|form",
                                              f"{name}|form+market", m)})
    ceil = pl.DataFrame(ceil_rows)
    print(ceil)
    ceil.write_csv(OUT / "qb021_ceiling.csv")

    # ------------------------------------------------------------- attack 4
    print(f"\n{BAR}\n ATTACK 4  REDUNDANCY: IS IT ALL ONE NUMBER?\n{BAR}")
    cols = ["implied_team_total", "total_line", "team_spread", "spread_abs",
            "implied_opp_total"]
    M = p_all.select(cols).to_numpy().astype(float)
    C = np.corrcoef(M, rowvar=False)
    red_rows = [{"a": cols[i], "b": cols[j], "r": float(C[i, j])}
                for i in range(len(cols)) for j in range(i + 1, len(cols))]
    red = pl.DataFrame(red_rows).sort("r", descending=True)
    print(red)
    ident = float(np.abs(p_all["implied_team_total"].to_numpy()
                         - (p_all["total_line"].to_numpy()
                            + p_all["team_spread"].to_numpy()) / 2).max())
    print(f"\nmax |implied_team_total - (total_line + team_spread)/2| = {ident:.2e}"
          "  -> it is an exact algebraic restatement, not a third opinion")
    red.write_csv(OUT / "qb021_market_redundancy.csv")

    sub_rows = []
    m_all = P.mask("all")
    m_close = P.mask("close", gap_key="P2_qb016")
    for lab in ("market", "itt", "total", "spread_abs", "team_spread",
                "opp_total", "itt+total"):
        base, aug = "P2_qb016|base", f"P2_qb016|{lab}"
        ps = P.per_season_delta(base, aug, m_all, TEST_SEASONS)
        _, se, t = se_t(ps)
        bse, blo, bhi = boot_blocks(P, base, aug, m_all, P.block)
        sub_rows.append({"spec": lab, "acc_all": P.acc(aug, m_all),
                         "delta_all_pp": P.delta(base, aug, m_all),
                         "se_seasons": se, "t": t, "boot_se": bse,
                         "boot_lo": blo, "boot_hi": bhi,
                         "delta_close_pp": P.delta(base, aug, m_close),
                         "seasons_positive": int((ps > 0).sum())})
    subs = pl.DataFrame(sub_rows)
    print("\nMarket subsets on top of the QB-016 proxy "
          f"(baseline {P.acc('P2_qb016|base', m_all):.2f}%):")
    print(subs)
    subs.write_csv(OUT / "qb021_market_variable_subsets.csv")

    # ------------------------------------------------------------- attack 6
    print(f"\n{BAR}\n ATTACK 6  PLACEBO: IS IT THE MARKET, OR ANY WEEKLY VARIABLE?\n{BAR}")
    pl_rows = []
    for lab, base, aug in (
        ("non-market weekly context on proxy", "P2_qb016|base", "P2_qb016|ctx"),
        ("market on proxy", "P2_qb016|base", "P2_qb016|market"),
        ("market on proxy + context", "P2_qb016|ctx", "P2_qb016|ctx+market"),
        ("context on proxy + market", "P2_qb016|market", "P2_qb016|ctx+market"),
    ):
        ps = P.per_season_delta(base, aug, m_all, TEST_SEASONS)
        _, se, t = se_t(ps)
        pl_rows.append({"contrast": lab, "base_acc": P.acc(base, m_all),
                        "aug_acc": P.acc(aug, m_all),
                        "delta_pp": P.delta(base, aug, m_all),
                        "se_seasons": se, "t": t,
                        "seasons_positive": int((ps > 0).sum())})
    plac = pl.DataFrame(pl_rows)
    print(plac)
    plac.write_csv(OUT / "qb021_weekly_placebo.csv")

    # ------------------------------------------------------------- attack 5
    print(f"\n{BAR}\n ATTACK 5  UNCERTAINTY UNDER FOUR CLUSTERINGS\n{BAR}")
    unc_rows = []
    targets = [
        ("market on P2 proxy (QB-016 headline)", "P2_qb016|base",
         "P2_qb016|market", "all"),
        ("market on P2 proxy, close calls", "P2_qb016|base",
         "P2_qb016|market", "close"),
        ("market on P2 proxy + form", "P2_qb016|form",
         "P2_qb016|form+market", "all"),
        ("market on STRONGEST honest proxy P5", "P5_plus_multiseason|base",
         "P5_plus_multiseason|market", "all"),
        ("market on hindsight oracle (LOO)", "H2_oracle_loo|base",
         "H2_oracle_loo|market", "all"),
        ("projection on form + market (QB-016 secondary)",
         "P2_qb016|form_only+market", "P2_qb016|form+market", "all"),
    ]
    for lab, base, aug, sl in targets:
        m = P.mask(sl, gap_key="P2_qb016")
        d = P.delta(base, aug, m)
        ps = P.per_season_delta(base, aug, m, TEST_SEASONS)
        mu, se, t = se_t(ps)
        crit = 2.571                                  # t(0.975, 5 df)
        r = {"contrast": lab, "slice": sl, "delta_pp": d,
             "se_seasons": se, "t_seasons": t,
             "ci_seasons_lo": mu - crit * se, "ci_seasons_hi": mu + crit * se}
        for tag, key in (("blockwk", P.block), ("season", P.season)):
            bse, lo, hi = boot_blocks(P, base, aug, m, key)
            r[f"se_{tag}"], r[f"ci_{tag}_lo"], r[f"ci_{tag}_hi"] = bse, lo, hi
        bse, lo, hi = boot_players(P, base, aug, m)
        r["se_player"], r["ci_player_lo"], r["ci_player_hi"] = bse, lo, hi
        jlo, jhi = jackknife_seasons(P, base, aug, m, TEST_SEASONS)
        r["jack_min"], r["jack_max"] = jlo, jhi
        r["excludes_zero_all_clusterings"] = bool(
            (r["ci_seasons_lo"] > 0) and (r["ci_blockwk_lo"] > 0)
            and (r["ci_season_lo"] > 0) and (r["ci_player_lo"] > 0)
        ) if d > 0 else bool(
            (r["ci_seasons_hi"] < 0) and (r["ci_blockwk_hi"] < 0)
            and (r["ci_season_hi"] < 0) and (r["ci_player_hi"] < 0))
        unc_rows.append(r)
    unc = pl.DataFrame(unc_rows)
    print(unc.select("contrast", "delta_pp", "se_seasons", "se_blockwk",
                     "se_season", "se_player", "excludes_zero_all_clusterings"))
    print("\n95% intervals:")
    print(unc.select("contrast", "ci_seasons_lo", "ci_seasons_hi",
                     "ci_blockwk_lo", "ci_blockwk_hi", "ci_season_lo",
                     "ci_season_hi", "ci_player_lo", "ci_player_hi"))
    print("\nleave-one-season-out range of the pooled delta:")
    print(unc.select("contrast", "delta_pp", "jack_min", "jack_max"))
    unc.write_csv(OUT / "qb021_uncertainty.csv")

    del P
    # ------------------------------------------------------------- attack 3
    print(f"\n{BAR}\n ATTACK 3  PRE-REGISTERED WEEK 10-18 CLAIM ON HELD-OUT SEASONS\n{BAR}")
    print("PRE-REGISTRATION, fixed before looking at the held-out data:")
    print("  H1  the market's increment over the QB-016 proxy is LARGER in")
    print("      weeks >= 10 than in weeks 1-9 (one-sided, alpha = 0.05).")
    print("  H2  the market's increment in weeks >= 10 is positive.")
    print("  Test set: seasons 2014-2019, which QB-016 never used. The panel is")
    print("  extended back to 2012 so those seasons have two prior seasons and")
    print("  real preseason ADP. Statistic: per-season delta, SE across the six")
    print("  held-out seasons (5 df), plus a (season, week) cluster bootstrap.")

    ext_panel = build_panel(EXT_SEASONS, EXT_TEST)
    print(f"\nextended panel rows {ext_panel.height:,}  "
          f"seasons {EXT_SEASONS[0]}-{EXT_SEASONS[-1]}")
    ext_ladder = {k: LADDER[k] for k in ("P2_qb016", "P5_plus_multiseason")}
    PE, ext_err, _ = run_ladder(ext_panel, EXT_TEST, ext_ladder)
    ext_err.write_csv(OUT / "qb021_extended_error_metrics.csv")
    print(f"extended decided pairs: {len(PE.season):,}  "
          f"seasons present: {sorted(set(PE.season.tolist()))}")

    early, late = set(range(1, 10)), set(range(10, 19))
    pre_rows = []
    for proxy in ext_ladder:
        base, aug = f"{proxy}|base", f"{proxy}|market"
        for lab, seas in (("HELD-OUT 2014-2019", HELD_OUT),
                          ("generating 2020-2025", GENERATING)):
            seas = [s for s in seas if s in set(PE.season.tolist())]
            m_e = PE.mask("all", seasons=seas, weeks=early)
            m_l = PE.mask("all", seasons=seas, weeks=late)
            d_e, d_l = PE.delta(base, aug, m_e), PE.delta(base, aug, m_l)
            ps_e = PE.per_season_delta(base, aug, m_e, seas)
            ps_l = PE.per_season_delta(base, aug, m_l, seas)
            did = ps_l - ps_e
            mu_d, se_d, t_d = se_t(did)
            mu_l, se_l, t_l = se_t(ps_l)
            bse_l, lo_l, hi_l = boot_blocks(PE, base, aug, m_l, PE.block)
            pre_rows.append({
                "proxy": proxy, "set": lab, "n_seasons": len(seas),
                "base_acc_early": PE.acc(base, m_e),
                "base_acc_late": PE.acc(base, m_l),
                "delta_early_pp": d_e, "delta_late_pp": d_l,
                "did_late_minus_early_pp": d_l - d_e,
                "did_mean_of_seasons": mu_d, "did_se": se_d, "did_t": t_d,
                "late_se": se_l, "late_t": t_l,
                "late_boot_lo": lo_l, "late_boot_hi": hi_l,
                "late_seasons_positive": int((ps_l > 0).sum()),
                "did_seasons_positive": int((did > 0).sum()),
            })
    pre = pl.DataFrame(pre_rows)
    print(pre.select("proxy", "set", "n_seasons", "base_acc_early",
                     "base_acc_late", "delta_early_pp", "delta_late_pp",
                     "did_late_minus_early_pp", "did_se", "did_t",
                     "did_seasons_positive"))
    print("\nH2 (late increment > 0):")
    print(pre.select("proxy", "set", "delta_late_pp", "late_se", "late_t",
                     "late_boot_lo", "late_boot_hi", "late_seasons_positive"))

    # three-bucket view for comparability with QB-016 section 4
    buck = []
    for proxy in ext_ladder:
        base, aug = f"{proxy}|base", f"{proxy}|market"
        for lab, seas in (("HELD-OUT 2014-2019", HELD_OUT),
                          ("generating 2020-2025", GENERATING)):
            for bl, wks in (("1-4", set(range(1, 5))), ("5-9", set(range(5, 10))),
                            ("10-18", set(range(10, 19)))):
                m = PE.mask("all", seasons=seas, weeks=wks)
                buck.append({"proxy": proxy, "set": lab, "bucket": bl,
                             "n_pairs": int(m.sum()),
                             "base_acc": PE.acc(base, m),
                             "market_acc": PE.acc(aug, m),
                             "delta_pp": PE.delta(base, aug, m)})
    bk = pl.DataFrame(buck)
    print("\nby week bucket:")
    print(bk)
    pre.write_csv(OUT / "qb021_prereg_week_late.csv")
    bk.write_csv(OUT / "qb021_prereg_week_buckets.csv")

    # headline increment on the held-out seasons, all weeks -- a bonus
    # out-of-sample replication of the main claim itself
    rep_rows = []
    for proxy in ext_ladder:
        base, aug = f"{proxy}|base", f"{proxy}|market"
        for lab, seas in (("HELD-OUT 2014-2019", HELD_OUT),
                          ("generating 2020-2025", GENERATING),
                          ("all extended 2014-2025", EXT_TEST)):
            m = PE.mask("all", seasons=seas)
            ps = PE.per_season_delta(base, aug, m, seas)
            _, se, t = se_t(ps)
            rep_rows.append({"proxy": proxy, "set": lab,
                             "base_acc": PE.acc(base, m),
                             "market_acc": PE.acc(aug, m),
                             "delta_pp": PE.delta(base, aug, m),
                             "se_seasons": se, "t": t,
                             "seasons_positive": int((ps > 0).sum()),
                             "n_seasons": len(ps)})
    rep = pl.DataFrame(rep_rows)
    print("\nBonus: the headline claim itself, replicated on seasons QB-016 never saw:")
    print(rep)
    rep.write_csv(OUT / "qb021_extended_replication.csv")

    print(f"\n{BAR}\n VERDICT INPUTS\n{BAR}")
    key = ladder_tab.filter(pl.col("proxy") == "P2_qb016").to_dicts()[0]
    strong = ladder_tab.filter(
        pl.col("proxy") == "P5_plus_multiseason").to_dicts()[0]
    print(f"QB-016 proxy  base {key['base_acc_all']:.2f}%  "
          f"market {key['delta_all_pp']:+.2f} pp (t {key['t_seasons']:.2f})")
    print(f"strongest honest proxy  base {strong['base_acc_all']:.2f}%  "
          f"market {strong['delta_all_pp']:+.2f} pp (t {strong['t_seasons']:.2f})")
    print(f"hindsight LOO oracle  base {ora['base_acc_all']:.2f}%  "
          f"market {ora['delta_all_pp']:+.2f} pp")
    print(f"\nwrote CSVs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
