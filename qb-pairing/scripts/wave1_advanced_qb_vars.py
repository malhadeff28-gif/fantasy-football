"""QB-018: do advanced (charting / tracking) QB variables beat box-score form?

The deployed projection uses nine box-score-and-market features and reaches
~64% out-of-sample pairwise start/sit accuracy. Nobody had yet tested whether
charting data -- pressure, blitz, sack conversion, time to throw,
aggressiveness, scrambling, red-zone rushing -- adds anything on top.

This script does the whole test end to end:

  1. probes the real schemas and coverage of PFR advanced stats and Next Gen
     Stats and prints them (they do NOT go back as far as box scores);
  2. builds strictly as-of advanced features using the same shift-then-roll
     discipline as `features/panel.py` -- every rolling statistic is computed
     on `.shift(1)`, so a game can never enter its own feature;
  3. joins them to the existing panel and reports the join rate honestly;
  4. measures the increment walk-forward (train seasons < S, test S) with MAE,
     RMSE and within-week pairwise ranking accuracy, per season and pooled,
     with a block bootstrap over weeks for the deltas;
  5. ranks individual advanced variables by correlation with the baseline
     model's out-of-sample residual;
  6. separates STABLE TRAITS from NOISY OUTCOMES via season-N to season-N+1
     correlation, at both quarterback and team level.

Memory note: this machine has ~1.8 GB free, so play-by-play is never loaded
whole. Each season's parquet is fetched once and read with column pruning
(16 of 372 columns, ~4.5 MB), aggregated immediately, and the per-QB-game
aggregate is cached to the shared processed cache.

Usage:  ./.venv/Scripts/python.exe qb-pairing/scripts/wave1_advanced_qb_vars.py
        [--no-probe]   skip the schema dump
"""
from __future__ import annotations

import os

# OpenBLAS pre-allocates per-thread buffers and fails to load on this machine
# with the default thread count. Must be set before numpy/polars are imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys                                        # noqa: E402
import warnings                                   # noqa: E402
from pathlib import Path                          # noqa: E402

import numpy as np                                # noqa: E402
import polars as pl                               # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)

PROJECT = Path(__file__).resolve().parents[1]      # qb-pairing/
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))

import bootstrap                                  # noqa: E402
bootstrap.setup()
import nfl_data                                   # noqa: E402

from qbpair.features import panel as pm           # noqa: E402
from qbpair.models import projection              # noqa: E402

warnings.filterwarnings("ignore")

BAR = "=" * 78
OUT = PROJECT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED = REPO / "shared" / "nfl-data" / "cache" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2018, 2026))          # panel window; PFR starts 2018
HIST_SEASON = 2017                          # one extra season of history
TEST_SEASONS = list(range(2020, 2026))     # matches the existing backtest
FORM_WINDOW = pm.FORM_WINDOW                # 8 games, same as the baseline
DEF_WINDOW = pm.DEF_WINDOW                  # 17 games, same as the baseline
BOOT = 500
BOOT_SEED = 20260818     # re-seeded per comparison: common random numbers, so
                         # every model is bootstrapped on the same week draws
                         # and a result does not depend on call order

PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
           "play_by_play_{s}.parquet")


# ---------------------------------------------------------------------------
# 1. schema + coverage probe
# ---------------------------------------------------------------------------

def _schema_table(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return pl.DataFrame({
        "source": [name] * df.width,
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "null_pct": [round(df[c].null_count() / max(df.height, 1) * 100, 2)
                     for c in df.columns],
    })


def probe() -> pl.DataFrame:
    """Print the ACTUAL schemas and coverage. Nothing here is assumed."""
    print(BAR)
    print(" 1. SOURCE SCHEMAS AND COVERAGE (probed, not assumed)")
    print(BAR)

    frames = []

    # --- PFR advanced passing ------------------------------------------------
    # The loader validates its own season range up front, so ask it where the
    # wall is rather than guessing. This is the binding coverage constraint.
    for probe_season in (2017, 2026):
        try:
            nfl_data.load_pfr_advstats([probe_season], stat_type="pass")
            print(f"  PFR accepts {probe_season}")
        except ValueError as e:
            print(f"  PFR rejects {probe_season}: {e}")

    pfr = nfl_data.load_pfr_advstats(SEASONS, stat_type="pass")
    print(f"\n  PFR advstats(stat_type='pass'): {pfr.height:,} rows x {pfr.width} cols")
    print(_schema_table(pfr, "pfr_pass"))
    frames.append(_schema_table(pfr, "pfr_pass"))
    print("\n  PFR rows by season / game_type:")
    print(pfr.group_by(["season", "game_type"]).agg(pl.len().alias("rows"))
          .sort(["season", "game_type"]))
    print("  NOTE: this table is one row per PLAYER-game, not per QB-game --"
          " receivers appear\n        with drop stats and zeros elsewhere."
          " It is keyed by pfr_player_id, NOT gsis_id.")

    # --- Next Gen Stats ------------------------------------------------------
    ngs_all = nfl_data.load_nextgen_stats(list(range(2016, 2026)),
                                          stat_type="passing")
    print(f"\n  NGS passing: {ngs_all.height:,} rows x {ngs_all.width} cols")
    print(_schema_table(ngs_all, "ngs_passing"))
    frames.append(_schema_table(ngs_all, "ngs_passing"))
    print("\n  NGS rows by season / season_type (week 0 = season aggregate):")
    print(ngs_all.group_by(["season", "season_type"]).agg(
        pl.len().alias("rows"),
        (pl.col("week") == 0).sum().alias("week0_rows"),
        pl.col("week").max().alias("max_week")).sort(["season", "season_type"]))
    reg = ngs_all.filter((pl.col("season_type") == "REG") & (pl.col("week") >= 1))
    print(f"  NGS weekly REG rows: {reg.height:,};"
          f" min attempts on a row = {reg['attempts'].min()}"
          "  <- NGS publishes only qualified passers")

    return pl.concat(frames)


# ---------------------------------------------------------------------------
# 2. per QB-game source tables
# ---------------------------------------------------------------------------

def crosswalk() -> pl.DataFrame:
    """gsis_id <-> pfr_id. PFR is keyed by pfr_player_id, the panel by gsis."""
    import nflreadpy as nfl

    return (nfl.load_players()
            .select(["gsis_id", "pfr_id", "position"])
            .drop_nulls(subset=["gsis_id", "pfr_id"]))


def qb_box(seasons: list[int]) -> pl.DataFrame:
    """One row per QB-game with the denominators the rate features need."""
    return (nfl_data.load_player_stats(seasons)
            .filter((pl.col("position") == "QB")
                    & (pl.col("season_type") == "REG")
                    & (pl.col("attempts") >= 1))
            .select(["player_id", "season", "week", "team", "opponent_team",
                     "attempts", "sacks_suffered", "carries", "rushing_tds",
                     "fantasy_points"])
            .with_columns(
                # PFR's pressure/blitz denominator is dropbacks = attempts +
                # sacks. Verified on 2018 wk1 Nick Foles: 9 pressures / 0.250
                # = 36 dropbacks = 34 attempts + 2 sacks.
                (pl.col("attempts") + pl.col("sacks_suffered")).alias("dropbacks")))


def pfr_qb_game(seasons: list[int], xwalk: pl.DataFrame) -> pl.DataFrame:
    """PFR charting counts per QB-game, keyed on gsis_id."""
    qb_ids = xwalk.filter(pl.col("position") == "QB").select(["gsis_id", "pfr_id"])
    d = (nfl_data.load_pfr_advstats(seasons, stat_type="pass")
         .filter(pl.col("game_type") == "REG")
         .join(qb_ids, left_on="pfr_player_id", right_on="pfr_id", how="inner"))
    return d.select(
        pl.col("gsis_id").alias("player_id"), "season", "week",
        pl.col("times_pressured").alias("pfr_pressured"),
        pl.col("times_blitzed").alias("pfr_blitzed"),
        pl.col("times_hurried").alias("pfr_hurried"),
        pl.col("times_hit").alias("pfr_hit"),
        pl.col("times_sacked").alias("pfr_sacked"),
        pl.col("passing_bad_throws").alias("pfr_bad_throws"),
        pl.col("passing_drops").alias("pfr_drops"),
    ).unique(subset=["player_id", "season", "week"], keep="first")


def ngs_qb_week(seasons: list[int]) -> pl.DataFrame:
    """NGS weekly tracking metrics per QB-week, keyed on gsis_id."""
    return (nfl_data.load_nextgen_stats(seasons, stat_type="passing")
            .filter((pl.col("season_type") == "REG") & (pl.col("week") >= 1))
            .select(
                pl.col("player_gsis_id").alias("player_id"), "season", "week",
                pl.col("avg_time_to_throw").alias("ngs_ttt"),
                pl.col("aggressiveness").alias("ngs_aggr"),
                pl.col("avg_intended_air_yards").alias("ngs_iay"),
                pl.col("avg_air_yards_differential").alias("ngs_ayd"),
                pl.col("avg_air_yards_to_sticks").alias("ngs_sticks"),
                pl.col("expected_completion_percentage").alias("ngs_xcomp"),
                pl.col("completion_percentage_above_expectation").alias("ngs_cpoe"),
                pl.col("attempts").alias("ngs_attempts"))
            .unique(subset=["player_id", "season", "week"], keep="first"))


def pbp_qb_game(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """QB scrambles and red-zone rushing per QB-game, from play-by-play.

    Play-by-play is 372 columns; loading a season whole is not affordable at
    1.8 GB free. Each season parquet is downloaded once and read with column
    pruning (16 columns, ~4.5 MB), aggregated, then released. The aggregate is
    cached so a rerun costs nothing.
    """
    tag = PROCESSED / f"qb018_pbp_rush_{min(seasons)}_{max(seasons)}.parquet"
    if tag.exists() and not refresh:
        return pl.read_parquet(tag)

    import requests

    cols = ["game_id", "season", "week", "season_type", "posteam", "play_type",
            "rush_attempt", "qb_scramble", "yardline_100", "rusher_player_id",
            "rush_touchdown"]
    raw_dir = PROCESSED / "_pbp_tmp"
    raw_dir.mkdir(exist_ok=True)
    frames_qb, frames_team = [], []

    for s in seasons:
        f = raw_dir / f"pbp_{s}.parquet"
        if not f.exists():
            r = requests.get(PBP_URL.format(s=s), timeout=600,
                             headers={"User-Agent": "qb-pairing-lab/1.0"})
            r.raise_for_status()
            f.write_bytes(r.content)
        p = (pl.read_parquet(f, columns=cols)
             .filter((pl.col("season_type") == "REG")
                     & (pl.col("play_type") == "run")
                     & (pl.col("rush_attempt") == 1)))
        rz = pl.col("yardline_100") <= 20
        gl = pl.col("yardline_100") <= 5
        scr = pl.col("qb_scramble").fill_null(0) == 1

        frames_qb.append(
            p.filter(pl.col("rusher_player_id").is_not_null())
            .group_by(["rusher_player_id", "season", "week"]).agg(
                pl.len().alias("pbp_rush_att"),
                scr.sum().alias("pbp_scrambles"),
                (~scr).sum().alias("pbp_designed_rush"),
                rz.sum().alias("pbp_rz_rush"),
                gl.sum().alias("pbp_gl_rush"),
                (rz & ~scr).sum().alias("pbp_rz_designed"),
                pl.col("rush_touchdown").fill_null(0).sum().alias("pbp_rush_td")))
        frames_team.append(
            p.filter(pl.col("posteam").is_not_null())
            .group_by(["posteam", "season", "week"]).agg(
                rz.sum().alias("team_rz_rush"),
                gl.sum().alias("team_gl_rush"),
                pl.len().alias("team_rush_att")))
        del p

    # Team red-zone totals are the denominator for red-zone rush SHARE; they
    # are cached alongside and attached via the QB's own team in build_qb_game.
    qb = pl.concat(frames_qb).rename({"rusher_player_id": "player_id"})
    team = pl.concat(frames_team).rename({"posteam": "team"})
    team.write_parquet(
        PROCESSED / f"qb018_pbp_team_rz_{min(seasons)}_{max(seasons)}.parquet")
    qb.write_parquet(tag)
    return qb


def pbp_team_rz(seasons: list[int]) -> pl.DataFrame:
    p = PROCESSED / f"qb018_pbp_team_rz_{min(seasons)}_{max(seasons)}.parquet"
    if not p.exists():
        pbp_qb_game(seasons, refresh=True)
    return pl.read_parquet(p)


# ---------------------------------------------------------------------------
# 3. as-of feature construction (shift-then-roll, exactly as panel.py)
# ---------------------------------------------------------------------------

def _prior_sum(expr: pl.Expr, w: int, over: str) -> pl.Expr:
    """Sum of `expr` over the previous `w` rows of `over`. Never the current row."""
    return expr.fill_null(0).shift(1).rolling_sum(window_size=w, min_samples=1).over(over)


def _prior_n(col: str, w: int, over: str) -> pl.Expr:
    """How many of the previous `w` rows actually carried a value for `col`."""
    return (pl.col(col).is_not_null().cast(pl.Int32)
            .shift(1).rolling_sum(window_size=w, min_samples=1).over(over))


def _ratio(num: str, den: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)


# feature blocks -------------------------------------------------------------
PRESSURE = ["adv_pressure_rate", "adv_blitz_rate", "adv_hurry_rate",
            "adv_qbhit_rate", "adv_sack_rate", "adv_press_to_sack",
            "adv_bad_throw_rate"]
NGS_F = ["adv_time_to_throw", "adv_aggressiveness", "adv_intended_air_yards",
         "adv_air_yards_diff", "adv_air_yards_to_sticks", "adv_xcomp",
         "adv_cpoe_ngs"]
RUSH = ["adv_scramble_rate", "adv_rz_rush_pg", "adv_rz_rush_share",
        "adv_gl_rush_pg", "adv_designed_rush_pg"]
OPPDEF = ["opp_def_pressure_rate", "opp_def_blitz_rate", "opp_def_sack_rate"]
ADVANCED = PRESSURE + NGS_F + RUSH + OPPDEF

BASELINE = list(projection.FEATURES)


def build_qb_game(seasons_hist: list[int]) -> pl.DataFrame:
    """Raw per-QB-game charting table (no rolling yet). One row per QB-game."""
    xw = crosswalk()
    box = qb_box(seasons_hist)
    pfr = pfr_qb_game([s for s in seasons_hist if s >= 2018], xw)
    ngs = ngs_qb_week(seasons_hist)
    rush = pbp_qb_game(seasons_hist)
    team_rz = pbp_team_rz(seasons_hist)

    g = (box
         .join(pfr, on=["player_id", "season", "week"], how="left")
         .join(ngs, on=["player_id", "season", "week"], how="left")
         .join(rush, on=["player_id", "season", "week"], how="left")
         .join(team_rz, on=["team", "season", "week"], how="left"))
    return g.with_columns(
        pl.col(["pbp_rush_att", "pbp_scrambles", "pbp_designed_rush",
                "pbp_rz_rush", "pbp_gl_rush", "pbp_rush_td"]).fill_null(0),
        # a dropback that ends in a scramble is still a dropback
        (pl.col("dropbacks") + pl.col("pbp_scrambles").fill_null(0))
        .alias("dropbacks_full"),
    ).sort(["player_id", "season", "week"])


def qb_advanced_features(g: pl.DataFrame) -> pl.DataFrame:
    """As-of rolling advanced features. Every input passes through .shift(1).

    History is restricted to >= MIN_ATT-attempt games, exactly as the baseline
    `_qb_form_panel` does, so a two-throw relief cameo cannot swing a rate.
    """
    w = FORM_WINDOW
    s = g.filter(pl.col("attempts") >= pm.MIN_ATT).sort(
        ["player_id", "season", "week"]).with_columns(
        # --- PFR pressure block, prior-8-game sums --------------------------
        _prior_sum(pl.col("pfr_pressured"), w, "player_id").alias("_p_press"),
        _prior_sum(pl.col("pfr_blitzed"), w, "player_id").alias("_p_blitz"),
        _prior_sum(pl.col("pfr_hurried"), w, "player_id").alias("_p_hurry"),
        _prior_sum(pl.col("pfr_hit"), w, "player_id").alias("_p_hit"),
        _prior_sum(pl.col("pfr_sacked"), w, "player_id").alias("_p_sack"),
        _prior_sum(pl.col("pfr_bad_throws"), w, "player_id").alias("_p_bad"),
        _prior_sum(pl.when(pl.col("pfr_pressured").is_not_null())
                   .then(pl.col("dropbacks")).otherwise(None), w, "player_id")
        .alias("_p_db_pfr"),
        _prior_sum(pl.when(pl.col("pfr_pressured").is_not_null())
                   .then(pl.col("attempts")).otherwise(None), w, "player_id")
        .alias("_p_att_pfr"),
        _prior_n("pfr_pressured", w, "player_id").alias("adv_n_pfr"),
        # --- NGS block, prior-8-game attempt-weighted means ------------------
        _prior_sum(pl.col("ngs_ttt") * pl.col("ngs_attempts"), w, "player_id").alias("_p_ttt"),
        _prior_sum(pl.col("ngs_aggr") * pl.col("ngs_attempts"), w, "player_id").alias("_p_aggr"),
        _prior_sum(pl.col("ngs_iay") * pl.col("ngs_attempts"), w, "player_id").alias("_p_iay"),
        _prior_sum(pl.col("ngs_ayd") * pl.col("ngs_attempts"), w, "player_id").alias("_p_ayd"),
        _prior_sum(pl.col("ngs_sticks") * pl.col("ngs_attempts"), w, "player_id").alias("_p_stk"),
        _prior_sum(pl.col("ngs_xcomp") * pl.col("ngs_attempts"), w, "player_id").alias("_p_xc"),
        _prior_sum(pl.col("ngs_cpoe") * pl.col("ngs_attempts"), w, "player_id").alias("_p_cpoe"),
        _prior_sum(pl.when(pl.col("ngs_ttt").is_not_null())
                   .then(pl.col("ngs_attempts")).otherwise(None), w, "player_id")
        .alias("_p_ngs_att"),
        _prior_n("ngs_ttt", w, "player_id").alias("adv_n_ngs"),
        # --- rushing block ---------------------------------------------------
        _prior_sum(pl.col("pbp_scrambles"), w, "player_id").alias("_p_scr"),
        _prior_sum(pl.col("pbp_rz_rush"), w, "player_id").alias("_p_rz"),
        _prior_sum(pl.col("pbp_gl_rush"), w, "player_id").alias("_p_gl"),
        _prior_sum(pl.col("pbp_designed_rush"), w, "player_id").alias("_p_des"),
        _prior_sum(pl.col("team_rz_rush"), w, "player_id").alias("_p_team_rz"),
        _prior_sum(pl.col("dropbacks_full"), w, "player_id").alias("_p_dbf"),
        pl.col("player_id").is_not_null().cast(pl.Int32).shift(1)
        .rolling_sum(window_size=w, min_samples=1).over("player_id").alias("_p_games"),
    )

    return s.with_columns(
        # pressure faced -- an offensive-line / scheme property
        _ratio("_p_press", "_p_db_pfr").alias("adv_pressure_rate"),
        _ratio("_p_blitz", "_p_db_pfr").alias("adv_blitz_rate"),
        _ratio("_p_hurry", "_p_db_pfr").alias("adv_hurry_rate"),
        _ratio("_p_hit", "_p_db_pfr").alias("adv_qbhit_rate"),
        _ratio("_p_sack", "_p_db_pfr").alias("adv_sack_rate"),
        # pressure-to-sack conversion: given pressure, how often does he go down
        _ratio("_p_sack", "_p_press").alias("adv_press_to_sack"),
        _ratio("_p_bad", "_p_att_pfr").alias("adv_bad_throw_rate"),
        # NGS
        _ratio("_p_ttt", "_p_ngs_att").alias("adv_time_to_throw"),
        _ratio("_p_aggr", "_p_ngs_att").alias("adv_aggressiveness"),
        _ratio("_p_iay", "_p_ngs_att").alias("adv_intended_air_yards"),
        _ratio("_p_ayd", "_p_ngs_att").alias("adv_air_yards_diff"),
        _ratio("_p_stk", "_p_ngs_att").alias("adv_air_yards_to_sticks"),
        _ratio("_p_xc", "_p_ngs_att").alias("adv_xcomp"),
        _ratio("_p_cpoe", "_p_ngs_att").alias("adv_cpoe_ngs"),
        # scrambling and red-zone rushing usage
        _ratio("_p_scr", "_p_dbf").alias("adv_scramble_rate"),
        _ratio("_p_rz", "_p_games").alias("adv_rz_rush_pg"),
        _ratio("_p_rz", "_p_team_rz").alias("adv_rz_rush_share"),
        _ratio("_p_gl", "_p_games").alias("adv_gl_rush_pg"),
        _ratio("_p_des", "_p_games").alias("adv_designed_rush_pg"),
    ).select(["player_id", "season", "week", "adv_n_pfr", "adv_n_ngs"]
             + PRESSURE + NGS_F + RUSH)


def defence_pressure_features(g: pl.DataFrame) -> pl.DataFrame:
    """Opponent defence pressure profile, as-of, from the same PFR charting."""
    tw = (g.filter(pl.col("pfr_pressured").is_not_null())
          .group_by(["opponent_team", "season", "week"]).agg(
              pl.col("pfr_pressured").sum().alias("d_press"),
              pl.col("pfr_blitzed").sum().alias("d_blitz"),
              pl.col("pfr_sacked").sum().alias("d_sack"),
              pl.col("dropbacks").sum().alias("d_db"))
          .sort(["opponent_team", "season", "week"]))
    w = DEF_WINDOW
    tw = tw.with_columns(
        _prior_sum(pl.col("d_press"), w, "opponent_team").alias("_dp"),
        _prior_sum(pl.col("d_blitz"), w, "opponent_team").alias("_db_"),
        _prior_sum(pl.col("d_sack"), w, "opponent_team").alias("_ds"),
        _prior_sum(pl.col("d_db"), w, "opponent_team").alias("_dd"),
    )
    return tw.with_columns(
        _ratio("_dp", "_dd").alias("opp_def_pressure_rate"),
        _ratio("_db_", "_dd").alias("opp_def_blitz_rate"),
        _ratio("_ds", "_dd").alias("opp_def_sack_rate"),
    ).select([pl.col("opponent_team").alias("opponent"), "season", "week"] + OPPDEF)


# ---------------------------------------------------------------------------
# 4. evaluation
# ---------------------------------------------------------------------------

def walk_forward(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Train on seasons < S, predict S. Returns OOS predictions for all S."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    frames = []
    for s in TEST_SEASONS:
        tr = df.filter(pl.col("season") < s)
        te = df.filter(pl.col("season") == s)
        if tr.height < 200 or te.is_empty():
            continue
        y_tr = tr["fantasy_points"].to_numpy().astype(float)
        X_tr = tr.select(feats).to_numpy().astype(float)
        X_te = te.select(feats).to_numpy().astype(float)
        sc = StandardScaler().fit(X_tr)
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X_tr), y_tr)
        frames.append(te.select(["season", "week", "player_id", "fantasy_points"])
                      .with_columns(pl.Series("pred", m.predict(sc.transform(X_te))),
                                    pl.lit(float(y_tr.mean())).alias("_trmean")))
    return pl.concat(frames)


def walk_forward_gbm(df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Same walk-forward with gradient boosting.

    Ridge is the deployed model and the fair comparison, but a linear model
    can only miss a nonlinear signal. If the advanced variables interact with
    form or the market in some way ridge cannot express, a shallow GBM should
    find at least a trace of it. Deliberately small: 4k training rows would
    over-fit anything larger.
    """
    import lightgbm as lgb

    frames = []
    for s in TEST_SEASONS:
        tr = df.filter(pl.col("season") < s)
        te = df.filter(pl.col("season") == s)
        if tr.height < 200 or te.is_empty():
            continue
        y_tr = tr["fantasy_points"].to_numpy().astype(float)
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=7,
            min_child_samples=60, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1, verbose=-1,
            random_state=0)
        m.fit(tr.select(feats).to_numpy().astype(float), y_tr)
        frames.append(te.select(["season", "week", "player_id", "fantasy_points"])
                      .with_columns(
                          pl.Series("pred", m.predict(te.select(feats).to_numpy()
                                                      .astype(float))),
                          pl.lit(float(y_tr.mean())).alias("_trmean")))
    return pl.concat(frames)


def week_pair_counts(pred: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Per (season, week): correct pairwise calls and total comparisons.

    Ties in realised points are skipped, matching `phase2_signal`. Prediction
    ties are impossible with a continuous ridge output and are counted as
    misses, so two models are always scored on an identical comparison set.
    """
    keys, corr, tot = [], [], []
    for (s, wk), g in pred.group_by(["season", "week"], maintain_order=True):
        y = g["fantasy_points"].to_numpy().astype(float)
        p = g["pred"].to_numpy().astype(float)
        n = len(y)
        if n < 2:
            continue
        iu = np.triu_indices(n, 1)
        dy = (y[:, None] - y[None, :])[iu]
        dp = (p[:, None] - p[None, :])[iu]
        m = dy != 0
        keys.append((s, wk))
        tot.append(int(m.sum()))
        corr.append(int((np.sign(dy[m]) == np.sign(dp[m])).sum()))
    return np.array(corr), np.array(tot), keys


def score(pred: pl.DataFrame) -> dict:
    y = pred["fantasy_points"].to_numpy().astype(float)
    p = pred["pred"].to_numpy().astype(float)
    tm = pred["_trmean"].to_numpy().astype(float)
    c, t, _ = week_pair_counts(pred)
    per_season = []
    for s in sorted(pred["season"].unique().to_list()):
        sub = pred.filter(pl.col("season") == s)
        ys = sub["fantasy_points"].to_numpy().astype(float)
        ps = sub["pred"].to_numpy().astype(float)
        cs, ts, _ = week_pair_counts(sub)
        per_season.append({
            "season": s, "n": sub.height,
            "mae": float(np.abs(ys - ps).mean()),
            "rmse": float(np.sqrt(((ys - ps) ** 2).mean())),
            "pair_acc": float(cs.sum() / ts.sum() * 100),
        })
    return {
        "mae": float(np.abs(y - p).mean()),
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "r2": float(1 - ((y - p) ** 2).sum() / ((y - tm) ** 2).sum()),
        "pair_acc": float(c.sum() / t.sum() * 100),
        "pairs": int(t.sum()),
        "per_season": per_season,
    }


def boot_delta(base: pl.DataFrame, aug: pl.DataFrame) -> dict:
    """Block bootstrap over weeks for the paired accuracy and MAE deltas."""
    cb, tb, kb = week_pair_counts(base)
    ca, ta, ka = week_pair_counts(aug)
    assert kb == ka and np.array_equal(tb, ta), "models must share a row set"

    # per-week absolute error sums, for the MAE delta
    def week_ae(df):
        d = (df.with_columns((pl.col("fantasy_points") - pl.col("pred")).abs().alias("ae"))
             .group_by(["season", "week"], maintain_order=True)
             .agg(pl.col("ae").sum().alias("ae"), pl.len().alias("n")))
        return d["ae"].to_numpy(), d["n"].to_numpy()

    aeb, nb = week_ae(base)
    aea, na = week_ae(aug)
    nweeks = len(tb)
    rng = np.random.default_rng(BOOT_SEED)
    acc_d, mae_d = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, nweeks, nweeks)
        acc_d.append(ca[idx].sum() / ta[idx].sum() - cb[idx].sum() / tb[idx].sum())
        mae_d.append(aea[idx].sum() / na[idx].sum() - aeb[idx].sum() / nb[idx].sum())
    acc_d = np.array(acc_d) * 100
    mae_d = np.array(mae_d)
    return {
        "acc_delta": float(ca.sum() / ta.sum() * 100 - cb.sum() / tb.sum() * 100),
        "acc_lo": float(np.percentile(acc_d, 2.5)),
        "acc_hi": float(np.percentile(acc_d, 97.5)),
        "acc_p_better": float((acc_d > 0).mean()),
        "mae_delta": float(aea.sum() / na.sum() - aeb.sum() / nb.sum()),
        "mae_lo": float(np.percentile(mae_d, 2.5)),
        "mae_hi": float(np.percentile(mae_d, 97.5)),
    }


# ---------------------------------------------------------------------------
# 5. stability: trait or outcome?
# ---------------------------------------------------------------------------

# (name, numerator column, denominator column). Season values are RATIOS OF
# SUMS, never means of per-game ratios -- a mean of ratios is dominated by
# small-denominator games and would understate every rate's true stability.
STAB_SPECS = [
    ("pressure_rate_faced", "pfr_pressured", "dropbacks"),
    ("blitz_rate_faced", "pfr_blitzed", "dropbacks"),
    ("hurry_rate_faced", "pfr_hurried", "dropbacks"),
    ("qbhit_rate_faced", "pfr_hit", "dropbacks"),
    ("sack_rate_taken", "pfr_sacked", "dropbacks"),
    ("press_to_sack", "pfr_sacked", "pfr_pressured"),
    ("bad_throw_rate", "pfr_bad_throws", "attempts"),
    ("time_to_throw", "w_ttt", "ngs_den"),
    ("aggressiveness", "w_aggr", "ngs_den"),
    ("intended_air_yards", "w_iay", "ngs_den"),
    ("air_yards_diff", "w_ayd", "ngs_den"),
    ("air_yards_to_sticks", "w_stk", "ngs_den"),
    ("xcomp", "w_xc", "ngs_den"),
    ("cpoe_ngs", "w_cpoe", "ngs_den"),
    ("scramble_rate", "pbp_scrambles", "dropbacks_full"),
    ("rz_rush_per_game", "pbp_rz_rush", "one"),
    ("rz_rush_share", "pbp_rz_rush", "team_rz_rush"),
    ("gl_rush_per_game", "pbp_gl_rush", "one"),
    ("designed_rush_per_game", "pbp_designed_rush", "one"),
    ("fantasy_points_per_game", "fantasy_points", "one"),
]


def stability_inputs(g: pl.DataFrame) -> pl.DataFrame:
    """Attach the attempt-weighted numerators the NGS rates need."""
    return g.with_columns(
        pl.lit(1.0).alias("one"),
        (pl.col("ngs_ttt") * pl.col("ngs_attempts")).alias("w_ttt"),
        (pl.col("ngs_aggr") * pl.col("ngs_attempts")).alias("w_aggr"),
        (pl.col("ngs_iay") * pl.col("ngs_attempts")).alias("w_iay"),
        (pl.col("ngs_ayd") * pl.col("ngs_attempts")).alias("w_ayd"),
        (pl.col("ngs_sticks") * pl.col("ngs_attempts")).alias("w_stk"),
        (pl.col("ngs_xcomp") * pl.col("ngs_attempts")).alias("w_xc"),
        (pl.col("ngs_cpoe") * pl.col("ngs_attempts")).alias("w_cpoe"),
        pl.when(pl.col("ngs_ttt").is_not_null())
        .then(pl.col("ngs_attempts")).otherwise(0.0).alias("ngs_den"),
    )


def season_stability(g: pl.DataFrame, key: str, min_games: int = 8) -> pl.DataFrame:
    """Correlate a season-N value against the same unit's season-N+1 value.

    High r = the variable describes a persistent property of the unit (a TRAIT).
    Low r = last season's value tells you little about next season's (an
    OUTCOME, mostly noise plus circumstance).
    """
    cols = sorted({c for _, n, d in STAB_SPECS for c in (n, d)})
    agg = (g.group_by([key, "season"])
           .agg([pl.col(c).sum().alias(c) for c in cols] + [pl.len().alias("games")])
           .filter(pl.col("games") >= min_games)
           .with_columns([
               pl.when(pl.col(d) > 0).then(pl.col(n) / pl.col(d))
               .otherwise(None).alias(name)
               for name, n, d in STAB_SPECS]))
    nxt = agg.with_columns((pl.col("season") - 1).alias("season"))
    j = agg.join(nxt, on=[key, "season"], how="inner", suffix="_next")
    rows = []
    for name, _, _ in STAB_SPECS:
        sub = j.drop_nulls(subset=[name, f"{name}_next"]).filter(
            pl.col(name).is_finite() & pl.col(f"{name}_next").is_finite())
        if sub.height < 12:
            rows.append({"variable": name, "unit": key, "n_pairs": sub.height,
                         "r_season_to_season": None, "ci_lo": None, "ci_hi": None})
            continue
        a = sub[name].to_numpy().astype(float)
        b = sub[f"{name}_next"].to_numpy().astype(float)
        r = float(np.corrcoef(a, b)[0, 1])
        # Fisher z interval, so a small n cannot masquerade as a firm answer
        z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1 / np.sqrt(sub.height - 3)
        rows.append({"variable": name, "unit": key, "n_pairs": sub.height,
                     "r_season_to_season": r,
                     "ci_lo": float(np.tanh(z - 1.96 * se)),
                     "ci_hi": float(np.tanh(z + 1.96 * se))})
    return pl.DataFrame(rows).sort("r_season_to_season", descending=True,
                                   nulls_last=True)


# ---------------------------------------------------------------------------
# leakage audit: the one failure mode that would invalidate everything
# ---------------------------------------------------------------------------

def audit_no_lookahead(g: pl.DataFrame, feat: pl.DataFrame, n: int = 40) -> int:
    """Recompute one advanced feature by explicit filtering and compare.

    `qb_advanced_features` is a vectorised shift-then-roll. This rebuilds
    `adv_pressure_rate` for a random sample of rows the slow, obvious way --
    take that quarterback's previous FORM_WINDOW games, strictly earlier than
    this one, and divide summed pressures by summed dropbacks -- and asserts
    the two agree. Mirrors `panel.validate_against_pregame()`.
    """
    src = (g.filter((pl.col("attempts") >= pm.MIN_ATT)
                    & pl.col("pfr_pressured").is_not_null())
           .sort(["player_id", "season", "week"]))
    chk = feat.drop_nulls(subset=["adv_pressure_rate"])
    sample = chk.sample(n=min(n, chk.height), seed=7)
    worst, checked = 0.0, 0
    for row in sample.to_dicts():
        prior = src.filter(
            (pl.col("player_id") == row["player_id"])
            & ((pl.col("season") < row["season"])
               | ((pl.col("season") == row["season"]) & (pl.col("week") < row["week"])))
        ).tail(FORM_WINDOW)
        if prior.is_empty():
            continue
        manual = prior["pfr_pressured"].sum() / prior["dropbacks"].sum()
        worst = max(worst, abs(manual - row["adv_pressure_rate"]))
        checked += 1
    print(f"  as-of audit: recomputed adv_pressure_rate for {checked} random "
          f"rows by explicit\n  history filtering; max |vectorised - explicit| "
          f"= {worst:.2e}")
    assert worst < 1e-9, "LOOK-AHEAD OR WINDOW MISMATCH in advanced features"
    return checked


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    do_probe = "--no-probe" not in sys.argv

    if do_probe:
        sch = probe()
        sch.write_csv(OUT / "qb018_source_schemas.csv")
        print(f"\n  wrote outputs/qb018_source_schemas.csv")

    hist = [HIST_SEASON] + SEASONS
    print(f"\n{BAR}\n 2. BUILDING AS-OF ADVANCED FEATURES\n{BAR}")
    g = build_qb_game(hist)
    print(f"  per QB-game charting table: {g.height:,} rows "
          f"({min(hist)}-{max(hist)}, REG, >=1 attempt)")
    for label, col in [("PFR charting", "pfr_pressured"), ("NGS", "ngs_ttt"),
                       ("PBP rushing", "pbp_rush_att")]:
        n = g[col].is_not_null().sum()
        print(f"    {label:<14} present on {n:,}/{g.height:,} "
              f"({n / g.height * 100:.1f}%) QB-games")

    adv = qb_advanced_features(g)
    dfn = defence_pressure_features(g)
    audit_no_lookahead(g, adv)

    panel = pm.build_panel(SEASONS)
    print(f"\n  baseline panel: {panel.height:,} QB-weeks "
          f"({min(SEASONS)}-{max(SEASONS)}, >= {pm.MIN_ATT} attempts)")

    j = (panel.join(adv, on=["player_id", "season", "week"], how="left")
         .join(dfn, on=["opponent", "season", "week"], how="left"))
    assert j.height == panel.height, "join duplicated rows"

    print(f"\n{BAR}\n 3. JOIN RATES (honest)\n{BAR}")
    rows = []
    for c in ADVANCED:
        n = j[c].is_not_null().sum()
        rows.append({"feature": c, "matched": n, "panel_rows": j.height,
                     "join_rate_pct": round(n / j.height * 100, 2)})
    jr = pl.DataFrame(rows)
    print(jr)
    jr.write_csv(OUT / "qb018_join_rates.csv")

    by_season = j.group_by("season").agg(
        pl.len().alias("panel_rows"),
        pl.col("adv_pressure_rate").is_not_null().sum().alias("with_pressure"),
        pl.col("adv_time_to_throw").is_not_null().sum().alias("with_ngs"),
        pl.col("adv_rz_rush_pg").is_not_null().sum().alias("with_rush"),
    ).sort("season")
    print("\n  coverage by season:")
    print(by_season)
    by_season.write_csv(OUT / "qb018_coverage_by_season.csv")

    # analysis row set: baseline + every advanced feature present
    need = BASELINE + ADVANCED
    m = j.drop_nulls(subset=need)
    print(f"\n  rows usable by every model: {m.height:,} / {panel.height:,} "
          f"({m.height / panel.height * 100:.1f}%)")
    lost = panel.height - m.height
    print(f"  {lost:,} rows dropped: debut/near-debut QB-weeks with no prior "
          f"charting history,\n  plus 10-14 attempt games NGS never publishes.")

    # The restricted row set is not free: quote the baseline on the full panel
    # too, so the reader can see how much of any change is the row set alone.
    full = j.drop_nulls(subset=BASELINE)
    full_base = score(walk_forward(full, BASELINE))
    print(f"\n  reference: baseline on the FULL panel ({full.height:,} rows) "
          f"scores {full_base['pair_acc']:.2f}% pair accuracy, "
          f"MAE {full_base['mae']:.4f}")
    print(f"             baseline on the RESTRICTED row set is quoted below; "
          f"any difference\n             is the row set, not the features.")

    # ---- model comparison --------------------------------------------------
    print(f"\n{BAR}\n 4. WALK-FORWARD INCREMENT (train < S, test = S; S = "
          f"{min(TEST_SEASONS)}-{max(TEST_SEASONS)})\n{BAR}")
    MODELS = {
        "baseline (9 box+market)": BASELINE,
        "baseline + pressure": BASELINE + PRESSURE,
        "baseline + NGS tracking": BASELINE + NGS_F,
        "baseline + scramble/RZ rush": BASELINE + RUSH,
        "baseline + opp def pressure": BASELINE + OPPDEF,
        "baseline + ALL advanced": BASELINE + ADVANCED,
        "advanced only": ADVANCED,
    }
    preds, res = {}, {}
    for name, feats in MODELS.items():
        preds[name] = walk_forward(m, feats)
        res[name] = score(preds[name])

    print(f"{'model':<30}{'k':>4}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'pair acc':>11}")
    print("-" * 78)
    b = res["baseline (9 box+market)"]
    for name, feats in MODELS.items():
        r = res[name]
        d = "" if name.startswith("baseline (") else \
            f"  ({r['mae'] - b['mae']:+.4f} MAE, {r['pair_acc'] - b['pair_acc']:+.2f} pp)"
        print(f"{name:<30}{len(feats):>4}{r['mae']:>8.4f}{r['rmse']:>8.4f}"
              f"{r['r2']:>8.4f}{r['pair_acc']:>10.2f}%{d}")
    print("-" * 78)
    print(f"pairwise comparisons per model: {b['pairs']:,}")

    print(f"\n  PER-SEASON SPREAD (pair accuracy %, so the uncertainty is visible)")
    hdr = f"{'model':<30}" + "".join(f"{s:>8}" for s in TEST_SEASONS) + f"{'mean':>8}{'sd':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        vals = [x["pair_acc"] for x in res[name]["per_season"]]
        print(f"{name:<30}" + "".join(f"{v:>8.2f}" for v in vals)
              + f"{np.mean(vals):>8.2f}{np.std(vals, ddof=1):>7.2f}")

    print(f"\n  PER-SEASON SPREAD (MAE)")
    print(hdr)
    print("-" * len(hdr))
    for name in MODELS:
        vals = [x["mae"] for x in res[name]["per_season"]]
        print(f"{name:<30}" + "".join(f"{v:>8.3f}" for v in vals)
              + f"{np.mean(vals):>8.3f}{np.std(vals, ddof=1):>7.3f}")

    print(f"\n  PAIRED BLOCK BOOTSTRAP vs BASELINE ({BOOT} resamples over weeks)")
    print(f"{'model':<30}{'d pair acc':>12}{'95% CI':>20}{'P(better)':>11}"
          f"{'d MAE':>9}{'95% CI':>18}")
    print("-" * 100)
    boots = {}
    for name in MODELS:
        if name.startswith("baseline ("):
            continue
        bd = boot_delta(preds["baseline (9 box+market)"], preds[name])
        boots[name] = bd
        acc_ci = "[{:+.2f}, {:+.2f}]".format(bd["acc_lo"], bd["acc_hi"])
        mae_ci = "[{:+.4f}, {:+.4f}]".format(bd["mae_lo"], bd["mae_hi"])
        print(f"{name:<30}{bd['acc_delta']:>10.2f}pp{acc_ci:>20}"
              f"{bd['acc_p_better']:>11.2f}{bd['mae_delta']:>9.4f}{mae_ci:>18}")

    # ---- robustness: is the null a linear-model artefact? -------------------
    print(f"\n{BAR}\n 4b. ROBUSTNESS -- SAME TEST WITH GRADIENT BOOSTING\n{BAR}")
    gbm = {}
    for name in ("baseline (9 box+market)", "baseline + ALL advanced"):
        gp = walk_forward_gbm(m, MODELS[name])
        gbm[name] = (score(gp), gp)
    gb_base, gb_adv = gbm["baseline (9 box+market)"], gbm["baseline + ALL advanced"]
    gbd = boot_delta(gb_base[1], gb_adv[1])
    print(f"{'model (LightGBM)':<30}{'MAE':>9}{'RMSE':>9}{'pair acc':>11}")
    print("-" * 60)
    for name in gbm:
        r = gbm[name][0]
        print(f"{name:<30}{r['mae']:>9.4f}{r['rmse']:>9.4f}{r['pair_acc']:>10.2f}%")
    print("-" * 60)
    acc_ci = "[{:+.2f}, {:+.2f}]".format(gbd["acc_lo"], gbd["acc_hi"])
    print(f"  advanced increment under GBM: {gbd['acc_delta']:+.2f}pp {acc_ci}, "
          f"P(better) = {gbd['acc_p_better']:.2f}, d MAE {gbd['mae_delta']:+.4f}")
    print(f"  ridge remains the deployed model; this only checks that a "
          f"nonlinear learner\n  finds nothing a linear one missed.")

    rows = []
    for name, (r, _) in gbm.items():
        rows.append({"model": f"GBM: {name}", "n_features": len(MODELS[name]),
                     "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"],
                     "pair_acc_pct": r["pair_acc"], "pairs": r["pairs"],
                     "mae_delta": r["mae"] - gb_base[0]["mae"],
                     "pair_acc_delta_pp": r["pair_acc"] - gb_base[0]["pair_acc"],
                     "acc_ci_lo": gbd["acc_lo"] if "ALL" in name else None,
                     "acc_ci_hi": gbd["acc_hi"] if "ALL" in name else None,
                     "acc_p_better": gbd["acc_p_better"] if "ALL" in name else None})
    for name in MODELS:
        r = res[name]
        rows.append({
            "model": name, "n_features": len(MODELS[name]), "mae": r["mae"],
            "rmse": r["rmse"], "r2": r["r2"], "pair_acc_pct": r["pair_acc"],
            "pairs": r["pairs"],
            "mae_delta": r["mae"] - b["mae"],
            "pair_acc_delta_pp": r["pair_acc"] - b["pair_acc"],
            "acc_ci_lo": boots.get(name, {}).get("acc_lo"),
            "acc_ci_hi": boots.get(name, {}).get("acc_hi"),
            "acc_p_better": boots.get(name, {}).get("acc_p_better"),
        })
    pl.DataFrame(rows).write_csv(OUT / "qb018_model_comparison.csv")
    pl.DataFrame([{"model": n, **ps} for n in MODELS
                  for ps in res[n]["per_season"]]).write_csv(
        OUT / "qb018_per_season.csv")

    # ---- residual correlations --------------------------------------------
    print(f"\n{BAR}\n 5. WHICH ADVANCED VARIABLES CARRY SIGNAL"
          f"\n    (corr with the baseline model's out-of-sample residual)\n{BAR}")
    base_pred = preds["baseline (9 box+market)"]
    rdf = (m.join(base_pred.select(["season", "week", "player_id", "pred"]),
                  on=["season", "week", "player_id"], how="inner")
           .with_columns((pl.col("fantasy_points") - pl.col("pred")).alias("resid")))
    print(f"  residual rows: {rdf.height:,}")
    rr = []
    for c in ADVANCED:
        sub = rdf.drop_nulls(subset=[c])
        x = sub[c].to_numpy().astype(float)
        y = sub["resid"].to_numpy().astype(float)
        if x.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        se = 1.0 / np.sqrt(sub.height - 3)
        rr.append({"variable": c, "n": sub.height, "corr_with_residual": r,
                   "abs_r": abs(r), "ci_lo": r - 1.96 * se, "ci_hi": r + 1.96 * se,
                   "pct_var_explained": r * r * 100})
    rr = pl.DataFrame(rr).sort("abs_r", descending=True)

    # ... and the same question asked the way a modeller would ask it: add the
    # variable to the deployed feature set, one at a time, and re-run the
    # whole walk-forward. Residual correlation and marginal value can disagree.
    single = {}
    for c in ADVANCED:
        p1 = walk_forward(m, BASELINE + [c])
        s1 = score(p1)
        bd = boot_delta(base_pred, p1)
        single[c] = {"pair_acc": s1["pair_acc"], "mae": s1["mae"],
                     "acc_delta": bd["acc_delta"], "mae_delta": bd["mae_delta"],
                     "acc_lo": bd["acc_lo"], "acc_hi": bd["acc_hi"],
                     "p_better": bd["acc_p_better"]}

    print(f"{'variable':<26}{'r resid':>9}{'95% CI':>20}{'var %':>7}"
          f"{'+1 d acc':>10}{'+1 d MAE':>10}")
    print("-" * 84)
    for row in rr.to_dicts():
        ci = "[{:+.3f}, {:+.3f}]".format(row["ci_lo"], row["ci_hi"])
        sg = single[row["variable"]]
        print(f"{row['variable']:<26}{row['corr_with_residual']:>9.4f}{ci:>20}"
              f"{row['pct_var_explained']:>7.2f}{sg['acc_delta']:>9.2f}pp"
              f"{sg['mae_delta']:>10.4f}")
    print("-" * 84)
    print("  '+1 d acc' / '+1 d MAE' = change when that ONE variable is added to")
    print("  the deployed nine-feature model and the walk-forward is re-run.")
    best = max(single.items(), key=lambda kv: kv[1]["acc_delta"])
    print(f"  best single addition: {best[0]} "
          f"{best[1]['acc_delta']:+.2f}pp "
          f"[{best[1]['acc_lo']:+.2f}, {best[1]['acc_hi']:+.2f}], "
          f"P(better) = {best[1]['p_better']:.2f}")
    print(f"  Note: 22 variables were scanned, so the best of 22 is expected to")
    print(f"  look positive by chance alone. Read the interval, not the point.")

    rr = rr.join(
        pl.DataFrame([{"variable": k,
                       "solo_pair_acc_pct": v["pair_acc"],
                       "solo_acc_delta_pp": v["acc_delta"],
                       "solo_acc_ci_lo": v["acc_lo"],
                       "solo_acc_ci_hi": v["acc_hi"],
                       "solo_p_better": v["p_better"],
                       "solo_mae_delta": v["mae_delta"]} for k, v in single.items()]),
        on="variable", how="left")
    rr.write_csv(OUT / "qb018_residual_correlations.csv")

    # ---- stability ---------------------------------------------------------
    print(f"\n{BAR}\n 6. STABLE TRAIT vs NOISY OUTCOME"
          f"\n    (season N value vs the same unit's season N+1 value)\n{BAR}")
    raw = stability_inputs(g.filter((pl.col("season") >= 2018)
                                    & (pl.col("attempts") >= pm.MIN_ATT)))
    qb_st = season_stability(raw, "player_id")
    tm_st = season_stability(raw.rename({"team": "team_off"}), "team_off")

    def show(tbl: pl.DataFrame, label: str, tag: bool) -> None:
        print(f"  {label}")
        head = f"{'variable':<26}{'r(N, N+1)':>11}{'95% CI':>18}{'n':>6}"
        print(head + ("  verdict" if tag else ""))
        print("-" * 78)
        for row in tbl.to_dicts():
            r = row["r_season_to_season"]
            if r is None:
                print(f"{row['variable']:<26}{'n/a':>11}{'':>18}{row['n_pairs']:>6}")
                continue
            ci = "[{:+.2f}, {:+.2f}]".format(row["ci_lo"], row["ci_hi"])
            v = ("  STABLE TRAIT" if r >= 0.55 else
                 "  semi-stable" if r >= 0.35 else "  NOISY OUTCOME") if tag else ""
            print(f"{row['variable']:<26}{r:>11.3f}{ci:>18}{row['n_pairs']:>6}{v}")

    show(qb_st, "QUARTERBACK level (>= 8 games in both seasons):", True)
    print()
    show(tm_st, "TEAM (offence) level -- is it the line, or the quarterback?", False)
    print("\n  A rate can be a stable trait and still be worthless for weekly")
    print("  prediction; stability is a necessary condition, not a sufficient one.")

    pl.concat([qb_st, tm_st]).write_csv(OUT / "qb018_stability.csv")

    # feature table for anyone who wants to re-use it
    m.select(["player_id", "player_display_name", "season", "week", "team",
              "opponent", "fantasy_points"] + BASELINE + ADVANCED).write_csv(
        OUT / "qb018_advanced_panel.csv")

    print(f"\n{BAR}\n WROTE\n{BAR}")
    for f in ["qb018_source_schemas.csv", "qb018_join_rates.csv",
              "qb018_coverage_by_season.csv", "qb018_model_comparison.csv",
              "qb018_per_season.csv", "qb018_residual_correlations.csv",
              "qb018_stability.csv", "qb018_advanced_panel.csv"]:
        p = OUT / f
        if p.exists():
            print(f"  outputs/{f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
