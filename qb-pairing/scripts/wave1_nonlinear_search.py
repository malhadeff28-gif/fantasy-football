"""QB-017: does nonlinear structure or interaction effects exist in QB weekly
fantasy scoring that a regularised linear model missed?

Wave 0 fitted ridge regression and stopped at out-of-sample R^2 ~= 0.177 with
64% pairwise start/sit accuracy. Hand-picked interactions made it slightly
worse. That is a claim about *hand-picked* interactions, not about interactions
in general, so this experiment searches properly and guards the search.

The design separates three distinct questions that "is it nonlinear?" usually
conflates:

    ridge          linear and additive                      (the incumbent)
    GBM-additive   depth-1 stumps: nonlinear, NO interactions possible
    GBM-full       depth-unconstrained trees: nonlinear AND interactions

`GBM-additive - ridge` isolates univariate curvature. `GBM-full - GBM-additive`
isolates interaction structure. If neither gap is positive out of sample, the
relationship is essentially linear and the ceiling is the data, not the model
class.

Guards against finding structure that is not there:

* Walk-forward only: train seasons < S, test season S, S = 2020..2025. Every
  hyperparameter is chosen on inner folds carved out of the *training* seasons.
  The test season is never seen during selection.
* Every model sees an identical row set and an identical feature set, so a
  difference cannot be a feature-availability artefact.
* Uncertainty is reported. Six test seasons is a small number; the paired
  block bootstrap resamples weeks, which is the unit of correlated error.
* PERMUTATION NULL on the interaction search. The target is shuffled within
  week and the entire SHAP interaction search is re-run, many times, to measure
  how large an "interaction" appears purely by chance. An interaction that does
  not clear its own null is noise and is reported as noise.
* A second, harder null: each candidate interaction is added to ridge as an
  explicit product term and scored walk-forward, against a null in which the
  product column is permuted within week. The family-wise statistic is the
  maximum improvement over all candidates in a draw, which prices in the fact
  that many interactions were examined.

Usage:  ./.venv/Scripts/python.exe qb-pairing/scripts/wave1_nonlinear_search.py

Outputs: qb-pairing/outputs/qb017_*.csv and qb-pairing/outputs/qb017_*.png
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys                                          # noqa: E402
import time                                         # noqa: E402
import warnings                                     # noqa: E402
from itertools import product as iproduct           # noqa: E402
from pathlib import Path                            # noqa: E402

import matplotlib                                   # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import polars as pl                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]          # qb-pairing/
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb                              # noqa: E402
import shap                                         # noqa: E402
import xgboost as xgb                               # noqa: E402
from sklearn.linear_model import RidgeCV            # noqa: E402
from sklearn.preprocessing import StandardScaler    # noqa: E402

from qbpair.features import panel as pm             # noqa: E402
from qbpair.models import projection                # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
TARGET = "fantasy_points"
BAR = "=" * 78
SEED = 20260818

# Full pregame feature set available from the panel. The ridge baseline uses a
# 9-feature subset (projection.FEATURES); every model here is additionally run
# on this wider set so that "trees win" can never be confused with "trees were
# given more features".
FULL_FEATURES = [
    "form_fp_mean", "form_fp_sd", "career_fp_mean", "games_prior",
    "form_pass_att", "form_rush_att", "rush_fp_share", "form_adot",
    "form_cpoe", "form_sack_rate",
    "implied_team_total", "total_line", "spread_abs",
    "is_home", "indoors", "rest", "div_game",
    "def_adj", "def_fp_allowed",
]

# Interactions Wave 0 hand-picked and rejected, carried along so the search's
# discoveries can be compared against them on the same footing.
HANDPICKED = [
    ("rush_fp_share", "def_adj"),
    ("rush_fp_share", "spread_abs"),
    ("form_fp_mean", "implied_team_total"),
    ("form_pass_att", "def_fp_allowed"),
]

RUN = dict(
    n_shap_rows=800,      # rows subsampled for SHAP interaction values
    n_null_a=200,         # permutation draws for the SHAP interaction null
    n_null_b=200,         # permutation draws for the OOS incremental null
    n_boot=2000,          # block-bootstrap draws over weeks
    n_candidates=10,      # top SHAP pairs carried into the OOS test
)

# `--quick` runs the identical pipeline with far fewer resampling draws. Useful
# for checking the script end to end; the reported nulls are then too coarse to
# quote, so every published number comes from a full run.
QUICK = dict(n_shap_rows=300, n_null_a=6, n_null_b=4, n_boot=200, n_candidates=4)
if "--quick" in sys.argv:
    RUN.update(QUICK)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def week_blocks(df: pl.DataFrame) -> list[np.ndarray]:
    """Row indices grouped by (season, week). The week is the correlated unit."""
    idx = np.arange(df.height)
    key = (df["season"].to_numpy().astype(np.int64) * 100
           + df["week"].to_numpy().astype(np.int64))
    return [idx[key == k] for k in np.unique(key)]


def per_week_stats(y: np.ndarray, yhat: np.ndarray,
                   blocks: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Sufficient statistics per week for MAE, RMSE and pairwise accuracy.

    MAE and accuracy are both ratios of sums over weeks, so the bootstrap can
    resample these precomputed blocks instead of refitting anything.
    """
    n, sae, sse, corr, tot = [], [], [], [], []
    for b in blocks:
        yb, pb = y[b], yhat[b]
        n.append(len(b))
        sae.append(np.abs(yb - pb).sum())
        sse.append(((yb - pb) ** 2).sum())
        if len(b) >= 2:
            iu = np.triu_indices(len(b), 1)
            dy = (yb[:, None] - yb[None, :])[iu]
            dp = (pb[:, None] - pb[None, :])[iu]
            m = dy != 0
            tot.append(int(m.sum()))
            corr.append(int((np.sign(dy[m]) == np.sign(dp[m])).sum()))
        else:
            tot.append(0)
            corr.append(0)
    return {k: np.asarray(v, dtype=float) for k, v in
            dict(n=n, sae=sae, sse=sse, corr=corr, tot=tot).items()}


def pooled(st: dict[str, np.ndarray]) -> tuple[float, float, float]:
    return (st["sae"].sum() / st["n"].sum(),
            float(np.sqrt(st["sse"].sum() / st["n"].sum())),
            st["corr"].sum() / st["tot"].sum() * 100)


def r2_scores(y: np.ndarray, yhat: np.ndarray,
              train_mean: np.ndarray) -> tuple[float, float]:
    """R^2 against the honest out-of-sample null (the training-fold mean) and
    against the test-season mean, which is not knowable in advance."""
    sse = ((y - yhat) ** 2).sum()
    return (1 - sse / ((y - train_mean) ** 2).sum(),
            1 - sse / ((y - y.mean()) ** 2).sum())


# --------------------------------------------------------------------------
# walk-forward engines
# --------------------------------------------------------------------------

def ridge_walk_forward(df: pl.DataFrame, feats: list[str],
                       extra: np.ndarray | None = None) -> np.ndarray:
    """Out-of-sample predictions for the test seasons, in df row order.

    `extra` is an optional additional column (an explicit interaction term)
    aligned to df; it is standardised on the training fold like everything else.
    """
    X_all = df.select(feats).to_numpy().astype(float)
    if extra is not None:
        X_all = np.column_stack([X_all, extra])
    y_all = df[TARGET].to_numpy().astype(float)
    season = df["season"].to_numpy()

    pred = np.full(df.height, np.nan)
    for s in TEST_SEASONS:
        tr, te = season < s, season == s
        sc = StandardScaler().fit(X_all[tr])
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X_all[tr]),
                                                       y_all[tr])
        pred[te] = m.predict(sc.transform(X_all[te]))
    return pred


class FastRidgeWF:
    """Walk-forward ridge with the penalty pinned per fold.

    Used for the exhaustive interaction scan and its permutation null, where
    tens of thousands of refits are needed. The alpha is chosen once per fold by
    RidgeCV on the base features and then held fixed while candidate columns are
    added, so a candidate cannot win by quietly buying itself a different amount
    of regularisation -- and observed and permuted candidates are scored by
    exactly the same procedure.
    """

    def __init__(self, df: pl.DataFrame, feats: list[str]):
        self.X = df.select(feats).to_numpy().astype(float)
        self.y = df[TARGET].to_numpy().astype(float)
        season = df["season"].to_numpy()
        self.folds = [(season < s, season == s) for s in TEST_SEASONS]
        self.alpha = []
        for tr, _ in self.folds:
            sc = StandardScaler().fit(self.X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(self.X[tr]),
                                                           self.y[tr])
            self.alpha.append(float(m.alpha_))

    def predict(self, extra: np.ndarray | None = None) -> np.ndarray:
        X = self.X if extra is None else np.column_stack([self.X, extra])
        pred = np.full(len(self.y), np.nan)
        for (tr, te), a in zip(self.folds, self.alpha):
            Xtr, Xte = X[tr], X[te]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            ym = self.y[tr].mean()
            G = Ztr.T @ Ztr + a * np.eye(Ztr.shape[1])
            b = np.linalg.solve(G, Ztr.T @ (self.y[tr] - ym))
            pred[te] = Zte @ b + ym
        return pred


def train_mean_baseline(df: pl.DataFrame) -> np.ndarray:
    y = df[TARGET].to_numpy().astype(float)
    season = df["season"].to_numpy()
    out = np.full(df.height, np.nan)
    for s in TEST_SEASONS:
        out[season == s] = y[season < s].mean()
    return out


LGB_FIXED = dict(n_jobs=1, verbose=-1, random_state=SEED,
                 force_col_wise=True, deterministic=True,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def lgb_grid(additive: bool) -> list[dict]:
    """The loss is tuned too, so a loss/metric mismatch cannot be blamed for a
    tree model losing."""
    if additive:
        return [dict(num_leaves=2, max_depth=1, min_child_samples=mcs,
                     reg_lambda=l2, learning_rate=lr, objective=obj)
                for mcs, l2, lr, obj in iproduct((20, 60), (1.0, 10.0),
                                                 (0.03, 0.06), ("l1", "l2"))]
    return [dict(num_leaves=nl, max_depth=-1, min_child_samples=mcs,
                 reg_lambda=l2, learning_rate=lr, objective=obj)
            for nl, mcs, l2, lr, obj in iproduct((3, 7, 15), (20, 60),
                                                 (1.0, 10.0), (0.03, 0.06),
                                                 ("l1", "l2"))]


def _inner_folds(seasons_tr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Up to two expanding inner folds inside the training seasons."""
    uniq = np.unique(seasons_tr)
    folds = []
    for v in uniq[-2:]:
        if (seasons_tr < v).sum() >= 300:
            folds.append((seasons_tr < v, seasons_tr == v))
    if not folds:                       # 2020 fold: train 2018, validate 2019
        v = uniq[-1]
        folds.append((seasons_tr < v, seasons_tr == v))
    return folds


def tune_lgb(X: np.ndarray, y: np.ndarray, seasons_tr: np.ndarray,
             additive: bool, max_rounds: int) -> tuple[dict, int, float]:
    """Pick hyperparameters on inner folds of the training seasons only.

    The test season is not visible here, so nothing selected can be an artefact
    of the season it is later scored on.
    """
    folds = _inner_folds(seasons_tr)
    best, best_mae, best_iter = None, np.inf, 200
    for cfg in lgb_grid(additive):
        maes, iters = [], []
        for itr, iva in folds:
            m = lgb.LGBMRegressor(n_estimators=max_rounds, **LGB_FIXED, **cfg)
            m.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(60, verbose=False)])
            bi = m.best_iteration_ or max_rounds
            maes.append(np.abs(y[iva] - m.predict(X[iva], num_iteration=bi)).mean())
            iters.append(bi)
        mae = float(np.mean(maes))
        if mae < best_mae:
            best, best_mae, best_iter = cfg, mae, int(np.mean(iters))
    return best, max(30, best_iter), best_mae


def lgb_walk_forward(df: pl.DataFrame, feats: list[str], additive: bool,
                     ridge_offset_feats: list[str] | None = None,
                     max_rounds: int = 2000) -> tuple[np.ndarray, list[dict]]:
    """Walk-forward gradient boosting.

    With `ridge_offset_feats`, the trees are boosted on the *ridge residuals*
    instead of the raw target: the linear model keeps whatever it explains and
    the trees are asked only for what it missed. That is the sharpest available
    test of "did the regularised linear model leave structure on the table".
    """
    X = df.select(feats).to_numpy().astype(float)
    y = df[TARGET].to_numpy().astype(float)
    season = df["season"].to_numpy()
    Xr = (df.select(ridge_offset_feats).to_numpy().astype(float)
          if ridge_offset_feats else None)

    pred = np.full(df.height, np.nan)
    chosen = []
    for s in TEST_SEASONS:
        tr, te = season < s, season == s
        if Xr is None:
            tgt_tr, off_tr, off_te = y[tr], 0.0, 0.0
        else:
            sc = StandardScaler().fit(Xr[tr])
            rm = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(Xr[tr]), y[tr])
            off_tr = rm.predict(sc.transform(Xr[tr]))
            off_te = rm.predict(sc.transform(Xr[te]))
            tgt_tr = y[tr] - off_tr
        cfg, n_it, inner_mae = tune_lgb(X[tr], tgt_tr, season[tr], additive, max_rounds)
        # inner folds trained on fewer rows than the final fit; scale the round
        # count by the row ratio rather than re-tuning on the test season.
        scale = tr.sum() / max(1, (season[tr] < np.unique(season[tr])[-1]).sum())
        n_final = int(min(max_rounds, max(30, n_it * min(1.5, scale))))
        m = lgb.LGBMRegressor(n_estimators=n_final, **LGB_FIXED, **cfg)
        m.fit(X[tr], tgt_tr)
        pred[te] = off_te + m.predict(X[te])
        chosen.append({"season": s, "n_estimators": n_final,
                       "inner_mae": round(inner_mae, 4), **cfg})
    return pred, chosen


def xgb_walk_forward(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    X = df.select(feats).to_numpy().astype(float)
    y = df[TARGET].to_numpy().astype(float)
    season = df["season"].to_numpy()
    grid = [dict(max_depth=d, min_child_weight=w, reg_lambda=l2)
            for d, w, l2 in iproduct((2, 3, 4), (5, 20), (1.0, 10.0))]

    pred = np.full(df.height, np.nan)
    for s in TEST_SEASONS:
        tr, te = season < s, season == s
        folds = _inner_folds(season[tr])
        best, best_mae, best_iter = None, np.inf, 200
        for cfg in grid:
            maes, iters = [], []
            for itr, iva in folds:
                m = xgb.XGBRegressor(
                    n_estimators=1500, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, objective="reg:absoluteerror",
                    eval_metric="mae", early_stopping_rounds=60, n_jobs=1,
                    random_state=SEED, verbosity=0, **cfg)
                m.fit(X[tr][itr], y[tr][itr],
                      eval_set=[(X[tr][iva], y[tr][iva])], verbose=False)
                bi = int(m.best_iteration or 200)
                iters.append(bi)
                maes.append(np.abs(y[tr][iva] - m.predict(X[tr][iva])).mean())
            mae = float(np.mean(maes))
            if mae < best_mae:
                best, best_mae, best_iter = cfg, mae, int(np.mean(iters))
        m = xgb.XGBRegressor(
            n_estimators=max(30, best_iter), learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="reg:absoluteerror", n_jobs=1,
            random_state=SEED, verbosity=0, **best)
        m.fit(X[tr], y[tr], verbose=False)
        pred[te] = m.predict(X[te])
    return pred


# --------------------------------------------------------------------------
# permutation helper
# --------------------------------------------------------------------------

def permute_within_week(v: np.ndarray, blocks: list[np.ndarray],
                        rng: np.random.Generator) -> np.ndarray:
    """Shuffle a column inside each (season, week) block.

    Within-week permutation preserves the pooled marginal distribution and the
    week-level composition exactly, and destroys only the row-level association
    -- which is precisely the thing a claimed interaction asserts.
    """
    out = v.copy()
    for b in blocks:
        out[b] = v[rng.permutation(b)]
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # -- panel --------------------------------------------------------------
    print(BAR)
    print(" QB-017  NONLINEAR STRUCTURE AND INTERACTION SEARCH")
    print(BAR)
    raw = pm.build_panel(SEASONS)
    panel = raw.drop_nulls(subset=FULL_FEATURES).sort(["season", "week", "player_id"])
    test_rows = panel.filter(pl.col("season").is_in(TEST_SEASONS))
    print(f"panel                 : {raw.height:,} QB-weeks, {SEASONS[0]}-{SEASONS[-1]}")
    print(f"complete-case rows    : {panel.height:,} "
          f"(dropped {raw.height - panel.height} debut/short-history rows)")
    print(f"features              : {len(FULL_FEATURES)} "
          f"(ridge baseline uses {len(projection.FEATURES)})")
    print(f"test seasons          : {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}, "
          f"{test_rows.height:,} rows")
    print("every model below sees the identical row set and identical features")

    y_test = test_rows[TARGET].to_numpy().astype(float)
    blocks_test = week_blocks(test_rows)
    is_test = panel["season"].is_in(TEST_SEASONS).to_numpy()
    tmean = train_mean_baseline(panel)[is_test]

    # -- model ladder -------------------------------------------------------
    print(f"\n{BAR}\n 1. MODEL LADDER, WALK-FORWARD (train < S, test S)\n{BAR}")
    preds: dict[str, np.ndarray] = {}

    t0 = time.time()
    preds["train mean"] = tmean
    preds["ridge (9 feat)"] = ridge_walk_forward(panel, projection.FEATURES)[is_test]
    preds["ridge (19 feat)"] = ridge_walk_forward(panel, FULL_FEATURES)[is_test]
    print(f"ridge fits            : {time.time() - t0:.1f}s")

    t0 = time.time()
    p_add, cfg_add = lgb_walk_forward(panel, FULL_FEATURES, additive=True)
    preds["GBM additive"] = p_add[is_test]
    print(f"GBM-additive (stumps) : {time.time() - t0:.1f}s   tuned per fold")

    t0 = time.time()
    p_full, cfg_full = lgb_walk_forward(panel, FULL_FEATURES, additive=False)
    preds["GBM full (lightgbm)"] = p_full[is_test]
    print(f"GBM-full (lightgbm)   : {time.time() - t0:.1f}s   tuned per fold")

    t0 = time.time()
    preds["GBM full (xgboost)"] = xgb_walk_forward(panel, FULL_FEATURES)[is_test]
    print(f"GBM-full (xgboost)    : {time.time() - t0:.1f}s   tuned per fold")

    t0 = time.time()
    p_res, cfg_res = lgb_walk_forward(panel, FULL_FEATURES, additive=False,
                                      ridge_offset_feats=projection.FEATURES)
    preds["ridge + GBM residual"] = p_res[is_test]
    print(f"GBM on ridge residuals: {time.time() - t0:.1f}s   tuned per fold")

    preds["ridge+GBM blend"] = 0.5 * preds["ridge (9 feat)"] + 0.5 * preds["GBM full (lightgbm)"]

    order = ["train mean", "ridge (9 feat)", "ridge (19 feat)", "GBM additive",
             "GBM full (lightgbm)", "GBM full (xgboost)",
             "ridge + GBM residual", "ridge+GBM blend"]
    stats = {k: per_week_stats(y_test, preds[k], blocks_test) for k in order}

    rows = []
    for k in order:
        mae, rmse, acc = pooled(stats[k])
        r2_oos, r2_ts = r2_scores(y_test, preds[k], tmean)
        rows.append({"model": k, "MAE": mae, "RMSE": rmse, "R2_vs_trainmean": r2_oos,
                     "R2_vs_testmean": r2_ts, "pair_acc_pct": acc,
                     "pred_sd": float(preds[k].std())})
    comp = pl.DataFrame(rows)
    print(f"\n{'model':<22}{'MAE':>8}{'RMSE':>8}{'R2(oos)':>10}"
          f"{'pair acc':>11}{'pred sd':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['model']:<22}{r['MAE']:>8.4f}{r['RMSE']:>8.4f}"
              f"{r['R2_vs_trainmean']:>10.4f}{r['pair_acc_pct']:>10.2f}%"
              f"{r['pred_sd']:>10.3f}")
    print("-" * 78)
    print(f"actual sd = {y_test.std():.3f}.  n pairwise comparisons = "
          f"{int(stats['ridge (9 feat)']['tot'].sum()):,}")

    # -- paired block bootstrap over weeks ----------------------------------
    print(f"\n{BAR}\n 2. UNCERTAINTY: PAIRED BLOCK BOOTSTRAP OVER WEEKS "
          f"({RUN['n_boot']:,} draws)\n{BAR}")
    nweeks = len(blocks_test)
    draws = rng.integers(0, nweeks, size=(RUN["n_boot"], nweeks))
    base_key = "ridge (9 feat)"

    def boot_metric(st, d):
        return (st["sae"][d].sum(1) / st["n"][d].sum(1),
                st["corr"][d].sum(1) / st["tot"][d].sum(1) * 100)

    b_base_mae, b_base_acc = boot_metric(stats[base_key], draws)
    print(f"vs {base_key}: negative dMAE = better, positive dAcc = better")
    print(f"\n{'model':<22}{'dMAE':>9}{'95% CI':>20}{'P(better)':>11}"
          f"{'dAcc pp':>10}{'P(better)':>11}")
    print("-" * 78)
    boot_rows = []
    for k in order:
        if k in ("train mean", base_key):
            continue
        b_mae, b_acc = boot_metric(stats[k], draws)
        dm, da = b_mae - b_base_mae, b_acc - b_base_acc
        obs_m = pooled(stats[k])[0] - pooled(stats[base_key])[0]
        obs_a = pooled(stats[k])[2] - pooled(stats[base_key])[2]
        lo, hi = np.percentile(dm, [2.5, 97.5])
        boot_rows.append({"model": k, "dMAE": obs_m, "dMAE_lo": lo, "dMAE_hi": hi,
                          "p_better_mae": float((dm < 0).mean()),
                          "dAcc_pp": obs_a,
                          "dAcc_lo": float(np.percentile(da, 2.5)),
                          "dAcc_hi": float(np.percentile(da, 97.5)),
                          "p_better_acc": float((da > 0).mean())})
        print(f"{k:<22}{obs_m:>+9.4f}   [{lo:>+6.4f},{hi:>+6.4f}]"
              f"{(dm < 0).mean():>11.2f}{obs_a:>+10.2f}{(da > 0).mean():>11.2f}")
    boot = pl.DataFrame(boot_rows)

    # -- per-season stability -----------------------------------------------
    print(f"\n{BAR}\n 3. STABILITY ACROSS SEASONS\n{BAR}")
    season_test = test_rows["season"].to_numpy()
    key_models = [base_key, "GBM additive", "GBM full (lightgbm)",
                  "ridge + GBM residual", "ridge+GBM blend"]
    srows = []
    for s in TEST_SEASONS:
        m = season_test == s
        bl = week_blocks(test_rows.filter(pl.col("season") == s))
        rec = {"season": s, "n": int(m.sum())}
        for k in key_models:
            st = per_week_stats(y_test[m], preds[k][m], bl)
            mae, _, acc = pooled(st)
            rec[f"{k}|MAE"] = mae
            rec[f"{k}|acc"] = acc
        srows.append(rec)
    per_season = pl.DataFrame(srows)

    print(f"{'season':<8}" + "".join(f"{k.split(' (')[0][:13]:>14}" for k in key_models))
    print(f"{'':<8}" + "".join(f"{'MAE':>14}" for _ in key_models))
    print("-" * 78)
    for r in srows:
        print(f"{r['season']:<8}" + "".join(f"{r[f'{k}|MAE']:>14.4f}" for k in key_models))
    print("-" * 78)
    for k in key_models[1:]:
        d = np.array([r[f"{k}|MAE"] - r[f"{base_key}|MAE"] for r in srows])
        da = np.array([r[f"{k}|acc"] - r[f"{base_key}|acc"] for r in srows])
        print(f"{k:<22} dMAE {d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f} (SE), "
              f"beats ridge in {(d < 0).sum()}/{len(d)} seasons; "
              f"dAcc {da.mean():+.2f}pp, {(da > 0).sum()}/{len(da)}")

    # -- calibration ---------------------------------------------------------
    print(f"\n{BAR}\n 4. CALIBRATION\n{BAR}")
    cal_models = [base_key, "GBM additive", "GBM full (lightgbm)"]
    cal_rows, cal_curves = [], {}
    for k in cal_models:
        p = preds[k]
        A = np.column_stack([np.ones_like(p), p])
        icept, slope = np.linalg.lstsq(A, y_test, rcond=None)[0]
        qs = np.quantile(p, np.linspace(0, 1, 11))
        qs[-1] += 1e-9
        b = np.clip(np.digitize(p, qs[1:-1]), 0, 9)
        curve = []
        for i in range(10):
            m = b == i
            curve.append({"bin": i + 1, "n": int(m.sum()),
                          "mean_pred": float(p[m].mean()),
                          "mean_actual": float(y_test[m].mean()),
                          "se_actual": float(y_test[m].std(ddof=1) / np.sqrt(m.sum()))})
        cal_curves[k] = curve
        gap = float(np.mean([abs(c["mean_actual"] - c["mean_pred"]) for c in curve]))
        cal_rows.append({"model": k, "cal_slope": float(slope),
                         "cal_intercept": float(icept), "mean_abs_bin_gap": gap,
                         "bias": float((p - y_test).mean()),
                         "pred_sd": float(p.std())})
    calib = pl.DataFrame(cal_rows)
    print(f"{'model':<22}{'slope':>9}{'intercept':>11}{'bias':>9}"
          f"{'|bin gap|':>11}{'pred sd':>10}")
    print("-" * 78)
    for r in cal_rows:
        print(f"{r['model']:<22}{r['cal_slope']:>9.3f}{r['cal_intercept']:>11.3f}"
              f"{r['bias']:>9.3f}{r['mean_abs_bin_gap']:>11.3f}{r['pred_sd']:>10.3f}")
    print("-" * 78)
    print("slope 1.0 / intercept 0.0 = perfectly calibrated. A slope < 1 means the")
    print("model's spread is too wide for the signal it actually has.")

    # -- SHAP interaction discovery -----------------------------------------
    print(f"\n{BAR}\n 5. SHAP INTERACTION SEARCH\n{BAR}")
    X_all = panel.select(FULL_FEATURES).to_numpy().astype(float)
    y_all = panel[TARGET].to_numpy().astype(float)
    season_all = panel["season"].to_numpy()

    # discovery model: the hyperparameters the walk-forward tuner chose most
    # often, refit on the whole panel for maximum search power. Discovery is
    # descriptive; sections 6 and 7 are the tests.
    def modal_cfg(chosen: list[dict]) -> dict:
        keys = [tuple(sorted((k, v) for k, v in c.items()
                             if k not in ("season", "inner_mae", "n_estimators")))
                for c in chosen]
        return dict(max(set(keys), key=keys.count))

    disc_cfg = modal_cfg(cfg_full)
    n_disc = int(np.median([c["n_estimators"] for c in cfg_full]))
    disc = lgb.LGBMRegressor(n_estimators=n_disc, **LGB_FIXED, **disc_cfg)
    disc.fit(X_all, y_all)
    print(f"discovery model       : lightgbm, {n_disc} trees, "
          + ", ".join(f"{k}={v}" for k, v in disc_cfg.items()))

    sub_idx = rng.choice(panel.height, RUN["n_shap_rows"], replace=False)
    X_sub = X_all[sub_idx]
    t0 = time.time()
    ex = shap.TreeExplainer(disc)
    iv = ex.shap_interaction_values(X_sub)
    print(f"SHAP interaction vals : {time.time() - t0:.1f}s on "
          f"{RUN['n_shap_rows']} rows x {len(FULL_FEATURES)} features")

    def rank_pairs(inter: np.ndarray) -> np.ndarray:
        """mean |interaction| per unordered pair, in fantasy points."""
        a = np.abs(inter).mean(axis=0)
        return (a + a.T) - np.diag(np.diag(a) * 2)   # off-diag doubled, diag zeroed

    obs_pair = rank_pairs(iv)
    p = len(FULL_FEATURES)
    iu = np.triu_indices(p, 1)
    obs_vec = obs_pair[iu]
    main_mag = float(np.abs(iv).mean(axis=0).diagonal().sum())
    inter_mag = float(obs_vec.sum())
    print(f"main-effect magnitude : {main_mag:.3f} pts   "
          f"interaction magnitude : {inter_mag:.3f} pts   "
          f"interaction share : {inter_mag / (main_mag + inter_mag) * 100:.1f}%")

    ranked = np.argsort(obs_vec)[::-1]
    pair_names = [(FULL_FEATURES[iu[0][t]], FULL_FEATURES[iu[1][t]])
                  for t in range(len(obs_vec))]

    # -- permutation null A: re-run the whole search on shuffled targets -----
    print(f"\n{BAR}\n 6. PERMUTATION NULL A -- HOW BIG IS AN 'INTERACTION' BY CHANCE?\n{BAR}")
    print(f"target shuffled within (season, week); model refit with identical")
    print(f"hyperparameters and tree count; SHAP interaction search re-run.")
    print(f"{RUN['n_null_a']} draws. Conservative by construction: the tree count is")
    print("held at the value the real signal justified, so the null model is free")
    print("to overfit noise, which inflates its apparent interactions.")
    blocks_all = week_blocks(panel)
    null_vecs = np.empty((RUN["n_null_a"], len(obs_vec)))
    t0 = time.time()
    for i in range(RUN["n_null_a"]):
        y_perm = permute_within_week(y_all, blocks_all, rng)
        mnull = lgb.LGBMRegressor(n_estimators=n_disc, **LGB_FIXED, **disc_cfg)
        mnull.fit(X_all, y_perm)
        ivn = shap.TreeExplainer(mnull).shap_interaction_values(X_sub)
        null_vecs[i] = rank_pairs(ivn)[iu]
        if (i + 1) % 50 == 0:
            print(f"  draw {i + 1}/{RUN['n_null_a']}  ({time.time() - t0:.0f}s)")
    null_max = null_vecs.max(axis=1)
    print(f"null A elapsed        : {time.time() - t0:.0f}s")

    obs_top = float(obs_vec.max())
    fw_p = float((null_max >= obs_top).mean())
    print(f"\nfamily-wise test on the strongest pair found:")
    print(f"  observed strongest interaction : {obs_top:.4f} pts "
          f"({pair_names[int(np.argmax(obs_vec))][0]} x "
          f"{pair_names[int(np.argmax(obs_vec))][1]})")
    print(f"  null max-statistic  mean       : {null_max.mean():.4f} pts")
    print(f"  null max-statistic  95th pct   : {np.percentile(null_max, 95):.4f} pts")
    print(f"  family-wise p-value            : {fw_p:.3f}")

    shap_rows = []
    for t in ranked[:20]:
        nv = null_vecs[:, t]
        shap_rows.append({
            "feature_a": pair_names[t][0], "feature_b": pair_names[t][1],
            "shap_interaction_pts": float(obs_vec[t]),
            "null_mean": float(nv.mean()), "null_p95": float(np.percentile(nv, 95)),
            "p_pairwise": float((nv >= obs_vec[t]).mean()),
            "p_familywise": float((null_max >= obs_vec[t]).mean()),
            "exceeds_own_null_p95": bool(obs_vec[t] > np.percentile(nv, 95)),
        })
    shap_tab = pl.DataFrame(shap_rows)
    print(f"\n{'rank':<5}{'interaction':<44}{'obs':>8}{'null p95':>10}"
          f"{'p(pair)':>9}{'p(FW)':>8}")
    print("-" * 78)
    for i, r in enumerate(shap_rows[:15], 1):
        nm = f"{r['feature_a']} x {r['feature_b']}"
        print(f"{i:<5}{nm:<44}{r['shap_interaction_pts']:>8.4f}"
              f"{r['null_p95']:>10.4f}{r['p_pairwise']:>9.3f}{r['p_familywise']:>8.3f}")

    # -- null B: does a candidate interaction pay off out of sample? ---------
    print(f"\n{BAR}\n 7. NULL B -- EXHAUSTIVE INTERACTION SCAN, OUT OF SAMPLE\n{BAR}")
    print(f"Every one of the {len(obs_vec)} feature pairs is added to ridge as an")
    print("explicit product term and scored walk-forward. This is a second, fully")
    print("independent search: it does not consult the tree model at all, so it")
    print("cannot inherit the tree model's biases -- and because it is exhaustive,")
    print("the family-wise null below covers the entire search, not a shortlist.")
    print("null: the product column permuted within week -- same marginal, same")
    print("week composition, row-level link destroyed.")

    fast = FastRidgeWF(panel, FULL_FEATURES)
    base_pred = fast.predict()[is_test]
    base_st = per_week_stats(y_test, base_pred, blocks_test)
    base_mae, _, base_acc = pooled(base_st)
    ref_pred = ridge_walk_forward(panel, FULL_FEATURES)[is_test]
    ref_mae = pooled(per_week_stats(y_test, ref_pred, blocks_test))[0]
    print(f"\nbase ridge(19)        : MAE {base_mae:.4f}  acc {base_acc:.2f}%")
    print(f"  (fixed-alpha fast path vs RidgeCV path: {abs(base_mae - ref_mae):.5f} "
          f"MAE apart -- same model)")

    prod_cols = {}
    for t in range(len(obs_vec)):
        a, b = pair_names[t]
        prod_cols[t] = (X_all[:, FULL_FEATURES.index(a)]
                        * X_all[:, FULL_FEATURES.index(b)])

    t0 = time.time()
    obs_delta = np.empty(len(obs_vec))
    obs_dacc = np.empty(len(obs_vec))
    for t in range(len(obs_vec)):
        st = per_week_stats(y_test, fast.predict(prod_cols[t])[is_test], blocks_test)
        mae, _, acc = pooled(st)
        obs_delta[t] = base_mae - mae       # positive = the interaction helped
        obs_dacc[t] = acc - base_acc
    print(f"exhaustive scan       : {time.time() - t0:.0f}s, "
          f"{(obs_delta > 0).sum()}/{len(obs_vec)} pairs improved MAE at all")

    t0 = time.time()
    nullB = np.empty((RUN["n_null_b"], len(obs_vec)))
    nullB_acc = np.empty((RUN["n_null_b"], len(obs_vec)))
    for i in range(RUN["n_null_b"]):
        for t in range(len(obs_vec)):
            perm = permute_within_week(prod_cols[t], blocks_all, rng)
            st = per_week_stats(y_test, fast.predict(perm)[is_test], blocks_test)
            mae, _, acc = pooled(st)
            nullB[i, t] = base_mae - mae
            nullB_acc[i, t] = acc - base_acc
        if (i + 1) % 25 == 0:
            print(f"  draw {i + 1}/{RUN['n_null_b']}  ({time.time() - t0:.0f}s)")
    nullB_max = nullB.max(axis=1)
    nullB_acc_max = nullB_acc.max(axis=1)
    print(f"null B elapsed        : {time.time() - t0:.0f}s")

    shap_top = set(ranked[:RUN["n_candidates"]].tolist())
    hand_idx = {}
    for a, b in HANDPICKED:
        for t, (pa, pb) in enumerate(pair_names):
            if {pa, pb} == {a, b}:
                hand_idx[t] = True

    def source_of(t: int) -> str:
        tags = []
        if t in shap_top:
            tags.append("SHAP top-K")
        if t in hand_idx:
            tags.append("Wave-0 hand-picked")
        return " + ".join(tags) if tags else "scan only"

    irows = []
    by_gain = np.argsort(obs_delta)[::-1]
    report_idx = list(dict.fromkeys(by_gain[:10].tolist()
                                    + sorted(shap_top) + sorted(hand_idx)))
    for t in report_idx:
        irows.append({
            "feature_a": pair_names[t][0], "feature_b": pair_names[t][1],
            "source": source_of(t),
            "scan_rank": int(np.where(by_gain == t)[0][0]) + 1,
            "dMAE_gain": float(obs_delta[t]),
            "null_mean": float(nullB[:, t].mean()),
            "null_p95": float(np.percentile(nullB[:, t], 95)),
            "p_pairwise": float((nullB[:, t] >= obs_delta[t]).mean()),
            "p_familywise": float((nullB_max >= obs_delta[t]).mean()),
            "dAcc_pp": float(obs_dacc[t]),
            "p_acc_familywise": float((nullB_acc_max >= obs_dacc[t]).mean()),
        })
    inter_tab = pl.DataFrame(irows)
    print(f"\n{'interaction':<40}{'dMAE':>9}{'null p95':>10}{'p(FW)':>8}"
          f"{'dAcc':>7}{'p(FW)':>7}")
    print("-" * 78)
    for r in irows:
        nm = f"{r['feature_a']} x {r['feature_b']}"[:39]
        print(f"{nm:<40}{r['dMAE_gain']:>+9.4f}{r['null_p95']:>10.4f}"
              f"{r['p_familywise']:>8.3f}{r['dAcc_pp']:>+7.2f}"
              f"{r['p_acc_familywise']:>7.3f}")
    print("-" * 78)
    best_j = int(np.argmax(obs_delta))
    fw_b = float((nullB_max >= obs_delta[best_j]).mean())
    print(f"best of all {len(obs_vec)} pairs   : {pair_names[best_j][0]} x "
          f"{pair_names[best_j][1]}")
    print(f"  gain                : dMAE {obs_delta[best_j]:+.4f} pts "
          f"({obs_delta[best_j] / base_mae * 100:+.2f}% of baseline MAE), "
          f"dAcc {obs_dacc[best_j]:+.2f} pp")
    print(f"  family-wise p-value : {fw_b:.3f}   (null max mean "
          f"{nullB_max.mean():+.4f}, p95 {np.percentile(nullB_max, 95):+.4f})")

    # does the winner replicate season by season, or is it one season?
    wcol = prod_cols[best_j]
    wpred = fast.predict(wcol)[is_test]
    print(f"\n  season-by-season dMAE for the winning term "
          f"(negative = worse than base):")
    rep = []
    for s in TEST_SEASONS:
        m = season_test == s
        bl = week_blocks(test_rows.filter(pl.col("season") == s))
        bm = pooled(per_week_stats(y_test[m], base_pred[m], bl))[0]
        wm = pooled(per_week_stats(y_test[m], wpred[m], bl))[0]
        rep.append(bm - wm)
        print(f"    {s}: {bm - wm:+.4f}")
    print(f"  replicates in {(np.array(rep) > 0).sum()}/6 seasons; "
          f"mean {np.mean(rep):+.4f} +/- {np.std(rep, ddof=1) / np.sqrt(6):.4f} (SE)")

    # -- honest-split robustness check --------------------------------------
    print(f"\n{BAR}\n 8. ROBUSTNESS: DISCOVERY ON <=2021 ONLY, TESTED ON 2022-2025\n{BAR}")
    print("Sections 5-7 search and score on overlapping rows, which favours the")
    print("search. Here the search sees only 2018-2021 and is graded on 2022-2025,")
    print("so nothing it found can have come from the seasons it is judged on.")
    early = season_all < 2022
    d2 = lgb.LGBMRegressor(n_estimators=n_disc, **LGB_FIXED, **disc_cfg)
    d2.fit(X_all[early], y_all[early])
    sub2 = rng.choice(int(early.sum()), min(600, int(early.sum())), replace=False)
    iv2 = shap.TreeExplainer(d2).shap_interaction_values(X_all[early][sub2])
    vec2 = rank_pairs(iv2)[iu]
    top2 = np.argsort(vec2)[::-1][:5]
    late_seasons = [2022, 2023, 2024, 2025]
    late_mask = np.isin(test_rows["season"].to_numpy(), late_seasons)
    bl_late = week_blocks(test_rows.filter(pl.col("season").is_in(late_seasons)))
    b_mae_late = pooled(per_week_stats(y_test[late_mask], base_pred[late_mask], bl_late))[0]
    print(f"\nbase ridge(19) on 2022-2025 : MAE {b_mae_late:.4f}")
    print(f"\n{'rank':<5}{'interaction discovered pre-2022':<46}{'dMAE 2022-25':>14}")
    print("-" * 78)
    honest_rows = []
    for i, t in enumerate(top2, 1):
        a, b = pair_names[t]
        col = X_all[:, FULL_FEATURES.index(a)] * X_all[:, FULL_FEATURES.index(b)]
        pr = fast.predict(col)[is_test]
        mae = pooled(per_week_stats(y_test[late_mask], pr[late_mask], bl_late))[0]
        honest_rows.append({"rank": i, "feature_a": a, "feature_b": b,
                            "dMAE_2022_2025": float(b_mae_late - mae)})
        print(f"{i:<5}{a + ' x ' + b:<46}{b_mae_late - mae:>+14.4f}")

    # -- plots ---------------------------------------------------------------
    print(f"\n{BAR}\n 9. PLOTS AND ARTEFACTS\n{BAR}")
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25})

    # 9a calibration
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lo = min(y_test.min(), min(p.min() for p in preds.values()))
    hi = max(y_test.max(), max(p.max() for p in preds.values()))
    ax[0].plot([0, 40], [0, 40], color="0.4", lw=1, ls="--", label="perfect")
    for k, c in zip(cal_models, ("#1f77b4", "#2ca02c", "#d62728")):
        cv = cal_curves[k]
        ax[0].errorbar([c_["mean_pred"] for c_ in cv], [c_["mean_actual"] for c_ in cv],
                       yerr=[c_["se_actual"] for c_ in cv], marker="o", ms=4,
                       lw=1.4, capsize=2, color=c, label=k)
    ax[0].set_xlabel("predicted fantasy points (decile bin mean)")
    ax[0].set_ylabel("actual fantasy points")
    ax[0].set_title("Calibration, out-of-sample 2020-2025")
    ax[0].legend(fontsize=8)
    bins = np.linspace(lo, hi, 45)
    ax[1].hist(y_test, bins=bins, alpha=0.35, color="0.5", label="actual")
    for k, c in zip(cal_models, ("#1f77b4", "#2ca02c", "#d62728")):
        ax[1].hist(preds[k], bins=bins, histtype="step", lw=1.5, color=c, label=k)
    ax[1].set_xlabel("fantasy points")
    ax[1].set_ylabel("QB-weeks")
    ax[1].set_title("Prediction spread vs actual spread")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "qb017_calibration.png")
    plt.close(fig)

    # 9b season stability
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    w = 0.25
    xs = np.arange(len(TEST_SEASONS))
    for i, k in enumerate(["GBM additive", "GBM full (lightgbm)",
                           "ridge + GBM residual"]):
        ax[0].bar(xs + (i - 1) * w,
                  [r[f"{k}|MAE"] - r[f"{base_key}|MAE"] for r in srows],
                  width=w, label=k)
        ax[1].bar(xs + (i - 1) * w,
                  [r[f"{k}|acc"] - r[f"{base_key}|acc"] for r in srows],
                  width=w, label=k)
    for a_, t_, yl in ((ax[0], "MAE vs ridge (lower is better)", "dMAE (points)"),
                       (ax[1], "Pairwise accuracy vs ridge (higher is better)",
                        "dAccuracy (pp)")):
        a_.axhline(0, color="0.2", lw=1)
        a_.set_xticks(xs)
        a_.set_xticklabels(TEST_SEASONS)
        a_.set_title(t_)
        a_.set_ylabel(yl)
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "qb017_season_stability.png")
    plt.close(fig)

    # 9c SHAP interactions vs null
    fig, ax = plt.subplots(figsize=(9, 5.2))
    top_n = 15
    names = [f"{shap_rows[i]['feature_a']} x {shap_rows[i]['feature_b']}"
             for i in range(top_n)]
    ypos = np.arange(top_n)[::-1]
    ax.barh(ypos, [shap_rows[i]["shap_interaction_pts"] for i in range(top_n)],
            color="#1f77b4", label="observed")
    ax.plot([shap_rows[i]["null_p95"] for i in range(top_n)], ypos, "o",
            color="#d62728", ms=5, label="permutation null, 95th pct")
    ax.plot([shap_rows[i]["null_mean"] for i in range(top_n)], ypos, "|",
            color="0.3", ms=10, label="permutation null, mean")
    ax.axvline(np.percentile(null_max, 95), color="#d62728", ls="--", lw=1,
               label="family-wise null threshold")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("mean |SHAP interaction| (fantasy points)")
    ax.set_title(f"Top interactions vs permutation null "
                 f"({RUN['n_null_a']} draws, target shuffled within week)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "qb017_shap_interactions.png")
    plt.close(fig)

    # 9d null B distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].hist(nullB_max, bins=40, color="0.6", edgecolor="none")
    ax[0].axvline(obs_delta.max(), color="#d62728", lw=2,
                  label=f"best observed {obs_delta.max():+.4f}")
    ax[0].axvline(np.percentile(nullB_max, 95), color="#1f77b4", ls="--", lw=1.4,
                  label=f"null 95th pct {np.percentile(nullB_max, 95):+.4f}")
    ax[0].set_xlabel("best dMAE over all candidates (points, positive = helps)")
    ax[0].set_ylabel("draws")
    ax[0].set_title("Null B: OOS gain from an interaction, by chance")
    ax[0].legend(fontsize=8)
    ax[1].hist(null_max, bins=40, color="0.6", edgecolor="none")
    ax[1].axvline(obs_top, color="#d62728", lw=2, label=f"observed {obs_top:.4f}")
    ax[1].axvline(np.percentile(null_max, 95), color="#1f77b4", ls="--", lw=1.4,
                  label=f"null 95th pct {np.percentile(null_max, 95):.4f}")
    ax[1].set_xlabel("strongest SHAP interaction (points)")
    ax[1].set_ylabel("draws")
    ax[1].set_title("Null A: strongest interaction found on shuffled targets")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "qb017_permutation_nulls.png")
    plt.close(fig)

    # 9e learned shape functions: is the response curved?
    add_cfg = modal_cfg(cfg_add)
    add_disc = lgb.LGBMRegressor(n_estimators=int(np.median(
        [c["n_estimators"] for c in cfg_add])), **LGB_FIXED, **add_cfg)
    add_disc.fit(X_all, y_all)
    sv = shap.TreeExplainer(add_disc).shap_values(X_sub)
    scal = StandardScaler().fit(X_all)
    rmod = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(scal.transform(X_all), y_all)
    show = ["form_fp_mean", "implied_team_total", "rush_fp_share", "form_pass_att",
            "spread_abs", "career_fp_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for a_, f in zip(axes.ravel(), show):
        j = FULL_FEATURES.index(f)
        o = np.argsort(X_sub[:, j])
        a_.plot(X_sub[o, j], sv[o, j], lw=1.6, color="#2ca02c",
                label="GBM-additive shape")
        lin = rmod.coef_[j] * (X_sub[o, j] - scal.mean_[j]) / scal.scale_[j]
        a_.plot(X_sub[o, j], lin, lw=1.4, ls="--", color="#1f77b4", label="ridge linear")
        a_.set_title(f, fontsize=9)
        a_.set_ylabel("contribution (pts)")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Learned univariate shape functions: additive GBM vs ridge", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "qb017_shape_functions.png")
    plt.close(fig)

    # -- artefacts -----------------------------------------------------------
    comp.write_csv(OUT / "qb017_model_comparison.csv")
    boot.write_csv(OUT / "qb017_bootstrap_deltas.csv")
    per_season.write_csv(OUT / "qb017_by_season.csv")
    calib.write_csv(OUT / "qb017_calibration.csv")
    pl.DataFrame([c for k in cal_curves for c in
                  [{**r, "model": k} for r in cal_curves[k]]]
                 ).write_csv(OUT / "qb017_calibration_curves.csv")
    shap_tab.write_csv(OUT / "qb017_shap_interactions.csv")
    inter_tab.write_csv(OUT / "qb017_interaction_oos_test.csv")
    pl.DataFrame(honest_rows).write_csv(OUT / "qb017_honest_split.csv")
    pl.concat([
        pl.DataFrame(cfg_full).with_columns(pl.lit("GBM full").alias("model")),
        pl.DataFrame(cfg_add).with_columns(pl.lit("GBM additive").alias("model")),
        pl.DataFrame(cfg_res).with_columns(pl.lit("GBM residual").alias("model")),
    ], how="diagonal").write_csv(OUT / "qb017_tuned_params.csv")
    for f in ("qb017_model_comparison.csv", "qb017_bootstrap_deltas.csv",
              "qb017_by_season.csv", "qb017_calibration.csv",
              "qb017_calibration_curves.csv", "qb017_shap_interactions.csv",
              "qb017_interaction_oos_test.csv", "qb017_honest_split.csv",
              "qb017_tuned_params.csv", "qb017_calibration.png",
              "qb017_season_stability.png", "qb017_shap_interactions.png",
              "qb017_permutation_nulls.png", "qb017_shape_functions.png"):
        print(f"  wrote outputs/{f}")

    # -- verdict -------------------------------------------------------------
    print(f"\n{BAR}\n VERDICT\n{BAR}")
    gbm_key = "GBM full (lightgbm)"
    d_mae = pooled(stats[gbm_key])[0] - pooled(stats[base_key])[0]
    d_acc = pooled(stats[gbm_key])[2] - pooled(stats[base_key])[2]
    d_add = pooled(stats["GBM additive"])[0] - pooled(stats[base_key])[0]
    d_int = pooled(stats[gbm_key])[0] - pooled(stats["GBM additive"])[0]
    d_res = pooled(stats["ridge + GBM residual"])[0] - pooled(stats[base_key])[0]
    print("dMAE is signed so that negative = better than the ridge baseline.")
    print(f"univariate curvature (GBM-additive - ridge) : dMAE {d_add:+.4f} pts")
    print(f"interaction structure (GBM-full - additive) : dMAE {d_int:+.4f} pts")
    print(f"trees on the ridge residual                 : dMAE {d_res:+.4f} pts")
    print(f"total (GBM-full - ridge)                    : dMAE {d_mae:+.4f} pts, "
          f"dAcc {d_acc:+.2f} pp")
    print(f"\nstrongest SHAP interaction, family-wise p   : {fw_p:.3f}")
    print(f"best of {len(obs_vec)} OOS interaction gains, family-wise p : {fw_b:.3f}")
    print(f"  that best gain is {obs_delta[best_j]:+.4f} MAE points = "
          f"{obs_delta[best_j] / base_mae * 100:+.2f}% of baseline error, "
          f"replicating in {(np.array(rep) > 0).sum()}/6 seasons")

    p_gbm = float(boot.filter(pl.col("model") == gbm_key)["p_better_mae"][0])
    helps = (d_mae < 0) and (p_gbm > 0.95)
    stat_survives = (fw_p < 0.05) or (fw_b < 0.05)
    # a term can clear a null and still be worthless. 0.05 MAE points is under
    # 1% of the baseline error and far inside the season-to-season spread.
    material = abs(obs_delta[best_j]) >= 0.05 and (np.array(rep) > 0).sum() >= 5

    print(f"\nnonlinearity helps out of sample     : {'YES' if helps else 'NO'} "
          f"(P(GBM beats ridge) = {p_gbm:.2f})")
    print(f"an interaction clears its permutation null : "
          f"{'YES' if stat_survives else 'NO'}")
    print(f"...and is large enough to matter           : "
          f"{'YES' if material else 'NO'}")
    verdict = ("KEEP" if helps and stat_survives and material
               else "UNCERTAIN" if stat_survives and not helps and material
               else "REJECT")
    print(f"\nVERDICT: {verdict} -- "
          + ("tree models and interactions earn their place"
             if verdict == "KEEP" else
             "the linear model is not leaving usable structure on the table"))
    print(f"\ntotal runtime: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
