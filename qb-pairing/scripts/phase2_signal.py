"""Phase 2: measure which pregame features actually predict QB fantasy points.

The brief's own stated risk is that schedule difficulty may be less predictive
than expected, so it must be tested independently *before* the pairing model
gives it weight. This script does that with nested feature sets evaluated
walk-forward, never in-sample.

Two questions matter more than raw error:

1. Does opponent difficulty add anything over simply knowing how good the
   quarterback has been lately?
2. Does it add anything *on top of* the betting market -- or is schedule
   difficulty just a noisy proxy for information the market already prices?

Question 2 has a twist specific to this project: at draft time there is no
market, so even a fully-subsumed schedule signal may still be the best
available preseason input. Both views are reported.

Usage:  python scripts/phase2_signal.py
"""
from __future__ import annotations

import os

# OpenBLAS pre-allocates per-thread buffers and fails to load on this machine
# with the default thread count. Must be set before numpy/scipy are imported.
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

from qbpair.features import panel as panel_mod   # noqa: E402

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))   # need >=2 seasons of training history
BAR = "=" * 78

FORM = ["form_fp_mean", "career_fp_mean", "form_pass_att", "form_rush_att"]
SCHED = ["def_adj", "def_fp_allowed"]
MARKET = ["implied_team_total", "spread_abs", "total_line"]
CONTEXT = ["is_home", "indoors", "rest", "div_game"]

MODELS = {
    "M0 league mean": [],
    "M1 form only": FORM,
    "M2 form + schedule": FORM + SCHED,
    "M3 form + market": FORM + MARKET,
    "M4 form + market + schedule": FORM + MARKET + SCHED,
    "M5 M4 + context": FORM + MARKET + SCHED + CONTEXT,
    "M6 M5 + QB x matchup": None,   # built below (interactions)
}

INTERACTIONS = [
    ("rush_fp_share", "def_adj"),
    ("rush_fp_share", "spread_abs"),
    ("rush_fp_share", "implied_team_total"),
    ("form_fp_mean", "def_adj"),
    ("form_pass_att", "spread_abs"),
]


def add_interactions(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    names = []
    exprs = []
    for a, b in INTERACTIONS:
        nm = f"{a}_X_{b}"
        names.append(nm)
        exprs.append((pl.col(a) * pl.col(b)).alias(nm))
    return df.with_columns(exprs), names


def pairwise_accuracy(df: pl.DataFrame, pred_col: str) -> tuple[float, int]:
    """Within each week, how often does the model rank two QBs correctly?

    This is the metric the project actually needs: a start/sit decision is a
    pairwise comparison, not an absolute projection.
    """
    correct = total = 0
    for (_s, _w), g in df.group_by(["season", "week"], maintain_order=True):
        pred = g[pred_col].to_numpy()
        act = g["fantasy_points"].to_numpy()
        for i, j in combinations(range(len(pred)), 2):
            if act[i] == act[j] or pred[i] == pred[j]:
                continue
            total += 1
            if (pred[i] > pred[j]) == (act[i] > act[j]):
                correct += 1
    return (correct / total if total else float("nan")), total


def run() -> int:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    p = panel_mod.build_panel(SEASONS)
    p, inter_names = add_interactions(p)
    MODELS["M6 M5 + QB x matchup"] = (
        FORM + MARKET + SCHED + CONTEXT + ["rush_fp_share"] + inter_names
    )

    # Debut quarterbacks have no prior form; drop rather than impute so the
    # comparison across models uses an identical row set.
    before = p.height
    p = p.drop_nulls(subset=["form_fp_mean", "form_pass_att", "rush_fp_share"])
    print(f"panel: {before} QB-weeks -> {p.height} with prior form "
          f"({before - p.height} debuts dropped)")
    print(f"target: mean {p['fantasy_points'].mean():.2f}, "
          f"sd {p['fantasy_points'].std():.2f}\n")

    results = {name: [] for name in MODELS}
    preds_store: dict[str, pl.DataFrame] = {}

    for name, feats in MODELS.items():
        rows = []
        oos_frames = []
        for test_s in TEST_SEASONS:
            tr = p.filter(pl.col("season") < test_s)
            te = p.filter(pl.col("season") == test_s)
            if tr.height < 200:
                continue
            y_tr = tr["fantasy_points"].to_numpy()
            y_te = te["fantasy_points"].to_numpy()

            if not feats:                      # M0: predict the training mean
                pred = np.full(len(y_te), y_tr.mean())
            else:
                X_tr = tr.select(feats).to_numpy().astype(float)
                X_te = te.select(feats).to_numpy().astype(float)
                sc = StandardScaler().fit(X_tr)
                model = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X_tr), y_tr)
                pred = model.predict(sc.transform(X_te))

            mae = float(np.mean(np.abs(pred - y_te)))
            rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
            # R^2 against the naive "training mean" forecast
            ss_res = float(np.sum((y_te - pred) ** 2))
            ss_tot = float(np.sum((y_te - y_tr.mean()) ** 2))
            rows.append((mae, rmse, 1 - ss_res / ss_tot))
            oos_frames.append(te.select(["season", "week", "fantasy_points"])
                              .with_columns(pl.Series("pred", pred)))

        arr = np.array(rows)
        oos = pl.concat(oos_frames)
        preds_store[name] = oos
        acc, npairs = pairwise_accuracy(oos, "pred")
        results[name] = {
            "mae": arr[:, 0].mean(), "rmse": arr[:, 1].mean(),
            "r2": arr[:, 2].mean(), "acc": acc, "pairs": npairs,
        }

    print(BAR)
    print(" WALK-FORWARD OUT-OF-SAMPLE RESULTS (train < S, test = S; S = 2020-2025)")
    print(BAR)
    print(f"{'model':<30}{'MAE':>7}{'RMSE':>8}{'R2':>8}{'pair acc':>10}")
    print("-" * 78)
    base_mae = results["M1 form only"]["mae"]
    for name in MODELS:
        r = results[name]
        delta = "" if name == "M1 form only" else f"  ({r['mae'] - base_mae:+.3f})"
        print(f"{name:<30}{r['mae']:>7.3f}{r['rmse']:>8.3f}{r['r2']:>8.4f}"
              f"{r['acc'] * 100:>9.2f}%{delta}")
    print("-" * 78)
    print(f"pairwise comparisons evaluated: {results['M1 form only']['pairs']:,}")

    # --- the decisive contrasts --------------------------------------------
    print(f"\n{BAR}\n MARGINAL CONTRIBUTION\n{BAR}")

    def contrast(label: str, a: str, b: str) -> None:
        ra, rb = results[a], results[b]
        print(f"{label}")
        print(f"    MAE {ra['mae']:.3f} -> {rb['mae']:.3f}  ({rb['mae'] - ra['mae']:+.3f})"
              f"   pair acc {ra['acc'] * 100:.2f}% -> {rb['acc'] * 100:.2f}% "
              f"({(rb['acc'] - ra['acc']) * 100:+.2f} pp)")

    contrast("schedule difficulty added to form:", "M1 form only", "M2 form + schedule")
    contrast("market added to form:", "M1 form only", "M3 form + market")
    contrast("schedule added ON TOP OF market:", "M3 form + market", "M4 form + market + schedule")
    contrast("QB x matchup interactions added:", "M5 M4 + context", "M6 M5 + QB x matchup")

    # --- which individual features carry signal ----------------------------
    print(f"\n{BAR}\n FEATURE SIGNAL vs FORM-ONLY RESIDUAL (full panel)\n{BAR}")
    resid_frames = []
    for test_s in TEST_SEASONS:
        tr = p.filter(pl.col("season") < test_s)
        te = p.filter(pl.col("season") == test_s)
        X_tr = tr.select(FORM).to_numpy().astype(float)
        X_te = te.select(FORM).to_numpy().astype(float)
        sc = StandardScaler().fit(X_tr)
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X_tr),
                                                       tr["fantasy_points"].to_numpy())
        r = te["fantasy_points"].to_numpy() - m.predict(sc.transform(X_te))
        resid_frames.append(te.with_columns(pl.Series("resid", r)))
    rdf = pl.concat(resid_frames)

    cand = MARKET + SCHED + CONTEXT + ["rush_fp_share", "form_fp_sd"]
    print(f"{'feature':<24}{'corr w/ residual':>18}{'|r|':>8}")
    print("-" * 78)
    scored = []
    for c in cand:
        sub = rdf.drop_nulls(subset=[c])
        v = float(np.corrcoef(sub[c].to_numpy().astype(float),
                              sub["resid"].to_numpy())[0, 1])
        scored.append((c, v))
    for c, v in sorted(scored, key=lambda t: -abs(t[1])):
        print(f"{c:<24}{v:>18.4f}{abs(v):>8.4f}")

    print(f"\n{BAR}")
    print(" Correlations near zero mean the feature adds nothing beyond knowing")
    print(" how the quarterback has been playing. Read alongside pair accuracy:")
    print(" a tiny MAE gain that moves start/sit accuracy is still useful.")
    print(BAR)

    out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
    pl.DataFrame([
        {"model": k, **{kk: float(vv) for kk, vv in v.items()}}
        for k, v in results.items()
    ]).write_csv(out / "phase2_model_comparison.csv")
    rdf.select(["season", "week", "player_display_name", "opponent",
                "fantasy_points", "resid"] + cand).write_csv(
        out / "phase2_residuals.csv")
    print(f"\nwrote outputs/phase2_model_comparison.csv and phase2_residuals.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
