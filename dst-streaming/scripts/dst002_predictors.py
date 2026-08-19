"""DST-002 / DST-003: what predicts weekly D/ST points, and what is just noise.

The benchmark that matters is not another model. It is the manager who each
week simply starts whichever defense faces the lowest opponent implied total.
That takes thirty seconds and needs no model. Anything here must beat it or
admit that it did not.
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
sys.path.insert(0, str(PROJECT / "src"))
import dst_data                                   # noqa: E402
import dst_features                               # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_width_chars(150)
SEASONS = list(range(2018, 2026))
TEST = list(range(2020, 2026))
BAR = "=" * 84


def pair_acc(df: pl.DataFrame, col: str) -> float:
    """Within-week pairwise ranking accuracy: the start/sit decision metric."""
    ok = tot = 0
    for _k, g in df.group_by(["season", "week"], maintain_order=True):
        p, a = g[col].to_numpy(), g["dst_points"].to_numpy()
        for i, j in combinations(range(len(p)), 2):
            if a[i] == a[j] or p[i] == p[j]:
                continue
            tot += 1
            ok += (p[i] > p[j]) == (a[i] > a[j])
    return ok / tot if tot else float("nan")


def main() -> int:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    d = dst_features.add_dst_features(
        dst_data.build_dst_panel(SEASONS)).drop_nulls(subset=dst_features.FEATURES)
    print(f"D/ST panel with features: {d.height} defense-weeks")

    # ---- DST-003: separate stable traits from noisy outcomes ----
    print(f"\n{BAR}\n DST-003: STABLE TRAIT OR NOISY OUTCOME?\n{BAR}")
    print("Season-to-season correlation of each team rate (season N vs N+1):\n")
    per = (d.group_by(["team", "season"]).agg(
        pl.col("def_sacks").mean().alias("sacks"),
        pl.col("def_interceptions").mean().alias("ints"),
        pl.col("fumble_recovery_opp").mean().alias("fum"),
        pl.col("def_tds").mean().alias("def_td"),
        pl.col("points_allowed").mean().alias("pa"),
        pl.col("dst_points").mean().alias("dst")))
    nxt = per.with_columns((pl.col("season") - 1).alias("prev"))
    j = per.join(nxt, left_on=["team", "season"], right_on=["team", "prev"],
                 suffix="_next", how="inner")
    print(f"{'metric':<16}{'r(N, N+1)':>12}{'n':>6}   verdict")
    print("-" * 84)
    for m in ["sacks", "ints", "fum", "def_td", "pa", "dst"]:
        r = float(np.corrcoef(j[m].to_numpy(), j[f"{m}_next"].to_numpy())[0, 1])
        v = ("stable trait" if r >= 0.4
             else "weak" if r >= 0.2 else "essentially random")
        print(f"{m:<16}{r:>12.3f}{j.height:>6}   {v}")
    print("\n=> anything near zero is a coin flip and must not be projected forward")

    # ---- DST-002: predicting weekly points ----
    print(f"\n{BAR}\n DST-002: PREDICTING WEEKLY D/ST POINTS (walk-forward)\n{BAR}")
    sets = {
        "M0 league mean": [],
        "M1 market only (implied opp total)": ["implied_opp_total"],
        "M2 defensive form only": ["form_dst_pts", "form_sacks", "form_ints",
                                   "form_fum", "form_pa"],
        "M3 form + opponent offence": ["form_dst_pts", "form_sacks", "form_ints",
                                       "form_fum", "form_pa",
                                       "opp_generosity_shrunk", "opp_ppg_prior"],
        "M4 form + market": ["form_dst_pts", "form_sacks", "form_ints", "form_fum",
                             "form_pa", "implied_opp_total", "spread_abs"],
        "M5 everything": dst_features.FEATURES,
    }
    rows = []
    for name, feats in sets.items():
        maes, frames = [], []
        for s in TEST:
            tr, te = d.filter(pl.col("season") < s), d.filter(pl.col("season") == s)
            if tr.height < 200:
                continue
            y_tr, y_te = tr["dst_points"].to_numpy(), te["dst_points"].to_numpy()
            if not feats:
                pred = np.full(len(y_te), y_tr.mean())
            else:
                X_tr = tr.select(feats).to_numpy().astype(float)
                X_te = te.select(feats).to_numpy().astype(float)
                sc = StandardScaler().fit(X_tr)
                m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X_tr), y_tr)
                pred = m.predict(sc.transform(X_te))
            maes.append(float(np.mean(np.abs(pred - y_te))))
            frames.append(te.select(["season", "week", "dst_points"])
                          .with_columns(pl.Series("pred", pred)))
        oos = pl.concat(frames)
        rows.append((name, float(np.mean(maes)), pair_acc(oos, "pred")))

    print(f"{'model':<38}{'MAE':>8}{'pair acc':>11}")
    print("-" * 84)
    for n, mae, acc in rows:
        print(f"{n:<38}{mae:>8.3f}{acc * 100:>10.2f}%")

    # ---- the benchmark that actually matters ----
    print(f"\n{BAR}\n THE BENCHMARK THAT MATTERS\n{BAR}")
    triv = d.filter(pl.col("season").is_in(TEST)).with_columns(
        (-pl.col("implied_opp_total")).alias("pred"))
    tacc = pair_acc(triv, "pred")
    # M0 predicts a constant, so every within-week pair ties and its
    # accuracy is nan; excluding it keeps max() meaningful
    scored = [r for r in rows if r[2] == r[2]]
    best = max(scored, key=lambda r: r[2])
    print(f"trivial rule (lowest implied opponent total): {tacc * 100:.2f}%")
    print(f"best model ({best[0]}): {best[2] * 100:.2f}%")
    print(f"=> model beats the thirty-second rule by {(best[2] - tacc) * 100:+.2f} pp")
    print("\nIf that gap is not clearly positive, the honest conclusion is that")
    print("the trivial rule is the strategy and the model is decoration.")

    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    pl.DataFrame([{"model": n, "mae": m, "pair_acc": a} for n, m, a in rows]
                 ).write_csv(out / "dst002_model_comparison.csv")
    print(f"\nwrote {out / 'dst002_model_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
