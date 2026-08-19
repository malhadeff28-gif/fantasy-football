"""DST-004: does the opposing QUARTERBACK add anything beyond his team?

The brief puts heavy emphasis on opposing-QB vulnerability - the idea that a
defense facing a turnover-prone or sack-prone quarterback is a better fantasy
start than the team-level matchup alone suggests. That is a hypothesis, not a
fact, and DST-002 already showed a single market variable beats every richer
model, so the prior here should be sceptical.

Guards carried over from the QB lab, which are not optional:
  - Stability does not imply incremental value (QB-018 found six variables more
    stable than fantasy points itself, none of which helped).
  - A single permutation null is not enough when the statistic is
    scale-dependent (QB-017 shipped a false positive at p=0.000 on its first
    null). Interactions here face an exhaustive scan with a max-statistic null.
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
import nfl_data                                   # noqa: E402
import dst_data                                   # noqa: E402
import dst_features                               # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_width_chars(150)
SEASONS = list(range(2018, 2026))
TEST = list(range(2020, 2026))
BAR = "=" * 84
RNG = np.random.default_rng(20260818)

QB_FEATS = ["oqb_int_rate", "oqb_sack_rate", "oqb_fum_rate",
            "oqb_fp_mean", "oqb_games"]
BASE = ["implied_opp_total", "opp_generosity_shrunk", "is_home"]


def opposing_qb_panel() -> pl.DataFrame:
    """Prior-only features for the quarterback each defense faces.

    Identity comes from who actually took the most attempts that week, which is
    knowable pregame in the overwhelming majority of cases. Every FEATURE is
    strictly prior, so nothing about the game itself leaks in.
    """
    st = nfl_data.load_player_stats(SEASONS).filter(
        (pl.col("position") == "QB") & (pl.col("season_type") == "REG"))

    starters = (st.sort(["team", "season", "week", "attempts"],
                        descending=[False, False, False, True])
                  .group_by(["team", "season", "week"], maintain_order=True)
                  .head(1))

    s = starters.sort(["player_id", "season", "week"])

    def roll_sum(col: str, alias: str, w: int = 12) -> pl.Expr:
        return (pl.col(col).shift(1).rolling_sum(window_size=w, min_samples=1)
                .over("player_id").alias(alias))

    s = s.with_columns(
        roll_sum("passing_interceptions", "_int"),
        roll_sum("attempts", "_att"),
        roll_sum("sacks_suffered", "_sack"),
        roll_sum("sack_fumbles_lost", "_fum"),
        pl.col("fantasy_points").shift(1).rolling_mean(window_size=12, min_samples=1)
        .over("player_id").alias("oqb_fp_mean"),
        pl.col("fantasy_points").shift(1).is_not_null().cum_sum()
        .over("player_id").alias("oqb_games"),
    ).with_columns(
        (pl.col("_int") / pl.col("_att").clip(lower_bound=1)).alias("oqb_int_rate"),
        (pl.col("_sack") / (pl.col("_att") + pl.col("_sack")).clip(lower_bound=1))
        .alias("oqb_sack_rate"),
        (pl.col("_fum") / pl.col("_att").clip(lower_bound=1)).alias("oqb_fum_rate"),
    )
    return s.select(["team", "season", "week", "player_display_name"] + QB_FEATS)


def pair_acc(df: pl.DataFrame, col: str) -> float:
    ok = tot = 0
    for _k, g in df.group_by(["season", "week"], maintain_order=True):
        p, a = g[col].to_numpy(), g["dst_points"].to_numpy()
        for i, j in combinations(range(len(p)), 2):
            if a[i] == a[j] or p[i] == p[j]:
                continue
            tot += 1
            ok += (p[i] > p[j]) == (a[i] > a[j])
    return ok / tot if tot else float("nan")


def walk_forward(d: pl.DataFrame, feats: list[str]) -> tuple[float, float]:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    maes, frames = [], []
    for s in TEST:
        tr, te = d.filter(pl.col("season") < s), d.filter(pl.col("season") == s)
        if tr.height < 200 or te.is_empty():
            continue
        X_tr = tr.select(feats).to_numpy().astype(float)
        X_te = te.select(feats).to_numpy().astype(float)
        y_tr, y_te = tr["dst_points"].to_numpy(), te["dst_points"].to_numpy()
        sc = StandardScaler().fit(X_tr)
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X_tr), y_tr)
        pred = m.predict(sc.transform(X_te))
        maes.append(float(np.mean(np.abs(pred - y_te))))
        frames.append(te.select(["season", "week", "dst_points"])
                      .with_columns(pl.Series("pred", pred)))
    oos = pl.concat(frames)
    return float(np.mean(maes)), pair_acc(oos, "pred")


def main() -> int:
    d = dst_features.add_dst_features(dst_data.build_dst_panel(SEASONS))
    oqb = opposing_qb_panel()
    d = d.join(oqb, left_on=["opponent", "season", "week"],
               right_on=["team", "season", "week"], how="left")

    before = d.height
    d = d.drop_nulls(subset=BASE + QB_FEATS)
    print(f"D/ST panel joined to opposing QB: {before} -> {d.height} rows "
          f"({d.height / before:.1%} retained)")

    print(f"\n{BAR}\n DST-004a: DOES OPPOSING-QB IDENTITY ADD ANYTHING?\n{BAR}")
    sets = {
        "market only": ["implied_opp_total"],
        "baseline (market + opp offence + home)": BASE,
        "baseline + opposing QB": BASE + QB_FEATS,
        "opposing QB only": QB_FEATS,
    }
    rows = []
    for name, f in sets.items():
        mae, acc = walk_forward(d, f)
        rows.append((name, mae, acc))
    print(f"{'feature set':<42}{'MAE':>8}{'pair acc':>11}")
    print("-" * 84)
    for n, mae, acc in rows:
        print(f"{n:<42}{mae:>8.3f}{acc * 100:>10.2f}%")
    base_acc = [r for r in rows if r[0].startswith("baseline (")][0][2]
    add_acc = [r for r in rows if r[0].endswith("opposing QB")][0][2]
    print("-" * 84)
    print(f"opposing-QB increment: {(add_acc - base_acc) * 100:+.2f} pp")

    print(f"\n{BAR}\n DST-004b: WHICH QB TRAITS CORRELATE WITH D/ST OUTCOMES?\n{BAR}")
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    res_frames = []
    for s in TEST:
        tr, te = d.filter(pl.col("season") < s), d.filter(pl.col("season") == s)
        if tr.height < 200:
            continue
        sc = StandardScaler().fit(tr.select(BASE).to_numpy().astype(float))
        m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(
            sc.transform(tr.select(BASE).to_numpy().astype(float)),
            tr["dst_points"].to_numpy())
        r = te["dst_points"].to_numpy() - m.predict(
            sc.transform(te.select(BASE).to_numpy().astype(float)))
        res_frames.append(te.with_columns(pl.Series("resid", r)))
    rd = pl.concat(res_frames)

    print("Correlation with the residual from the team-level baseline, i.e. what")
    print("the quarterback adds beyond who he plays for.\n")
    print(f"{'opposing-QB trait':<24}{'corr with residual':>20}{'n':>8}")
    print("-" * 84)
    scored = []
    for c in QB_FEATS:
        sub = rd.drop_nulls(subset=[c])
        v = float(np.corrcoef(sub[c].to_numpy().astype(float),
                              sub["resid"].to_numpy())[0, 1])
        scored.append((c, v, sub.height))
    for c, v, n in sorted(scored, key=lambda t: -abs(t[1])):
        print(f"{c:<24}{v:>20.4f}{n:>8}")

    print(f"\n{BAR}\n DST-004c: INTERACTION SEARCH WITH A MAX-STATISTIC NULL\n{BAR}")
    print("The brief proposes: strong pass rush + QB with a poor pressure-to-sack")
    print("profile + elevated interception expectation + low implied total.")
    print("Every pairwise product is tested, not just the proposed one, so the")
    print("winner is judged against the whole search rather than on its own.\n")

    cand = [c for c in BASE + QB_FEATS + ["form_sacks", "form_ints"] if c in d.columns]
    # products of a null are null, so the interaction candidates need the
    # same completeness the base features already have
    d = d.drop_nulls(subset=cand)
    print(f"rows with all interaction candidates present: {d.height}")
    pairs = list(combinations(cand, 2))
    base_mae, _ = walk_forward(d, BASE + QB_FEATS)
    gains = []
    for a, b in pairs:
        dd = d.with_columns((pl.col(a) * pl.col(b)).alias("_x"))
        mae, _ = walk_forward(dd, BASE + QB_FEATS + ["_x"])
        gains.append((f"{a} x {b}", base_mae - mae))
    gains.sort(key=lambda t: -t[1])
    print(f"{'interaction':<52}{'MAE gain':>12}")
    print("-" * 84)
    for nm, g in gains[:6]:
        print(f"{nm:<52}{g:>+12.4f}")

    print("\nrunning max-statistic permutation null (target shuffled within week)")
    null_max = []
    for _ in range(10):
        dd = d.with_columns(
            pl.col("dst_points").shuffle(seed=int(RNG.integers(1_000_000_000)))
            .over(["season", "week"]).alias("dst_points"))
        bm, _ = walk_forward(dd, BASE + QB_FEATS)
        best = -9e9
        for a, b in pairs:
            ddd = dd.with_columns((pl.col(a) * pl.col(b)).alias("_x"))
            mae, _ = walk_forward(ddd, BASE + QB_FEATS + ["_x"])
            best = max(best, bm - mae)
        null_max.append(best)
    obs = gains[0][1]
    p = float(np.mean([n >= obs for n in null_max]))
    print(f"\nobserved best gain: {obs:+.4f}")
    print(f"null max gains: mean {np.mean(null_max):+.4f}, max {np.max(null_max):+.4f}")
    print(f"p-value against the max-statistic null: {p:.3f}")
    print("\nAn interaction that does not clear its own search-wide null is noise,")
    print("however good its individual p-value looks.")

    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    pl.DataFrame([{"model": n, "mae": m, "pair_acc": a} for n, m, a in rows]
                 ).write_csv(out / "dst004_model_comparison.csv")
    pl.DataFrame([{"feature": c, "r": v, "n": n} for c, v, n in scored]
                 ).write_csv(out / "dst004_qb_trait_correlations.csv")
    pl.DataFrame([{"interaction": nm, "mae_gain": g} for nm, g in gains]
                 ).write_csv(out / "dst004_interactions.csv")
    print(f"\nwrote dst004_*.csv to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
