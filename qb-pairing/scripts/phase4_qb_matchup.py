"""Phase 4: do quarterbacks respond to opponents differently?

The brief's central matchup concept is that the same defence should affect two
quarterbacks differently depending on their characteristics. Phase 2 tested a
few hand-picked interactions and found nothing; this phase tests the idea
properly, at three levels of granularity, and asks which generalises.

1. League-wide  -- one matchup coefficient for every quarterback.
2. Archetype    -- quarterbacks clustered by measurable play style, each
                   cluster getting its own matchup coefficient.
3. Individual   -- a matchup coefficient per quarterback.

Granularity that does not beat the simpler model out of sample is unstable
complexity and gets rejected, per the brief.

The decisive diagnostic is the split-half reliability test: if a quarterback's
matchup sensitivity is a real trait, it should be similar in two random halves
of his own games. If the halves disagree, the "individual effect" is noise
being fitted, and no amount of model tuning will fix it.

Usage:  python scripts/phase4_qb_matchup.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                      # noqa: E402
import warnings                                 # noqa: E402
from itertools import combinations              # noqa: E402
from pathlib import Path                        # noqa: E402

import numpy as np                              # noqa: E402
import polars as pl                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.features import panel as panel_mod  # noqa: E402

warnings.filterwarnings("ignore")

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
BAR = "=" * 78
RNG = np.random.default_rng(20260817)

STYLE = ["rush_fp_share", "form_adot", "form_pass_att", "form_sack_rate"]
BASE = ["form_fp_mean", "career_fp_mean", "form_pass_att", "form_rush_att",
        "implied_team_total", "spread_abs", "total_line",
        "is_home", "indoors", "rest", "div_game"]
MATCHUP = "def_adj"
N_ARCH = 3


def load() -> pl.DataFrame:
    p = panel_mod.build_panel(SEASONS)
    p = p.drop_nulls(subset=BASE + STYLE + [MATCHUP])
    return p.with_columns(
        (pl.col("fantasy_points") - pl.col("form_fp_mean")).alias("form_resid"))


def describe_archetypes(p: pl.DataFrame) -> None:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = p.select(STYLE).to_numpy().astype(float)
    sc = StandardScaler().fit(X)
    km = KMeans(n_clusters=N_ARCH, n_init=10, random_state=0).fit(sc.transform(X))
    q = p.with_columns(pl.Series("arch", km.labels_))

    print(f"\n{'archetype':<12}{'n':>7}{'rush share':>12}{'aDOT':>8}"
          f"{'pass att':>10}{'sack rt':>9}{'mean FP':>9}")
    print("-" * 78)
    for a in range(N_ARCH):
        g = q.filter(pl.col("arch") == a)
        print(f"{a:<12}{g.height:>7}{g['rush_fp_share'].mean():>12.3f}"
              f"{g['form_adot'].mean():>8.2f}{g['form_pass_att'].mean():>10.1f}"
              f"{g['form_sack_rate'].mean():>9.3f}{g['fantasy_points'].mean():>9.2f}")
    for a in range(N_ARCH):
        g = q.filter(pl.col("arch") == a)
        top = (g.group_by("player_display_name").agg(pl.len().alias("n"))
               .sort("n", descending=True).head(5)["player_display_name"].to_list())
        print(f"  archetype {a}: {', '.join(top)}")


def pairwise_accuracy(df: pl.DataFrame, pred: str) -> float:
    correct = total = 0
    for _k, g in df.group_by(["season", "week"], maintain_order=True):
        pr, ac = g[pred].to_numpy(), g["fantasy_points"].to_numpy()
        for i, j in combinations(range(len(pr)), 2):
            if ac[i] == ac[j] or pr[i] == pr[j]:
                continue
            total += 1
            correct += (pr[i] > pr[j]) == (ac[i] > ac[j])
    return correct / total if total else float("nan")


def compare_granularity(p: pl.DataFrame) -> None:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    print(f"\n{BAR}\n GRANULARITY COMPARISON (walk-forward, train < S)\n{BAR}")

    results = {}
    for variant in ["no matchup", "league-wide", "archetype", "individual"]:
        maes, frames = [], []
        for s in TEST_SEASONS:
            tr, te = p.filter(pl.col("season") < s), p.filter(pl.col("season") == s)

            def design(df: pl.DataFrame, km=None, sc_style=None, ids=None):
                cols = [df.select(BASE).to_numpy().astype(float)]
                if variant != "no matchup":
                    m = df[MATCHUP].to_numpy().astype(float).reshape(-1, 1)
                    cols.append(m)
                if variant == "archetype":
                    Z = sc_style.transform(df.select(STYLE).to_numpy().astype(float))
                    lab = km.predict(Z)
                    D = np.zeros((df.height, N_ARCH))
                    D[np.arange(df.height), lab] = 1.0
                    cols.append(D * m)                      # archetype x matchup
                if variant == "individual":
                    pid = df["player_id"].to_list()
                    D = np.zeros((df.height, len(ids)))
                    for r, q in enumerate(pid):
                        if q in ids:
                            D[r, ids[q]] = 1.0
                    cols.append(D * m)                      # per-QB x matchup
                return np.hstack(cols)

            km = sc_style = ids = None
            if variant == "archetype":
                sc_style = StandardScaler().fit(tr.select(STYLE).to_numpy().astype(float))
                km = KMeans(n_clusters=N_ARCH, n_init=10, random_state=0).fit(
                    sc_style.transform(tr.select(STYLE).to_numpy().astype(float)))
            if variant == "individual":
                cnt = tr.group_by("player_id").agg(pl.len().alias("n"))
                keep = cnt.filter(pl.col("n") >= 16)["player_id"].to_list()
                ids = {q: i for i, q in enumerate(sorted(keep))}

            Xtr, Xte = design(tr, km, sc_style, ids), design(te, km, sc_style, ids)
            ytr, yte = tr["fantasy_points"].to_numpy(), te["fantasy_points"].to_numpy()
            sc = StandardScaler().fit(Xtr)
            mdl = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(sc.transform(Xtr), ytr)
            pr = mdl.predict(sc.transform(Xte))
            maes.append(np.mean(np.abs(pr - yte)))
            frames.append(te.select(["season", "week", "fantasy_points"])
                          .with_columns(pl.Series("pred", pr)))
        oos = pl.concat(frames)
        results[variant] = (float(np.mean(maes)), pairwise_accuracy(oos, "pred"))

    print(f"{'matchup granularity':<24}{'MAE':>9}{'pair acc':>11}{'vs simplest':>14}")
    print("-" * 78)
    base = results["no matchup"][0]
    for k, (mae, acc) in results.items():
        d = "" if k == "no matchup" else f"{mae - base:+.4f}"
        print(f"{k:<24}{mae:>9.4f}{acc * 100:>10.2f}%{d:>14}")
    print("-" * 78)
    print("Any variant that fails to beat 'no matchup' is complexity without payoff.")


def split_half_reliability(p: pl.DataFrame) -> None:
    print(f"\n{BAR}\n SPLIT-HALF RELIABILITY OF INDIVIDUAL MATCHUP SENSITIVITY\n{BAR}")
    print("For each QB, regress his form residual on opponent difficulty in two")
    print("random halves of his games. A real trait shows up in both halves.\n")

    rows = []
    for pid, g in p.group_by("player_id"):
        if g.height < 40:
            continue
        idx = RNG.permutation(g.height)
        h1, h2 = idx[: g.height // 2], idx[g.height // 2:]
        x = g[MATCHUP].to_numpy().astype(float)
        y = g["form_resid"].to_numpy().astype(float)
        s = []
        ok = True
        for h in (h1, h2):
            if x[h].std() < 1e-9:
                ok = False
                break
            s.append(np.polyfit(x[h], y[h], 1)[0])
        if ok:
            rows.append({"player_id": pid[0], "s1": s[0], "s2": s[1],
                         "name": g["player_display_name"][0], "n": g.height})

    d = pl.DataFrame(rows)
    r = float(np.corrcoef(d["s1"].to_numpy(), d["s2"].to_numpy())[0, 1])
    print(f"quarterbacks with >= 40 games: {d.height}")
    print(f"correlation between half-1 and half-2 matchup slopes: r = {r:+.4f}")
    print(f"    => split-half reliability explains {max(r, 0) ** 2 * 100:.1f}% of variance")
    if abs(r) < 0.2:
        print("\nVERDICT: individual matchup sensitivity does not replicate within a")
        print("quarterback's own games. Per-QB matchup coefficients would be fitting")
        print("noise, and must be rejected as unstable complexity.")
    else:
        print("\nVERDICT: some individual signal replicates; worth modelling with"
              " shrinkage.")

    out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
    d.write_csv(out / "phase4_split_half.csv")
    print(f"\nwrote outputs/phase4_split_half.csv")


def main() -> int:
    p = load()
    print(f"panel: {p.height} QB-weeks with full style features")
    print(f"{BAR}\n QB ARCHETYPES (k-means on measurable play style)\n{BAR}")
    describe_archetypes(p)
    compare_granularity(p)
    split_half_reliability(p)
    print(f"\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
