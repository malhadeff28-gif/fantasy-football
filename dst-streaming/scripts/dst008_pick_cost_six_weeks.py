"""DST-008: what a draft pick returns over WEEKS 1-6, so §8 compares like with like.

`qb-pairing/outputs/phase6_pick_value_curve.csv` measures a pick's usable points
above replacement over the league's whole fantasy season (weeks 1-16). The D/ST
marginal test measures six weeks, because that is as far as betting lines reach.
Comparing the two directly is a units error, and prorating by 6/16 is wrong too:
the six-week share turns out to be ~58%, not 37.5%.

So the same curve is rebuilt with the season truncated at week 6 and replacement
levels re-derived on six-week totals - the same code path, one parameter moved.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                        # noqa: E402
from pathlib import Path                          # noqa: E402

import polars as pl                               # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))
sys.path.insert(0, str(REPO / "qb-pairing" / "src"))
import bootstrap                                  # noqa: E402
bootstrap.setup()
from qbpair.benchmarks import draft_value as dv   # noqa: E402
from qbpair.data import loaders                   # noqa: E402

pl.Config.set_tbl_formatting("ASCII_FULL")
SEASONS = list(range(2018, 2026))
BUCKET = 12
BAR = "=" * 84


def curve(last_week: int, bucket: int = BUCKET) -> pl.DataFrame:
    """dv.pick_value_curve, but with the season cut at `last_week`.

    Replacement is recomputed on the truncated totals, so PAR is measured
    against the bar that actually applies over that window.
    """
    st = loaders.load_player_stats(SEASONS).filter(
        (pl.col("season_type") == "REG") & (pl.col("week") <= last_week)
        & pl.col("position").is_in(list(dv.REPLACEMENT_RANK)))
    totals = (st.group_by(["season", "player_id", "player_display_name", "position"])
              .agg(pl.col("fantasy_points").sum().alias("season_points"),
                   pl.len().alias("games"))
              .with_columns(pl.col("player_display_name")
                            .map_elements(dv.normalise_name, return_dtype=pl.String)
                            .alias("key")))
    par = (totals.join(dv.replacement_levels(totals), on=["season", "position"],
                       how="left")
           .with_columns((pl.col("season_points")
                          - pl.col("replacement_points")).alias("par"))
           .with_columns(pl.col("par").clip(lower_bound=0.0).alias("usable_par")))
    d = dv.load_adp(SEASONS).join(par, on=["season", "key"], how="inner").filter(
        pl.col("position") != "QB")
    return (d.with_columns(((pl.col("adp") - 1) // bucket).cast(pl.Int32).alias("bkt"))
            .group_by("bkt").agg(
                pl.len().alias("n"),
                (pl.col("bkt").first() * bucket + 1).alias("pick_from"),
                (pl.col("bkt").first() * bucket + bucket).alias("pick_to"),
                pl.col("usable_par").mean().alias("mean_usable_par"),
                pl.col("usable_par").median().alias("median_usable_par"),
                (pl.col("par") > 0).mean().alias("hit_rate"))
            .sort("bkt"))


def main() -> int:
    print(BAR)
    print(" DST-008: PICK COST OVER WEEKS 1-6 vs THE FULL FANTASY SEASON")
    print(BAR)

    c16, c6 = curve(16), curve(6)
    j = (c16.select("pick_from", "pick_to", "n",
                    pl.col("mean_usable_par").alias("mean_w1_16"))
         .join(c6.select("pick_from", "mean_usable_par", "median_usable_par",
                         "hit_rate"), on="pick_from", how="inner")
         .rename({"mean_usable_par": "mean_w1_6",
                  "median_usable_par": "median_w1_6",
                  "hit_rate": "hit_rate_w1_6"}))
    print("\nDefenses are drafted between picks 84 and 173 on the 2026 board.\n")
    print(j.filter((pl.col("pick_to") >= 73) & (pl.col("pick_from") <= 168)))

    span = j.filter((pl.col("pick_from") >= 85) & (pl.col("pick_to") <= 168))
    lo, hi = span["mean_w1_6"].min(), span["mean_w1_6"].max()
    share = span["mean_w1_6"].sum() / span["mean_w1_16"].sum()
    print(f"\nover picks 85-168, a non-QB pick returns {lo:.2f} to {hi:.2f} usable "
          f"PAR in weeks 1-6")
    print(f"the six-week share of full-season value is {share:.1%}, against the "
          f"{6 / 16:.1%} a uniform proration would assume")
    print(f"median usable PAR in every one of those buckets: "
          f"{sorted(set(span['median_w1_6'].to_list()))}")
    print("\n=> the second D/ST (+1.61 over weeks 1-6) is compared against THIS,")
    print("   not against the 16-week number, and against a lottery whose")
    print("   median payoff is zero rather than against a certainty.")

    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    j.write_csv(out / "dst008_pick_cost_six_weeks.csv")
    print(f"\nwrote {out / 'dst008_pick_cost_six_weeks.csv'}")
    return 0


def demo() -> None:
    """Self-check: truncation must reduce value and preserve the curve's shape."""
    c16, c6 = curve(16), curve(6)
    m16 = dict(zip(c16["pick_from"], c16["mean_usable_par"]))
    m6 = dict(zip(c6["pick_from"], c6["mean_usable_par"]))
    shared = [p for p in m16 if p in m6 and p <= 168]
    assert shared, "no overlapping buckets"
    assert all(m6[p] <= m16[p] + 1e-9 for p in shared), \
        "six weeks cannot be worth more than sixteen"
    assert m16[1] > m16[157] and m6[1] > m6[157], "curve must fall with pick number"
    tot6 = sum(m6[p] for p in shared)
    tot16 = sum(m16[p] for p in shared)
    assert 6 / 16 < tot6 / tot16 < 1.0, \
        f"six-week share {tot6 / tot16:.3f} outside (0.375, 1.0)"
    print(f"demo ok: {len(shared)} buckets, six-week share {tot6 / tot16:.1%}")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
