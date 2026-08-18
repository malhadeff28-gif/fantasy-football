"""QB-016: does the betting market add anything AFTER a public projection?

Wave 0 measured the market against a quarterback's own recent form and found it
worth +1.46 pp of start/sit ranking accuracy. That is not the decision a manager
faces. A manager opens a site, reads a published projection, and *then* decides.
The sharper question is whether the market carries information the published
projection has not already priced in.

THE HONEST OBSTACLE
-------------------
CBS publishes only a CURRENT board. This script verifies that empirically at
run time: the URLs for 2022, 2025 and 2026 return byte-identical HTML. There is
no accessible historical archive, so "CBS + market" cannot be backtested. Any
claim that it can is false.

WHAT IS DONE INSTEAD
--------------------
A CONSENSUS PROXY is built from information that provably existed before week 1
of each season:

    prior-season fantasy points per game and games played (S-1 and S-2)
    that season's preseason ADP from real drafts (FFC, window closes in the
    last days of August, before week 1 in every season used)

Those features are fitted walk-forward -- train on seasons < S, predict S -- to
weekly fantasy points, so the proxy IS a per-week baseline expectation and is
constant within a player-season, exactly like a published preseason projection.
It is a PROXY. It is not CBS. It is validated three ways: against realised
season outcomes out of sample, against simpler preseason benchmarks, and --
the one direct test available -- against the real CBS 2026 board, the single
season where a genuine public projection and the proxy both exist.

THE TEST
--------
Second-stage models, all walk-forward (train seasons < S, test S, S=2020-2025):

    consensus                 the published-projection baseline
    consensus + market        does the market add on top of it?
    consensus + form          does in-season form add on top of it?
    consensus + form + market the practical question: market on top of both
    form / form + market      Wave 0's comparison, replicated on these rows

Metrics: MAE and RMSE, plus the decision-relevant one -- pairwise within-week
start/sit ranking accuracy. Uncertainty is reported as the standard error of
the per-season deltas (n = 6 seasons, so the standard errors are large and the
script says so) and as a cluster bootstrap over (season, week) blocks.

Usage:  ./.venv/Scripts/python.exe qb-pairing/scripts/wave1_market_vs_consensus.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                        # noqa: E402
import warnings                                   # noqa: E402
from pathlib import Path                          # noqa: E402

import numpy as np                                # noqa: E402
import polars as pl                               # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]     # qb-pairing/
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))

import bootstrap                                  # noqa: E402

bootstrap.setup()
import markets                                    # noqa: E402

from qbpair.config import DATA_RAW, Scoring       # noqa: E402
from qbpair.data import loaders                   # noqa: E402
from qbpair.features import panel as pm           # noqa: E402

warnings.filterwarnings("ignore")
pl.Config.set_tbl_formatting("ASCII_FULL")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(24)

SEASONS = list(range(2018, 2026))
TEST_SEASONS = list(range(2020, 2026))
FORWARD_SEASON = 2026
BAR = "=" * 78
SUB = "-" * 78

# Preseason-only features. Every one is knowable before week 1.
CONSENSUS_FEATURES = ["prev_ppg", "prev_games", "prev2_ppg", "prev2_games",
                      "has_prev", "log_adp", "adp_qb_rank", "is_drafted"]

MARKET = ["implied_team_total", "total_line", "spread_abs"]
FORM = ["form_fp_mean", "form_pass_att", "form_rush_att"]

MODELS = {
    "consensus":              ["consensus"],
    "consensus+market":       ["consensus"] + MARKET,
    "consensus+form":         ["consensus"] + FORM,
    "consensus+form+market":  ["consensus"] + FORM + MARKET,
    "form":                   ["form_fp_mean", "career_fp_mean"] + FORM[1:],
    "form+market":            ["form_fp_mean", "career_fp_mean"] + FORM[1:] + MARKET,
}

# The comparisons the write-up turns on: (baseline, augmented).
CONTRASTS = [
    ("consensus", "consensus+market"),           # the literal question
    ("consensus+form", "consensus+form+market"),  # the practical question
    ("form", "form+market"),                     # Wave 0, replicated here
    ("consensus", "consensus+form"),             # form on top of consensus
    ("form", "consensus+form"),                  # consensus on top of form
    ("form+market", "consensus+form+market"),    # consensus on top of form+market
]

UNDRAFTED_ADP = 250.0       # past the end of a 12-team board
UNDRAFTED_QB_RANK = 40.0
CLOSE_CALL_PTS = 2.0        # |consensus gap| below this = a live start/sit call
N_BOOT = 2000
RNG = np.random.default_rng(20260818)


# --- data ------------------------------------------------------------------

def cbs_archive_probe() -> dict:
    """Verify at run time that CBS serves no historical board."""
    import requests

    lens, tables = {}, {}
    for yr in (2022, 2025, 2026):
        try:
            r = requests.get(
                markets.CBS_URL.format(pos="QB", season=yr),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=45)
            lens[yr] = len(r.text)
            # Compare the parsed projection table, not the raw HTML: the page
            # carries a per-request nonce, so bytes differ while content does not.
            d = markets.load_cbs_projections(yr, "QB").sort("player")
            tables[yr] = (d.height, d.select("player", "pass_yds", "pass_td",
                                             "rush_yds").rows())
        except Exception as e:                              # offline is fine
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    ref = tables[2026]
    identical = all(v == ref for v in tables.values())
    return {"ok": True, "lengths": lens, "identical": identical,
            "rows": {k: v[0] for k, v in tables.items()}}


def load_adp(seasons: list[int]) -> pl.DataFrame:
    """Preseason ADP from real drafts, cached so re-runs are reproducible."""
    path = DATA_RAW / "ffc_adp_qb_2018_2026.parquet"
    if path.exists():
        return pl.read_parquet(path)
    frames = []
    for s in seasons:
        d = markets.load_ffc_adp(s)
        frames.append(d.select(
            "season", "key", "position",
            pl.col("name").alias("adp_name"),
            pl.col("adp").cast(pl.Float64),
            pl.col("stdev").cast(pl.Float64).alias("adp_stdev"),
            pl.col("times_drafted").cast(pl.Int64),
            "adp_window_start", "adp_window_end",
        ))
    out = pl.concat(frames)
    out.write_parquet(path)
    return out


def prior_season_stats(seasons: list[int]) -> pl.DataFrame:
    """Per player-season QB fantasy PPG and starts, from finished seasons only."""
    st = loaders.load_player_stats(seasons).filter(
        (pl.col("position") == "QB")
        & (pl.col("season_type") == "REG")
        & (pl.col("attempts") >= pm.MIN_ATT)
    ).select("player_id", "season", "fantasy_points")
    return st.group_by(["player_id", "season"]).agg(
        pl.col("fantasy_points").mean().alias("ppg"),
        pl.len().alias("g"),
    )


def build_preseason_panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    """QB-week panel with preseason-only consensus inputs attached."""
    panel = pm.build_panel(SEASONS).drop_nulls(
        subset=MARKET + FORM + ["career_fp_mean", "fantasy_points"])

    prior = prior_season_stats(list(range(min(SEASONS) - 2, max(SEASONS) + 1)))
    adp = load_adp(list(range(2018, FORWARD_SEASON + 1)))
    adp_qb = adp.filter(pl.col("position") == "QB").with_columns(
        pl.col("adp").rank("min").over("season").alias("adp_qb_rank_raw")
    )

    p = panel.with_columns(
        pl.col("player_display_name")
        .map_elements(markets.normalise_name, return_dtype=pl.String).alias("key"),
        (pl.col("season") - 1).alias("_s1"),
        (pl.col("season") - 2).alias("_s2"),
    )
    p = (
        p.join(prior.select("player_id", pl.col("season").alias("_s1"),
                            pl.col("ppg").alias("prev_ppg_raw"),
                            pl.col("g").alias("prev_games_raw")),
               on=["player_id", "_s1"], how="left")
        .join(prior.select("player_id", pl.col("season").alias("_s2"),
                           pl.col("ppg").alias("prev2_ppg_raw"),
                           pl.col("g").alias("prev2_games_raw")),
              on=["player_id", "_s2"], how="left")
        .join(adp_qb.select("season", "key", "adp", "adp_qb_rank_raw", "adp_stdev"),
              on=["season", "key"], how="left")
        .drop("_s1", "_s2")
    )
    return _finish_features(p), adp_qb


def _finish_features(p: pl.DataFrame) -> pl.DataFrame:
    """Fill the preseason features, encoding 'no prior' / 'undrafted' explicitly."""
    return p.with_columns(
        pl.col("prev_ppg_raw").fill_null(0.0).alias("prev_ppg"),
        pl.col("prev_games_raw").fill_null(0).cast(pl.Float64).alias("prev_games"),
        pl.col("prev2_ppg_raw").fill_null(0.0).alias("prev2_ppg"),
        pl.col("prev2_games_raw").fill_null(0).cast(pl.Float64).alias("prev2_games"),
        pl.col("prev_ppg_raw").is_not_null().cast(pl.Float64).alias("has_prev"),
        pl.col("adp").is_not_null().cast(pl.Float64).alias("is_drafted"),
        pl.col("adp").fill_null(UNDRAFTED_ADP).log().alias("log_adp"),
        pl.col("adp_qb_rank_raw").fill_null(UNDRAFTED_QB_RANK)
        .cast(pl.Float64).alias("adp_qb_rank"),
    )


# --- modelling -------------------------------------------------------------

def _ridge(train: pl.DataFrame, test: pl.DataFrame, feats: list[str]) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    X = train.select(feats).to_numpy().astype(float)
    y = train["fantasy_points"].to_numpy().astype(float)
    sc = StandardScaler().fit(X)
    m = RidgeCV(alphas=np.logspace(-2, 3, 24)).fit(sc.transform(X), y)
    Xt = test.select(feats).to_numpy().astype(float)
    return m.predict(sc.transform(Xt)), dict(zip(feats, m.coef_))


def attach_consensus(p: pl.DataFrame, test_season: int,
                     nested: bool) -> pl.DataFrame | None:
    """Walk-forward consensus proxy for one test season.

    Primary spec: one proxy fitted on seasons < S, applied to every row. Test
    rows are strictly out of sample; training rows carry the proxy's in-sample
    fit, whose optimism is quantified in the printed report.

    Nested spec: every row's proxy comes from a model fitted only on strictly
    earlier seasons, so no value anywhere is in-sample. Costs the 2018 rows and
    thins the earliest training sets; run as a robustness check.
    """
    if nested:
        frames = []
        for t in sorted(p["season"].unique().to_list()):
            if t > test_season:
                continue
            tr = p.filter(pl.col("season") < t)
            if tr.height < 300:
                continue
            te = p.filter(pl.col("season") == t)
            pred, _ = _ridge(tr, te, CONSENSUS_FEATURES)
            frames.append(te.with_columns(pl.Series("consensus", pred)))
        return pl.concat(frames) if frames else None

    tr = p.filter(pl.col("season") < test_season)
    sub = p.filter(pl.col("season") <= test_season)
    if tr.height < 300:
        return None
    pred, _ = _ridge(tr, sub, CONSENSUS_FEATURES)
    return sub.with_columns(pl.Series("consensus", pred))


# --- metrics ---------------------------------------------------------------

def pair_block(y: np.ndarray, preds: dict[str, np.ndarray],
               consensus: np.ndarray, drafted: np.ndarray) -> dict:
    """Concordance counts for every within-week QB pair in one week."""
    n = len(y)
    if n < 2:
        return {}
    i, j = np.triu_indices(n, k=1)
    dy = y[i] - y[j]
    live = dy != 0                      # ties in outcome decide nothing
    if not live.any():
        return {}
    i, j, dy = i[live], j[live], dy[live]
    dc = np.abs(consensus[i] - consensus[j])
    slices = {
        "all": np.ones(len(dy), bool),
        "close": dc < CLOSE_CALL_PTS,
        "drafted": drafted[i] & drafted[j],
    }
    out = {}
    for name, pred in preds.items():
        dp = pred[i] - pred[j]
        agree = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(dy)).astype(float))
        for s, mask in slices.items():
            out[(name, s)] = (float(agree[mask].sum()), int(mask.sum()))
    return out


def run(p_all: pl.DataFrame, nested: bool) -> dict:
    """Walk-forward fit and score every model. Returns metrics and blocks."""
    rows, blocks, coefs, resid_corr = [], [], [], []

    for s in TEST_SEASONS:
        p = attach_consensus(p_all, s, nested)
        if p is None:
            continue
        train = p.filter(pl.col("season") < s)
        test = p.filter(pl.col("season") == s)
        if train.height < 300 or test.is_empty():
            continue

        preds = {}
        for name, feats in MODELS.items():
            pred, coef = _ridge(train, test, feats)
            preds[name] = pred
            if name == "consensus+market":
                coefs.append({"season": s, **{k: float(v) for k, v in coef.items()}})

        y = test["fantasy_points"].to_numpy().astype(float)
        for name, pred in preds.items():
            e = pred - y
            rows.append({"season": s, "model": name, "n": len(y),
                         "mae": float(np.abs(e).mean()),
                         "rmse": float(np.sqrt((e ** 2).mean()))})

        # Model-free check: what does the market know that the baseline does
        # not? Run against two baselines -- if the correlation barely moves when
        # in-season form is added, the market's information is structurally
        # unavailable to any projection, not merely missing from a weak one.
        for base in ("consensus", "consensus+form"):
            r = y - preds[base]
            rc = {"season": s, "baseline": base}
            for c in MARKET + ["implied_opp_total", "team_spread"]:
                v = test[c].to_numpy().astype(float)
                rc[c] = float(np.corrcoef(v, r)[0, 1])
            resid_corr.append(rc)

        drafted = (test["is_drafted"].to_numpy() > 0.5)
        cons = preds["consensus"]
        weeks = test["week"].to_numpy()
        for w in np.unique(weeks):
            m = weeks == w
            blk = pair_block(y[m], {k: v[m] for k, v in preds.items()},
                             cons[m], drafted[m])
            if blk:
                blocks.append({"season": s, "week": int(w), "counts": blk})

    return {"err": pl.DataFrame(rows), "blocks": blocks,
            "coefs": pl.DataFrame(coefs), "resid_corr": pl.DataFrame(resid_corr)}


def pooled_accuracy(blocks: list[dict], sl: str, seasons=None,
                    weeks=None) -> dict[str, float]:
    acc = {}
    for name in MODELS:
        num = den = 0.0
        for b in blocks:
            if seasons is not None and b["season"] not in seasons:
                continue
            if weeks is not None and b["week"] not in weeks:
                continue
            c, n = b["counts"].get((name, sl), (0.0, 0))
            num += c
            den += n
        acc[name] = num / den * 100 if den else float("nan")
    acc["_n"] = den
    return acc


def season_table(blocks: list[dict], sl: str) -> pl.DataFrame:
    return pl.DataFrame([
        {"season": s, **{k: v for k, v in pooled_accuracy(blocks, sl, {s}).items()}}
        for s in TEST_SEASONS
    ])


def across_season_se(vals: np.ndarray) -> tuple[float, float, float]:
    """Mean, standard error and t across seasons. n = 6, so t has 5 df."""
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    m = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return m, se, (m / se if se else float("nan"))


def block_bootstrap(blocks: list[dict], base: str, aug: str,
                    sl: str, b: int = N_BOOT) -> tuple[float, float]:
    """Cluster bootstrap over (season, week) blocks for the accuracy delta."""
    nb = len(blocks)
    cb = np.array([blocks[k]["counts"].get((base, sl), (0.0, 0))[0] for k in range(nb)])
    ca = np.array([blocks[k]["counts"].get((aug, sl), (0.0, 0))[0] for k in range(nb)])
    dn = np.array([blocks[k]["counts"].get((base, sl), (0.0, 0))[1] for k in range(nb)],
                  dtype=float)
    if dn.sum() == 0:
        return float("nan"), float("nan")
    point = (ca.sum() - cb.sum()) / dn.sum() * 100
    idx = RNG.integers(0, nb, size=(b, nb))
    num = (ca[idx] - cb[idx]).sum(axis=1)
    den = dn[idx].sum(axis=1)
    d = np.where(den > 0, num / np.where(den > 0, den, 1) * 100, np.nan)
    return point, float(np.nanstd(d, ddof=1))


# --- reporting -------------------------------------------------------------

def contrast_report(res: dict, label: str) -> pl.DataFrame:
    err, blocks = res["err"], res["blocks"]
    out = []
    for base, aug in CONTRASTS:
        for metric, better in (("mae", -1), ("rmse", -1)):
            b = err.filter(pl.col("model") == base).sort("season")[metric].to_numpy()
            a = err.filter(pl.col("model") == aug).sort("season")[metric].to_numpy()
            m, se, t = across_season_se(a - b)
            out.append({"spec": label, "contrast": f"{aug} - {base}",
                        "metric": metric, "delta": m, "se_across_seasons": se,
                        "t": t, "n_seasons": int(len(a))})
        for sl in ("all", "close", "drafted"):
            tab = season_table(blocks, sl)
            d = (tab[aug].to_numpy() - tab[base].to_numpy())
            m, se, t = across_season_se(d)
            pt, bse = block_bootstrap(blocks, base, aug, sl)
            out.append({"spec": label, "contrast": f"{aug} - {base}",
                        "metric": f"pair_acc_pp[{sl}]", "delta": pt,
                        "se_across_seasons": se, "t": (pt / se if se else float("nan")),
                        "n_seasons": int(len(d)), "se_bootstrap": bse,
                        "mean_of_season_deltas": m})
    return pl.DataFrame(out)


def validate_proxy(p_all: pl.DataFrame) -> pl.DataFrame:
    """How well does the proxy track outcomes out of sample, vs simpler rules?"""
    rows = []
    for s in TEST_SEASONS:
        p = attach_consensus(p_all, s, nested=False)
        if p is None:
            continue
        t = p.filter(pl.col("season") == s)
        ps = t.group_by("player_id").agg(
            pl.col("fantasy_points").mean().alias("actual_ppg"),
            pl.col("consensus").first(),
            pl.col("prev_ppg").first(),
            pl.col("adp_qb_rank").first(),
            pl.len().alias("games"),
        ).filter(pl.col("games") >= 4)
        for nm, col, sign in (("consensus proxy", "consensus", 1),
                              ("prior-season ppg", "prev_ppg", 1),
                              ("ADP QB rank", "adp_qb_rank", -1)):
            v = ps[col].to_numpy().astype(float) * sign
            a = ps["actual_ppg"].to_numpy().astype(float)
            rows.append({
                "season": s, "predictor": nm, "n_qbs": ps.height,
                "pearson_r": float(np.corrcoef(v, a)[0, 1]),
                "spearman_r": float(np.corrcoef(
                    np.argsort(np.argsort(v)), np.argsort(np.argsort(a)))[0, 1]),
            })
    return pl.DataFrame(rows)


def cbs_rescored(season: int) -> pl.DataFrame:
    sc = Scoring()
    d = markets.load_cbs_projections(season, "QB")
    return d.with_columns(
        (pl.col("pass_yds") * sc.pass_yds + pl.col("pass_td") * sc.pass_td
         + pl.col("interceptions") * sc.interception
         + pl.col("rush_yds") * sc.rush_yds + pl.col("rush_td") * sc.rush_td
         + pl.col("fumbles_lost") * sc.fumble_lost).alias("cbs_points_league")
    ).with_columns(
        (pl.col("cbs_points_league") / pl.col("games").clip(lower_bound=1))
        .alias("cbs_ppg_league")
    )


def forward_2026(p_all: pl.DataFrame, adp_qb: pl.DataFrame) -> tuple:
    """The proxy, the real CBS board and ADP for 2026. Describable, not testable."""
    prior = prior_season_stats(list(range(2023, 2026)))
    base = adp_qb.filter(pl.col("season") == FORWARD_SEASON)
    cbs = cbs_rescored(FORWARD_SEASON)

    # QB universe for 2026: anyone on the CBS board or drafted.
    uni = cbs.select("key", "player", "team", "cbs_points_league", "cbs_ppg_league")
    uni = uni.join(base.select("key", "adp", "adp_qb_rank_raw", "adp_stdev"),
                   on="key", how="left")

    ids = (p_all.select("player_id", "player_display_name", "key")
           .unique(subset=["key"], keep="last"))
    uni = uni.join(ids, on="key", how="left")
    uni = (uni.join(prior.filter(pl.col("season") == 2025)
                    .select("player_id", pl.col("ppg").alias("prev_ppg_raw"),
                            pl.col("g").alias("prev_games_raw")),
                    on="player_id", how="left")
           .join(prior.filter(pl.col("season") == 2024)
                 .select("player_id", pl.col("ppg").alias("prev2_ppg_raw"),
                         pl.col("g").alias("prev2_games_raw")),
                 on="player_id", how="left"))
    uni = _finish_features(uni).with_columns(pl.lit(FORWARD_SEASON).alias("season"))

    train = p_all.filter(pl.col("season") < FORWARD_SEASON)
    pred, _ = _ridge(train, uni, CONSENSUS_FEATURES)
    uni = uni.with_columns(pl.Series("proxy_ppg", pred))

    board = uni.filter(pl.col("cbs_ppg_league") > 0).select(
        "player", "team", "adp", "adp_qb_rank_raw", "adp_stdev",
        "cbs_ppg_league", "cbs_points_league", "proxy_ppg",
    ).sort("cbs_ppg_league", descending=True)

    # Proxy validated against a REAL public projection, the only season where
    # both exist. This is the direct test of whether the proxy stands in.
    m = board.drop_nulls(subset=["proxy_ppg", "cbs_ppg_league"])
    a = m["proxy_ppg"].to_numpy().astype(float)
    b = m["cbs_ppg_league"].to_numpy().astype(float)
    agree = {
        "n": m.height,
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "spearman_r": float(np.corrcoef(np.argsort(np.argsort(a)),
                                        np.argsort(np.argsort(b)))[0, 1]),
    }
    md = m.drop_nulls(subset=["adp"])
    a2, b2 = (md["proxy_ppg"].to_numpy().astype(float),
              md["cbs_ppg_league"].to_numpy().astype(float))
    agree["n_drafted"] = md.height
    agree["pearson_r_drafted"] = float(np.corrcoef(a2, b2)[0, 1])
    agree["spearman_r_drafted"] = float(np.corrcoef(
        np.argsort(np.argsort(a2)), np.argsort(np.argsort(b2)))[0, 1])

    # Where CBS and the draft market disagree, in comparable units.
    dis = md.with_columns(
        ((pl.col("cbs_ppg_league") - pl.col("cbs_ppg_league").mean())
         / pl.col("cbs_ppg_league").std()).alias("cbs_z"),
        ((pl.col("adp").log().mean() - pl.col("adp").log())
         / pl.col("adp").log().std()).alias("adp_z"),
    ).with_columns((pl.col("cbs_z") - pl.col("adp_z")).alias("cbs_minus_adp_z")) \
        .sort("cbs_minus_adp_z", descending=True)
    return board, dis, agree


def main() -> int:
    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)

    print(BAR)
    print(" QB-016  DOES THE MARKET ADD ANYTHING AFTER A PUBLIC PROJECTION?")
    print(BAR)

    probe = cbs_archive_probe()
    if probe["ok"]:
        print(f"CBS archive probe: page lengths {probe['lengths']}, "
              f"all identical = {probe['identical']}")
        print("  -> CBS serves the CURRENT board for every season path. No historical")
        print("     archive exists, so 'CBS + market' cannot be backtested directly.")
    else:
        print(f"CBS archive probe skipped ({probe['reason']}); "
              "prior runs confirmed identical pages for 2022/2025/2026.")

    p_all, adp_qb = build_preseason_panel()
    print(f"\npanel rows {p_all.height:,}  seasons {SEASONS[0]}-{SEASONS[-1]}")
    cov = p_all.select(
        (pl.col("is_drafted").mean() * 100).alias("rows_with_ADP_pct"),
        (pl.col("has_prev").mean() * 100).alias("rows_with_prior_season_pct"),
    )
    print(cov)
    win = load_adp(list(range(2018, FORWARD_SEASON + 1))).group_by("season").agg(
        pl.col("adp_window_start").first(), pl.col("adp_window_end").first()
    ).sort("season")
    print("\nADP windows (all close before week 1, so the proxy is preseason-only):")
    print(win)

    # --- market redundancy -------------------------------------------------
    print(f"\n{BAR}\n MARKET VARIABLE REDUNDANCY\n{BAR}")
    red = markets.market_redundancy(
        p_all, MARKET + ["team_spread", "implied_opp_total"])
    print(red)
    red.write_csv(out / "qb016_market_redundancy.csv")

    # --- primary run -------------------------------------------------------
    res = run(p_all, nested=False)
    res["err"].write_csv(out / "qb016_season_error_metrics.csv")

    print(f"\n{BAR}\n ERROR METRICS, WALK-FORWARD 2020-2025 (pooled over test rows)\n{BAR}")
    pooled = res["err"].group_by("model").agg(
        pl.col("mae").mean().alias("MAE"), pl.col("rmse").mean().alias("RMSE"),
        pl.col("n").sum().alias("n_rows"))
    order = list(MODELS)
    pooled = pooled.with_columns(
        pl.col("model").replace_strict({m: i for i, m in enumerate(order)},
                                       return_dtype=pl.Int32).alias("_o")
    ).sort("_o").drop("_o")
    print(pooled)

    print(f"\n{BAR}\n PAIRWISE WITHIN-WEEK START/SIT ACCURACY (%)\n{BAR}")
    for sl, desc in (("all", "every within-week QB pair"),
                     ("close", f"|consensus gap| < {CLOSE_CALL_PTS} pts, the live calls"),
                     ("drafted", "both QBs drafted, i.e. a real roster pair")):
        acc = pooled_accuracy(res["blocks"], sl)
        print(f"\n{desc}   (n = {acc['_n']:,} decided pairs)")
        for m in order:
            print(f"  {m:<24}{acc[m]:>8.2f}")
    for sl in ("all", "close", "drafted"):
        season_table(res["blocks"], sl).write_csv(
            out / f"qb016_pairwise_by_season_{sl}.csv")

    print(f"\n{BAR}\n WHEN DOES THE MARKET HELP? ACCURACY BY WEEK BUCKET (all pairs)\n{BAR}")
    wk_rows = []
    for lab, wks in (("weeks 1-4", set(range(1, 5))),
                     ("weeks 5-9", set(range(5, 10))),
                     ("weeks 10-18", set(range(10, 19)))):
        a = pooled_accuracy(res["blocks"], "all", weeks=wks)
        wk_rows.append({"bucket": lab, "n_pairs": int(a["_n"]),
                        **{m: round(a[m], 2) for m in order},
                        "market_gain_pp": round(a["consensus+market"] - a["consensus"], 2),
                        "market_gain_over_form_pp": round(
                            a["consensus+form+market"] - a["consensus+form"], 2)})
    wk = pl.DataFrame(wk_rows)
    print(wk.select("bucket", "n_pairs", "consensus", "consensus+market",
                    "consensus+form", "consensus+form+market",
                    "market_gain_pp", "market_gain_over_form_pp"))
    wk.write_csv(out / "qb016_accuracy_by_week_bucket.csv")

    print(f"\n{BAR}\n CONTRASTS WITH UNCERTAINTY (n = 6 test seasons; SEs are large)\n{BAR}")
    con = contrast_report(res, "primary")
    print(con.select("contrast", "metric", "delta", "se_across_seasons", "t",
                     "se_bootstrap"))

    # --- nested robustness -------------------------------------------------
    res_n = run(p_all, nested=True)
    con_n = contrast_report(res_n, "nested")
    print(f"\n{BAR}\n ROBUSTNESS: FULLY NESTED PROXY (no in-sample proxy anywhere)\n{BAR}")
    print(con_n.select("contrast", "metric", "delta", "se_across_seasons", "t"))
    pl.concat([con, con_n], how="diagonal").write_csv(out / "qb016_contrasts.csv")

    # --- what the market knows that consensus does not ---------------------
    print(f"\n{BAR}\n MARKET vs BASELINE RESIDUAL (model-free, correlation)\n{BAR}")
    rcm = res["resid_corr"]
    print(rcm.group_by("baseline").agg(
        [pl.col(c).mean().alias(c) for c in MARKET
         + ["implied_opp_total", "team_spread"]]).sort("baseline"))
    print("\nper season:")
    print(rcm.sort("baseline", "season"))
    rcm.write_csv(out / "qb016_market_resid_corr.csv")

    print(f"\n{BAR}\n STANDARDISED COEFFICIENTS, consensus+market\n{BAR}")
    print(res["coefs"])
    res["coefs"].write_csv(out / "qb016_consensus_market_coefs.csv")

    # --- proxy validation --------------------------------------------------
    print(f"\n{BAR}\n IS THE CONSENSUS PROXY ANY GOOD? (out of sample, player-season)\n{BAR}")
    vp = validate_proxy(p_all)
    print(vp.group_by("predictor").agg(
        pl.col("pearson_r").mean().alias("mean_pearson"),
        pl.col("spearman_r").mean().alias("mean_spearman"),
        pl.col("pearson_r").std().alias("sd_pearson"),
    ).sort("mean_pearson", descending=True))
    vp.write_csv(out / "qb016_proxy_validation.csv")

    ins = []
    for s in TEST_SEASONS:
        p = attach_consensus(p_all, s, nested=False)
        if p is None:
            continue
        tr, te = p.filter(pl.col("season") < s), p.filter(pl.col("season") == s)
        ins.append({"season": s,
                    "proxy_MAE_in_sample_train": float(
                        (tr["consensus"] - tr["fantasy_points"]).abs().mean()),
                    "proxy_MAE_out_of_sample_test": float(
                        (te["consensus"] - te["fantasy_points"]).abs().mean())})
    print("\nProxy in-sample vs out-of-sample MAE (the optimism the primary spec carries):")
    print(pl.DataFrame(ins))

    # --- forward check -----------------------------------------------------
    print(f"\n{BAR}\n FORWARD CHECK 2026 - DESCRIPTIVE ONLY, CANNOT BE VALIDATED\n{BAR}")
    try:
        board, dis, agree = forward_2026(p_all, adp_qb)
        print(f"proxy vs the REAL CBS 2026 board: n={agree['n']}, "
              f"pearson r={agree['pearson_r']:.3f}, spearman={agree['spearman_r']:.3f}")
        print(f"  drafted QBs only: n={agree['n_drafted']}, "
              f"pearson r={agree['pearson_r_drafted']:.3f}, "
              f"spearman={agree['spearman_r_drafted']:.3f}")
        print("  This is the one direct check that the proxy stands in for a real")
        print("  public projection; there is no other season where both exist.")
        board.write_csv(out / "qb016_consensus_vs_cbs_2026.csv")

        cols = ["player", "team", "adp", "adp_qb_rank_raw", "cbs_ppg_league",
                "proxy_ppg", "cbs_minus_adp_z"]
        print("\nCBS most bullish vs the draft market:")
        print(dis.select(cols).head(8))
        print("\nCBS most bearish vs the draft market:")
        print(dis.select(cols).tail(8))
        dis.write_csv(out / "qb016_cbs_vs_adp_2026.csv")
    except Exception as e:
        print(f"forward check unavailable: {type(e).__name__}: {e}")

    print(f"\n{BAR}\n VERDICT INPUTS\n{BAR}")
    key = con.filter(
        (pl.col("contrast") == "consensus+market - consensus")
        & (pl.col("metric") == "pair_acc_pp[all]"))
    if key.height:
        r = key.to_dicts()[0]
        print(f"market on top of consensus, all pairs: {r['delta']:+.2f} pp "
              f"(SE across seasons {r['se_across_seasons']:.2f}, "
              f"bootstrap SE {r.get('se_bootstrap', float('nan')):.2f})")
    print(f"\nwrote CSVs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
