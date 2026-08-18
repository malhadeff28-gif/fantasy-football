"""QB-019 driver: export the panel, fit hierarchical models in R, report."""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                       # noqa: E402
from pathlib import Path                         # noqa: E402

import polars as pl                              # noqa: E402

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO / "shared"))
import bootstrap                                 # noqa: E402
bootstrap.setup()
import r_bridge                                  # noqa: E402
from qbpair.features import panel as pm          # noqa: E402

COLS = ["player_id", "fantasy_points", "form_fp_mean", "career_fp_mean",
        "form_pass_att", "form_rush_att", "implied_team_total", "total_line",
        "spread_abs", "is_home", "indoors", "def_adj"]


def main() -> int:
    p = pm.build_panel(list(range(2018, 2026))).drop_nulls(subset=COLS)
    # keep QBs with enough games for a slope to be estimable at all
    keep = (p.group_by("player_id").agg(pl.len().alias("n"))
            .filter(pl.col("n") >= 16)["player_id"].to_list())
    p = p.filter(pl.col("player_id").is_in(keep))

    out = PROJECT / "outputs"
    out.mkdir(exist_ok=True)
    csv = out / "_qb019_panel.csv"
    p.select(COLS).write_csv(csv)
    print(f"exported {p.height} QB-weeks, {len(keep)} quarterbacks -> {csv.name}\n")
    print(r_bridge.run_r(HERE.with_suffix(".R"), [str(csv)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
