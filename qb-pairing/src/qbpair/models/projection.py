"""QB-week projection layer.

The feature set is deliberately lean, and that leanness is a research result
rather than a shortcut. Phases 2-4 established:

* opponent difficulty adds nothing at any granularity, so it is excluded;
* QB-characteristic x opponent interactions are unstable noise, so they are
  excluded;
* the betting market's implied team total is the single strongest weekly
  signal beyond a quarterback's own recent form;
* venue (home, indoors) carries a small but statistically real effect.

Adding the rejected features back would make the model look more sophisticated
and predict very slightly worse.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Evidence-selected feature set. See research/METHODOLOGY.md.
FEATURES = [
    "form_fp_mean",        # strongest single predictor
    "career_fp_mean",
    "form_pass_att",
    "form_rush_att",
    "implied_team_total",  # strongest market signal (r = 0.150 with residual)
    "total_line",
    "spread_abs",
    "is_home",             # +1.1 pt home/away spread
    "indoors",             # +1.1 pt indoor/outdoor spread
]

TARGET = "fantasy_points"


class QBWeekProjector:
    """Ridge projection of a quarterback's fantasy points in a given week."""

    def __init__(self, features: list[str] | None = None, alphas=None):
        self.features = features or FEATURES
        self.alphas = alphas if alphas is not None else np.logspace(-2, 3, 24)
        self._model = None
        self._scaler = None

    def fit(self, train: pl.DataFrame) -> "QBWeekProjector":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        X = train.select(self.features).to_numpy().astype(float)
        y = train[TARGET].to_numpy().astype(float)
        self._scaler = StandardScaler().fit(X)
        self._model = RidgeCV(alphas=self.alphas).fit(self._scaler.transform(X), y)
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        X = df.select(self.features).to_numpy().astype(float)
        return self._model.predict(self._scaler.transform(X))

    @property
    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.features, self._model.coef_))


def fit_walk_forward(
    panel: pl.DataFrame,
    test_seasons: list[int],
    features: list[str] | None = None,
) -> pl.DataFrame:
    """Attach an out-of-sample `proj` column, training only on earlier seasons.

    This is the only projection entry point the backtest should use; it makes
    look-ahead structurally impossible at the model level, complementing the
    as-of guarantees at the feature level.
    """
    frames = []
    for s in test_seasons:
        train = panel.filter(pl.col("season") < s)
        test = panel.filter(pl.col("season") == s)
        if train.height < 200 or test.is_empty():
            continue
        model = QBWeekProjector(features).fit(train)
        frames.append(test.with_columns(pl.Series("proj", model.predict(test))))
    return pl.concat(frames) if frames else panel.head(0)
