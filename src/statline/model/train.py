"""Quantile gradient-boosted model: one LightGBM regressor per quantile."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from statline.data.nflverse import POSITIONS
from statline.model.evaluate import Predictor

QUANTILES = (0.1, 0.5, 0.9)
FEATURES = [
    "as_of_week",
    "games_so_far",
    "std_ppg",
    "std_targets_pg",
    "std_carries_pg",
    "std_pass_att_pg",
    "std_target_share",
    "std_snap_share",
    "l3_ppg",
    "l3_targets_pg",
    "l3_carries_pg",
    "l3_snap_share",
    "prev_games",
    "prev_ppg",
    "prev_targets_pg",
    "prev_carries_pg",
    "prev_target_share",
    "age",
    "years_exp",
    "position",
]
# tuned by walk-forward pinball on 2016-2021 only (before any test season)
PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 400,
    "verbose": -1,
}
CAL_SEASONS = 2
MIN_CAL_ROWS = 50
GLOBAL = "*"
Offsets = dict[str, tuple[float, float]]


def to_matrix(table: pd.DataFrame) -> pd.DataFrame:
    x = table[FEATURES].copy()
    x["position"] = pd.Categorical(x["position"], categories=POSITIONS)
    return x


def rearrange(pred: np.ndarray) -> np.ndarray:
    """Sort each row's quantiles so p10 <= p50 <= p90 (separately fit models can cross)."""
    return np.sort(pred, axis=1)


def split_calibration(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last CAL_SEASONS whole seasons; rows of one player-season stay together."""
    cutoff = sorted(train["season"].unique())[-CAL_SEASONS]
    return train[train["season"] < cutoff], train[train["season"] >= cutoff]


def calibration_offsets(
    p10: pd.Series, p90: pd.Series, actual: pd.Series, position: pd.Series
) -> Offsets:
    """CQR widening per position: how far actuals escape the interval, at the 0.9 quantile.

    Positions with too few rows borrow the pooled offsets, also stored under GLOBAL.
    """

    def offsets(mask: pd.Series | slice) -> tuple[float, float]:
        return (
            float(np.quantile((p10 - actual)[mask], 0.9)),
            float(np.quantile((actual - p90)[mask], 0.9)),
        )

    pooled = offsets(slice(None))
    result = {GLOBAL: pooled}
    for pos, count in position.value_counts().items():
        result[str(pos)] = offsets(position == pos) if count >= MIN_CAL_ROWS else pooled
    return result


def fit_models(train: pd.DataFrame) -> list[LGBMRegressor]:
    x = to_matrix(train)
    return [
        LGBMRegressor(objective="quantile", alpha=q, **PARAMS).fit(x, train["ros_ppg"])
        for q in QUANTILES
    ]


def raw_quantiles(models: list[LGBMRegressor], table: pd.DataFrame) -> np.ndarray:
    xt = to_matrix(table)
    return np.column_stack([np.asarray(m.predict(xt)) for m in models])


def quantile_gbm(train: pd.DataFrame) -> Predictor:
    offsets: Offsets = {GLOBAL: (0.0, 0.0)}
    if train["season"].nunique() > CAL_SEASONS:
        fit, cal = split_calibration(train)
        raw = raw_quantiles(fit_models(fit), cal)
        offsets = calibration_offsets(
            pd.Series(raw[:, 0], index=cal.index),
            pd.Series(raw[:, 2], index=cal.index),
            cal["ros_ppg"],
            cal["position"],
        )
    models = fit_models(train)

    def predict(test: pd.DataFrame) -> pd.DataFrame:
        raw = raw_quantiles(models, test)
        lo, hi = zip(*(offsets.get(str(p), offsets[GLOBAL]) for p in test["position"]), strict=True)
        raw[:, 0] -= np.array(lo)
        raw[:, 2] += np.array(hi)
        return pd.DataFrame(rearrange(raw), columns=["p10", "p50", "p90"], index=test.index)

    return predict
