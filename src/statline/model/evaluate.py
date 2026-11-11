"""Walk-forward backtest: baselines, metrics and the report table the model must beat.

A method is `fit(train) -> predict`, where `predict(test)` returns a DataFrame with a
`p50` column and, for quantile models, `p10` and `p90`.
"""

from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

Predictor = Callable[[pd.DataFrame], pd.DataFrame]
Method = Callable[[pd.DataFrame], Predictor]

MIN_ROS_GAMES = 2
BUCKETS = {"t 0-4": (0, 4), "t 5-9": (5, 9), "t 10-14": (10, 14)}
BLEND_K_GRID = np.arange(0.5, 16.5, 0.5)


# ---------------------------------------------------------------- split + metrics
def trainable(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["ros_games"] >= MIN_ROS_GAMES]


def walk_forward(
    table: pd.DataFrame, test_seasons: Iterable[int]
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    rows = trainable(table)
    for season in test_seasons:
        yield season, rows[rows["season"] < season], rows[rows["season"] == season]


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float((actual - predicted).abs().mean())


def pinball_loss(actual: pd.Series, predicted: pd.Series, q: float) -> float:
    diff = actual - predicted
    return float(np.maximum(q * diff, (q - 1) * diff).mean())


def coverage(actual: pd.Series, low: pd.Series, high: pd.Series) -> float:
    return float(((actual >= low) & (actual <= high)).mean())


# ---------------------------------------------------------------------- baselines
def season_to_date(train: pd.DataFrame) -> Predictor:
    return lambda test: pd.DataFrame({"p50": test["std_ppg"].fillna(test["prev_ppg"])})


def prior_season(train: pd.DataFrame) -> Predictor:
    return lambda test: pd.DataFrame({"p50": test["prev_ppg"].fillna(test["std_ppg"])})


def blend_prediction(test: pd.DataFrame, k: float) -> pd.Series:
    """Prior season counts as k games of evidence; rookies fall back to season to date."""
    g = test["games_so_far"]
    blended = (g * test["std_ppg"].fillna(0) + k * test["prev_ppg"]) / (g + k)
    return blended.fillna(test["std_ppg"])


def blend(train: pd.DataFrame) -> Predictor:
    """k is chosen on the training seasons only."""
    errors = [mae(train["ros_ppg"], blend_prediction(train, k)) for k in BLEND_K_GRID]
    k = float(BLEND_K_GRID[int(np.argmin(errors))])
    return lambda test: pd.DataFrame({"p50": blend_prediction(test, k)})


BASELINES: dict[str, Method] = {
    "season to date": season_to_date,
    "prior season": prior_season,
    "blend": blend,
}


# ------------------------------------------------------------------------- report
def backtest(
    table: pd.DataFrame, methods: dict[str, Method], test_seasons: Iterable[int]
) -> pd.DataFrame:
    """Out-of-sample predictions for every method, pooled over the test seasons."""
    frames = []
    for _, train, test in walk_forward(table, test_seasons):
        for name, fit in methods.items():
            pred = fit(train)(test).set_index(test.index)
            frames.append(test[["season", "as_of_week", "ros_ppg"]].join(pred).assign(method=name))
    return pd.concat(frames, ignore_index=True)


def report(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE by method and t bucket; pinball loss and coverage where quantiles exist."""
    rows = []
    for name, group in predictions.groupby("method", sort=False):
        row: dict[str, object] = {"method": name}
        for label, (lo, hi) in BUCKETS.items():
            bucket = group[group["as_of_week"].between(lo, hi)]
            row[f"MAE {label}"] = mae(bucket["ros_ppg"], bucket["p50"])
        row["MAE all"] = mae(group["ros_ppg"], group["p50"])
        if {"p10", "p90"} <= set(group.columns) and group["p10"].notna().all():
            row["pinball"] = np.mean(
                [pinball_loss(group["ros_ppg"], group[f"p{q}"], q / 100) for q in (10, 50, 90)]
            )
            row["p10-p90 coverage"] = coverage(group["ros_ppg"], group["p10"], group["p90"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").round(3)
