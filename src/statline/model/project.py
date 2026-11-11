"""Live projections: train on every completed season, predict one (season, as-of week)."""

import pandas as pd

from statline.model.evaluate import Method, trainable
from statline.model.train import quantile_gbm

OUTPUT_COLUMNS = [
    "player_id",
    "season",
    "name",
    "position",
    "team",
    "as_of_week",
    "p10",
    "p50",
    "p90",
]


def project(
    table: pd.DataFrame, season: int, as_of_week: int, method: Method = quantile_gbm
) -> pd.DataFrame:
    """Unlike training rows, live rows are not filtered on remaining games: the future is
    unknown, so every player with a row at this week gets a projection."""
    train = trainable(table[table["season"] < season])
    live = table[(table["season"] == season) & (table["as_of_week"] == as_of_week)]
    pred = method(train)(live)
    out = live.join(pred)[OUTPUT_COLUMNS]
    return out.sort_values("p50", ascending=False).reset_index(drop=True)
