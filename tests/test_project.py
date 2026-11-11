from datetime import UTC, datetime

import pandas as pd

from statline.model.evaluate import Predictor
from statline.model.project import project
from statline.model.schema import OUTPUT_COLUMNS


def mean_method(train: pd.DataFrame) -> Predictor:
    """Stub model: predicts the training mean, so the test is about row selection."""
    mean = train["ros_ppg"].mean()
    return lambda test: pd.DataFrame(
        {"p10": mean - 1, "p50": mean, "p90": mean + 1}, index=test.index
    )


def test_project_trains_on_past_seasons_and_keeps_players_with_unknown_futures() -> None:
    table = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d", "e"],
            "name": list("ABCDE"),
            "position": ["WR"] * 5,
            "team": ["DET"] * 5,
            "season": [2024, 2024, 2025, 2026, 2026],
            "as_of_week": [9, 9, 9, 9, 8],
            "ros_ppg": [10.0, 99.0, 20.0, float("nan"), float("nan")],
            "ros_games": [5, 1, 5, 0, 0],  # b is untrainable (1 game left)
        }
    )
    out = project(table, season=2026, as_of_week=9, method=mean_method)
    assert out["player_id"].tolist() == ["d"]  # live row kept despite ros_games == 0
    assert out.loc[0, "p50"] == 15.0  # mean of a and c only: b excluded, 2026 excluded
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.loc[0, "season"] == 2026
    assert out.loc[0, "built_on"] == datetime.now(UTC).date().isoformat()
