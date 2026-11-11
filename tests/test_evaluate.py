import numpy as np
import pandas as pd
import pytest

from statline.model.evaluate import (
    blend_prediction,
    coverage,
    mae,
    pinball_loss,
    walk_forward,
)


def test_metric_math() -> None:
    actual = pd.Series([10.0, 20.0])
    assert mae(actual, pd.Series([12.0, 17.0])) == 2.5
    # under-predicting by 4 at q=0.9 costs 0.9 * 4; over-predicting by 4 costs 0.1 * 4
    assert pinball_loss(pd.Series([10.0]), pd.Series([6.0]), 0.9) == pytest.approx(3.6)
    assert pinball_loss(pd.Series([10.0]), pd.Series([14.0]), 0.9) == pytest.approx(0.4)
    assert coverage(actual, pd.Series([5.0, 21.0]), pd.Series([15.0, 30.0])) == 0.5


def test_walk_forward_never_trains_on_the_test_season_or_later() -> None:
    table = pd.DataFrame({"season": [2020, 2021, 2022, 2023], "ros_games": [5, 5, 5, 1]})
    for season, train, test in walk_forward(table, [2021, 2022, 2023]):
        assert (train["season"] < season).all()
        assert (test["season"] == season).all()
        assert (test["ros_games"] >= 2).all()


def test_blend_moves_from_prior_to_season_to_date() -> None:
    test = pd.DataFrame(
        {
            "games_so_far": [0, 4, 1000, 3],
            "std_ppg": [np.nan, 20.0, 20.0, 12.0],
            "prev_ppg": [10.0, 10.0, 10.0, np.nan],  # last row: rookie
        }
    )
    pred = blend_prediction(test, k=4.0)
    assert pred[0] == 10.0  # no games yet: all prior
    assert pred[1] == 15.0  # 4 games vs k=4: halfway
    assert pred[2] == pytest.approx(20.0, abs=0.05)  # many games: season to date
    assert pred[3] == 12.0  # no prior: season to date
