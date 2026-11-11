import numpy as np
import pandas as pd

from statline.model.train import (
    CAL_SEASONS,
    FEATURES,
    calibration_offsets,
    quantile_gbm,
    rearrange,
    split_calibration,
)


def test_rearrange_orders_crossed_quantiles() -> None:
    out = rearrange(np.array([[5.0, 3.0, 9.0], [1.0, 2.0, 3.0]]))
    assert (np.diff(out, axis=1) >= 0).all()
    assert out[0].tolist() == [3.0, 5.0, 9.0]


def test_identifiers_and_targets_are_not_features() -> None:
    leaky = {"player_id", "name", "team", "season", "ros_ppg", "ros_games"}
    assert not leaky & set(FEATURES)


def test_model_learns_a_known_signal_and_returns_ordered_quantiles() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    table = pd.DataFrame({c: rng.uniform(0, 1, n) for c in FEATURES})
    table["position"] = rng.choice(["QB", "RB", "WR", "TE"], n)
    table["season"] = rng.choice([2018, 2019, 2020, 2021], n)
    table["std_ppg"] = rng.uniform(0, 25, n)
    table["ros_ppg"] = table["std_ppg"] + rng.normal(0, 2, n)

    pred = quantile_gbm(table)(table.head(500))
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    assert np.corrcoef(pred["p50"], table["std_ppg"].head(500))[0, 1] > 0.95
    assert (
        0.7
        < (
            (table["ros_ppg"].head(500) >= pred["p10"])
            & (table["ros_ppg"].head(500) <= pred["p90"])
        ).mean()
        < 0.9
    )


def test_calibration_offsets_widen_narrow_intervals_and_fall_back_for_small_positions() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    position = pd.Series(["WR"] * (n - 10) + ["QB"] * 10)
    actual = pd.Series(rng.normal(10, 3, n))
    p10, p90 = pd.Series(np.full(n, 9.0)), pd.Series(np.full(n, 11.0))  # far too narrow

    offsets = calibration_offsets(p10, p90, actual, position)

    lo, hi = offsets["WR"]
    assert lo > 0 and hi > 0
    assert 0.78 < ((actual >= p10 - lo) & (actual <= p90 + hi)).mean() < 0.9
    assert offsets["QB"] == offsets["*"] != offsets["WR"]
    assert set(offsets) == {"*", "WR", "QB"}


def test_calibration_split_holds_out_only_the_last_training_seasons() -> None:
    train = pd.DataFrame({"season": np.repeat([2016, 2017, 2018, 2019, 2020], 3)})
    fit, cal = split_calibration(train)
    assert sorted(cal["season"].unique()) == [2019, 2020][-CAL_SEASONS:]
    assert sorted(fit["season"].unique()) == [2016, 2017, 2018]
    assert len(fit) + len(cal) == len(train)


def test_calibrated_model_trains_on_seasons_and_keeps_quantiles_ordered() -> None:
    rng = np.random.default_rng(2)
    n = 600
    table = pd.DataFrame({c: rng.uniform(0, 1, n) for c in FEATURES})
    table["position"] = rng.choice(["QB", "RB", "WR", "TE"], n)
    table["season"] = rng.choice([2018, 2019, 2020, 2021], n)
    table["ros_ppg"] = table["std_ppg"] * 10 + rng.normal(0, 2, n)

    pred = quantile_gbm(table)(table.head(100))
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
