import numpy as np
import pandas as pd
import pytest

from statline.features.build import TARGET_COLUMNS, build_features

STATS = ["ppr_points", "targets", "carries", "pass_attempts", "target_share", "snap_share"]


def game(player: str, season: int, week: int, ppr: float, **stats: float) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(STATS, 0.0)
    row.update(
        player_id=player,
        name=player.upper(),
        position="WR",
        team="DET",
        season=season,
        week=week,
        ppr_points=ppr,
    )
    row.update(stats)
    return row


ROSTERS = pd.DataFrame(
    {
        "player_id": ["a", "b"],
        "pfr_id": ["A", "B"],
        "season": [2024, 2024],
        "birth_date": pd.to_datetime(["2000-09-01", "1995-09-01"]),
        "rookie_year": [2022, 2018],
    }
)


def row_for(table: pd.DataFrame, player: str, season: int, t: int) -> pd.Series:
    match = table[(table.player_id == player) & (table.season == season) & (table.as_of_week == t)]
    assert len(match) == 1
    return match.iloc[0]


def test_feature_math_on_a_hand_built_season() -> None:
    games = pd.DataFrame(
        [
            game("a", 2023, 1, 10.0, targets=5),
            game("a", 2023, 2, 20.0, targets=7),
            game("a", 2024, 1, 4.0, targets=2),
            game("a", 2024, 2, 8.0, targets=4),
            game("a", 2024, 3, 12.0, targets=6),
            game("a", 2024, 4, 16.0, targets=8),
            game("a", 2024, 5, 30.0, targets=10),
        ]
    )
    table = build_features(games, ROSTERS, seasons=[2024], max_week=5)

    pre = row_for(table, "a", 2024, 0)
    assert pre.games_so_far == 0 and np.isnan(pre.std_ppg)
    assert pre.prev_ppg == 15.0 and pre.prev_games == 2
    assert pre.ros_ppg == pytest.approx(14.0) and pre.ros_games == 5
    assert pre.age == pytest.approx(24.0, abs=0.01) and pre.years_exp == 2

    wk4 = row_for(table, "a", 2024, 4)
    assert wk4.games_so_far == 4
    assert wk4.std_ppg == 10.0  # (4 + 8 + 12 + 16) / 4
    assert wk4.l3_ppg == 12.0  # last three: 8, 12, 16
    assert wk4.std_targets_pg == 5.0
    assert wk4.ros_ppg == 30.0 and wk4.ros_games == 1


def test_rookie_has_no_preseason_row_and_inactive_veteran_drops_midseason() -> None:
    games = pd.DataFrame(
        [
            game("a", 2023, 1, 10.0),  # a: played 2023, nothing in 2024
            game("b", 2024, 1, 5.0),  # b: rookie in 2024
            game("b", 2024, 2, 7.0),
        ]
    )
    table = build_features(games, ROSTERS, seasons=[2024], max_week=2)
    keys = set(zip(table.player_id, table.as_of_week, strict=True))
    assert ("a", 0) in keys and ("a", 1) not in keys
    assert ("b", 0) not in keys and ("b", 1) in keys


def test_features_never_see_the_future() -> None:
    """Scramble every stat after week t (and every later season); as-of-t features
    must not change. Only the target may."""
    rng = np.random.default_rng(0)
    games = pd.DataFrame(
        [
            game(
                p,
                s,
                w,
                float(rng.uniform(0, 30)),
                targets=float(rng.integers(0, 12)),
                carries=float(rng.integers(0, 20)),
                target_share=float(rng.uniform()),
                snap_share=float(rng.uniform()),
            )
            for p in ("a", "b")
            for s in (2023, 2024, 2025)
            for w in range(1, 18)
        ]
    )
    for t in (0, 3, 9):
        future = (games.season > 2024) | ((games.season == 2024) & (games.week > t))
        scrambled = games.copy()
        scrambled.loc[future, STATS] = rng.uniform(0, 50, size=(future.sum(), len(STATS)))

        def features(g: pd.DataFrame, t: int = t) -> pd.DataFrame:
            table = build_features(g, ROSTERS, seasons=[2024], max_week=t)
            return table[table.as_of_week == t].drop(columns=TARGET_COLUMNS)

        pd.testing.assert_frame_equal(features(games), features(scrambled))
