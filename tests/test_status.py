import pandas as pd

from statline.data.nflverse import clean_injuries, clean_roster_status
from statline.model.status import attach_status


def projections() -> pd.DataFrame:
    return pd.DataFrame({"player_id": ["a", "b", "c", "d"], "p50": [1.0, 2.0, 3.0, 4.0]})


def roster(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["player_id", "week", "roster_status"]).assign(team="AAA")


def injury(rows: list[tuple[str, int, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["player_id", "week", "report_status", "injury"])


def test_attach_status_maps_codes_and_unknown_or_missing_to_null() -> None:
    rosters = roster([("a", 10, "ACT"), ("b", 10, "RES"), ("c", 10, "ZZZ"), ("a", 9, "RES")])
    out = attach_status(projections(), rosters, injury([]))
    assert out["availability"].tolist() == ["active", "injured reserve", None, None]
    assert out["status_as_of"].tolist() == [10, 10, 10, 10]
    assert out["p50"].tolist() == [1.0, 2.0, 3.0, 4.0]  # nothing else touched


def test_attach_status_falls_back_to_a_bye_teams_latest_week() -> None:
    rosters = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "team": ["AAA", "BBB"],  # BBB is on bye in week 10
            "week": [10, 9],
            "roster_status": ["RES", "ACT"],
        }
    )
    out = attach_status(projections(), rosters, injury([]))
    assert out["availability"].tolist()[:2] == ["injured reserve", "active"]
    assert out["status_as_of"].iloc[0] == 10


def test_attach_status_uses_latest_injury_week_only() -> None:
    reports = injury(
        [
            ("a", 10, "Questionable", "Hamstring"),
            ("b", 9, "Out", "Knee"),  # stale: week 9 is not the latest report week
            ("c", 10, "Out", None),
        ]
    )
    out = attach_status(projections(), roster([]), reports)
    assert out["injury"].tolist() == [
        "Questionable (hamstring), week 10",
        None,
        "Out, week 10",
        None,
    ]


def test_attach_status_with_empty_frames_is_all_null() -> None:
    out = attach_status(projections(), roster([]), injury([]))
    assert out["availability"].isna().all()
    assert out["injury"].isna().all()
    assert out["status_as_of"].isna().all()


def test_clean_roster_status_and_injuries_filter_and_rename() -> None:
    raw_rosters = pd.DataFrame(
        {
            "gsis_id": ["a", None, "b"],
            "team": ["X", "X", "X"],
            "week": [1, 1, 1],
            "status": ["ACT", "ACT", "RES"],
            "game_type": ["REG", "REG", "POST"],
        }
    )
    assert clean_roster_status(raw_rosters)["player_id"].tolist() == ["a"]
    raw_injuries = pd.DataFrame(
        {
            "gsis_id": ["a", "b", "c"],
            "week": [1, 1, 1],
            "report_status": ["Out", None, "Doubtful"],
            "report_primary_injury": ["Knee", None, "Ankle"],
            "game_type": ["REG", "REG", "REG"],
        }
    )
    assert clean_injuries(raw_injuries)["player_id"].tolist() == ["a", "c"]
