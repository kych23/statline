import pandas as pd

from statline.data.nflverse import (
    WEEKLY_COLUMNS,
    clean_rosters,
    clean_snaps,
    clean_weekly,
)


def weekly_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(WEEKLY_COLUMNS, 0)
    row.update(
        player_id="00-1",
        player_display_name="A",
        position="WR",
        team="DET",
        season=2025,
        week=1,
        season_type="REG",
    )
    row.update(overrides)
    return row


def test_clean_weekly_keeps_regular_season_fantasy_positions_and_renames() -> None:
    raw = pd.DataFrame(
        [
            weekly_row(),
            weekly_row(season_type="POST"),
            weekly_row(position="K"),
        ]
    )
    out = clean_weekly(raw)
    assert len(out) == 1
    assert list(out.columns) == list(WEEKLY_COLUMNS.values())
    assert out.loc[0, "name"] == "A"


def test_clean_snaps_filters_and_renames() -> None:
    raw = pd.DataFrame(
        {
            "pfr_player_id": ["X", "Y"],
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["DET", "DET"],
            "offense_pct": [0.8, 0.9],
            "game_type": ["REG", "REG"],
            "position": ["RB", "OT"],
        }
    )
    out = clean_snaps(raw)
    assert out["pfr_id"].tolist() == ["X"]
    assert out["snap_share"].tolist() == [0.8]


def test_clean_rosters_drops_missing_ids_and_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "gsis_id": ["00-1", "00-1", "", None],
            "pfr_id": ["A", "A", "B", "C"],
            "season": [2025] * 4,
            "position": ["WR"] * 4,
            "birth_date": pd.to_datetime(["2000-01-01"] * 4),
            "rookie_year": [2022] * 4,
        }
    )
    out = clean_rosters(raw)
    assert out["player_id"].tolist() == ["00-1"]
