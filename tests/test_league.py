from pathlib import Path

import pytest

from statline.draft.league import League, load_league

REPO = Path(__file__).resolve().parents[1]


def league(**overrides: object) -> League:
    fields: dict = dict(
        teams=12,
        rounds=4,
        slots={"QB": 1, "RB": 1, "FLEX": 1, "BENCH": 1},
        flex_eligible=("RB", "WR", "TE"),
        max_per_position={"QB": 2},
    )
    fields.update(overrides)
    return League(**fields)


def test_shipped_config_loads_and_is_consistent() -> None:
    lg = load_league(REPO / "config" / "league.yaml")
    assert lg.teams == 12 and lg.rounds == 15
    assert "BENCH" not in lg.starters and lg.starters["FLEX"] == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"slots": {"QB": 1, "SS": 3}}, "unknown roster slots"),
        ({"teams": 1}, "at least 2 teams"),
        ({"rounds": 5}, "sum to 4"),
        ({"flex_eligible": ("RB", "XX")}, "flex_eligible"),
        ({"max_per_position": {"QB": 0}}, "below its starting slots"),
    ],
)
def test_invalid_settings_fail_loudly(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        league(**overrides)
