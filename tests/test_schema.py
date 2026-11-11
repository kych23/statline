import pandas as pd
import pytest

from statline.model.schema import MIN_PLAYERS, OUTPUT_COLUMNS, projection_problems


def projections(rows: int = MIN_PLAYERS, week: int = 9, season: int = 2026) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "player_id": [f"00-{i}" for i in range(rows)],
            "season": season,
            "name": "A",
            "position": "WR",
            "team": "DET",
            "as_of_week": week,
            "p10": 4.0,
            "p50": 8.0,
            "p90": 14.0,
            "built_on": "2026-11-10",
        }
    )
    assert list(df.columns) == OUTPUT_COLUMNS
    return df


def test_good_frame_has_no_problems() -> None:
    assert projection_problems(projections(), 9, season=2026) == []


def test_season_is_optional() -> None:
    assert projection_problems(projections(season=1999), 9) == []


def test_too_few_rows() -> None:
    problems = projection_problems(projections(rows=MIN_PLAYERS - 1), 9)
    assert len(problems) == 1 and "rows" in problems[0]


def test_null_quantile() -> None:
    df = projections()
    df.loc[0, "p50"] = None
    problems = projection_problems(df, 9)
    assert problems == ["1 null values in p50"]


def test_crossed_quantiles() -> None:
    df = projections()
    df.loc[0, "p10"] = 9.0
    df.loc[1, "p90"] = 7.0
    assert projection_problems(df, 9) == ["2 rows violate p10 <= p50 <= p90"]


def test_wrong_week() -> None:
    assert "as_of_week" in projection_problems(projections(week=8), 9)[0]


def test_mixed_weeks() -> None:
    df = projections()
    df.loc[0, "as_of_week"] = 8
    assert "as_of_week" in projection_problems(df, 9)[0]


def test_wrong_season() -> None:
    problems = projection_problems(projections(season=2025), 9, season=2026)
    assert len(problems) == 1 and "season" in problems[0]


def test_reports_every_failure() -> None:
    df = projections(rows=MIN_PLAYERS - 1, week=8)
    df.loc[0, "p50"] = None
    assert len(projection_problems(df, 9)) == 3


@pytest.mark.parametrize("week", [0, 18])
def test_week_bounds_are_not_special(week: int) -> None:
    assert projection_problems(projections(week=week), week) == []
