from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statline.draft.board import build_board
from statline.draft.league import League, load_league
from statline.draft.mock import _my_pick, lineup_value, mock_draft

REPO = Path(__file__).resolve().parents[1]
Synthetic = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]
LEAGUE = load_league(REPO / "config/league.yaml")


def test_every_team_gets_a_full_legal_roster(synthetic: Synthetic) -> None:
    board = build_board(*synthetic(), LEAGUE)
    positions = board["position"]
    for seed in range(30):
        result = mock_draft(board, LEAGUE, seat=4, seed=seed)
        assert result.picks["player"].is_unique
        for rows in result.rosters.values():
            assert len(rows) == LEAGUE.rounds
            counts = Counter(positions.iloc[rows])
            for pos, minimum in {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}.items():
                assert counts[pos] >= minimum
            assert counts["RB"] + counts["WR"] + counts["TE"] >= 5
            for pos, cap in LEAGUE.max_per_position.items():
                assert counts[pos] <= cap


def test_same_seed_same_picks_and_different_seeds_differ(synthetic: Synthetic) -> None:
    board = build_board(*synthetic(), LEAGUE)
    a = mock_draft(board, LEAGUE, 4, 1).picks
    pd.testing.assert_frame_equal(a, mock_draft(board, LEAGUE, 4, 1).picks)
    assert not a.equals(mock_draft(board, LEAGUE, 4, 2).picks)


def test_seat_avoids_qb_with_top_raw_points_but_low_vor(synthetic: Synthetic) -> None:
    proj, adp = synthetic()
    proj.loc[proj["position"] == "QB", "p50"] += 30  # every QB outscores every RB/WR
    proj["p10"], proj["p90"] = proj["p50"] - 3, proj["p50"] + 3
    board = build_board(proj, adp, LEAGUE)
    for seed in range(10):
        picks = mock_draft(board, LEAGUE, seat=1, seed=seed).picks
        first = picks[picks["team"] == 1].iloc[0]
        assert first["position"] in {"RB", "WR", "TE"}


def test_fallback_when_only_capped_positions_remain() -> None:
    league = League(
        teams=2,
        rounds=3,
        slots={"QB": 1, "RB": 1, "BENCH": 1},
        flex_eligible=("RB", "WR", "TE"),
        max_per_position={"QB": 1, "RB": 1},
    )
    board = pd.DataFrame(
        {
            "name": [f"p{i}" for i in range(6)],
            "position": ["QB", "RB"] * 3,
            "adp": [1.0, 2, 3, 4, 5, 6],
            "p50": [15.0, 14, 13, 12, 11, 10],
            "vor": [5.0, 4, 3, 2, 1, 0],
        }
    )
    board.attrs["replacement"] = {"QB": 10.0, "RB": 10.0, "FLEX": 10.0}
    result = mock_draft(board, league, seat=1, seed=0)
    assert {t: len(r) for t, r in result.rosters.items()} == {1: 3, 2: 3}


LEVELS = {"QB": 15.0, "RB": 10.2, "WR": 10.4, "TE": 8.6, "FLEX": 10.4}


def test_lineup_value_counts_only_starters() -> None:
    nan = float("nan")
    # the second QB has no slot
    assert lineup_value(["QB", "QB"], [20.0, 19.0], LEAGUE, LEVELS) == 5.0
    assert lineup_value(["WR"], [3.0], LEAGUE, LEVELS) == 0.0
    # K/DST (no level) and unprojected players add nothing
    assert lineup_value(["DST", "TE"], [9.0, nan], LEAGUE, LEVELS) == 0.0
    assert lineup_value(["TE", "RB"], [15.3, 20.2], LEAGUE, LEVELS) == pytest.approx(6.7 + 10.0)


def test_second_te_in_flex_is_valued_against_the_flex_level() -> None:
    levels = {"TE": 8.6, "RB": 10.2, "WR": 10.4, "FLEX": 10.4}
    value = lineup_value(["TE", "TE"], [15.3, 11.6], LEAGUE, levels)
    assert value == pytest.approx((15.3 - 8.6) + (11.6 - 10.4))


def test_seat_prefers_starting_wr_over_second_qb() -> None:
    board = pd.DataFrame(
        {
            "position": ["QB", "QB", "WR"],
            "adp": [1.0, 2, 3],
            "p50": [25.0, 23.0, 15.0],
            "vor": [10.0, 8.0, 5.0],
        }
    )
    levels = {"QB": 15.0, "WR": 10.0, "FLEX": 10.0}
    rows = np.array([1, 2])
    args = (rows, board, False)
    assert _my_pick(*args, roster=[0], league=LEAGUE, levels=levels) == 2
    assert _my_pick(*args, roster=[], league=LEAGUE, levels=levels) == 1


def test_mock_draft_requires_replacement_levels() -> None:
    board = pd.DataFrame({"position": ["QB"], "adp": [1.0]})
    with pytest.raises(ValueError, match="replacement"):
        mock_draft(board, LEAGUE, seat=1, seed=0)


def test_seat_never_prefers_an_unprojected_player() -> None:
    """Late in a draft every candidate scores ~0; the tie must not go to a player with no
    projection just because he has a better ADP."""
    league = load_league(REPO / "config" / "league.yaml")
    board = pd.DataFrame(
        {
            "position": ["RB", "RB"],
            "adp": [150.0, 160.0],
            "p50": [np.nan, 5.0],
            "vor": [np.nan, -3.0],
        }
    )
    levels = {"QB": 16.0, "RB": 10.0, "WR": 10.0, "TE": 8.0, "FLEX": 10.0}
    assert _my_pick(np.array([0, 1]), board, False, [], league, levels) == 1
