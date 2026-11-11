from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statline.data.adp import name_key
from statline.draft.board import build_board
from statline.draft.league import League, load_league

Synthetic = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]
REPO = Path(__file__).resolve().parents[1]


def test_join_matches_suffix_variants(synthetic: Synthetic) -> None:
    league = load_league(REPO / "config/league.yaml")
    proj, adp = synthetic()
    proj.loc[0, "name"] = "Chris Godwin"
    adp.loc[0, "name"] = "Chris Godwin Jr."
    adp.loc[0, "name_key"] = name_key("Chris Godwin Jr.")
    board = build_board(proj, adp, league)
    row = board[board["name"].str.startswith("Chris Godwin")]
    assert len(row) == 1
    assert row["p50"].notna().all()
    assert row["adp"].notna().all()


def test_kicker_and_dst_get_zero_vor_and_no_projection(synthetic: Synthetic) -> None:
    board = build_board(*synthetic(), load_league(REPO / "config/league.yaml"))
    special = board[board["position"].isin(["K", "DST"])]
    assert len(special) == 40
    assert (special["vor"] == 0.0).all()
    assert special["p50"].isna().all()


def test_unranked_projected_players_follow_max_adp_in_vor_order(synthetic: Synthetic) -> None:
    league = load_league(REPO / "config/league.yaml")
    proj, adp = synthetic()
    dropped = adp[adp["position"].isin(["RB", "WR"]) & (adp["adp"] > 200)].index
    assert len(dropped) >= 3
    max_adp = adp.drop(dropped)["adp"].max()
    board = build_board(proj, adp.drop(dropped), league)
    tail = board[board["adp"] > max_adp]
    assert len(tail) == len(dropped)
    assert list(tail["adp"]) == [max_adp + i for i in range(1, len(dropped) + 1)]
    assert tail["vor"].is_monotonic_decreasing


def test_adp_player_without_projection_is_kept_with_nan_vor(synthetic: Synthetic) -> None:
    league = load_league(REPO / "config/league.yaml")
    proj, adp = synthetic()
    board = build_board(proj.iloc[1:], adp, league)
    row = board[board["name"] == "QB0 Player"]
    assert len(row) == 1
    assert np.isnan(row["vor"].item())


def test_too_small_board_raises(synthetic: Synthetic) -> None:
    league = League(
        teams=12,
        rounds=15,
        slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1, "BENCH": 6},
        flex_eligible=("RB", "WR", "TE"),
        max_per_position={},
    )
    proj, adp = synthetic()
    with pytest.raises(ValueError, match="need at least"):
        build_board(proj.head(50), adp[adp["position"].isin(["K", "DST"])].head(10), league)
