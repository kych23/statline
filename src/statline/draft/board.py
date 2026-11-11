"""Draft board: projections and ADP joined into one row per draftable player."""

import pandas as pd

from statline.data.adp import name_key
from statline.draft.league import League
from statline.draft.value import add_vor, replacement_levels

MIN_EXTRA_PLAYERS = 20  # board must hold every pick plus a little slack
BOARD_COLUMNS = ["player_id", "name", "position", "team", "adp", "p50", "vor"]


def build_board(projections: pd.DataFrame, adp: pd.DataFrame, league: League) -> pd.DataFrame:
    """One row per player, sorted by ADP. K/DST have no projection: vor 0, p50 NaN.
    Projected players missing from ADP are placed after the last ranked player, best vor
    first; ADP players without a projection keep a NaN vor so opponents can still take them."""
    levels = replacement_levels(projections, league)
    proj = add_vor(projections, league)
    proj = proj.assign(name_key=proj["name"].map(name_key))
    board = proj[["name_key", "position", "player_id", "name", "team", "p50", "vor"]].merge(
        adp[["name_key", "position", "name", "team", "adp"]],
        on=["name_key", "position"],
        how="outer",
        suffixes=("", "_adp"),
    )
    board["name"] = board["name"].fillna(board["name_adp"])
    board["team"] = board["team"].fillna(board["team_adp"])
    board.loc[board["position"].isin(["K", "DST"]), "vor"] = 0.0

    unranked = board.index[board["adp"].isna()]
    order = board.loc[unranked].sort_values("vor", ascending=False).index
    board.loc[order, "adp"] = board["adp"].max() + 1 + pd.RangeIndex(len(order))

    board = board.sort_values("adp", kind="stable").reset_index(drop=True)
    needed = league.teams * league.rounds + MIN_EXTRA_PLAYERS
    if len(board) < needed:
        raise ValueError(f"board has {len(board)} players; need at least {needed}")
    board = board[BOARD_COLUMNS].copy()
    board.attrs["replacement"] = levels  # mock reads these to value FLEX slots
    return board
