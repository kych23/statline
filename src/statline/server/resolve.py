"""Turn a user-typed player name into a row of the projections table."""

import difflib
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from statline.data.adp import name_key

MAX_CANDIDATES = 5
FUZZY_CUTOFF = 0.85


@dataclass(frozen=True)
class Resolution:
    status: Literal["resolved", "ambiguous", "unresolved"]
    player_index: int | None
    candidates: list[int]  # row indices, highest p50 first


def resolve(query: str, players: pd.DataFrame) -> Resolution:
    key = name_key(query)
    keys = players["name"].map(name_key)

    def pick(rows: pd.Index) -> Resolution:
        if len(rows) == 1:
            return Resolution("resolved", int(rows[0]), [int(rows[0])])
        top = players.loc[rows].sort_values("p50", ascending=False).index[:MAX_CANDIDATES]
        return Resolution("ambiguous", None, [int(i) for i in top])

    if not key:
        return Resolution("unresolved", None, [])
    if (exact := players.index[keys == key]).size:
        return pick(exact)
    if " " not in key:
        last = keys.str.rsplit(" ", n=1).str[-1]
        if (rows := players.index[last == key]).size:
            return pick(rows)
    close = difflib.get_close_matches(
        key, keys.unique().tolist(), n=MAX_CANDIDATES, cutoff=FUZZY_CUTOFF
    )
    if close:
        return pick(players.index[keys.isin(close)])
    return Resolution("unresolved", None, [])
