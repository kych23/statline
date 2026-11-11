"""Replacement level and value over replacement (VOR).

A player's draft value is not his raw points but his margin over what a team could get
for free: the best player left at his position once every team's starting lineup is
filled. In a 1-QB league a 20-point QB can be worth less than a 15-point RB.
"""

import pandas as pd

from statline.draft.league import League


def replacement_levels(
    players: pd.DataFrame, league: League, value: str = "p50"
) -> dict[str, float]:
    """Greedy league-wide lineup fill: each position's dedicated slots first, then FLEX
    from the best remaining flex-eligible players. Replacement = best player not starting.

    `players` has `position` and `value` columns; positions with no rows are omitted.
    """
    ranked = players.sort_values(value, ascending=False)
    starting: set = set()
    for pos, n in league.starters.items():
        if pos != "FLEX":
            starting.update(ranked[ranked["position"] == pos].index[: league.teams * n])
    flex_pool = ranked[ranked["position"].isin(league.flex_eligible) & ~ranked.index.isin(starting)]
    starting.update(flex_pool.index[: league.teams * league.starters.get("FLEX", 0)])

    bench = ranked[~ranked.index.isin(starting)]
    levels = {}
    for pos in ranked["position"].unique():
        on_bench = bench.loc[bench["position"] == pos, value]
        # if every player at the position starts, the worst starter is the replacement
        levels[str(pos)] = float(
            on_bench.max() if len(on_bench) else ranked.loc[ranked["position"] == pos, value].min()
        )
    return levels


def add_vor(players: pd.DataFrame, league: League, value: str = "p50") -> pd.DataFrame:
    levels = replacement_levels(players, league, value)
    return players.assign(vor=players[value] - players["position"].map(levels))
