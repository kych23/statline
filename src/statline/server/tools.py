"""Projection tools: plain functions over ServerData, plus their MCP registration."""

import math
import secrets
from typing import Annotated, Any, Literal

import anyio.to_thread
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from pydantic import Field

from statline.draft.mock import mock_draft as run_mock_draft
from statline.server.data import ServerData
from statline.server.limits import Busy, HeavyToolGate, OverQuota, caller_id
from statline.server.resolve import resolve

Position = Literal["QB", "RB", "WR", "TE"]
SortBy = Literal["vor", "p50"]
MAX_NAMES = 8
# Claude read p10/p90 as "bad week / great week"; they bound the season-long average.
PROJECTION_NOTE = (
    "p10/p50/p90 are the 10th/50th/90th percentiles of the player's AVERAGE PPR points per game "
    "over the rest of the season (an 80% range for that average), not single-week scores, "
    "which swing far more."
)
MAX_TOP_N = 50
MAX_SEED = 2**31 - 1
BUSY_RETRY_AFTER_S = 15


def bad_request(message: str) -> dict[str, Any]:
    return {"error": "bad_request", "message": message}


def as_of(data: ServerData) -> dict[str, Any]:
    return {"season": data.season, "week": data.as_of_week, "data_date": data.data_date}


def get_player(data: ServerData, names: list[str]) -> dict[str, Any]:
    if not 1 <= len(names) <= MAX_NAMES:
        return bad_request(f"pass between 1 and {MAX_NAMES} names")
    table = data.projections
    position_rank = table.groupby("position")["p50"].rank(ascending=False, method="min")

    players: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for query in names:
        res = resolve(query, table)
        if res.status == "resolved" and res.player_index is not None:
            row: Any = table.loc[res.player_index]
            players.append(
                {
                    "name": row["name"],
                    "position": row["position"],
                    "team": row["team"],
                    "projection": {k: round(float(row[k]), 1) for k in ("p10", "p50", "p90")},
                    "vor": round(float(row["vor"]), 1),
                    "position_rank": int(position_rank[res.player_index]),
                }
            )
        elif res.status == "ambiguous":
            ambiguous.append(
                {
                    "query": query,
                    "candidates": [
                        {k: table.loc[i, k] for k in ("name", "position", "team")}
                        for i in res.candidates
                    ],
                }
            )
        else:
            unresolved.append(query)

    result: dict[str, Any] = {
        "players": players,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
        "as_of": as_of(data),
        "projection_note": PROJECTION_NOTE,
    }
    if len(players) >= 2:
        ordered = sorted(players, key=lambda p: p["projection"]["p50"], reverse=True)
        best = ordered[0]["projection"]["p50"]
        result["comparison"] = [
            {
                "name": p["name"],
                "p50": p["projection"]["p50"],
                "delta_p50": round(p["projection"]["p50"] - best, 1),
            }
            for p in ordered
        ]
    return result


def rankings(
    data: ServerData,
    position: Position | None = None,
    top_n: int = 20,
    sort_by: SortBy = "vor",
) -> dict[str, Any]:
    if not 1 <= top_n <= MAX_TOP_N:
        return bad_request(f"top_n must be between 1 and {MAX_TOP_N}")
    table = data.projections
    if position is not None:
        table = table[table["position"] == position]
    top = table.sort_values(sort_by, ascending=False).head(top_n)
    rows = [
        {
            "rank": rank,
            "name": row["name"],
            "position": row["position"],
            "team": row["team"],
            **{k: round(float(row[k]), 1) for k in ("p10", "p50", "p90", "vor")},
        }
        for rank, (_, row) in enumerate(top.iterrows(), start=1)
    ]
    return {
        "rankings": rows,
        "position": position,
        "sort_by": sort_by,
        "as_of": as_of(data),
        "projection_note": PROJECTION_NOTE,
    }


def _round1(value: Any) -> float | None:
    return None if math.isnan(value) else round(float(value), 1)


async def mock_draft(
    data: ServerData, gate: HeavyToolGate, caller: str, seat: int, seed: int | None
) -> dict[str, Any]:
    board = data.board
    if board is None:
        return {"error": "unavailable", "message": "mock drafts are not configured on this server"}
    league = data.league
    if not 1 <= seat <= league.teams:
        return bad_request(f"seat must be between 1 and {league.teams}")
    if seed is None:
        seed = secrets.randbelow(2**31)
    elif not 0 <= seed <= MAX_SEED:
        return bad_request(f"seed must be between 0 and {MAX_SEED}")

    try:
        async with gate.slot(caller):
            result = await anyio.to_thread.run_sync(run_mock_draft, board, league, seat, seed)
    except Busy:
        return {"status": "busy", "retry_after_s": BUSY_RETRY_AFTER_S}
    except OverQuota as exc:
        return {"status": "rate_limited", "resets_in_s": exc.resets_in_s}

    mine = result.picks[result.picks["team"] == seat]
    roster = [
        {
            "round": int(pick["round"]),
            "pick": int(pick["pick_no"]),
            "name": pick["name"],
            "position": pick["position"],
            "p50": _round1(board["p50"].iloc[pick["player"]]),
            "vor": _round1(pick["vor"]),
        }
        for _, pick in mine.iterrows()
    ]
    return {
        "seat": seat,
        "seed": seed,
        "league": {"teams": league.teams, "rounds": league.rounds},
        "roster": roster,
        "note": ("Roster p50 and vor are preseason (week 0) PPR points per game."),
        "method": (
            "Opponents draft by average draft position with calibrated randomness; "
            "your picks maximize starting-lineup value over replacement."
        ),
    }


def register(server: FastMCP, data: ServerData, gate: HeavyToolGate) -> None:
    read_only = {"read_only_hint": True}

    @server.tool(name="get_player", title="Get player projections", annotations=read_only)
    def get_player_tool(
        names: Annotated[
            list[str],
            Field(description="1-8 player names; suffixes, last names and typos are tolerated"),
        ],
    ) -> dict[str, Any]:
        """Projected rest-of-season average PPR points per game for named players: p50 is the
        median, p10-p90 an 80% range for that season-long average (not single-week scores),
        plus value over replacement and position rank.
        Two or more players also get a comparison against the best one."""
        return get_player(data, names)

    @server.tool(name="rankings", title="Player rankings", annotations=read_only)
    def rankings_tool(
        position: Annotated[
            Position | None, Field(description="Restrict to one position; omit for all")
        ] = None,
        top_n: Annotated[int, Field(description="Number of players, 1-50", ge=1, le=50)] = 20,
        sort_by: Annotated[
            SortBy,
            Field(description="vor: value over replacement (cross-position); p50: median points"),
        ] = "vor",
    ) -> dict[str, Any]:
        """Top players by projected rest-of-season average PPR points per game (p10-p90 is an
        80% range for that average, not single-week scores), ranked by value over replacement
        or median projection."""
        return rankings(data, position, top_n, sort_by)

    @server.tool(name="mock_draft", title="Mock draft", annotations=read_only)
    async def mock_draft_tool(
        seat: Annotated[int, Field(description="Your draft position, 1 to the number of teams")],
        seed: Annotated[
            int | None,
            Field(description="Replay a previous draft by its seed; omit for a new draft"),
        ] = None,
    ) -> dict[str, Any]:
        """Simulate a full snake mock draft and return the roster picked from your seat.
        Opponents pick by average draft position with noise; your picks follow projected
        value. Limited per caller per hour."""
        caller = caller_id(get_http_headers().get("x-forwarded-for"))
        return await mock_draft(data, gate, caller, seat, seed)
