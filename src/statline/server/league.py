"""Yahoo league tools: plain async functions over ServerData and a YahooClient, plus MCP
registration. Projections are matched to Yahoo players by normalized name and position."""

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import pandas as pd
from fastmcp import FastMCP
from fastmcp.server.auth.oauth_proxy import OAuthProxy
from fastmcp.server.dependencies import get_access_token, get_http_headers
from httpx2 import AsyncBaseTransport
from pydantic import Field

from statline.data.adp import name_key
from statline.server import auth
from statline.server.data import ServerData
from statline.server.tools import Position, as_of, bad_request
from statline.yahoo.client import (
    TTLCache,
    YahooAuthError,
    YahooClient,
    YahooError,
    YahooThrottled,
    user_key,
)
from statline.yahoo.parse import (
    LEAGUE_KEY_RE,
    TEAM_KEY_RE,
    parse_leagues,
    parse_my_team,
    parse_players,
    parse_settings,
    parse_standings,
)

ATTRIBUTION = {
    "text": "Fantasy data provided by Yahoo Fantasy",
    "logo": "https://763445962456-brand-assets.s3.us-west-2.amazonaws.com/brandwebsite/s3fs-public/Yahoo_Fantasy.svg",
    "link": "https://football.fantasysports.yahoo.com/",
}
POSITIONS = ("QB", "RB", "WR", "TE")
MAX_FREE_AGENTS = 25
FREE_AGENT_FETCH = 50
THROTTLE_RETRY_AFTER_S = 60
NON_PPR_WARNING = "projections are full-PPR points per game"
APP_ACCESS_URL = "https://login.yahoo.com/account/security/app-access"


class _Matcher:
    """Yahoo player -> projections row: name key plus a listed position, else a unique name."""

    def __init__(self, projections: pd.DataFrame) -> None:
        self._rows = projections.assign(_key=projections["name"].map(name_key))
        self._by_key = self._rows.groupby("_key")

    def match(self, name: str, position: str | None) -> Any | None:
        key = name_key(name)
        if key not in self._by_key.groups:
            return None
        rows = self._by_key.get_group(key)
        for pos in (position or "").split(","):
            hit = rows[rows["position"] == pos.strip()]
            if len(hit) == 1:
                return hit.iloc[0]
        return rows.iloc[0] if len(rows) == 1 else None


def _projection_fields(row: Any | None) -> dict[str, Any]:
    if row is None:
        return {"projection": None, "vor": None}
    return {
        "projection": {k: round(float(row[k]), 1) for k in ("p10", "p50", "p90")},
        "vor": round(float(row["vor"]), 1),
    }


def _scoring(settings: dict[str, Any]) -> dict[str, Any]:
    scoring = {"ppr": settings["ppr"], "reception_points": settings["reception_points"]}
    return {
        "scoring": scoring,
        **({"warning": NON_PPR_WARNING} if settings["ppr"] != "full" else {}),
    }


async def _guarded(call: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    try:
        return await call()
    except YahooAuthError:
        return {
            "error": "yahoo_reauthorize",
            "how": "Reconnect the Statline connector and sign in with Yahoo again.",
        }
    except YahooThrottled:
        return {"error": "yahoo_throttled", "retry_after_s": THROTTLE_RETRY_AFTER_S}
    except YahooError:
        return {"error": "yahoo_unavailable"}
    except ValueError as exc:
        return bad_request(str(exc))


async def my_league(
    data: ServerData, client: YahooClient, league_key: str | None
) -> dict[str, Any]:
    if league_key is not None and not LEAGUE_KEY_RE.fullmatch(league_key):
        return bad_request("invalid league_key")
    return await _guarded(lambda: _my_league(data, client, league_key))


async def _my_league(
    data: ServerData, client: YahooClient, league_key: str | None
) -> dict[str, Any]:
    key = league_key
    if key is None:
        leagues = [
            lg
            for lg in parse_leagues(await client.leagues())
            if LEAGUE_KEY_RE.fullmatch(lg["league_key"])
        ]
        if len(leagues) != 1:
            return {"leagues": leagues, "next": "call my_league with a league_key"}
        key = leagues[0]["league_key"]

    settings = parse_settings(await client.settings(key))
    standings = parse_standings(await client.standings(key))
    team = parse_my_team(await client.teams(key))
    if team is not None and not TEAM_KEY_RE.fullmatch(team["team_key"]):
        team = None
    roster: list[dict[str, Any]] = []
    if team is not None:
        matcher = _Matcher(data.projections)
        for p in parse_players(await client.roster(team["team_key"])):
            row = matcher.match(p["name"], p["position"])
            roster.append(
                {
                    "name": p["name"],
                    "position": p["position"],
                    "nfl_team": p["nfl_team"],
                    "status": p["status"],
                    **_projection_fields(row),
                }
            )
    return {
        "league_key": key,
        **_scoring(settings),
        "roster_slots": settings["roster_slots"],
        "standings": standings,
        "my_team": team,
        "roster": roster,
        "as_of": as_of(data),
        "attribution": ATTRIBUTION,
    }


async def free_agents(
    data: ServerData,
    client: YahooClient,
    league_key: str,
    position: str | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    if not LEAGUE_KEY_RE.fullmatch(league_key):
        return bad_request("invalid league_key")
    if position is not None and position not in POSITIONS:
        return bad_request(f"position must be one of {', '.join(POSITIONS)}")
    if not 1 <= top_n <= MAX_FREE_AGENTS:
        return bad_request(f"top_n must be between 1 and {MAX_FREE_AGENTS}")
    return await _guarded(lambda: _free_agents(data, client, league_key, position, top_n))


async def _free_agents(
    data: ServerData, client: YahooClient, league_key: str, position: str | None, top_n: int
) -> dict[str, Any]:
    settings = parse_settings(await client.settings(league_key))
    available = parse_players(
        await client.free_agents(league_key, position, count=FREE_AGENT_FETCH)
    )
    matcher = _Matcher(data.projections)
    matched: list[tuple[float, dict[str, Any]]] = []
    for p in available:
        row = matcher.match(p["name"], p["position"])
        if row is None:
            continue
        fields = _projection_fields(row)
        matched.append(
            (
                fields["vor"],
                {
                    "name": p["name"],
                    "position": p["position"],
                    "nfl_team": p["nfl_team"],
                    "status": p["status"],
                    "percent_owned": p["percent_owned"],
                    **fields,
                },
            )
        )
    matched.sort(key=lambda item: item[0], reverse=True)
    return {
        "league_key": league_key,
        "position": position,
        **_scoring(settings),
        "free_agents": [row for _, row in matched[:top_n]],
        "unprojected": len(available) - len(matched),
        "as_of": as_of(data),
        "attribution": ATTRIBUTION,
    }


def register_league(
    server: FastMCP,
    data: ServerData,
    cache: TTLCache,
    *,
    proxy: OAuthProxy,
    transport: AsyncBaseTransport | None = None,
) -> None:
    def client_for() -> tuple[YahooClient, str] | None:
        token = get_access_token()
        if token is None or not token.subject:
            return None
        return YahooClient(token.token, user_key(token.subject), cache, transport=transport), (
            token.subject
        )

    login_required = {"error": "login_required"}

    @server.tool(
        name="my_league",
        title="My Yahoo league",
        annotations={"read_only_hint": True},
    )
    async def my_league_tool(
        league_key: Annotated[
            str | None,
            Field(
                description="Yahoo league key such as '461.l.12345'. Omit to list your NFL "
                "leagues; a single league is selected automatically."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Your Yahoo fantasy league: scoring, standings and your roster. Each rostered player
        carries a projection (p10/p50/p90 are rest-of-season PPR points per game) and vor
        (value over replacement, same unit); players without a projection have null."""
        if (ctx := client_for()) is None:
            return login_required
        return await my_league(data, ctx[0], league_key)

    @server.tool(
        name="free_agents",
        title="Yahoo free agents",
        annotations={"read_only_hint": True},
    )
    async def free_agents_tool(
        league_key: Annotated[str, Field(description="Yahoo league key such as '461.l.12345'")],
        position: Annotated[
            Position | None, Field(description="Restrict to one position; omit for all")
        ] = None,
        top_n: Annotated[
            int, Field(description="Number of players to return, 1-25", ge=1, le=25)
        ] = 15,
    ) -> dict[str, Any]:
        """Best available free agents in your Yahoo league, ranked by value over replacement
        (vor) computed from rest-of-season PPR points per game projections (p10/p50/p90). Draws
        from Yahoo's 50 most-owned available players; 'unprojected' counts the ones skipped
        for lacking a projection."""
        if (ctx := client_for()) is None:
            return login_required
        return await free_agents(data, ctx[0], league_key, position, top_n)

    @server.tool(
        name="delete_my_data",
        title="Delete my data",
        annotations={"read_only_hint": False, "destructive_hint": True},
    )
    async def delete_my_data_tool() -> dict[str, Any]:
        """Delete the Yahoo login tokens Statline stores for you and your cached Yahoo data.
        You will need to sign in again to use league tools."""
        if (ctx := client_for()) is None:
            return login_required
        header = get_http_headers(include={"authorization"}).get("authorization", "")
        scheme, _, bearer = header.partition(" ")
        # Revoke first: a failed cache drop must not leave a usable token chain behind.
        revoked = scheme.lower() == "bearer" and await auth.delete_chain(proxy, bearer.strip())
        cache.drop(user_key(ctx[1]))
        return {
            "deleted": True,
            "token_chain_revoked": revoked,
            "note": f"To fully revoke Yahoo access, remove Statline at {APP_ACCESS_URL}",
        }
