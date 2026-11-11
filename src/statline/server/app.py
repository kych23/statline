"""Assemble the MCP server."""

from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider

from statline.server.data import ServerData
from statline.server.limits import HeavyToolGate
from statline.server.tools import register
from statline.server.usage import UsageMiddleware

MOCK_PER_HOUR = 10
INSTRUCTIONS = (
    "Statline serves fantasy football projections as rest-of-season average PPR points per "
    "game, with a 10th-90th percentile range for that average (not single-week scores). "
    "Use get_player to look up and compare players and rankings to list the top players by "
    "value over replacement. mock_draft simulates a full snake draft from a seat and returns "
    "the roster."
)


def build_server(
    data: ServerData,
    gate: HeavyToolGate | None = None,
    *,
    name: str = "statline",
    auth: AuthProvider | None = None,
    usage: UsageMiddleware | None = None,
) -> FastMCP:
    server = FastMCP(name, instructions=INSTRUCTIONS, auth=auth)
    if usage is not None:
        server.add_middleware(usage)
    register(server, data, gate or HeavyToolGate(per_caller_per_hour=MOCK_PER_HOUR))
    return server
