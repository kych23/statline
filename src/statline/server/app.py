"""Assemble the MCP server."""

from fastmcp import FastMCP

from statline.server.data import ServerData
from statline.server.limits import HeavyToolGate
from statline.server.tools import register

MOCK_PER_HOUR = 10
INSTRUCTIONS = (
    "Statline serves fantasy football projections as rest-of-season PPR points per game "
    "with a 10th-90th percentile range. Use get_player to look up and compare players and "
    "rankings to list the top players by value over replacement. mock_draft simulates a "
    "full snake draft from a seat and returns the roster."
)


def build_server(data: ServerData, gate: HeavyToolGate | None = None) -> FastMCP:
    server = FastMCP("statline", instructions=INSTRUCTIONS)
    register(server, data, gate or HeavyToolGate(per_caller_per_hour=MOCK_PER_HOUR))
    return server
