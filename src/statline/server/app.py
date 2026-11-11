"""Assemble the MCP server."""

from fastmcp import FastMCP

from statline.server.data import ServerData
from statline.server.tools import register

INSTRUCTIONS = (
    "Statline serves fantasy football projections as rest-of-season PPR points per game "
    "with a 10th-90th percentile range. Use get_player to look up and compare players and "
    "rankings to list the top players by value over replacement."
)


def build_server(data: ServerData) -> FastMCP:
    server = FastMCP("statline", instructions=INSTRUCTIONS)
    register(server, data)
    return server
