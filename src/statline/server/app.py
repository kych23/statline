"""Assemble the MCP server."""

from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider

from statline.server.data import ServerData
from statline.server.instructions import BASE_INSTRUCTIONS
from statline.server.limits import HeavyToolGate
from statline.server.tools import register
from statline.server.usage import UsageMiddleware

MOCK_PER_HOUR = 10  # per signed-in Yahoo account
MOCK_PUBLIC_PER_HOUR = 60  # shared by all public callers (~0.4 s CPU each)


def mock_gate() -> HeavyToolGate:
    return HeavyToolGate(MOCK_PER_HOUR, public_per_hour=MOCK_PUBLIC_PER_HOUR)


def build_server(
    data: ServerData,
    gate: HeavyToolGate | None = None,
    *,
    name: str = "statline",
    auth: AuthProvider | None = None,
    usage: UsageMiddleware | None = None,
    instructions: str = BASE_INSTRUCTIONS,
) -> FastMCP:
    server = FastMCP(name, instructions=instructions, auth=auth)
    if usage is not None:
        server.add_middleware(usage)
    register(server, data, gate or mock_gate())
    return server
