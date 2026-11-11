from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx2
import pytest

from statline.server.data import ServerData
from statline.server.http import build_http_app

HOST = "statline-abc.a.run.app"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


@asynccontextmanager
async def http_client(data: ServerData) -> AsyncIterator[httpx2.AsyncClient]:
    app = build_http_app(data)
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url=f"https://{HOST}") as client:
            yield client


@pytest.mark.anyio
async def test_landing_page_shows_connector_url_and_season(data: ServerData) -> None:
    async with http_client(data) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert f"https://{HOST}/mcp" in resp.text
    assert "2026" in resp.text


@pytest.mark.anyio
async def test_healthz(data: ServerData) -> None:
    async with http_client(data) as client:
        resp = await client.get("/healthz")
    assert resp.json() == {"ok": True, "season": 2026, "as_of_week": 9, "data_date": "2026-11-01"}


@pytest.mark.anyio
async def test_mcp_initialize_accepts_public_host_and_origin(data: ServerData) -> None:
    async with http_client(data) as client:
        resp = await client.post(
            "/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Origin": "https://claude.ai"}
        )
    assert resp.status_code == 200
    assert "statline" in resp.text


@pytest.mark.anyio
async def test_mcp_is_stateless(data: ServerData) -> None:
    async with http_client(data) as client:
        resp = await client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
    assert "mcp-session-id" not in resp.headers
