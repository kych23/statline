from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest
from cryptography.fernet import Fernet
from fastmcp.server.auth.auth import AccessToken, TokenVerifier

from statline.server.auth import build_auth
from statline.server.auth_config import AuthConfig
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


class RejectAll(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        return None


def auth_config(tmp_path: Path) -> AuthConfig:
    return AuthConfig(
        yahoo_client_id="cid",
        yahoo_client_secret="csecret",
        jwt_signing_key="j" * 40,
        storage_encryption_key=Fernet.generate_key(),
        base_url=f"https://{HOST}",
        token_store="disk",
        token_dir=tmp_path / "oauth",
    )


@asynccontextmanager
async def http_client(
    data: ServerData, tmp_path: Path | None = None
) -> AsyncIterator[httpx2.AsyncClient]:
    if tmp_path is None:
        app = build_http_app(data)
    else:
        cfg = auth_config(tmp_path)
        app = build_http_app(data, cfg, auth=build_auth(cfg, verifier=RejectAll()))
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
async def test_health(data: ServerData) -> None:
    async with http_client(data) as client:
        resp = await client.get("/health")
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


@pytest.mark.anyio
async def test_league_endpoint_absent_without_auth(data: ServerData) -> None:
    async with http_client(data) as client:
        assert (await client.post("/league/mcp", json=INITIALIZE)).status_code == 404
        assert (await client.get("/.well-known/oauth-authorization-server")).status_code == 404
        landing = (await client.get("/")).text
    assert "/league/mcp" not in landing


@pytest.mark.anyio
async def test_league_endpoint_requires_bearer(data: ServerData, tmp_path: Path) -> None:
    async with http_client(data, tmp_path) as client:
        resp = await client.post("/league/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        meta = await client.get("/.well-known/oauth-protected-resource/league/mcp")
        server_meta = await client.get("/.well-known/oauth-authorization-server")
        rejected = await client.post(
            "/league/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": "Bearer x"}
        )
    assert resp.status_code == 401
    assert "resource_metadata" in resp.headers["www-authenticate"]
    assert rejected.status_code == 401
    assert meta.status_code == 200
    assert meta.json()["resource"] == f"https://{HOST}/league/mcp"
    assert server_meta.json()["authorization_endpoint"] == f"https://{HOST}/authorize"


@pytest.mark.anyio
async def test_public_mcp_and_pages_stay_open_with_auth(data: ServerData, tmp_path: Path) -> None:
    async with http_client(data, tmp_path) as client:
        mcp = await client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        landing = await client.get("/")
        health = await client.get("/health")
    assert mcp.status_code == 200
    assert health.status_code == 200
    assert f"https://{HOST}/league/mcp" in landing.text
    assert "Fantasy data provided by Yahoo Fantasy" in landing.text
    assert 'href="/privacy"' in landing.text


@pytest.mark.anyio
async def test_privacy_page(data: ServerData) -> None:
    async with http_client(data) as client:
        resp = await client.get("/privacy")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "delete_my_data" in resp.text and "Firestore" in resp.text
