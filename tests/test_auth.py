import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest
from cryptography.fernet import Fernet
from fastmcp.server.auth.oauth_proxy.models import JTIMapping, UpstreamTokenSet
from key_value.aio.stores.memory import MemoryStore
from starlette.applications import Starlette

from statline.server import auth
from statline.server.auth import YahooTokenVerifier, build_auth, delete_chain, make_store
from statline.server.auth_config import AuthConfig, load_auth_config

BASE_URL = "https://statline.example"
GUID = "ABCDEF123456"
IDENTITY_JSON = {"fantasy_content": {"users": {"0": {"user": [{"guid": GUID}]}, "count": 1}}}


def _env(key: bytes | None = None, **overrides: str) -> dict[str, str]:
    env = {
        "YAHOO_CLIENT_ID": "cid",
        "YAHOO_CLIENT_SECRET": "csecret",
        "JWT_SIGNING_KEY": "j" * 40,
        "STORAGE_ENCRYPTION_KEY": (key or Fernet.generate_key()).decode(),
        "STATLINE_BASE_URL": BASE_URL + "/",
        "STATLINE_TOKEN_STORE": "disk",
    }
    return {**env, **overrides}


@pytest.fixture
def cfg(tmp_path: Path) -> AuthConfig:
    loaded = load_auth_config(_env(STATLINE_TOKEN_DIR=str(tmp_path / "oauth")))
    assert loaded is not None
    return loaded


def test_config_disabled_when_nothing_set() -> None:
    assert load_auth_config({}) is None
    assert load_auth_config({"STATLINE_BASE_URL": BASE_URL}) is None


def test_config_loads_and_strips_trailing_slash(cfg: AuthConfig) -> None:
    assert cfg.base_url == BASE_URL
    assert cfg.token_store == "disk"


def test_config_partial_names_missing_vars_not_values() -> None:
    env = {"YAHOO_CLIENT_ID": "cid", "YAHOO_CLIENT_SECRET": "very-secret"}
    with pytest.raises(ValueError, match="JWT_SIGNING_KEY") as exc:
        load_auth_config(env)
    assert "STATLINE_BASE_URL" in str(exc.value)
    assert "very-secret" not in str(exc.value)


def test_config_firestore_needs_project() -> None:
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        load_auth_config(_env(STATLINE_TOKEN_STORE="firestore"))


def test_config_bad_fernet_key() -> None:
    with pytest.raises(ValueError, match="Fernet"):
        load_auth_config(_env(STORAGE_ENCRYPTION_KEY="nope"))


def _verifier(handler: httpx2.MockTransport) -> YahooTokenVerifier:
    return YahooTokenVerifier(transport=handler)


def _counting(statuses: list[int | httpx2.Response]) -> tuple[httpx2.MockTransport, list[int]]:
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nxt = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(1)
        if isinstance(nxt, httpx2.Response):
            return nxt
        return httpx2.Response(nxt, json=IDENTITY_JSON if nxt == 200 else {})

    return httpx2.MockTransport(handler), calls


@pytest.mark.anyio
async def test_verifier_returns_guid_subject_and_caches() -> None:
    transport, calls = _counting([200])
    verifier = _verifier(transport)
    first = await verifier.verify_token("tok")
    second = await verifier.verify_token("tok")
    assert first is not None and first.subject == GUID
    assert second is not None and second.subject == GUID
    assert len(calls) == 1


@pytest.mark.anyio
async def test_verifier_serves_cached_result_during_throttle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, calls = _counting([200, 999])
    verifier = _verifier(transport)
    await verifier.verify_token("tok")
    monkeypatch.setattr(auth, "FRESH_S", 0)  # force a refetch, still inside STALE_S
    result = await verifier.verify_token("tok")
    assert result is not None and result.subject == GUID
    assert len(calls) == 2


@pytest.mark.anyio
async def test_verifier_drops_cache_entry_past_stale_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, _ = _counting([200, 999])
    verifier = _verifier(transport)
    await verifier.verify_token("tok")
    monkeypatch.setattr(auth, "FRESH_S", 0)
    monkeypatch.setattr(auth, "STALE_S", 0)
    assert await verifier.verify_token("tok") is None


@pytest.mark.anyio
@pytest.mark.parametrize("status", [429, 999, 503])
async def test_verifier_throttle_without_cache_is_none(status: int) -> None:
    transport, _ = _counting([status])
    assert await _verifier(transport).verify_token("tok") is None


@pytest.mark.anyio
async def test_verifier_401_drops_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, _ = _counting([200, 401, 999])
    verifier = _verifier(transport)
    await verifier.verify_token("tok")
    monkeypatch.setattr(auth, "FRESH_S", 0)
    assert await verifier.verify_token("tok") is None
    # The 401 evicted the entry, so a later outage cannot resurrect it.
    assert await verifier.verify_token("tok") is None


@pytest.mark.anyio
async def test_verifier_html_200_is_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    html = httpx2.Response(200, text="<html>maintenance</html>")
    transport, _ = _counting([200, html])
    verifier = _verifier(transport)
    await verifier.verify_token("tok")
    monkeypatch.setattr(auth, "FRESH_S", 0)
    result = await verifier.verify_token("tok")
    assert result is not None and result.subject == GUID


@pytest.mark.anyio
async def test_verifier_200_without_guid_is_none() -> None:
    transport, _ = _counting([httpx2.Response(200, json={"fantasy_content": {}})])
    assert await _verifier(transport).verify_token("tok") is None


@pytest.mark.anyio
async def test_verifier_network_error_without_cache_is_none() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    assert await _verifier(httpx2.MockTransport(boom)).verify_token("tok") is None


def test_verifier_prunes_expired_entries_on_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    transport, _ = _counting([200])
    verifier = _verifier(transport)
    asyncio.run(verifier.verify_token("old"))
    monkeypatch.setattr(auth, "STALE_S", 0)
    asyncio.run(verifier.verify_token("new"))
    assert len(verifier._cache) == 1


def test_proxy_never_sends_scope_upstream(cfg: AuthConfig) -> None:
    proxy = build_auth(cfg, store=MemoryStore())
    url = proxy._build_upstream_authorize_url(
        "txn1", {"scopes": ["read", "write"], "proxy_code_verifier": None}
    )
    query = parse_qs(urlparse(url).query)
    assert "scope" not in query
    assert "code_challenge" not in query  # PKCE is not forwarded upstream
    assert query["client_id"] == ["cid"]
    assert query["state"] == ["txn1"]
    assert proxy._prepare_scopes_for_token_exchange(["read"]) == []
    assert proxy._prepare_scopes_for_upstream_refresh(["read"]) == []


@pytest.mark.anyio
async def test_dcr_accepts_client_requesting_a_scope(cfg: AuthConfig) -> None:
    proxy = build_auth(cfg, store=MemoryStore())
    app = Starlette(routes=proxy.get_routes("/league/mcp"))
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url=BASE_URL
    ) as client:
        resp = await client.post(
            "/register",
            json={
                "client_name": "claude",
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "scope": "fantasy.read other",
            },
        )
    assert resp.status_code == 201, resp.text


@pytest.mark.anyio
async def test_encrypted_store_round_trip_hides_plaintext(cfg: AuthConfig) -> None:
    store = make_store(cfg)
    await store.put("k", {"refresh": "plaintext-refresh-token"}, collection="c")
    assert await store.get("k", collection="c") == {"refresh": "plaintext-refresh-token"}
    on_disk = b"".join(p.read_bytes() for p in cfg.token_dir.rglob("*") if p.is_file())
    assert on_disk and b"plaintext-refresh-token" not in on_disk


@pytest.mark.anyio
async def test_rotated_key_reads_as_missing_not_error(cfg: AuthConfig) -> None:
    await make_store(cfg).put("k", {"v": 1}, collection="c")
    rotated = AuthConfig(**{**cfg.__dict__, "storage_encryption_key": Fernet.generate_key()})
    assert await make_store(rotated).get("k", collection="c") is None


@pytest.mark.anyio
async def test_delete_chain(cfg: AuthConfig) -> None:
    proxy = build_auth(cfg, store=MemoryStore())
    proxy.get_routes("/league/mcp")  # initializes the JWT issuer, as mounting the app does
    assert await delete_chain(proxy, "garbage") is False

    now = time.time()
    await proxy._upstream_token_store.put(
        key="up1",
        value=UpstreamTokenSet(
            upstream_token_id="up1",
            access_token="a",
            refresh_token="r",
            refresh_token_expires_at=None,
            expires_at=now + 3600,
            token_type="Bearer",
            scope="",
            client_id="c",
            created_at=now,
        ),
    )
    await proxy._jti_mapping_store.put(
        key="jti1", value=JTIMapping(jti="jti1", upstream_token_id="up1", created_at=now)
    )
    jwt = proxy.jwt_issuer.issue_access_token(client_id="c", scopes=[], jti="jti1")

    unknown = proxy.jwt_issuer.issue_access_token(client_id="c", scopes=[], jti="nope")
    assert await delete_chain(proxy, unknown) is False
    assert await delete_chain(proxy, jwt) is True
    assert await proxy._upstream_token_store.get(key="up1") is None
    assert await proxy._jti_mapping_store.get(key="jti1") is None
    assert await delete_chain(proxy, jwt) is False
