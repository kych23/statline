"""Yahoo OAuth for the MCP endpoint, via fastmcp's OAuthProxy.

Yahoo: 1-hour access tokens, rotating refresh tokens, HTTP Basic client auth, scopes
configured on the Yahoo app (sending `scope` fails with invalid_scope), PKCE undocumented.
Identity is the Fantasy API user GUID.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import httpx2
from cryptography.fernet import Fernet
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.oauth_proxy import OAuthProxy
from key_value.aio.protocols import AsyncKeyValue
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1CollectionSanitizationStrategy,
    FileTreeV1KeySanitizationStrategy,
)
from key_value.aio.stores.firestore import FirestoreStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

from statline.server.auth_config import AuthConfig
from statline.yahoo.client import BASE_URL
from statline.yahoo.parse import parse_guid

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
IDENTITY_URL = f"{BASE_URL}/users;use_login=1"
CLIENT_REDIRECTS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    "http://localhost:*",
    "http://127.0.0.1:*",
]
FRESH_S = 300  # a positive verification is trusted this long
STALE_S = 1800  # and served during a Yahoo throttle or outage up to this age


class YahooTokenVerifier(TokenVerifier):
    """Validates a Yahoo access token by fetching the caller's Fantasy GUID.

    On a throttle or outage it serves the last positive result (<= STALE_S old) rather than
    None: None makes the proxy attempt a refresh, which turns throttling into refresh storms.
    """

    def __init__(
        self, *, transport: httpx2.AsyncBaseTransport | None = None, timeout_s: float = 10.0
    ) -> None:
        super().__init__(required_scopes=None)
        self._transport = transport
        self._timeout = timeout_s
        self._cache: dict[str, tuple[float, AccessToken]] = {}

    async def verify_token(self, token: str) -> AccessToken | None:
        key = hashlib.sha256(token.encode()).hexdigest()
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < FRESH_S:
            return hit[1].model_copy()
        status, payload = await self._fetch_identity(token)
        if status == 200:
            guid = parse_guid(payload)
            if not guid:
                logger.warning("yahoo identity 200 without a guid")
                return None
            result = AccessToken(
                token=token, client_id="yahoo", scopes=[], subject=guid, claims={"sub": guid}
            )
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if now - v[0] < STALE_S}
            self._cache[key] = (now, result)
            return result.model_copy()
        if status in (401, 403):
            self._cache.pop(key, None)
            return None
        if hit and time.monotonic() - hit[0] < STALE_S:
            logger.info("yahoo identity unavailable (%s); serving cached verification", status)
            return hit[1].model_copy()
        logger.info("yahoo identity unavailable (%s) and no cached verification", status)
        return None

    async def _fetch_identity(self, token: str) -> tuple[int | None, Any]:
        """(status, json); status is None for network errors and non-JSON 200s."""
        try:
            async with httpx2.AsyncClient(timeout=self._timeout, transport=self._transport) as c:
                resp = await c.get(
                    IDENTITY_URL,
                    params={"format": "json"},
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx2.HTTPError as exc:
            logger.info("yahoo identity unreachable (%s)", type(exc).__name__)
            return None, None
        if resp.status_code != 200:
            return resp.status_code, None
        try:
            return 200, resp.json()
        except ValueError:
            return None, None


class YahooOAuthProxy(OAuthProxy):
    """OAuthProxy that never sends `scope` to Yahoo (authorize, exchange, refresh)."""

    def _build_upstream_authorize_url(self, txn_id: str, transaction: dict[str, Any]) -> str:
        # Base falls back to required_scopes when the transaction has none; both are empty here.
        return super()._build_upstream_authorize_url(txn_id, {**transaction, "scopes": []})

    def _prepare_scopes_for_token_exchange(self, scopes: list[str]) -> list[str]:
        return []

    def _prepare_scopes_for_upstream_refresh(self, scopes: list[str]) -> list[str]:
        return []


def make_store(cfg: AuthConfig) -> FernetEncryptionWrapper:
    inner: AsyncKeyValue
    if cfg.token_store == "firestore":
        inner = FirestoreStore(project=cfg.google_cloud_project)
    else:
        directory = Path(cfg.token_dir)
        directory.mkdir(parents=True, exist_ok=True)
        inner = FileTreeStore(
            data_directory=directory,
            key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(directory),
            collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(directory),
        )
    # A rotated key then logs users out instead of crashing every request.
    return FernetEncryptionWrapper(
        key_value=inner, fernet=Fernet(cfg.storage_encryption_key), raise_on_decryption_error=False
    )


def build_auth(
    cfg: AuthConfig, *, verifier: TokenVerifier | None = None, store: AsyncKeyValue | None = None
) -> YahooOAuthProxy:
    proxy = YahooOAuthProxy(
        upstream_authorization_endpoint=AUTHORIZE_URL,
        upstream_token_endpoint=TOKEN_URL,
        upstream_client_id=cfg.yahoo_client_id,
        upstream_client_secret=cfg.yahoo_client_secret,
        token_endpoint_auth_method="client_secret_basic",
        forward_pkce=False,
        token_verifier=verifier or YahooTokenVerifier(),
        base_url=cfg.base_url,
        allowed_client_redirect_uris=CLIENT_REDIRECTS,
        jwt_signing_key=cfg.jwt_signing_key,
        client_storage=store or make_store(cfg),
        fastmcp_access_token_expiry_seconds=3600,
        token_expiry_threshold_seconds=120,
    )
    # required_scopes normalizes to [] (4.x too; the valid_scopes arg treats [] as unset), so
    # DCR would reject a client that requests any scope. Scopes grant nothing here: they are
    # stripped before Yahoo and access is decided by the Yahoo identity alone.
    if proxy.client_registration_options is not None:
        proxy.client_registration_options.valid_scopes = None
        proxy.client_registration_options.default_scopes = None
    return proxy


async def delete_chain(proxy: OAuthProxy, bearer_jwt: str) -> bool:
    """Revoke the caller's whole token chain by deleting its upstream token record.

    Every access and refresh token, rotated ones included, resolves through the same
    upstream_token_id; without that record both token loading and refresh fail. Yahoo
    itself is not told (no revocation endpoint is configured).
    """
    try:
        jti = proxy.jwt_issuer.verify_token(bearer_jwt)["jti"]
    except Exception:  # noqa: BLE001  # any decode/signature/expiry failure means "not ours"
        return False
    mapping = await proxy._jti_mapping_store.get(key=jti)
    if mapping is None:
        return False
    await proxy._upstream_token_store.delete(key=mapping.upstream_token_id)
    await proxy._jti_mapping_store.delete(key=jti)
    return True
