"""Read-only Yahoo Fantasy API client.

Logging rule: status codes and timings only, never bodies, tokens or GUIDs.
"""

import hashlib
import logging
import time
from typing import Any

import httpx2 as httpx

from statline.yahoo.parse import LEAGUE_KEY_RE, TEAM_KEY_RE

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"


class YahooError(Exception):
    def __init__(self, status: int | None, message: str = ""):
        super().__init__(message or f"Yahoo returned {status}")
        self.status = status


class YahooAuthError(YahooError):
    """401/403: the user must re-authorize."""


class YahooThrottled(YahooError):
    """429 or Yahoo's 999 throttle."""


class TTLCache:
    """In-process cache keyed by (user, resource), droppable per user."""

    def __init__(self, ttl_s: float):
        self.ttl_s = ttl_s
        self._data: dict[tuple[str, str], tuple[float, Any]] = {}
        self._dropped_at: dict[str, float] = {}

    def get(self, user: str, resource: str) -> Any | None:
        hit = self._data.get((user, resource))
        if hit is None:
            return None
        if time.monotonic() - hit[0] >= self.ttl_s:
            del self._data[(user, resource)]
            return None
        return hit[1]

    def put(self, user: str, resource: str, value: Any, *, started: float) -> None:
        """`started` is when the fetch began. A write that began before the user's last
        drop is discarded, so an in-flight read cannot resurrect deleted data."""
        if started < self._dropped_at.get(user, float("-inf")):
            return
        now = time.monotonic()
        for key in [k for k, (t, _) in self._data.items() if now - t >= self.ttl_s]:
            del self._data[key]
        self._data[(user, resource)] = (now, value)

    def drop(self, user: str) -> int:
        self._dropped_at[user] = time.monotonic()
        keys = [k for k in self._data if k[0] == user]
        for key in keys:
            del self._data[key]
        return len(keys)


def user_key(subject: str) -> str:
    """The only per-user identifier stored or logged."""
    return hashlib.sha256(subject.encode()).hexdigest()[:16]


def _check_league_key(key: str) -> None:
    if not LEAGUE_KEY_RE.fullmatch(key):
        raise ValueError("invalid league key")


def _check_team_key(key: str) -> None:
    if not TEAM_KEY_RE.fullmatch(key):
        raise ValueError("invalid team key")


class YahooClient:
    def __init__(
        self,
        token: str,
        user: str,
        cache: TTLCache,
        timeout_s: float = 15,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._token = token
        self._user = user
        self._cache = cache
        self._timeout = timeout_s
        self._transport = transport

    async def get(self, resource: str) -> Any:
        cached = self._cache.get(self._user, resource)
        if cached is not None:
            return cached
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as http:
                resp = await http.get(
                    f"{BASE_URL}/{resource}",
                    params={"format": "json"},
                    headers={"Authorization": f"Bearer {self._token}"},
                )
        except httpx.HTTPError as exc:
            raise YahooError(None, f"Yahoo unreachable ({type(exc).__name__})") from exc
        logger.info(
            "yahoo %s -> %s in %.0f ms",
            resource.split("/")[0].split(";")[0],
            resp.status_code,
            1000 * (time.monotonic() - start),
        )
        if resp.status_code in (401, 403):
            raise YahooAuthError(resp.status_code)
        if resp.status_code in (429, 999):
            raise YahooThrottled(resp.status_code)
        if resp.status_code != 200:
            raise YahooError(resp.status_code)
        try:
            data = resp.json()
        except ValueError as exc:  # maintenance pages arrive as HTML with a 200
            raise YahooError(200, "non-JSON response") from exc
        self._cache.put(self._user, resource, data, started=start)
        return data

    async def leagues(self) -> Any:
        return await self.get("users;use_login=1/games;game_keys=nfl/leagues")

    async def settings(self, league_key: str) -> Any:
        _check_league_key(league_key)
        return await self.get(f"league/{league_key}/settings")

    async def standings(self, league_key: str) -> Any:
        _check_league_key(league_key)
        return await self.get(f"league/{league_key}/standings")

    async def teams(self, league_key: str) -> Any:
        _check_league_key(league_key)
        return await self.get(f"league/{league_key}/teams")

    async def roster(self, team_key: str) -> Any:
        _check_team_key(team_key)
        return await self.get(f"team/{team_key}/roster")

    async def free_agents(
        self, league_key: str, position: str | None = None, count: int = 25
    ) -> Any:
        _check_league_key(league_key)
        if position is not None and not position.isalnum():
            raise ValueError("invalid position")
        if not 1 <= count <= 100:
            raise ValueError("count must be 1-100")
        filters = f";position={position}" if position else ""
        resource = f"league/{league_key}/players;status=A;sort=AR;count={count}{filters}"
        return await self.get(resource)
