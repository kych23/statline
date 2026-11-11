"""Abuse limits for expensive tools: a per-caller hourly quota and one simulation at a time."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

WINDOW_S = 3600.0
# claude.ai relays every user's public requests from shared infrastructure, so the public
# server cannot tell callers apart: they share one quota. League callers are Yahoo accounts.
PUBLIC_CALLER = "public"


class OverQuota(Exception):
    def __init__(self, resets_in_s: int) -> None:
        super().__init__(f"quota exhausted; resets in {resets_in_s}s")
        self.resets_in_s = resets_in_s


class Busy(Exception):
    pass


class HeavyToolGate:
    def __init__(
        self,
        per_caller_per_hour: int,
        clock: Callable[[], float] = time.monotonic,
        *,
        public_per_hour: int | None = None,
    ) -> None:
        self._limit = per_caller_per_hour
        self._public_limit = per_caller_per_hour if public_per_hour is None else public_per_hour
        self._clock = clock
        self._calls: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        for caller in list(self._calls):
            history = self._calls[caller]
            while history and now - history[0] >= WINDOW_S:
                history.popleft()
            if not history:
                del self._calls[caller]

    @asynccontextmanager
    async def slot(self, caller: str) -> AsyncIterator[None]:
        now = self._clock()
        self._prune(now)
        history = self._calls.get(caller)
        limit = self._public_limit if caller == PUBLIC_CALLER else self._limit
        if history is not None and len(history) >= limit:
            raise OverQuota(max(1, int(history[0] + WINDOW_S - now + 0.999)))
        if self._lock.locked():
            raise Busy
        async with self._lock:
            self._calls.setdefault(caller, deque()).append(now)
            yield


def quota_caller(subject: str | None) -> str:
    """Quota key: the signed-in Yahoo account, else the shared public bucket."""
    return f"yahoo:{subject}" if subject else PUBLIC_CALLER


def caller_id(forwarded_for: str | None) -> str:
    # The rightmost hop is appended by the platform proxy; leftmost entries are client-controlled.
    hops = [h.strip() for h in (forwarded_for or "").split(",")]
    last = next((h for h in reversed(hops) if h), None)
    return f"ip:{last}" if last else "anon"
