"""Anonymous usage metrics: one JSON log line per tool call, with a keyed one-way caller hash.

Cloud Run turns a stdout line that is a JSON object into jsonPayload, and `severity` sets the
log level. Arguments, results, IPs and account ids are never logged.
"""

import hashlib
import hmac
import json
import logging
import os
import sys
import time
from typing import Any

import mcp.types as mt
from fastmcp.server.dependencies import get_access_token, get_http_headers
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

from statline.server.limits import caller_id

usage_log = logging.getLogger("statline.usage")
logger = logging.getLogger(__name__)

KEY_CONTEXT = b"statline-usage-v1"
HASH_HEX_CHARS = 12
ANON = "anon"


def _configure_usage_log() -> None:
    """Emit the bare JSON line on stdout, untouched by the root logger's format."""
    if usage_log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    usage_log.addHandler(handler)
    usage_log.setLevel(logging.INFO)
    usage_log.propagate = False


_configure_usage_log()


def usage_key(storage_encryption_key: bytes | None) -> bytes:
    """HMAC key for caller hashes, derived so the storage key itself is never used for it."""
    if storage_encryption_key is None:
        logger.warning("no storage key: usage hashes are per-process and change on restart")
        return os.urandom(32)
    return hmac.new(storage_encryption_key, KEY_CONTEXT, hashlib.sha256).digest()


def hash_identity(key: bytes, identity: str) -> str:
    if identity == ANON:
        return ANON
    return hmac.new(key, identity.encode(), hashlib.sha256).hexdigest()[:HASH_HEX_CHARS]


def _identity() -> str:
    token = get_access_token()
    if token is not None and token.subject:
        return f"yahoo:{token.subject}"
    return caller_id(get_http_headers(include={"x-forwarded-for"}).get("x-forwarded-for"))


def is_error_result(result: ToolResult) -> bool:
    content = result.structured_content
    return result.is_error or (isinstance(content, dict) and "error" in content)


class UsageMiddleware(Middleware):
    def __init__(self, server_label: str, key: bytes) -> None:
        self._server = server_label
        self._key = key

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            result = await call_next(context)
        except Exception:
            self._record(context.message.name, False, started)
            raise
        self._record(context.message.name, not is_error_result(result), started)
        return result

    def _record(self, tool: str, ok: bool, started: float) -> None:
        try:
            line: dict[str, Any] = {
                "severity": "INFO",
                "event": "tool_call",
                "tool": tool,
                "server": self._server,
                "user": hash_identity(self._key, _identity()),
                "ok": ok,
                "ms": round((time.perf_counter() - started) * 1000),
            }
            usage_log.info(json.dumps(line, separators=(",", ":")))
        except Exception:  # metrics must never break a tool call
            logger.exception("usage logging failed")
