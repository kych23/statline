import importlib.util
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.auth.auth import AccessToken
from starlette.testclient import TestClient

from statline.server.data import ServerData
from statline.server.http import PRIVACY, build_http_app
from statline.server.usage import UsageMiddleware, hash_identity, usage_key

KEY = b"k" * 32
RAW_IP = "203.0.113.9"
SUBJECT = "GUID123SECRET"


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    @property
    def events(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.lines]


@pytest.fixture
def captured() -> Iterator[Capture]:
    handler = Capture()
    log = logging.getLogger("statline.usage")
    log.addHandler(handler)
    yield handler
    log.removeHandler(handler)


def tiny_server(key: bytes = KEY) -> FastMCP:
    server = FastMCP("tiny")
    server.add_middleware(UsageMiddleware("public", key))

    @server.tool
    def ok_tool(secret_arg: str) -> dict[str, Any]:
        return {"value": secret_arg}

    @server.tool
    def error_tool() -> dict[str, Any]:
        return {"error": "bad_request", "message": "nope"}

    @server.tool
    def boom() -> dict[str, Any]:
        raise RuntimeError("kaboom")

    return server


async def call(server: FastMCP, tool: str, args: dict[str, Any] | None = None) -> None:
    async with Client(server) as client:
        await client.call_tool(tool, args or {}, raise_on_error=False)


@pytest.mark.anyio
async def test_successful_call_logs_one_line(captured: Capture) -> None:
    await call(tiny_server(), "ok_tool", {"secret_arg": "my private question"})
    assert len(captured.lines) == 1
    event = captured.events[0]
    assert event["severity"] == "INFO"
    assert event["event"] == "tool_call"
    assert event["tool"] == "ok_tool"
    assert event["server"] == "public"
    assert event["user"] == "anon"  # in-memory transport has no HTTP request
    assert event["ok"] is True
    assert isinstance(event["ms"], int)
    assert "my private question" not in captured.lines[0]


@pytest.mark.anyio
async def test_error_dict_is_not_ok(captured: Capture) -> None:
    await call(tiny_server(), "error_tool")
    assert captured.events[0]["ok"] is False


@pytest.mark.anyio
async def test_exception_is_not_ok_and_reraised(captured: Capture) -> None:
    async with Client(tiny_server()) as client:
        with pytest.raises(Exception, match="kaboom"):
            await client.call_tool("boom", {})
    assert captured.events[0]["ok"] is False


@pytest.mark.anyio
async def test_logging_failure_does_not_break_the_call(captured: Capture) -> None:
    with patch("statline.server.usage.hash_identity", side_effect=RuntimeError("x")):
        async with Client(tiny_server()) as client:
            result = await client.call_tool("ok_tool", {"secret_arg": "a"})
    assert result.data == {"value": "a"}
    assert captured.lines == []


@pytest.mark.anyio
async def test_user_is_keyed_hash_of_forwarded_ip(captured: Capture) -> None:
    headers = {"x-forwarded-for": f"198.51.100.1, {RAW_IP}"}
    with patch("statline.server.usage.get_http_headers", return_value=headers):
        await call(tiny_server(), "ok_tool", {"secret_arg": "a"})
        await call(tiny_server(), "ok_tool", {"secret_arg": "a"})
        await call(tiny_server(b"other" * 8), "ok_tool", {"secret_arg": "a"})
    same_a, same_b, other_key = (e["user"] for e in captured.events)
    assert same_a == same_b == hash_identity(KEY, f"ip:{RAW_IP}")
    assert other_key != same_a
    assert len(same_a) == 12
    assert all(RAW_IP not in line for line in captured.lines)


@pytest.mark.anyio
async def test_league_user_uses_access_token_subject(captured: Capture) -> None:
    token = AccessToken(token="t", client_id="yahoo", scopes=[], subject=SUBJECT, claims={})
    with patch("statline.server.usage.get_access_token", return_value=token):
        await call(tiny_server(), "ok_tool", {"secret_arg": "a"})
    assert captured.events[0]["user"] == hash_identity(KEY, f"yahoo:{SUBJECT}")
    assert SUBJECT not in captured.lines[0]


def test_usage_key_is_derived_and_deterministic() -> None:
    storage = b"s" * 44
    assert usage_key(storage) == usage_key(storage)
    assert usage_key(storage) != storage
    assert usage_key(b"t" * 44) != usage_key(storage)


def test_usage_key_without_storage_key_is_random(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        first, second = usage_key(None), usage_key(None)
    assert len(first) == 32
    assert first != second
    assert "per-process" in caplog.text


def test_privacy_page_discloses_usage_logs() -> None:
    assert "store nothing about you" not in PRIVACY
    assert "Usage logs" in PRIVACY


def test_public_server_logs_via_http_app(data: ServerData, captured: Capture) -> None:
    with TestClient(build_http_app(data)) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_player", "arguments": {"names": ["Jahmyr Gibbs"]}},
            },
            headers={
                "accept": "application/json, text/event-stream",
                "x-forwarded-for": RAW_IP,
            },
        )
    assert response.status_code == 200
    event = captured.events[0]
    assert (event["tool"], event["server"], event["ok"]) == ("get_player", "public", True)
    assert event["user"] != "anon"
    assert RAW_IP not in captured.lines[0]


# --- scripts/usage.py -------------------------------------------------------------------


@pytest.fixture(scope="module")
def usage_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "usage.py"
    spec = importlib.util.spec_from_file_location("usage_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(ts: str, tool: str, user: str, ok: bool, ms: int, server: str = "public") -> dict:
    payload = {"event": "tool_call", "tool": tool, "server": server, "user": user}
    return {"timestamp": ts, "jsonPayload": {**payload, "ok": ok, "ms": ms}}


def test_summarize(usage_script: ModuleType) -> None:
    entries = [
        entry("2026-11-01T10:00:00Z", "get_player", "aaa", True, 10),
        entry("2026-11-01T11:00:00Z", "get_player", "bbb", False, 30),
        entry("2026-11-02T10:00:00Z", "rankings", "aaa", True, 50),
        entry("2026-11-02T12:00:00Z", "rankings", "anon", True, 70),
        entry("2026-11-02T13:00:00Z", "my_league", "ccc", True, 90, server="league"),
        {"timestamp": "2026-11-02T13:00:00Z", "jsonPayload": {"event": "other"}},
    ]
    summary = usage_script.summarize(usage_script.parse_entries(entries))
    assert summary["total_calls"] == 5
    assert summary["distinct_users"] == 3
    assert summary["anon_calls"] == 1
    assert summary["by_server"].loc["public", "calls"] == 4
    assert summary["by_server"].loc["public", "users"] == 2
    assert summary["by_server"].loc["league", "users"] == 1
    tools = summary["by_tool"]
    assert tools.loc["get_player", "error_rate"] == pytest.approx(0.5)
    assert tools.loc["get_player", "users"] == 2
    assert tools.loc["get_player", "p50_ms"] == pytest.approx(20)
    assert tools.loc["rankings", "users"] == 1
    assert summary["per_day"].to_dict() == {"2026-11-01": 2, "2026-11-02": 3}
    assert "Total calls: 5" in usage_script.render(summary)


def test_summarize_empty(usage_script: ModuleType) -> None:
    summary = usage_script.summarize(usage_script.parse_entries([]))
    assert summary == {"total_calls": 0}
    assert usage_script.render(summary) == "No tool calls in this window."
