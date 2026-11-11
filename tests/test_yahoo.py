import json
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from statline.yahoo.client import (
    TTLCache,
    YahooAuthError,
    YahooClient,
    YahooError,
    YahooThrottled,
    user_key,
)
from statline.yahoo.parse import (
    LEAGUE_KEY_RE,
    TEAM_KEY_RE,
    iter_nodes,
    parse_guid,
    parse_leagues,
    parse_my_team,
    parse_players,
    parse_settings,
    parse_standings,
)

FIXTURES = Path(__file__).parent / "fixtures" / "yahoo"


def fx(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def settings_payload(*stats: tuple[int, str]) -> dict:
    return {
        "fantasy_content": {
            "league": [
                {"league_key": "461.l.1"},
                {
                    "settings": [
                        {
                            "stat_modifiers": {
                                "stats": [{"stat": {"stat_id": i, "value": v}} for i, v in stats]
                            }
                        }
                    ]
                },
            ]
        }
    }


# ---------------------------------------------------------------- parsers
def test_iter_nodes_merges_fragments_in_document_order() -> None:
    payload = {"a": {"0": {"x": [{"id": 1}, {"n": "one"}]}, "1": {"x": {"id": 2}}, "count": 2}}
    assert [n["id"] for n in iter_nodes(payload, "x")] == [1, 2]
    assert next(iter_nodes(payload, "x")) == {"id": 1, "n": "one"}


def test_guid_and_leagues_newest_season_first() -> None:
    assert parse_guid(fx("user.json")) == "ABCGUID123"
    assert parse_guid({}) is None
    payload = {
        "leagues": {
            "0": {"league": [{"league_key": "423.l.9", "name": "Old", "season": "2024"}]},
            "1": {
                "league": [
                    {"league_key": "461.l.1", "name": "New", "season": "2026", "num_teams": "12"}
                ]
            },
        }
    }
    got = parse_leagues(payload)
    assert [lg["league_key"] for lg in got] == ["461.l.1", "423.l.9"]
    assert got[0]["season"] == 2026 and got[0]["num_teams"] == 12
    assert parse_leagues(fx("leagues.json"))[0]["name"] == "Island League"


def test_settings_ppr_detection_and_slots() -> None:
    full = parse_settings(fx("settings_full.json"))
    assert full["ppr"] == "full" and full["reception_points"] == 1.0
    assert full["roster_slots"] == {"QB": 1, "WR": 2, "RB": 2, "TE": 1, "W/R/T": 1, "BN": 6}
    assert parse_settings(fx("settings_half.json"))["ppr"] == "half"
    none = parse_settings(settings_payload((4, "0.04")))
    assert none["ppr"] == "none" and none["reception_points"] == 0.0
    assert parse_settings(settings_payload((11, "0.75")))["ppr"] == "0.75"


def test_standings_sorted_by_rank_with_unranked_last() -> None:
    teams = parse_standings(fx("standings.json"))
    assert [t["name"] for t in teams] == ["Kyle Island", "Bravo"]
    assert teams[0]["wins"] == 8 and teams[0]["points_for"] == 1200.25
    unranked = {
        "teams": {
            "0": {"team": [{"team_key": "461.l.1.t.9", "name": "NoRank"}]},
            "1": {
                "team": [{"team_key": "461.l.1.t.8", "name": "R1", "team_standings": {"rank": "1"}}]
            },
        }
    }
    assert [t["name"] for t in parse_standings(unranked)] == ["R1", "NoRank"]


def test_my_team_uses_is_current_login() -> None:
    assert parse_my_team(fx("teams.json")) == {"team_key": "461.l.12345.t.1", "name": "Kyle Island"}
    assert parse_my_team({"teams": {}}) is None


def test_players_roster_and_free_agents() -> None:
    roster = parse_players(fx("roster.json"))
    assert [p["name"] for p in roster] == ["Josh Allen", "Zzzz Nobody"]
    assert roster[0]["position"] == "QB" and roster[0]["nfl_team"] == "Buf"
    assert roster[0]["status"] is None and roster[0]["percent_owned"] is None
    agents = parse_players(fx("players.json"))
    assert [p["name"] for p in agents] == ["Rico Dowdle", "Hurt Receiver"]
    assert agents[0]["percent_owned"] == 41.5
    assert agents[1]["status"] == "Q" and agents[1]["percent_owned"] == 12.0


@pytest.mark.parametrize("bad", ["461.l.123\n", "461.l.123/../x", "", "461.l.", "461.l.1.t.2"])
def test_league_key_rejects_malformed(bad: str) -> None:
    assert LEAGUE_KEY_RE.fullmatch(bad) is None


@pytest.mark.parametrize("bad", ["461.l.123.t.4\n", "461.l.123.t.4/../x", "461.l.123"])
def test_team_key_rejects_malformed(bad: str) -> None:
    assert TEAM_KEY_RE.fullmatch(bad) is None


# ----------------------------------------------------------------- cache
def test_cache_expiry_and_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr("statline.yahoo.client.time.monotonic", lambda: now[0])
    cache = TTLCache(10)
    cache.put("u", "a", 1, started=now[0])
    assert cache.get("u", "a") == 1
    now[0] += 10
    cache.put("u", "b", 2, started=now[0])  # prunes "a"
    assert cache.get("u", "a") is None
    assert ("u", "a") not in cache._data
    assert cache.get("u", "b") == 2


def test_cache_drop_fences_inflight_put(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr("statline.yahoo.client.time.monotonic", lambda: now[0])
    cache = TTLCache(900)
    cache.put("u", "a", 1, started=now[0])
    started = now[0]
    now[0] += 1
    assert cache.drop("u") == 1
    cache.put("u", "a", 2, started=started)  # fetch began before the drop
    assert cache.get("u", "a") is None
    cache.put("u", "a", 3, started=now[0])
    assert cache.get("u", "a") == 3


def test_user_key_is_16_hex() -> None:
    key = user_key("GUID")
    assert len(key) == 16 and int(key, 16) >= 0 and "GUID" not in key


# ---------------------------------------------------------------- client
def make_client(handler, cache: TTLCache | None = None) -> YahooClient:
    return YahooClient("tok", "u1", cache or TTLCache(900), transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_get_success_and_cache_hit() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"x": 1})

    client = make_client(handler)
    assert await client.get("league/461.l.1/settings") == {"x": 1}
    assert await client.get("league/461.l.1/settings") == {"x": 1}
    assert len(calls) == 1
    assert calls[0].headers["authorization"] == "Bearer tok"
    assert calls[0].url.params["format"] == "json"


@pytest.mark.anyio
async def test_method_urls() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.raw_path.decode().split("?")[0])
        return httpx.Response(200, json={})

    client = make_client(handler, TTLCache(0))
    await client.leagues()
    await client.roster("461.l.1.t.2")
    await client.free_agents("461.l.1", position="WR", count=10)
    await client.free_agents("461.l.1")
    assert paths == [
        "/fantasy/v2/users;use_login=1/games;game_keys=nfl/leagues",
        "/fantasy/v2/team/461.l.1.t.2/roster",
        "/fantasy/v2/league/461.l.1/players;status=A;sort=AR;count=10;position=WR",
        "/fantasy/v2/league/461.l.1/players;status=A;sort=AR;count=25",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, YahooAuthError),
        (403, YahooAuthError),
        (429, YahooThrottled),
        (999, YahooThrottled),
        (500, YahooError),
    ],
)
async def test_error_mapping(status: int, error: type[YahooError]) -> None:
    client = make_client(lambda request: httpx.Response(status))
    with pytest.raises(error) as exc:
        await client.get("league/461.l.1/settings")
    assert exc.value.status == status


@pytest.mark.anyio
async def test_html_200_raises_yahoo_error() -> None:
    client = make_client(lambda request: httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(YahooError) as exc:
        await client.get("league/461.l.1/settings")
    assert exc.value.status == 200


@pytest.mark.anyio
async def test_network_failure_raises_yahoo_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(YahooError) as exc:
        await make_client(handler).get("league/461.l.1/settings")
    assert exc.value.status is None


@pytest.mark.anyio
async def test_bad_key_raises_before_any_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    client = make_client(handler)
    for call in (
        client.settings("461.l.123\n"),
        client.standings("461.l.123/../x"),
        client.teams("bad"),
        client.roster("461.l.123"),
        client.free_agents("461.l.1", position="WR/../x"),
        client.free_agents("461.l.1", count=0),
    ):
        with pytest.raises(ValueError):
            await call
    assert calls == []
