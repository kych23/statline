import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pandas as pd
import pytest

from statline.server.data import ServerData
from statline.server.league import ATTRIBUTION, free_agents, my_league
from statline.yahoo.client import TTLCache, YahooClient

FIXTURES = Path(__file__).parent / "fixtures" / "yahoo"
KEY = "461.l.12345"
Handler = Callable[[str], httpx2.Response | None]


def fixture_response(path: str, settings: str = "settings_full.json") -> httpx2.Response:
    if path.endswith("/leagues"):
        name = "leagues.json"
    elif path.endswith("/settings"):
        name = settings
    elif path.endswith("/standings"):
        name = "standings.json"
    elif path.endswith("/teams"):
        name = "teams.json"
    elif path.endswith("/roster"):
        name = "roster.json"
    else:
        name = "players.json"
    return httpx2.Response(200, content=(FIXTURES / name).read_bytes())


def make_client(handler: Handler | None = None, settings: str = "settings_full.json"):
    calls: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix("/fantasy/v2/")
        calls.append(path)
        return (handler and handler(path)) or fixture_response(path, settings)

    client = YahooClient("tok", "u1", TTLCache(900), transport=httpx2.MockTransport(respond))
    return client, calls


def players_payload(rows: list[tuple[str, str, float, str | None]]) -> httpx2.Response:
    players = {}
    for i, (name, pos, owned, status) in enumerate(rows):
        frag: list[dict[str, Any]] = [{"name": {"full": name}}, {"display_position": pos}]
        if status:
            frag.append({"status": status})
        players[str(i)] = {"player": [frag, {"percent_owned": [{"value": str(owned)}]}]}
    players["count"] = len(rows)
    body = {"fantasy_content": {"league": [{"league_key": KEY}, {"players": players}]}}
    return httpx2.Response(200, json=body)


def two_leagues(path: str) -> httpx2.Response | None:
    if not path.endswith("/leagues"):
        return None
    leagues = {
        str(i): {"league": [{"league_key": f"461.l.{i}", "name": f"L{i}", "season": "2026"}]}
        for i in (1, 2)
    }
    body = {
        "fantasy_content": {
            "users": {"0": {"user": [{"games": {"0": {"game": [{}, {"leagues": leagues}]}}}]}}
        }
    }
    return httpx2.Response(200, json=body)


def no_adp(value: Any) -> None:
    assert "adp" not in json.dumps(value).lower()


@pytest.mark.anyio
async def test_my_league_without_key_lists_multiple(data: ServerData) -> None:
    client, _ = make_client(two_leagues)
    out = await my_league(data, client, None)
    assert [lg["league_key"] for lg in out["leagues"]] == ["461.l.2", "461.l.1"]
    assert out["next"] == "call my_league with a league_key"


@pytest.mark.anyio
async def test_my_league_auto_picks_single_league_and_matches_roster(data: ServerData) -> None:
    client, _ = make_client()
    out = await my_league(data, client, None)
    assert out["league_key"] == KEY
    assert out["scoring"] == {"ppr": "full", "reception_points": 1.0}
    assert "warning" not in out
    assert out["my_team"]["team_key"] == f"{KEY}.t.1"
    allen, nobody = out["roster"]
    allen_vor = float(data.projections.set_index("name")["vor"]["Josh Allen"])
    assert allen["projection"] == {"p10": 16.0, "p50": 20.0, "p90": 24.0}
    assert allen["vor"] == round(allen_vor, 1)
    assert nobody["projection"] is None and nobody["vor"] is None
    assert out["as_of"] == {"season": 2026, "week": 9, "data_date": "2026-11-01"}
    assert out["attribution"] == ATTRIBUTION
    assert out["standings"]
    assert out["current_week"] == 10
    assert out["ir_slots"] == {"total": 2, "used": 1}
    assert (allen["bye_week"], allen["ir_eligible"], allen["selected_position"]) == (7, False, "QB")
    assert (nobody["bye_week"], nobody["ir_eligible"], nobody["selected_position"]) == (
        None,
        True,
        "IR",
    )
    no_adp(out)


@pytest.mark.anyio
async def test_my_league_ir_slots_default_to_zero_without_ir(data: ServerData) -> None:
    def no_ir(path: str) -> httpx2.Response | None:
        if path.endswith("/settings"):
            text = (FIXTURES / "settings_full.json").read_text()
            text = text.replace(',{"roster_position":{"position":"IR","count":2}}', "")
            return httpx2.Response(200, text=text)
        return None

    client, _ = make_client(no_ir)
    out = await my_league(data, client, KEY)
    assert out["ir_slots"] == {"total": 0, "used": 1}


@pytest.mark.anyio
async def test_my_league_warns_when_not_full_ppr(data: ServerData) -> None:
    client, _ = make_client(settings="settings_half.json")
    out = await my_league(data, client, KEY)
    assert out["scoring"]["ppr"] == "half"
    assert out["warning"] == "projections are full-PPR points per game"


@pytest.mark.anyio
async def test_my_team_with_invalid_key_is_dropped(data: ServerData) -> None:
    def bad_team(path: str) -> httpx2.Response | None:
        if path.endswith("/teams"):
            text = (FIXTURES / "teams.json").read_text().replace(f"{KEY}.t.1", "evil/../x")
            return httpx2.Response(200, text=text)
        return None

    client, calls = make_client(bad_team)
    out = await my_league(data, client, KEY)
    assert out["my_team"] is None and out["roster"] == []
    assert not any("roster" in c for c in calls)


@pytest.mark.anyio
async def test_free_agents_rank_by_vor_and_count_unprojected(data: ServerData) -> None:
    rows = [
        ("Josh Jacobs", "RB", 55.0, None),
        ("Tyler Williams", "WR", 20.0, "Q"),
        ("Brock Bowers", "TE,WR", 99.0, None),
        ("Nobody Special", "WR", 1.0, None),
        ("Mike Williams", "WR,TE", 30.0, None),
    ]
    client, _ = make_client(lambda p: players_payload(rows) if "/players" in p else None)
    out = await free_agents(data, client, KEY, None, 3)
    vor = data.projections.set_index("name")["vor"]
    expected = sorted(
        ["Josh Jacobs", "Tyler Williams", "Brock Bowers", "Mike Williams"], key=lambda n: -vor[n]
    )
    assert [p["name"] for p in out["free_agents"]] == expected[:3]
    assert out["unprojected"] == 1
    top = out["free_agents"][0]
    assert set(top) == {
        "name",
        "position",
        "nfl_team",
        "status",
        "percent_owned",
        "bye_week",
        "ir_eligible",
        "projection",
        "vor",
    }
    assert out["attribution"] == ATTRIBUTION
    no_adp(out)
    full = await free_agents(data, client, KEY, None, 25)
    williams = next(p for p in full["free_agents"] if p["name"] == "Tyler Williams")
    assert williams["status"] == "Q" and williams["percent_owned"] == 20.0
    assert williams["projection"]["p50"] == 8.0
    assert out["current_week"] == 10


@pytest.mark.anyio
async def test_free_agents_carry_bye_week_and_ir_eligibility(data: ServerData) -> None:
    def payload(path: str) -> httpx2.Response | None:
        if "/players" not in path:
            return None
        extras: list[list[dict[str, Any]]] = [
            [{"bye_weeks": {"week": "6"}}, {"eligible_positions": [{"position": "RB"}]}],
            [{"eligible_positions": [{"position": "RB"}, {"position": "IR"}]}],
        ]
        players: dict[str, Any] = {
            str(i): {"player": [[{"name": {"full": name}}, {"display_position": "RB"}, *frags]]}
            for i, (name, frags) in enumerate(
                zip(("Josh Jacobs", "Brock Bowers"), extras, strict=True)
            )
        }
        players["count"] = 2
        body = {"fantasy_content": {"league": [{"league_key": KEY}, {"players": players}]}}
        return httpx2.Response(200, json=body)

    client, _ = make_client(payload)
    out = await free_agents(data, client, KEY, None, 25)
    by_name = {p["name"]: p for p in out["free_agents"]}
    assert (by_name["Josh Jacobs"]["bye_week"], by_name["Josh Jacobs"]["ir_eligible"]) == (6, False)
    assert out["current_week"] == 10
    assert by_name["Brock Bowers"]["bye_week"] is None
    assert by_name["Brock Bowers"]["ir_eligible"] is True


@pytest.mark.anyio
async def test_free_agent_position_disambiguates_shared_name(data: ServerData) -> None:
    extra = data.projections.iloc[[0]].assign(name="Mike Williams", position="RB", p50=5.0)
    shared = ServerData(
        pd.concat([data.projections, extra], ignore_index=True),
        data.league,
        2026,
        9,
        "2026-11-01",
    )
    client, _ = make_client(lambda p: players_payload([("Mike Williams", "WR", 30.0, None)]))
    out = await free_agents(shared, client, KEY, None, 5)
    assert out["free_agents"][0]["projection"]["p50"] == 9.0
    client, _ = make_client(lambda p: players_payload([("Mike Williams", "K", 30.0, None)]))
    assert (await free_agents(shared, client, KEY, None, 5))["unprojected"] == 1


@pytest.mark.anyio
async def test_free_agents_passes_position_filter(data: ServerData) -> None:
    client, calls = make_client()
    await free_agents(data, client, KEY, "WR", 5)
    assert any("position=WR" in c and "count=50" in c for c in calls)


@pytest.mark.anyio
@pytest.mark.parametrize("top_n", [0, 26, -1])
async def test_free_agents_top_n_bounds(data: ServerData, top_n: int) -> None:
    client, calls = make_client()
    out = await free_agents(data, client, KEY, None, top_n)
    assert out["error"] == "bad_request" and not calls


@pytest.mark.anyio
@pytest.mark.parametrize("bad", ["../x", "461.l.1\n", "461.l.", "abc"])
async def test_bad_league_key_makes_no_http_call(data: ServerData, bad: str) -> None:
    client, calls = make_client()
    assert (await my_league(data, client, bad))["error"] == "bad_request"
    assert (await free_agents(data, client, bad))["error"] == "bad_request"
    assert (await free_agents(data, client, KEY, "K"))["error"] == "bad_request"
    assert calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx2.Response(401), {"error": "yahoo_reauthorize"}),
        (httpx2.Response(999), {"error": "yahoo_throttled", "retry_after_s": 60}),
        (httpx2.Response(200, text="<html>maintenance</html>"), {"error": "yahoo_unavailable"}),
    ],
)
async def test_yahoo_errors_are_mapped(
    data: ServerData, response: httpx2.Response, expected: dict[str, Any]
) -> None:
    for call in (
        lambda c: my_league(data, c, KEY),
        lambda c: free_agents(data, c, KEY),
    ):
        client, _ = make_client(lambda path: response)
        out = await call(client)
        assert {k: out[k] for k in expected} == expected
        if expected["error"] == "yahoo_reauthorize":
            assert "Reconnect" in out["how"]
