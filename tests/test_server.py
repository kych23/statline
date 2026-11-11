import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client

from statline.draft.board import build_board
from statline.server.app import build_server
from statline.server.data import ServerData
from statline.server.limits import HeavyToolGate
from statline.server.resolve import resolve
from statline.server.tools import get_player, mock_draft, rankings


def name_of(data: ServerData, index: int | None) -> str:
    assert index is not None
    return str(data.projections.loc[index, "name"])


def test_resolve_exact_and_suffix_variant(data: ServerData) -> None:
    assert name_of(data, resolve("Jahmyr Gibbs", data.projections).player_index) == "Jahmyr Gibbs"
    res = resolve("Chris Godwin", data.projections)
    assert res.status == "resolved"
    assert name_of(data, res.player_index) == "Chris Godwin Jr."


def test_resolve_unique_last_name(data: ServerData) -> None:
    res = resolve("Gibbs", data.projections)
    assert res.status == "resolved"
    assert name_of(data, res.player_index) == "Jahmyr Gibbs"


def test_resolve_shared_last_name_is_ambiguous(data: ServerData) -> None:
    res = resolve("Williams", data.projections)
    assert res.status == "ambiguous"
    assert [name_of(data, i) for i in res.candidates] == ["Mike Williams", "Tyler Williams"]


def test_resolve_typo_and_garbage(data: ServerData) -> None:
    res = resolve("Jahmyr Gibs", data.projections)
    assert res.status == "resolved"
    assert name_of(data, res.player_index) == "Jahmyr Gibbs"
    assert resolve("zzzz qqqq", data.projections).status == "unresolved"


def test_get_player_rejects_bad_name_counts(data: ServerData) -> None:
    assert get_player(data, [])["error"] == "bad_request"
    assert get_player(data, ["Josh Allen"] * 9)["error"] == "bad_request"


def test_get_player_compares_and_ranks(data: ServerData) -> None:
    out = get_player(data, ["Bijan Robinson", "Jahmyr Gibbs"])
    assert [p["name"] for p in out["players"]] == ["Bijan Robinson", "Jahmyr Gibbs"]
    assert out["players"][0]["position_rank"] == 2
    assert out["players"][1]["position_rank"] == 1
    assert out["players"][1]["projection"] == {"p10": 14.0, "p50": 18.0, "p90": 22.0}
    assert out["comparison"] == [
        {"name": "Jahmyr Gibbs", "p50": 18.0, "delta_p50": 0.0},
        {"name": "Bijan Robinson", "p50": 17.5, "delta_p50": -0.5},
    ]
    assert out["as_of"] == {"season": 2026, "week": 9, "data_date": "2026-11-01"}


def test_get_player_reports_ambiguous_and_unresolved_without_comparison(
    data: ServerData,
) -> None:
    out = get_player(data, ["Williams", "zzzz qqqq", "Josh Allen"])
    assert out["ambiguous"][0]["query"] == "Williams"
    assert len(out["ambiguous"][0]["candidates"]) == 2
    assert out["unresolved"] == ["zzzz qqqq"]
    assert "comparison" not in out


def keys_of(obj: object) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in keys_of(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in keys_of(v)}
    return set()


def test_outputs_never_expose_adp(data: ServerData) -> None:
    outputs = [get_player(data, ["Josh Allen", "Gibbs"]), rankings(data)]
    assert all("adp" not in {k.lower() for k in keys_of(o)} for o in outputs)


def test_rankings_position_filter_and_sort_orders(data: ServerData) -> None:
    te = rankings(data, position="TE", top_n=5)
    assert [r["name"] for r in te["rankings"]] == ["Brock Bowers", "Trey McBride"]
    assert [r["rank"] for r in te["rankings"]] == [1, 2]
    by_vor = [r["vor"] for r in rankings(data)["rankings"]]
    assert by_vor == sorted(by_vor, reverse=True)
    by_p50 = rankings(data, sort_by="p50")
    assert by_p50["rankings"][0]["name"] == "Josh Allen"
    assert by_p50["sort_by"] == "p50"


def test_rankings_top_n_bounds(data: ServerData) -> None:
    assert rankings(data, top_n=0)["error"] == "bad_request"
    assert rankings(data, top_n=51)["error"] == "bad_request"
    assert len(rankings(data, top_n=3)["rankings"]) == 3


def test_load_adds_vor_and_rejects_mixed_weeks(
    tmp_path: Path, league_path: Path, make_projections: Callable[[], pd.DataFrame]
) -> None:
    path = tmp_path / "projections.parquet"
    make_projections().to_parquet(path, index=False)
    loaded = ServerData.load(path, league_path)
    assert (loaded.season, loaded.as_of_week) == (2026, 9)
    assert "vor" in loaded.projections.columns
    assert loaded.data_date == "2026-11-01"

    mixed = make_projections()
    mixed.loc[0, "as_of_week"] = 8
    mixed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exactly one"):
        ServerData.load(path, league_path)

    two_dates = make_projections().assign(built_on="2026-11-01")
    two_dates.loc[0, "built_on"] = "2026-11-02"
    two_dates.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exactly one built_on"):
        ServerData.load(path, league_path)


@pytest.mark.anyio
async def test_mcp_end_to_end(data: ServerData) -> None:
    async with Client(build_server(data)) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {"get_player", "rankings", "mock_draft"}
        assert all(t.annotations and t.annotations.read_only_hint for t in tools)
        result = await client.call_tool("get_player", {"names": ["Jahmyr Gibs"]})
    assert result.data["players"][0]["name"] == "Jahmyr Gibbs"
    json.dumps(result.data)


@pytest.fixture
def draft_data(
    data: ServerData, synthetic: Callable[..., tuple[pd.DataFrame, pd.DataFrame]]
) -> ServerData:
    proj, adp = synthetic()
    board = build_board(proj, adp, data.league)
    return ServerData(data.projections, data.league, 2026, 9, "2026-11-01", board=board)


def new_gate(limit: int = 10) -> HeavyToolGate:
    return HeavyToolGate(per_caller_per_hour=limit)


@pytest.mark.anyio
async def test_mock_draft_unavailable_without_board(data: ServerData) -> None:
    out = await mock_draft(data, new_gate(), "a", 1, 1)
    assert out["error"] == "unavailable"


@pytest.mark.anyio
@pytest.mark.parametrize(("seat", "seed"), [(0, 1), (13, 1), (1, -1), (1, 2**31)])
async def test_mock_draft_rejects_bad_seat_or_seed(
    draft_data: ServerData, seat: int, seed: int
) -> None:
    assert (await mock_draft(draft_data, new_gate(), "a", seat, seed))["error"] == "bad_request"


@pytest.mark.anyio
async def test_mock_draft_is_replayable_and_adp_free(draft_data: ServerData) -> None:
    first = await mock_draft(draft_data, new_gate(), "a", 4, 7)
    again = await mock_draft(draft_data, new_gate(), "a", 4, 7)
    assert first["roster"] == again["roster"]
    assert len(first["roster"]) == first["league"]["rounds"] == draft_data.league.rounds
    assert first["seat"] == 4
    assert "adp" not in {k.lower() for k in keys_of(first)}
    json.dumps(first)


@pytest.mark.anyio
async def test_mock_draft_generates_seed_when_omitted(draft_data: ServerData) -> None:
    out = await mock_draft(draft_data, new_gate(), "a", 1, None)
    assert 0 <= out["seed"] < 2**31


@pytest.mark.anyio
async def test_mock_draft_rate_limited_after_quota(draft_data: ServerData) -> None:
    gate = new_gate(1)
    await mock_draft(draft_data, gate, "a", 1, 1)
    out = await mock_draft(draft_data, gate, "a", 1, 1)
    assert out["status"] == "rate_limited"
    assert out["resets_in_s"] > 0


@pytest.mark.anyio
async def test_mock_draft_busy_while_another_runs(draft_data: ServerData) -> None:
    gate = new_gate()
    async with gate.slot("other"):
        out = await mock_draft(draft_data, gate, "a", 1, 1)
    assert out == {"status": "busy", "retry_after_s": 15}


@pytest.mark.anyio
async def test_mock_draft_over_mcp(draft_data: ServerData) -> None:
    async with Client(build_server(draft_data)) as client:
        result = await client.call_tool("mock_draft", {"seat": 3, "seed": 5})
    assert len(result.data["roster"]) == draft_data.league.rounds


def test_load_builds_board_from_preseason_files(
    tmp_path: Path,
    league_path: Path,
    make_projections: Callable[[], pd.DataFrame],
    synthetic: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    proj, adp = synthetic()
    proj = proj.assign(season=2026)
    proj_path, adp_path = tmp_path / "w0.parquet", tmp_path / "adp.parquet"
    proj.to_parquet(proj_path, index=False)
    adp.to_parquet(adp_path, index=False)
    current = tmp_path / "projections.parquet"
    make_projections().to_parquet(current, index=False)

    loaded = ServerData.load(current, league_path, proj_path, adp_path)
    assert loaded.board is not None
    assert "replacement" in loaded.board.attrs
    assert ServerData.load(current, league_path).board is None

    proj.assign(as_of_week=3).to_parquet(proj_path, index=False)
    with pytest.raises(ValueError, match="as_of_week 0"):
        ServerData.load(current, league_path, proj_path, adp_path)
