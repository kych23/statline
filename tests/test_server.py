import json
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client

from statline.draft.league import load_league
from statline.draft.value import add_vor
from statline.server.app import build_server
from statline.server.data import ServerData
from statline.server.resolve import resolve
from statline.server.tools import get_player, rankings

ROWS = [
    ("Josh Allen", "QB", 20.0),
    ("Lamar Jackson", "QB", 19.0),
    ("Jahmyr Gibbs", "RB", 18.0),
    ("Bijan Robinson", "RB", 17.5),
    ("Josh Jacobs", "RB", 12.0),
    ("Chris Godwin Jr.", "WR", 14.0),
    ("Justin Jefferson", "WR", 16.0),
    ("Amon-Ra St. Brown", "WR", 15.5),
    ("Mike Williams", "WR", 9.0),
    ("Tyler Williams", "WR", 8.0),
    ("Brock Bowers", "TE", 13.0),
    ("Trey McBride", "TE", 12.5),
]
LEAGUE_PATH = Path(__file__).resolve().parents[1] / "config" / "league.yaml"


def make_projections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [f"id{i}" for i in range(len(ROWS))],
            "season": 2026,
            "name": [r[0] for r in ROWS],
            "position": [r[1] for r in ROWS],
            "team": "AAA",
            "as_of_week": 9,
            "p10": [r[2] - 4 for r in ROWS],
            "p50": [r[2] for r in ROWS],
            "p90": [r[2] + 4 for r in ROWS],
        }
    )


@pytest.fixture
def data() -> ServerData:
    league = load_league(LEAGUE_PATH)
    return ServerData(add_vor(make_projections(), league), league, 2026, 9, "2026-11-01")


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


def test_load_adds_vor_and_rejects_mixed_weeks(tmp_path: Path) -> None:
    path = tmp_path / "projections.parquet"
    make_projections().to_parquet(path, index=False)
    loaded = ServerData.load(path, LEAGUE_PATH)
    assert (loaded.season, loaded.as_of_week) == (2026, 9)
    assert "vor" in loaded.projections.columns

    mixed = make_projections()
    mixed.loc[0, "as_of_week"] = 8
    mixed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="exactly one"):
        ServerData.load(path, LEAGUE_PATH)


@pytest.mark.anyio
async def test_mcp_end_to_end(data: ServerData) -> None:
    async with Client(build_server(data)) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {"get_player", "rankings"}
        assert all(t.annotations and t.annotations.read_only_hint for t in tools)
        result = await client.call_tool("get_player", {"names": ["Jahmyr Gibs"]})
    assert result.data["players"][0]["name"] == "Jahmyr Gibbs"
    json.dumps(result.data)
