"""Pure parsers over decoded Yahoo Fantasy API v2 JSON.

Yahoo nests resources under numerically keyed objects ("0", "1", ..., "count") and a
resource is often a list of fragment dicts, so parsers walk the payload for named
nodes instead of indexing a fixed shape.
"""

import re
from collections.abc import Iterator, Mapping
from typing import Any

# fullmatch, not match + "$": "$" accepts a trailing newline.
LEAGUE_KEY_RE = re.compile(r"\d+\.l\.\d+")
TEAM_KEY_RE = re.compile(r"\d+\.l\.\d+\.t\.\d+")

RECEPTION_STAT_ID = "11"


def _merge(value: Any) -> dict[str, Any]:
    """Collapse a dict, or a (nested) list of fragment dicts, into one dict."""
    merged: dict[str, Any] = {}
    if isinstance(value, Mapping):
        merged.update(value)
    elif isinstance(value, list):
        for part in value:
            merged.update(_merge(part))
    return merged


def iter_nodes(payload: Any, key: str) -> Iterator[dict]:
    """Yield every `key` node in document order, merging fragment lists into one dict."""
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if key in node and (merged := _merge(node[key])):
                yield merged
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))


def _first(payload: Any, key: str) -> dict:
    return next(iter_nodes(payload, key), {})


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_guid(payload: Any) -> str | None:
    guid = _first(payload, "user").get("guid")
    return str(guid) if guid else None


def parse_leagues(payload: Any) -> list[dict]:
    leagues = [
        {
            "league_key": str(lg["league_key"]),
            "name": lg.get("name"),
            "season": _int(lg.get("season")),
            "num_teams": _int(lg.get("num_teams")),
        }
        for lg in iter_nodes(payload, "league")
        if "league_key" in lg
    ]
    return sorted(leagues, key=lambda lg: (lg["season"] or 0, lg["league_key"]), reverse=True)


def parse_settings(payload: Any) -> dict:
    """Roster slots and a PPR label from the reception stat modifier (stat_id 11)."""
    slots: dict[str, int] = {}
    for rp in iter_nodes(payload, "roster_position"):
        if pos := rp.get("position"):
            slots[str(pos)] = slots.get(str(pos), 0) + (_int(rp.get("count")) or 1)
    reception = 0.0
    for stat in iter_nodes(payload, "stat"):
        if str(stat.get("stat_id")) == RECEPTION_STAT_ID:
            reception = _float(stat.get("value")) or 0.0
    ppr = {1.0: "full", 0.5: "half", 0.0: "none"}.get(reception, f"{reception:g}")
    return {"roster_slots": slots, "ppr": ppr, "reception_points": reception}


def parse_standings(payload: Any) -> list[dict]:
    teams = []
    for team in iter_nodes(payload, "team"):
        if "team_key" not in team:
            continue
        standing = _merge(team.get("team_standings"))
        totals = _merge(standing.get("outcome_totals"))
        teams.append(
            {
                "team_key": str(team["team_key"]),
                "name": team.get("name"),
                "rank": _int(standing.get("rank")),
                "wins": _int(totals.get("wins")),
                "losses": _int(totals.get("losses")),
                "ties": _int(totals.get("ties")),
                "points_for": _float(standing.get("points_for")),
            }
        )
    return sorted(teams, key=lambda t: (t["rank"] is None, t["rank"] or 0))


def parse_my_team(payload: Any) -> dict | None:
    """The logged-in manager's team, from league/{key}/teams."""
    for team in iter_nodes(payload, "team"):
        managers = iter_nodes(team.get("managers", []), "manager")
        if "team_key" in team and any(str(m.get("is_current_login")) == "1" for m in managers):
            return {"team_key": str(team["team_key"]), "name": team.get("name")}
    return None


def parse_players(payload: Any) -> list[dict]:
    """Players from team/{key}/roster or league/{key}/players."""
    players = []
    for p in iter_nodes(payload, "player"):
        name = p.get("name")
        full = name.get("full") if isinstance(name, Mapping) else name
        if not full:
            continue
        owned = p.get("percent_owned")
        owned_value = _merge(owned).get("value") if owned is not None else None
        players.append(
            {
                "name": str(full),
                "position": p.get("display_position") or p.get("primary_position"),
                "nfl_team": p.get("editorial_team_abbr"),
                "status": p.get("status") or None,
                "percent_owned": _float(owned_value),
            }
        )
    return players
