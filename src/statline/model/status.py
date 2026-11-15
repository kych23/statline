"""Player availability and injury text for display; never a model feature."""

import pandas as pd

# nflverse weekly-roster status code -> display text. A code missing here (new or rare)
# yields null rather than a guess.
AVAILABILITY = {
    "ACT": "active",
    "RES": "injured reserve",
    "INA": "inactive",
    "DEV": "practice squad",
    "CUT": "not on a roster",
}


def attach_status(
    projections: pd.DataFrame, roster_status: pd.DataFrame, injuries: pd.DataFrame
) -> pd.DataFrame:
    """Add availability, injury and status_as_of, joined on player_id.

    availability comes from each team's latest roster week (a team on bye has no rows in the
    newest week); players absent from it get null.
    injury comes only from the latest week present in `injuries`, so an earlier report is
    stale and ignored. status_as_of is the roster week (null when there is no roster data).
    Frames may be empty, so a missing nflverse feed degrades to nulls.
    """
    out = projections.copy()
    ids = out["player_id"]

    roster_week = int(roster_status["week"].max()) if len(roster_status) else None
    availability = pd.Series(None, index=out.index, dtype=object)
    if roster_week is not None:
        team_week = roster_status.groupby("team")["week"].transform("max")
        latest = roster_status[roster_status["week"] == team_week].sort_values("week")
        by_player = latest.drop_duplicates("player_id", keep="last").set_index("player_id")
        codes = ids.map(by_player["roster_status"])
        availability = codes.map(AVAILABILITY).astype(object).where(lambda s: s.notna(), None)
    out["availability"] = availability

    injury = pd.Series(None, index=out.index, dtype=object)
    if len(injuries):
        injury_week = int(injuries["week"].max())
        recent = injuries[(injuries["week"] == injury_week) & injuries["report_status"].notna()]
        texts = {
            row.player_id: _injury_text(str(row.report_status), row.injury, injury_week)
            for row in recent.itertuples()
        }
        injury = ids.map(texts).astype(object).where(lambda s: s.notna(), None)
    out["injury"] = injury

    out["status_as_of"] = pd.array([roster_week] * len(out), dtype="Int64")
    return out


def _injury_text(status: str, injury: object, week: int) -> str:
    detail = f" ({str(injury).lower()})" if isinstance(injury, str) and injury else ""
    return f"{status}{detail}, week {week}"
