"""Training/prediction table: one row per (player, season, as-of week t).

t is the number of regular-season weeks completed. Features use only prior seasons
and this season's weeks <= t; the target uses weeks > t. t = 0 is preseason.
"""

from collections.abc import Iterable

import pandas as pd

L3_GAMES = 3
SEASON_START = "09-01"  # month-day used to compute age

STAT_RATES = {  # weekly column -> per-game feature suffix
    "ppr_points": "ppg",
    "targets": "targets_pg",
    "carries": "carries_pg",
    "pass_attempts": "pass_att_pg",
    "target_share": "target_share",
    "snap_share": "snap_share",
}
L3_STATS = ["ppr_points", "targets", "carries", "snap_share"]
PREV_STATS = ["ppr_points", "targets", "carries", "target_share"]

TARGET_COLUMNS = ["ros_ppg", "ros_games"]


def attach_snap_share(
    weekly: pd.DataFrame, snaps: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Join snap share onto weekly stats. Snaps are keyed by PFR ID; rosters from any
    season bridge it to the nflverse player_id (same-season-only leaves ~4% more unmapped)."""
    bridge = rosters.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    mapped = snaps.assign(player_id=snaps["pfr_id"].map(bridge["player_id"]))
    mapped = mapped.dropna(subset=["player_id"])[["player_id", "season", "week", "snap_share"]]
    return weekly.merge(mapped, on=["player_id", "season", "week"], how="left")


def build_features(
    games: pd.DataFrame,
    rosters: pd.DataFrame,
    seasons: Iterable[int],
    max_week: int = 14,
) -> pd.DataFrame:
    """`games` is weekly stats with snap_share attached (one row per player-game)."""
    tables = [
        _season_rows(games, rosters, season, t) for season in seasons for t in range(max_week + 1)
    ]
    return pd.concat(tables, ignore_index=True)


def _season_rows(games: pd.DataFrame, rosters: pd.DataFrame, season: int, t: int) -> pd.DataFrame:
    this = games[games["season"] == season]
    past = this[this["week"] <= t].sort_values("week")
    future = this[this["week"] > t]
    prev = games[games["season"] == season - 1]

    players = pd.Index(past["player_id"].unique()).union(prev["player_id"].unique())
    if t > 0:  # mid-season, a player still has to be on the field this season to project
        players = players.intersection(this["player_id"].unique())
    out = pd.DataFrame(index=players.rename("player_id"))

    latest = pd.concat([prev, past]).sort_values(["season", "week"]).groupby("player_id").last()
    out = out.join(latest[["name", "position", "team"]])

    by_player = past.groupby("player_id")
    out["games_so_far"] = by_player.size()
    out = out.join(
        by_player[list(STAT_RATES)].mean().rename(columns=lambda c: f"std_{STAT_RATES[c]}")
    )
    l3 = by_player.tail(L3_GAMES).groupby("player_id")[L3_STATS].mean()
    out = out.join(l3.rename(columns=lambda c: f"l3_{STAT_RATES[c]}"))

    prev_by = prev.groupby("player_id")
    out["prev_games"] = prev_by.size()
    out = out.join(prev_by[PREV_STATS].mean().rename(columns=lambda c: f"prev_{STAT_RATES[c]}"))

    out = out.join(_bio(rosters, season))

    fut = future.groupby("player_id")["ppr_points"]
    out["ros_ppg"] = fut.mean()
    out["ros_games"] = fut.size()

    out[["games_so_far", "prev_games", "ros_games"]] = (
        out[["games_so_far", "prev_games", "ros_games"]].fillna(0).astype(int)
    )
    out = out[(out["games_so_far"] > 0) | (out["prev_games"] > 0)]
    return out.reset_index().assign(season=season, as_of_week=t)


def _bio(rosters: pd.DataFrame, season: int) -> pd.DataFrame:
    """Age at season start and years of experience; any season's roster row will do."""
    r = rosters.sort_values("season").groupby("player_id").last()
    start = pd.Timestamp(f"{season}-{SEASON_START}")
    return pd.DataFrame(
        {
            "age": (start - r["birth_date"]).dt.days / 365.25,
            "years_exp": season - r["rookie_year"],
        }
    )
