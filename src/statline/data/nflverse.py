"""Fetch nflverse tables, keep the regular season and fantasy positions, cache as parquet.

Each table has a pure `clean_*` function (tested on fixtures) and a `load_*` wrapper
that adds the network fetch and the per-season cache.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")
CACHE_DIR = Path("data/raw")

WEEKLY_COLUMNS = {
    "player_id": "player_id",
    "player_display_name": "name",
    "position": "position",
    "team": "team",
    "season": "season",
    "week": "week",
    "fantasy_points_ppr": "ppr_points",
    "attempts": "pass_attempts",
    "carries": "carries",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_air_yards": "air_yards",
    "target_share": "target_share",
    "air_yards_share": "air_yards_share",
}

SNAP_COLUMNS = {
    "pfr_player_id": "pfr_id",
    "season": "season",
    "week": "week",
    "team": "team",
    "offense_pct": "snap_share",
}

ROSTER_COLUMNS = {
    "gsis_id": "player_id",
    "pfr_id": "pfr_id",
    "season": "season",
    "birth_date": "birth_date",
    "rookie_year": "rookie_year",
}


def clean_weekly(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[(raw["season_type"] == "REG") & raw["position"].isin(POSITIONS)]
    return df[list(WEEKLY_COLUMNS)].rename(columns=WEEKLY_COLUMNS).reset_index(drop=True)


def clean_snaps(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[(raw["game_type"] == "REG") & raw["position"].isin(POSITIONS)]
    return df[list(SNAP_COLUMNS)].rename(columns=SNAP_COLUMNS).reset_index(drop=True)


def clean_rosters(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season); rows without an nflverse ID cannot be joined."""
    df = raw[raw["position"].isin(POSITIONS) & raw["gsis_id"].notna() & (raw["gsis_id"] != "")]
    df = df[list(ROSTER_COLUMNS)].rename(columns=ROSTER_COLUMNS)
    return df.drop_duplicates(["player_id", "season"]).reset_index(drop=True)


def load_weekly(seasons: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    fetch = lambda s: nfl.load_player_stats(seasons=[s], summary_level="week").to_pandas()  # noqa: E731
    return _load("weekly", seasons, fetch, clean_weekly, refresh)


def load_snaps(seasons: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    fetch = lambda s: nfl.load_snap_counts(seasons=[s]).to_pandas()  # noqa: E731
    return _load("snaps", seasons, fetch, clean_snaps, refresh)


def load_rosters(seasons: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    fetch = lambda s: nfl.load_rosters(seasons=[s]).to_pandas()  # noqa: E731
    return _load("rosters", seasons, fetch, clean_rosters, refresh)


def _load(
    name: str,
    seasons: Iterable[int],
    fetch: Callable[[int], pd.DataFrame],
    clean: Callable[[pd.DataFrame], pd.DataFrame],
    refresh: bool,
) -> pd.DataFrame:
    """Cache per season so a new week only re-downloads the current season (refresh=True)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = CACHE_DIR / f"{name}_{season}.parquet"
        if refresh or not path.exists():
            clean(fetch(season)).to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)
