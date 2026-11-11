"""Preseason ADP board from Fantasy Football Calculator (12-team PPR), cached permanently.

A preseason board never changes once the season starts, so a cached season is never
re-fetched. In-season FFC only aggregates the last week of mock drafts (a handful of
players), so a thin board is rejected rather than drafted against.

ADP is third-party market data: it drives mock-draft opponents and is never displayed.
"""

import re
from pathlib import Path

import pandas as pd
import requests

URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
CACHE_DIR = Path("data/raw")
MIN_PLAYERS = 150
POSITION_MAP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "PK": "K", "DEF": "DST"}
SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def name_key(name: str) -> str:
    """'Chris Godwin Jr.' and 'chris godwin' both become 'chris godwin'."""
    key = re.sub(r"[.'’-]", "", name.lower())
    return " ".join(SUFFIXES.sub("", key).split())


def clean_adp(payload: dict, season: int) -> pd.DataFrame:
    players = payload.get("players", [])
    if len(players) < MIN_PLAYERS:
        raise ValueError(
            f"no usable ADP board for {season}: FFC returned {len(players)} players "
            f"(need {MIN_PLAYERS}); in-season it only has recent mock drafts"
        )
    df = pd.DataFrame(players)
    df = df[df["position"].isin(POSITION_MAP)]
    return pd.DataFrame(
        {
            "name": df["name"],
            "name_key": df["name"].map(name_key),
            "position": df["position"].map(POSITION_MAP),
            "team": df["team"],
            "adp": df["adp"].astype(float),
        }
    ).reset_index(drop=True)


def load_adp(season: int, teams: int = 12) -> pd.DataFrame:
    path = CACHE_DIR / f"adp_{season}.parquet"
    if not path.exists():
        resp = requests.get(URL, params={"teams": teams, "year": season}, timeout=15)
        resp.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        clean_adp(resp.json(), season).to_parquet(path, index=False)
    return pd.read_parquet(path)
