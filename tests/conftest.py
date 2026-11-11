"""Shared fixtures."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statline.data.adp import name_key
from statline.draft.league import load_league
from statline.draft.value import add_vor
from statline.server.data import ServerData

COUNTS = {"QB": 30, "RB": 70, "WR": 80, "TE": 30}
EXTRA = {"K": 20, "DST": 20}


def _synthetic(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """~250 players: 210 projected skill players, all on the ADP board, plus K and DST."""
    rng = np.random.default_rng(seed)
    rows = [(f"{pos}{i} Player", pos) for pos, n in COUNTS.items() for i in range(n)]
    p50 = rng.uniform(2, 22, len(rows))
    proj = pd.DataFrame(
        {
            "player_id": [f"id{i}" for i in range(len(rows))],
            "name": [r[0] for r in rows],
            "position": [r[1] for r in rows],
            "team": "AAA",
            "as_of_week": 0,
            "p10": p50 - 3,
            "p50": p50,
            "p90": p50 + 3,
        }
    )
    skill_adp = pd.Series(p50).rank(ascending=False).to_numpy()
    adp_rows = [(n, p, a) for (n, p), a in zip(rows, skill_adp, strict=True)]
    for pos, n in EXTRA.items():
        adp_rows += [(f"{pos}{i} Unit", pos, 150.0 + i * 2 + (pos == "DST")) for i in range(n)]
    adp = pd.DataFrame(adp_rows, columns=["name", "position", "adp"])
    adp["name_key"] = adp["name"].map(name_key)
    adp["team"] = "AAA"
    return proj, adp


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def synthetic() -> Callable[..., tuple[pd.DataFrame, pd.DataFrame]]:
    """Factory: synthetic(seed=0) -> (projections, adp) for ~250 players."""
    return _synthetic


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


def _projections() -> pd.DataFrame:
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
            "built_on": "2026-11-01",
        }
    )


@pytest.fixture
def league_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "league.yaml"


@pytest.fixture
def make_projections() -> Callable[[], pd.DataFrame]:
    """Factory: a fresh 12-player week-9 projections frame in OUTPUT_COLUMNS shape."""
    return _projections


@pytest.fixture
def data(league_path: Path) -> ServerData:
    league = load_league(league_path)
    return ServerData(add_vor(_projections(), league), league, 2026, 9, "2026-11-01")
