"""Shared fixtures."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from statline.data.adp import name_key

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
def synthetic() -> Callable[..., tuple[pd.DataFrame, pd.DataFrame]]:
    """Factory: synthetic(seed=0) -> (projections, adp) for ~250 players."""
    return _synthetic
