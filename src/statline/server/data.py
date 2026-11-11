"""Everything the tools read, loaded once at startup."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from statline.draft.board import build_board
from statline.draft.league import League, load_league
from statline.draft.value import add_vor
from statline.model.schema import OUTPUT_COLUMNS


@dataclass(frozen=True)
class ServerData:
    projections: pd.DataFrame  # OUTPUT_COLUMNS plus vor
    league: League
    season: int
    as_of_week: int
    data_date: str  # ISO date the projections were built
    board: pd.DataFrame | None = None  # build_board output (carries attrs["replacement"])

    @classmethod
    def load(
        cls,
        projections_path: Path,
        league_path: Path,
        draft_projections_path: Path | None = None,
        adp_path: Path | None = None,
    ) -> "ServerData":
        projections = pd.read_parquet(projections_path)
        missing = [c for c in OUTPUT_COLUMNS if c not in projections.columns]
        if missing:
            raise ValueError(
                f"{projections_path} is missing columns {missing}; rebuild with scripts/build.py"
            )
        pairs = projections[["season", "as_of_week"]].drop_duplicates()
        if len(pairs) != 1:
            raise ValueError(
                f"{projections_path} must hold exactly one (season, as_of_week), found {len(pairs)}"
            )
        built = projections["built_on"].unique()
        if len(built) != 1:
            raise ValueError(
                f"{projections_path} must hold exactly one built_on, found {len(built)}"
            )
        league = load_league(league_path)
        board = None
        if draft_projections_path is not None and adp_path is not None:
            draft = pd.read_parquet(draft_projections_path)
            if (draft["as_of_week"] != 0).any():
                raise ValueError(f"{draft_projections_path} must be preseason (as_of_week 0)")
            # Read the cached file directly: the server must never hit the network.
            board = build_board(draft, pd.read_parquet(adp_path), league)
        return cls(
            projections=add_vor(projections.reset_index(drop=True), league),
            league=league,
            season=int(pairs["season"].iloc[0]),
            as_of_week=int(pairs["as_of_week"].iloc[0]),
            data_date=str(built[0]),
            board=board,
        )
