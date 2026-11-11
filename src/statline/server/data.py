"""Everything the tools read, loaded once at startup."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from statline.draft.league import League, load_league
from statline.draft.value import add_vor
from statline.model.project import OUTPUT_COLUMNS


@dataclass(frozen=True)
class ServerData:
    projections: pd.DataFrame  # OUTPUT_COLUMNS plus vor
    league: League
    season: int
    as_of_week: int
    data_date: str  # ISO date the projections file was written

    @classmethod
    def load(cls, projections_path: Path, league_path: Path) -> "ServerData":
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
        league = load_league(league_path)
        mtime = datetime.fromtimestamp(projections_path.stat().st_mtime, tz=UTC)
        return cls(
            projections=add_vor(projections.reset_index(drop=True), league),
            league=league,
            season=int(pairs["season"].iloc[0]),
            as_of_week=int(pairs["as_of_week"].iloc[0]),
            data_date=mtime.date().isoformat(),
        )
