"""Build live projections, or print the walk-forward backtest table.

uv run python scripts/build.py --season 2026 --as-of-week 9
uv run python scripts/build.py --backtest
"""

import argparse
from pathlib import Path

import pandas as pd

from statline.data.nflverse import load_rosters, load_snaps, load_weekly
from statline.features.build import attach_snap_share, build_features
from statline.model.evaluate import BASELINES, backtest, relevant, report
from statline.model.project import project
from statline.model.train import quantile_gbm

FIRST_SEASON = 2015  # earliest data; features start a season later (they need a prior year)
BACKTEST_SEASONS = range(2022, 2026)
OUTPUT = Path("data/projections.parquet")


def feature_table(last_season: int, refresh_last: bool) -> pd.DataFrame:
    """Past seasons come from cache; the last one is re-downloaded when it is in progress."""
    past = range(FIRST_SEASON, last_season)
    frames = []
    for load in (load_weekly, load_snaps, load_rosters):
        frames.append(pd.concat([load(past), load([last_season], refresh=refresh_last)]))
    weekly, snaps, rosters = frames
    games = attach_snap_share(weekly, snaps, rosters)
    return build_features(games, rosters, seasons=range(FIRST_SEASON + 1, last_season + 1))


def run_backtest() -> None:
    table = feature_table(max(BACKTEST_SEASONS), refresh_last=False)
    preds = backtest(table, {**BASELINES, "quantile gbm": quantile_gbm}, BACKTEST_SEASONS)
    print("All players\n")
    print(report(preds).to_markdown())
    print(f"\nAlready-relevant players ({len(relevant(preds)) // (len(BASELINES) + 1)} rows)\n")
    print(report(relevant(preds)).to_markdown())


def run_projections(season: int, as_of_week: int) -> None:
    table = feature_table(season, refresh_last=True)
    projections = project(table, season, as_of_week)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(OUTPUT, index=False)
    print(f"wrote {len(projections)} projections to {OUTPUT}\n")
    print(projections.head(15).round(1).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build live projections, or print the backtest table."
    )
    parser.add_argument("--backtest", action="store_true", help="print the backtest table")
    parser.add_argument("--season", type=int, help="season to project")
    parser.add_argument("--as-of-week", type=int, help="regular-season weeks completed")
    args = parser.parse_args()
    if args.backtest:
        run_backtest()
    elif args.season is not None and args.as_of_week is not None:
        run_projections(args.season, args.as_of_week)
    else:
        parser.error("pass --backtest, or both --season and --as-of-week")


if __name__ == "__main__":
    main()
