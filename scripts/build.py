"""Build live projections, or print the walk-forward backtest table.

uv run python scripts/build.py --season 2026 --as-of-week 9
uv run python scripts/build.py --season 2026 --as-of-week latest
uv run python scripts/build.py --season 2026 --print-week
uv run python scripts/build.py --check data/projections.parquet --expect-week 9
uv run python scripts/build.py --backtest
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from statline.data.nflverse import (
    completed_weeks,
    load_rosters,
    load_schedule,
    load_snaps,
    load_weekly,
)
from statline.features.build import attach_snap_share, build_features
from statline.model.evaluate import BASELINES, backtest, relevant, report
from statline.model.project import project
from statline.model.schema import projection_problems
from statline.model.train import quantile_gbm
from statline.server.data import ServerData

FIRST_SEASON = 2015  # earliest data; features start a season later (they need a prior year)
BACKTEST_SEASONS = range(2022, 2026)
DEFAULT_OUTPUT = Path("data/projections.parquet")
DEFAULT_LEAGUE = Path("config/league.yaml")  # the server's default, see scripts/serve.py


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


def run_projections(season: int, as_of_week: int, output: Path) -> None:
    table = feature_table(season, refresh_last=True)
    projections = project(table, season, as_of_week)
    output.parent.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(output, index=False)
    print(f"wrote {len(projections)} projections to {output}\n")
    print(projections.head(15).round(1).to_string(index=False))


def resolve_week(value: str, season: int) -> int:
    """`latest` means the weeks completed so far; anything else is an explicit week."""
    if value == "latest":
        return completed_weeks(load_schedule(season))
    return int(value)


def run_check(path: Path, expected_week: int, season: int | None) -> None:
    """Quality gate for a built file; exits non-zero listing every failure."""
    problems = []
    try:
        ServerData.load(path, DEFAULT_LEAGUE)  # the validation the server runs at startup
        projections = pd.read_parquet(path)
        problems = projection_problems(projections, expected_week, season)
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"server would not load {path}: {error}")
    if problems:
        sys.exit("projection check failed:\n" + "\n".join(f"- {p}" for p in problems))
    print(f"ok: {len(projections)} projections, as_of_week {expected_week}, {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build live projections, or print the backtest table."
    )
    parser.add_argument("--backtest", action="store_true", help="print the backtest table")
    parser.add_argument("--season", type=int, help="season to project")
    parser.add_argument(
        "--as-of-week", help="regular-season weeks completed: an integer or `latest`"
    )
    parser.add_argument(
        "--print-week", action="store_true", help="print the weeks completed in --season"
    )
    parser.add_argument("--check", type=Path, help="quality-check a projections parquet")
    parser.add_argument("--expect-week", type=int, help="as_of_week --check must find")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="projections parquet")
    args = parser.parse_args()
    if args.backtest:
        run_backtest()
    elif args.check is not None:
        if args.expect_week is None:
            parser.error("--check needs --expect-week")
        run_check(args.check, args.expect_week, args.season)
    elif args.print_week and args.season is not None:
        print(completed_weeks(load_schedule(args.season)))
    elif args.season is not None and args.as_of_week is not None:
        try:
            week = resolve_week(args.as_of_week, args.season)
        except ValueError:
            parser.error("--as-of-week must be an integer or `latest`")
        print(f"as of week {week}")
        run_projections(args.season, week, args.output)
    else:
        parser.error(
            "pass --backtest, --check, --season with --print-week, or --season with --as-of-week"
        )


if __name__ == "__main__":
    main()
