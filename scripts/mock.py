"""Run one mock draft and print the seat's picks.

uv run python scripts/mock.py --seat 4 --seed 7
"""

import argparse
import secrets
import sys
from pathlib import Path

import pandas as pd

from statline.data.adp import load_adp
from statline.draft.board import build_board
from statline.draft.league import DEFAULT_PATH, load_league
from statline.draft.mock import mock_draft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock draft and print one seat's picks.")
    parser.add_argument("--seat", type=int, required=True, help="1-based draft seat")
    parser.add_argument("--seed", type=int, help="RNG seed (random if omitted)")
    parser.add_argument("--projections", type=Path, default=Path("data/projections.parquet"))
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()

    league = load_league(DEFAULT_PATH)
    projections = pd.read_parquet(args.projections)
    if (projections["as_of_week"] != 0).any():
        print(
            "warning: mock drafts should use preseason (week 0) projections: "
            "run build.py --as-of-week 0",
            file=sys.stderr,
        )
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    board = build_board(projections, load_adp(args.season), league)
    result = mock_draft(board, league, args.seat, seed)

    mine = result.picks[result.picks["team"] == args.seat].copy()
    mine["p50"] = board["p50"].to_numpy()[mine["player"]].round(1)
    mine["vor"] = mine["vor"].round(1)
    print(f"seat {args.seat}, seed {seed}")
    print(mine[["round", "pick_no", "name", "position", "p50", "vor"]].to_string(index=False))


if __name__ == "__main__":
    main()
