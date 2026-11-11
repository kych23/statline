"""Snake mock draft: opponents follow ADP with noise, the user's seat follows value."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statline.draft.league import League

# Opponent pick noise: P ∝ exp(-adp / tau), tau = TAU0 + TAU_SLOPE * pick_no.
# Calibrated against FFC per-player pick stdev, 2016-2024 + 2026 boards.
TAU0 = 1.0
TAU_SLOPE = 0.07
NO_STARTER_POSITIONS = ("K", "DST")
# Bench players cover injuries and byes, so they are worth a fraction of their value.
DEPTH_WEIGHT = 0.25


@dataclass(frozen=True)
class MockResult:
    picks: pd.DataFrame
    rosters: dict[int, list[int]]  # team (1-based) -> board row indices


def _snake_order(league: League) -> list[int]:
    forward = list(range(1, league.teams + 1))
    return [t for r in range(league.rounds) for t in (forward if r % 2 == 0 else forward[::-1])]


def _must_fill_positions(league: League, counts: dict[str, int], picks_left: int) -> set[str]:
    """Positions the team must draft from now on, or empty if it still has slack."""
    slots = {pos: n for pos, n in league.starters.items() if pos != "FLEX"}
    need = {pos: max(0, n - counts.get(pos, 0)) for pos, n in slots.items()}
    flex_filled = sum(
        max(0, counts.get(pos, 0) - slots.get(pos, 0)) for pos in league.flex_eligible
    )
    flex_need = max(0, league.starters.get("FLEX", 0) - flex_filled)
    if picks_left > sum(need.values()) + flex_need:
        return set()
    required = {pos for pos, n in need.items() if n > 0}
    if flex_need:
        required |= set(league.flex_eligible)
    return required


def _candidates(
    available: np.ndarray,
    positions: np.ndarray,
    league: League,
    counts: dict[str, int],
    required: set[str],
) -> np.ndarray:
    """Available rows the team may take: under the position caps, inside the must-fill set.
    If nothing qualifies the caps are dropped; an invalid row is never returned silently."""
    under_cap = np.array(
        [counts.get(p, 0) < league.max_per_position.get(p, league.rounds) for p in positions]
    )
    wanted = np.isin(positions, list(required)) if required else np.ones(len(positions), bool)
    for mask in (available & under_cap & wanted, available & wanted, available):
        if mask.any():
            return np.flatnonzero(mask)
    raise ValueError("no players left to draft")


def lineup_value(
    positions: list[str], p50s: list[float], league: League, levels: dict[str, float]
) -> float:
    """Value of the best starting lineup from these players: each starter's p50 over the
    replacement level of the slot he fills (his position's, or FLEX's). Dedicated slots take
    the top p50 at the position; FLEX takes the best remaining by p50 over the FLEX level.
    Empty slots, NaN p50 (K/DST, unprojected) and positions without a level count as 0."""
    by_pos: dict[str, list[float]] = {}
    for pos, p50 in zip(positions, p50s, strict=True):
        if pos in levels and not np.isnan(p50):
            by_pos.setdefault(pos, []).append(p50)
    total = 0.0
    flex_pool: list[float] = []
    for pos, values in by_pos.items():
        values.sort(reverse=True)
        n = league.starters.get(pos, 0)
        total += sum(max(0.0, v - levels[pos]) for v in values[:n])
        if pos in league.flex_eligible:
            flex_pool.extend(values[n:])
    flex_level = levels.get("FLEX")
    if flex_level is None:
        return total
    flex_pool.sort(reverse=True)
    n_flex = league.starters.get("FLEX", 0)
    return total + sum(max(0.0, v - flex_level) for v in flex_pool[:n_flex])


def _my_pick(
    rows: np.ndarray,
    board: pd.DataFrame,
    restricted: bool,
    roster: list[int],
    league: League,
    levels: dict[str, float],
) -> int:
    """Highest marginal lineup value (plus a little depth credit) among the candidate rows."""
    all_positions = board["position"].to_numpy()
    if not restricted:
        # K/DST carry vor 0 by construction; they are not taken early unless forced
        skill = rows[~np.isin(all_positions[rows], NO_STARTER_POSITIONS)]
        rows = skill if len(skill) else rows
    all_vor = board["vor"].to_numpy(dtype=float)
    if np.isnan(all_vor[rows]).all():
        return int(rows[np.argmin(board["adp"].to_numpy()[rows])])
    all_p50 = board["p50"].to_numpy(dtype=float)
    have_pos = [str(all_positions[r]) for r in roster]
    have_p50 = [float(all_p50[r]) for r in roster]
    base = lineup_value(have_pos, have_p50, league, levels)
    scores = []
    for r in rows:
        gain = (
            lineup_value(
                [*have_pos, str(all_positions[r])], [*have_p50, all_p50[r]], league, levels
            )
            - base
        )
        vor = all_vor[r]
        scores.append(gain + DEPTH_WEIGHT * (0.0 if np.isnan(vor) else max(vor, 0.0)))
    return int(rows[int(np.argmax(scores))])


def mock_draft(board: pd.DataFrame, league: League, seat: int, seed: int) -> MockResult:
    """`board` is build_board output; returned row indices refer to its index."""
    if not 1 <= seat <= league.teams:
        raise ValueError(f"seat must be in 1..{league.teams}, got {seat}")
    levels = board.attrs.get("replacement")
    if levels is None:
        raise ValueError("board has no replacement levels in attrs; build it with build_board")
    rng = np.random.default_rng(seed)
    positions = board["position"].to_numpy()
    adp = board["adp"].to_numpy(dtype=float)
    available = np.ones(len(board), dtype=bool)
    rosters: dict[int, list[int]] = {t: [] for t in range(1, league.teams + 1)}
    counts: dict[int, dict[str, int]] = {t: {} for t in rosters}
    records = []

    for pick_no, team in enumerate(_snake_order(league), start=1):
        team_counts = counts[team]
        picks_left = league.rounds - len(rosters[team])
        required = _must_fill_positions(league, team_counts, picks_left)
        rows = _candidates(available, positions, league, team_counts, required)
        if team == seat:
            chosen = _my_pick(rows, board, bool(required), rosters[team], league, levels)
        else:
            tau = TAU0 + TAU_SLOPE * pick_no
            # Gumbel-max is a numerically stable draw from softmax(-adp / tau)
            scores = -adp[rows] / tau + rng.gumbel(size=len(rows))
            chosen = int(rows[np.argmax(scores)])
        available[chosen] = False
        rosters[team].append(chosen)
        team_counts[positions[chosen]] = team_counts.get(positions[chosen], 0) + 1
        records.append((pick_no, (pick_no - 1) // league.teams + 1, team, chosen))

    picks = pd.DataFrame(records, columns=["pick_no", "round", "team", "player"])
    picked = board.iloc[picks["player"]].reset_index(drop=True)
    picks = pd.concat([picks, picked[["name", "position", "adp", "vor"]]], axis=1)
    return MockResult(picks=picks, rosters=rosters)
