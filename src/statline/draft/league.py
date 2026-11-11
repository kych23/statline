"""League settings, read from config/league.yaml and validated once on load."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path("config/league.yaml")
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
SPECIAL_SLOTS = ("FLEX", "BENCH")


@dataclass(frozen=True)
class League:
    teams: int
    rounds: int
    slots: dict[str, int]
    flex_eligible: tuple[str, ...]
    max_per_position: dict[str, int]

    def __post_init__(self) -> None:
        unknown = set(self.slots) - set(POSITIONS) - set(SPECIAL_SLOTS)
        if unknown:
            raise ValueError(f"unknown roster slots: {sorted(unknown)}")
        if self.teams < 2:
            raise ValueError("a league needs at least 2 teams")
        if sum(self.slots.values()) != self.rounds:
            raise ValueError(
                f"roster slots sum to {sum(self.slots.values())} but rounds is {self.rounds}"
            )
        if not set(self.flex_eligible) <= set(POSITIONS):
            raise ValueError(f"flex_eligible must be positions: {self.flex_eligible}")
        for pos, cap in self.max_per_position.items():
            if cap < self.slots.get(pos, 0):
                raise ValueError(f"max_per_position[{pos}]={cap} is below its starting slots")

    @property
    def starters(self) -> dict[str, int]:
        return {pos: n for pos, n in self.slots.items() if pos != "BENCH"}


def load_league(path: Path = DEFAULT_PATH) -> League:
    raw = yaml.safe_load(path.read_text())
    roster = raw["roster"]
    return League(
        teams=int(raw["teams"]),
        rounds=int(raw["rounds"]),
        slots={str(k): int(v) for k, v in roster["slots"].items()},
        flex_eligible=tuple(roster["flex_eligible"]),
        max_per_position={str(k): int(v) for k, v in roster.get("max_per_position", {}).items()},
    )
