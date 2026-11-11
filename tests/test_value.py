import pandas as pd

from statline.draft.league import League
from statline.draft.value import add_vor, replacement_levels

# 2 teams, 1 QB / 1 RB / 1 WR / 1 FLEX starters
LEAGUE = League(
    teams=2,
    rounds=5,
    slots={"QB": 1, "RB": 1, "WR": 1, "FLEX": 1, "BENCH": 1},
    flex_eligible=("RB", "WR", "TE"),
    max_per_position={},
)
PLAYERS = pd.DataFrame(
    {
        "position": ["QB", "QB", "QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR"],
        "p50": [22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 6.0, 15.0, 11.0, 9.0],
    }
)


def test_replacement_is_the_best_player_left_after_starters_and_flex() -> None:
    # QB: 22, 20 start -> 18. RB: 16, 14 start; WR: 15, 11 start;
    # FLEX takes the best two leftovers, RB 12 and WR 9 -> RB 6, WR has none left.
    levels = replacement_levels(PLAYERS, LEAGUE)
    assert levels["QB"] == 18.0
    assert levels["RB"] == 6.0
    assert levels["WR"] == 9.0  # no bench WR: falls back to the worst WR


def test_vor_can_rank_a_lower_scoring_rb_above_a_qb() -> None:
    out = add_vor(PLAYERS, LEAGUE)
    qb1 = out.loc[out["p50"] == 22.0, "vor"].item()  # 22 - 18 = 4
    rb1 = out.loc[out["p50"] == 16.0, "vor"].item()  # 16 - 6 = 10
    assert rb1 > qb1
