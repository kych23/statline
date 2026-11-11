# Statline

[![CI](https://github.com/kych23/statline/actions/workflows/ci.yml/badge.svg)](https://github.com/kych23/statline/actions/workflows/ci.yml)

Fantasy football projections and league context for Claude. Statline is an
[MCP](https://modelcontextprotocol.io) server: add it to Claude as a connector and ask
about players, mock drafts, or your Yahoo league. Projections are quantile forecasts
(10th/50th/90th percentile points per game) from a gradient-boosted model trained on
nflverse data.

**Status:** milestone 3 of 5 (draft logic). Next: MCP server with a public demo endpoint.

## Projections

Rest-of-season PPR points per game for every QB, RB, WR and TE, at any point in the season:
a 10th percentile (floor), median and 90th percentile (ceiling). One LightGBM model per
quantile, trained on nflverse weekly data from 2016 on, with features computed strictly
from games played before the prediction (season-to-date and last-3-game usage, prior
season, age). Intervals are calibrated per position on held-out seasons (split conformal).

```sh
uv run python scripts/build.py --season 2026 --as-of-week 9   # -> data/projections.parquet
uv run python scripts/build.py --backtest                     # the table below
```

### Backtest

Walk-forward: each test season (2022-2025) is predicted by a model trained only on
earlier seasons, at every week t = 0 (preseason) to 14. Error is mean absolute error in
points per game; lower is better. The blend weights last season against this season by
games played, with its weight fitted on training seasons.

| Method | MAE t 0-4 | MAE t 5-9 | MAE t 10-14 | MAE all | p10-p90 coverage |
|:--|--:|--:|--:|--:|--:|
| Season to date | 3.28 | 2.84 | 2.88 | 3.01 | |
| Prior season | 3.00 | 3.13 | 3.40 | 3.17 | |
| Blend | 2.78 | 2.66 | 2.81 | 2.75 | |
| **Statline** | **2.51** | **2.50** | **2.70** | **2.56** | **0.80** |

On players already fantasy-relevant before the prediction (prior-season or season-to-date
PPG of 10 or more), Statline's MAE is 3.34 vs 3.60 for the blend. Coverage is the share of
outcomes that land between the floor and the ceiling; 0.80 is the target.

**Limitations**

- Backtest rows require 2+ remaining games, so misses below the floor will be more
  common live, where some players are about to get hurt or benched.
- No preseason projection for rookies (no prior-season data yet).
- Coverage is lower for QBs (0.76) and varies by season (0.74 in 2024).

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync              # create .venv and install dependencies
uv run pytest        # tests
uv run ruff check    # lint
uv run pyright       # type check
```


## Mock draft

Simulates a 12-team snake draft (settings in `config/league.yaml`) from your seat.

```sh
uv run python scripts/build.py --season 2026 --as-of-week 0 --output data/projections_2026_w0.parquet
uv run python scripts/mock.py --seat 4 --seed 7 --projections data/projections_2026_w0.parquet
```

```
 round  pick_no                name position  p50  vor
     1        4  Jaxon Smith-Njigba       WR 16.9  7.1
     2       21        Trey McBride       TE 15.3  5.9
     3       28       Davante Adams       WR 13.9  4.1
     4       45    Javonte Williams       RB 12.5  3.1
   ...
```

- **Your picks** maximize the gain to your starting lineup, measured as value over
  replacement (VOR): projected points minus the best player left unstarted league-wide at
  that slot. FLEX gets its own replacement level, since a TE there competes with RBs and
  WRs. A small credit for bench depth breaks late-round ties.
- **Opponents** pick from ADP (average draft position, Fantasy Football Calculator) with
  noise: P(player) ∝ exp(−ADP / τ), where τ = 1 + 0.07 × pick number. The randomness grows
  through the draft, as it does in real drafts. τ was fitted so simulated pick spreads
  match real ones from thousands of drafts, 2016-2026 (e.g. 1.6 vs 1.6 picks for ADP
  1-12, 13.8 vs 14.2 for ADP 121+).
- Every team fills a legal starting lineup (position caps, plus a must-fill rule near the
  end), and the same seed replays the same draft.
- ADP is used only to drive opponents and is never displayed. The cached board is local
  and not committed.
