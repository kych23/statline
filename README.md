# Statline

[![CI](https://github.com/kych23/statline/actions/workflows/ci.yml/badge.svg)](https://github.com/kych23/statline/actions/workflows/ci.yml)

Fantasy football projections and league context for Claude. Statline is an
[MCP](https://modelcontextprotocol.io) server: add it to Claude as a connector and ask
about players, mock drafts, or your Yahoo league. Projections are quantile forecasts
(10th/50th/90th percentile points per game) from a gradient-boosted model trained on
nflverse data.

**Status:** milestone 1 of 5 (project skeleton).

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync              # create .venv and install dependencies
uv run pytest        # tests
uv run ruff check    # lint
uv run pyright       # type check
```
