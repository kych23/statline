# Statline

MCP server giving Claude fantasy football projections, mock drafts and Yahoo league context.

## Layout and dependency rule

`src/statline/`: data -> features -> model -> draft -> server. A package imports only
packages to its left; `yahoo` is a low-level source that only `server` imports.
Enforced by `tests/test_dependencies.py`.

## Commands

    uv sync
    uv run pytest
    uv run ruff check && uv run ruff format --check
    uv run pyright

## Working agreement

- Kyle leads. Build only what was asked; propose anything else first.
- Work in small, reviewable pieces and explain each one.
- Never commit or push unless Kyle says to.
- Comments only for non-obvious why. No speculative abstractions.
- Point-in-time: every feature uses only data available before the prediction moment.
  Backtests walk forward.
