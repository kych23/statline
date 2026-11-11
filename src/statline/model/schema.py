"""Projection file schema, kept import-light so the server can use it."""

OUTPUT_COLUMNS = [
    "player_id",
    "season",
    "name",
    "position",
    "team",
    "as_of_week",
    "p10",
    "p50",
    "p90",
    "built_on",  # ISO date (UTC) the projections were built
]
