"""Projection file schema, kept import-light so the server can use it."""

import pandas as pd

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

MIN_PLAYERS = 300  # a real build has ~600; fewer means a broken data pull


def projection_problems(
    df: pd.DataFrame, expected_week: int, season: int | None = None
) -> list[str]:
    """Everything wrong with a projections frame, empty when it is fit to serve."""
    problems = []
    if len(df) < MIN_PLAYERS:
        problems.append(f"only {len(df)} rows, expected at least {MIN_PLAYERS}")
    quantiles = ["p10", "p50", "p90"]
    for column in quantiles:
        nulls = int(df[column].isna().sum())
        if nulls:
            problems.append(f"{nulls} null values in {column}")
    # Comparisons with NaN are False, so null rows are reported above, not here.
    crossed = int(((df["p10"] > df["p50"]) | (df["p50"] > df["p90"])).sum())
    if crossed:
        problems.append(f"{crossed} rows violate p10 <= p50 <= p90")
    weeks = sorted(df["as_of_week"].unique().tolist())
    if weeks != [expected_week]:
        problems.append(f"as_of_week is {weeks}, expected {expected_week}")
    if season is not None:
        seasons = sorted(df["season"].unique().tolist())
        if seasons != [season]:
            problems.append(f"season is {seasons}, expected {season}")
    return problems
