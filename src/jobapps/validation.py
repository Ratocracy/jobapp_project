"""Small, reusable data-contract checks for Spark DataFrames."""

from __future__ import annotations

from collections.abc import Iterable

from pyspark.sql import DataFrame, functions as F


def require_columns(dataframe: DataFrame, required: Iterable[str]) -> None:
    """Raise when a DataFrame is missing contract columns."""

    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def count_duplicate_keys(dataframe: DataFrame, key_columns: Iterable[str]) -> int:
    """Count rows beyond the first row for each duplicated key."""

    keys = list(key_columns)
    require_columns(dataframe, keys)
    duplicate_groups = dataframe.groupBy(*keys).count().filter(F.col("count") > 1)
    row = duplicate_groups.select(F.sum(F.col("count") - 1).alias("duplicates")).first()
    return int(row["duplicates"] or 0)
