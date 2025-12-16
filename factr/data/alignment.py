"""Point-in-time correctness helpers - pure functions."""

from typing import Literal

import polars as pl


def asof_join(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    on: str = "date",
    by: str = "asset",
    strategy: Literal["backward", "forward", "nearest"] = "backward",
) -> pl.LazyFrame:
    """As-of join for point-in-time correctness.

    Args:
        left: Primary dataset (e.g., daily prices)
        right: Dataset to join (e.g., quarterly fundamentals)
        on: Date column name
        by: Grouping column (typically 'asset')
        strategy: 'backward' = use last available, 'forward' = use next available

    Example:
        # Join daily prices with quarterly fundamentals
        # Each price date gets the most recent fundamental data
        result = asof_join(prices, fundamentals, on='date', by='asset')
    """
    return left.join_asof(right, on=on, by=by, strategy=strategy)


def apply_offset(
    lf: pl.LazyFrame,
    offset_days: int,
    date_col: str = "date",
    output_col: str = "available_date",
) -> pl.LazyFrame:
    """Apply reporting delay to dates.

    Example:
        # Fundamentals reported with 45-day lag
        # Q1 2024 (ends 2024-03-31) available 2024-05-15
        funds = apply_offset(funds, offset_days=45)
    """
    if offset_days == 0:
        return lf

    return lf.with_columns(
        [(pl.col(date_col).cast(pl.Date) + pl.duration(days=offset_days)).alias(output_col)]
    )


def forward_fill(
    lf: pl.LazyFrame, columns: list[str], by: str = "asset", order_by: str = "date"
) -> pl.LazyFrame:
    """Forward-fill values within groups.

    Example:
        # Fill quarterly fundamentals to daily frequency
        daily_funds = forward_fill(funds, ['market_cap', 'pe_ratio'])
    """
    return lf.with_columns(
        [pl.col(col).forward_fill().over(by, order_by=order_by) for col in columns]
    )
