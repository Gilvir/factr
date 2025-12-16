"""Data loader helpers for combining sources with point-in-time correctness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .binding import DataSource


def combine_sources(
    primary: DataSource,
    *secondary: tuple[DataSource, dict],
    date_col: str = "date",
    asset_col: str = "asset",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.LazyFrame:
    """Combine multiple sources with point-in-time correctness.

    Handles different frequencies and reporting lags automatically.

    Args:
        primary: Primary data source (e.g., daily prices)
        *secondary: Tuples of (source, config_dict) for additional data
        date_col: Date column name
        asset_col: Asset column name
        start_date: Optional start date
        end_date: Optional end date

    Config dict keys:
        - offset: Reporting lag in days (default: 0)
        - forward_fill: List of columns to forward-fill (default: [])

    Example:
        >>> from factr.data import ParquetSource, combine_sources
        >>> prices = ParquetSource('prices.parquet')
        >>> funds = ParquetSource('fundamentals.parquet')
        >>> sentiment = ParquetSource('sentiment.parquet')
        >>>
        >>> lf = combine_sources(
        ...     prices,
        ...     (funds, {'offset': 45, 'forward_fill': ['market_cap', 'pe_ratio']}),
        ...     (sentiment, {'offset': 1}),
        ...     start_date='2020-01-01'
        ... )
    """
    from .alignment import asof_join, apply_offset, forward_fill

    # Load primary data
    result = primary.read(
        date_col=date_col,
        asset_col=asset_col,
        start_date=start_date,
        end_date=end_date,
    )

    # Join each secondary source
    for source, config in secondary:
        # Load secondary data (no date filtering - we need all for asof join)
        source_lf = source.read(date_col=date_col, asset_col=asset_col)

        # Apply reporting offset if specified
        offset = config.get("offset", 0)
        if offset > 0:
            source_lf = apply_offset(source_lf, offset, date_col=date_col)
            source_lf = source_lf.drop(date_col).rename({"available_date": date_col})

        # Point-in-time join
        result = asof_join(result, source_lf, on=date_col, by=asset_col)

        # Forward-fill if specified
        ff_cols = config.get("forward_fill", [])
        if ff_cols:
            result = forward_fill(result, ff_cols, by=asset_col, order_by=date_col)

    return result
