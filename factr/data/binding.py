"""Protocol definitions for data sources - structural typing only."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources.

    Any object with a read() method returning LazyFrame is valid.
    No inheritance required - duck typing FTW.
    """

    def read(
        self,
        date_col: str = "date",
        asset_col: str = "asset",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.LazyFrame:
        """Read data as LazyFrame."""
        ...


@runtime_checkable
class ColumnMapper(Protocol):
    """Protocol for column mapping."""

    def map(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply column mapping to LazyFrame."""
        ...
