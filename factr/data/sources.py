"""Data source implementations - thin wrappers over Polars IO.

Each source is ~20 lines. Leverage Polars, don't rebuild it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from sqlalchemy.sql.selectable import Select


class DataFrameSource:
    """Wrap existing DataFrame/LazyFrame.

    The simplest source - you already have the data.

    Example:
        >>> df = pl.DataFrame({'date': [...], 'asset': [...], 'close': [...]})
        >>> source = DataFrameSource(df)
        >>> lf = source.read()
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column_mapping: dict[str, str] | None = None,
    ):
        """Initialize with data.

        Args:
            data: DataFrame or LazyFrame
            column_mapping: Optional {source_col: target_col} mapping
        """
        self.data = data.lazy() if isinstance(data, pl.DataFrame) else data
        self.column_mapping = column_mapping or {}

    def read(
        self,
        date_col: str = "date",
        asset_col: str = "asset",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.LazyFrame:
        """Read data with optional filtering."""
        lf = self.data

        if self.column_mapping:
            lf = lf.rename(self.column_mapping)

        if date_col in lf.collect_schema().names():
            lf = lf.with_columns(pl.col(date_col).cast(pl.Date))

        if start_date:
            lf = lf.filter(pl.col(date_col) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            lf = lf.filter(pl.col(date_col) <= pl.lit(end_date).cast(pl.Date))

        return lf


class ParquetSource:
    """Read from Parquet file(s).

    Delegates to pl.scan_parquet - gets predicate pushdown for free.

    Example:
        >>> source = ParquetSource('prices.parquet')
        >>> source = ParquetSource('data/*.parquet')  # glob pattern
        >>> lf = source.read(start_date='2020-01-01')  # pushed down to scan
    """

    def __init__(
        self,
        path: str | Path,
        column_mapping: dict[str, str] | None = None,
        **scan_kwargs: Any,
    ):
        """Initialize with path.

        Args:
            path: Path to parquet file(s), supports globs
            column_mapping: Optional {source_col: target_col} mapping
            **scan_kwargs: Additional args passed to pl.scan_parquet
        """
        self.path = path
        self.column_mapping = column_mapping or {}
        self.scan_kwargs = scan_kwargs

    def read(
        self,
        date_col: str = "date",
        asset_col: str = "asset",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.LazyFrame:
        """Scan parquet with optional filtering (predicate pushdown!)."""
        lf = pl.scan_parquet(self.path, **self.scan_kwargs)

        if self.column_mapping:
            lf = lf.rename(self.column_mapping)

        if start_date:
            lf = lf.filter(pl.col(date_col).cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            lf = lf.filter(pl.col(date_col).cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))

        return lf


class CSVSource:
    """Read from CSV file(s).

    Thin wrapper over pl.scan_csv.

    Example:
        >>> source = CSVSource('prices.csv', parse_dates=['date'])
        >>> lf = source.read()
    """

    def __init__(
        self,
        path: str | Path,
        column_mapping: dict[str, str] | None = None,
        **scan_kwargs: Any,
    ):
        """Initialize with path.

        Args:
            path: Path to CSV file(s)
            column_mapping: Optional {source_col: target_col} mapping
            **scan_kwargs: Additional args passed to pl.scan_csv
        """
        self.path = path
        self.column_mapping = column_mapping or {}
        self.scan_kwargs = scan_kwargs

    def read(
        self,
        date_col: str = "date",
        asset_col: str = "asset",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.LazyFrame:
        """Scan CSV with optional filtering."""
        lf = pl.scan_csv(self.path, **self.scan_kwargs)

        if self.column_mapping:
            lf = lf.rename(self.column_mapping)

        if start_date:
            lf = lf.filter(pl.col(date_col).cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            lf = lf.filter(pl.col(date_col).cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))

        return lf


def _compile_sqlalchemy_query(query: Select, connection: Any) -> str:
    """Compile a SQLAlchemy Select to a SQL string.

    Args:
        query: SQLAlchemy Select object
        connection: Connection string or SQLAlchemy engine/connection

    Returns:
        Compiled SQL string
    """
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine

    if isinstance(connection, str):
        engine = create_engine(connection)
        dialect = engine.dialect
    elif isinstance(connection, Engine):
        dialect = connection.dialect
    elif hasattr(connection, "engine"):
        dialect = connection.engine.dialect
    elif hasattr(connection, "dialect"):
        dialect = connection.dialect
    else:
        dialect = None

    compiled = query.compile(
        dialect=dialect,
        compile_kwargs={"literal_binds": True},
    )
    return str(compiled)


class SQLSource:
    """Read from SQL database.

    Uses pl.read_database_uri for string URIs or pl.read_database for connection objects.
    Delegates connection handling to Polars.

    Example:
        >>> # String URI (uses read_database_uri)
        >>> source = SQLSource(
        ...     connection='sqlite:///data.db',
        ...     table='prices'
        ... )
        >>> # Or with custom query string
        >>> source = SQLSource(
        ...     connection='postgresql://user:pass@host/db',
        ...     query='SELECT * FROM prices WHERE asset = ?'
        ... )
        >>> # Or with SQLAlchemy query
        >>> from sqlalchemy import select
        >>> from mymodels import prices_table
        >>> source = SQLSource(
        ...     connection='postgresql://user:pass@host/db',
        ...     query=select(prices_table).where(prices_table.c.asset == 'AAPL')
        ... )
        >>> lf = source.read(start_date='2020-01-01')
    """

    def __init__(
        self,
        connection: str | Any,
        query: str | Select | None = None,
        table: str | None = None,
        column_mapping: dict[str, str] | None = None,
        **read_kwargs: Any,
    ):
        """Initialize with connection.

        Args:
            connection: Connection string or connection object (e.g., SQLAlchemy engine)
            query: SQL query string or SQLAlchemy Select object (if None, reads entire table)
            table: Table name (alternative to query)
            column_mapping: Optional {source_col: target_col} mapping
            **read_kwargs: Additional args passed to pl.read_database
        """
        if query is None and table is None:
            raise ValueError("Must provide either 'query' or 'table'")

        self.connection = connection
        self._sqlalchemy_query: Select | None = None

        if query is not None and not isinstance(query, str):
            self._sqlalchemy_query = query
            self.query = _compile_sqlalchemy_query(query, connection)
        else:
            self.query = query or f"SELECT * FROM {table}"

        self.column_mapping = column_mapping or {}
        self.read_kwargs = read_kwargs

    def read(
        self,
        date_col: str = "date",
        asset_col: str = "asset",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.LazyFrame:
        """Read from database and filter.

        Note: Filtering happens in-memory after read. For large datasets,
        include WHERE clause in your query for database-side filtering.
        """
        query = self.query

        reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        source_date_col = reverse_mapping.get(date_col, date_col)

        if start_date or end_date:
            if "WHERE" not in query.upper():
                conditions = []
                if start_date:
                    conditions.append(f"{source_date_col} >= '{start_date}'")
                if end_date:
                    conditions.append(f"{source_date_col} <= '{end_date}'")
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

        if isinstance(self.connection, str):
            df = pl.read_database_uri(query, uri=self.connection, **self.read_kwargs)
        else:
            df = pl.read_database(query, connection=self.connection, **self.read_kwargs)
        lf = df.lazy()

        if self.column_mapping:
            lf = lf.rename(self.column_mapping)

        if date_col in lf.collect_schema().names():
            lf = lf.with_columns(pl.col(date_col).cast(pl.Date))

        if start_date and "WHERE" in self.query.upper():
            lf = lf.filter(pl.col(date_col) >= pl.lit(start_date).cast(pl.Date))
        if end_date and "WHERE" in self.query.upper():
            lf = lf.filter(pl.col(date_col) <= pl.lit(end_date).cast(pl.Date))

        return lf
