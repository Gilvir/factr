"""Tests for SQLSource with SQLAlchemy support."""

import polars as pl
import pytest

from factr.data import SQLSource


def test_sql_source_with_string_query():
    """Test SQLSource with a string query."""
    pytest.importorskip("sqlalchemy")

    from sqlalchemy import create_engine

    # Create in-memory SQLite database with test data
    test_df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset": ["AAPL", "AAPL", "GOOGL"],
            "close": [150.0, 152.0, 2800.0],
            "volume": [1000000, 1100000, 2000000],
        }
    )

    # Write to SQLite for testing
    engine = create_engine("sqlite:///:memory:")
    test_df.write_database("prices", connection=engine, if_table_exists="replace")

    # Create SQLSource with string query
    source = SQLSource(connection=engine, query="SELECT * FROM prices")

    # Read data
    lf = source.read()
    df = lf.collect()

    assert len(df) == 3
    assert "close" in df.columns
    assert "volume" in df.columns
    assert df["close"][0] == 150.0

    engine.dispose()


def test_sql_source_with_table():
    """Test SQLSource with table name."""
    pytest.importorskip("sqlalchemy")

    from sqlalchemy import create_engine

    # Create in-memory SQLite database with test data
    test_df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "asset": ["AAPL", "AAPL"],
            "close": [150.0, 152.0],
        }
    )

    engine = create_engine("sqlite:///:memory:")
    test_df.write_database("prices", connection=engine, if_table_exists="replace")

    # Create SQLSource with table name
    source = SQLSource(connection=engine, table="prices")

    # Read data
    lf = source.read()
    df = lf.collect()

    assert len(df) == 2
    assert "close" in df.columns

    engine.dispose()


def test_sql_source_with_sqlalchemy_select():
    """Test SQLSource with SQLAlchemy Select object."""
    pytest.importorskip("sqlalchemy")

    from sqlalchemy import (
        Column,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        create_engine,
        select,
    )

    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    # Define table schema
    prices_table = Table(
        "prices",
        metadata,
        Column("date", String),
        Column("asset", String),
        Column("close", Float),
        Column("volume", Integer),
    )

    # Create table
    metadata.create_all(engine)

    # Insert test data using Polars
    test_df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset": ["AAPL", "AAPL", "GOOGL"],
            "close": [150.0, 152.0, 2800.0],
            "volume": [1000000, 1100000, 2000000],
        }
    )
    test_df.write_database("prices", connection=engine, if_table_exists="replace")

    # Create SQLAlchemy Select query
    query = select(prices_table).where(prices_table.c.asset == "AAPL")

    # Create SQLSource with SQLAlchemy query
    source = SQLSource(connection=engine, query=query)

    # Read data
    lf = source.read()
    df = lf.collect()

    # Should only have AAPL data (2 rows)
    assert len(df) == 2
    assert all(df["asset"] == "AAPL")
    assert df["close"][0] == 150.0
    assert df["close"][1] == 152.0

    engine.dispose()


def test_sql_source_with_sqlalchemy_select_and_connection_string():
    """Test SQLSource with SQLAlchemy Select object and connection string."""
    pytest.importorskip("sqlalchemy")

    from sqlalchemy import (
        Column,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        create_engine,
        select,
    )

    # Create in-memory SQLite database
    connection_string = "sqlite:///:memory:"
    engine = create_engine(connection_string)
    metadata = MetaData()

    # Define table schema
    prices_table = Table(
        "prices",
        metadata,
        Column("date", String),
        Column("asset", String),
        Column("close", Float),
        Column("volume", Integer),
    )

    # Create table and insert data
    metadata.create_all(engine)

    test_df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset": ["AAPL", "GOOGL", "MSFT"],
            "close": [150.0, 2800.0, 380.0],
            "volume": [1000000, 2000000, 1500000],
        }
    )
    test_df.write_database("prices", connection=engine, if_table_exists="replace")

    # Create SQLAlchemy Select query with filter
    query = select(prices_table).where(prices_table.c.close > 200.0)

    # Create SQLSource with connection string (not engine)
    # Note: For in-memory SQLite, we need to use the engine, not the string
    # because in-memory databases don't persist across connections
    source = SQLSource(connection=engine, query=query)

    # Read data
    lf = source.read()
    df = lf.collect()

    # Should only have GOOGL and MSFT (close > 200)
    assert len(df) == 2
    assert set(df["asset"]) == {"GOOGL", "MSFT"}

    engine.dispose()


def test_sql_source_with_sqlalchemy_complex_query():
    """Test SQLSource with complex SQLAlchemy query (joins, ordering, etc.)."""
    pytest.importorskip("sqlalchemy")

    from sqlalchemy import (
        Column,
        Float,
        MetaData,
        String,
        Table,
        create_engine,
        select,
    )

    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    # Define tables
    prices_table = Table(
        "prices",
        metadata,
        Column("date", String),
        Column("asset", String),
        Column("close", Float),
    )

    sectors_table = Table(
        "sectors",
        metadata,
        Column("asset", String),
        Column("sector", String),
    )

    # Create tables
    metadata.create_all(engine)

    # Insert test data
    prices_df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "asset": ["AAPL", "GOOGL", "XOM"],
            "close": [150.0, 2800.0, 100.0],
        }
    )
    sectors_df = pl.DataFrame(
        {
            "asset": ["AAPL", "GOOGL", "XOM"],
            "sector": ["Technology", "Technology", "Energy"],
        }
    )

    prices_df.write_database("prices", connection=engine, if_table_exists="replace")
    sectors_df.write_database("sectors", connection=engine, if_table_exists="replace")

    # Create complex query with join
    query = (
        select(
            prices_table.c.date,
            prices_table.c.asset,
            prices_table.c.close,
            sectors_table.c.sector,
        )
        .select_from(
            prices_table.join(sectors_table, prices_table.c.asset == sectors_table.c.asset)
        )
        .where(sectors_table.c.sector == "Technology")
        .order_by(prices_table.c.close)
    )

    # Create SQLSource
    source = SQLSource(connection=engine, query=query)

    # Read data
    lf = source.read()
    df = lf.collect()

    # Should only have Technology sector stocks, ordered by close
    assert len(df) == 2
    assert all(df["sector"] == "Technology")
    assert df["asset"][0] == "AAPL"  # Lower close price comes first
    assert df["asset"][1] == "GOOGL"

    engine.dispose()


def test_sql_source_no_query_or_table_raises():
    """Test that SQLSource raises error when neither query nor table is provided."""
    pytest.importorskip("sqlalchemy")

    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")

    with pytest.raises(ValueError, match="Must provide either 'query' or 'table'"):
        SQLSource(connection=engine)

    engine.dispose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
