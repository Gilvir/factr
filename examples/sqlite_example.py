"""SQLite integration example.

This example demonstrates:
1. Creating an SQLite database with sample data
2. Loading with SQLSource (table-based and custom queries)
3. Column mapping for schema translation
4. Factor pipelines with SQL data

Run with: python examples/sqlite_example.py
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from factr.data import DataContext, SQLSource
from factr.datasets import Column, DataSet, EquityPricing
from factr.pipeline import Pipeline

# ==============================================================================
# Setup: Create sample SQLite database
# ==============================================================================

DB_PATH = "example_data.db"


def create_sample_database():
    """Create sample SQLite database with pricing data."""
    Path(DB_PATH).unlink(missing_ok=True)

    start = date(2020, 1, 1)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    dates = [start + timedelta(days=i) for i in range(60)]

    pricing_data = []
    for ticker_idx, ticker in enumerate(tickers):
        base_price = 100 + ticker_idx * 50
        for day_idx, d in enumerate(dates):
            price = base_price + day_idx * 0.1 + (day_idx % 10) * 0.5
            pricing_data.append(
                {
                    "trade_date": d.isoformat(),
                    "symbol": ticker,
                    "close_price": price + 0.5,
                    "trade_volume": 1_000_000 + day_idx * 10_000,
                }
            )

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE equity_prices (
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            close_price REAL,
            trade_volume INTEGER,
            PRIMARY KEY (trade_date, symbol)
        )
    """)

    for row in pricing_data:
        conn.execute(
            "INSERT INTO equity_prices VALUES (:trade_date, :symbol, :close_price, :trade_volume)",
            row,
        )

    conn.commit()
    conn.close()
    print(f"Created {DB_PATH} with {len(pricing_data)} rows")


# ==============================================================================
# Dataset Definition (no Config.source - bind at runtime)
# ==============================================================================


class SQLPricing(DataSet):
    """Pricing from SQLite with column mapping."""

    close = Column(pl.Float64)
    volume = Column(pl.Int64)


# Column mapping used across examples
COLUMN_MAPPING = {
    "trade_date": "date",
    "symbol": "asset",
    "close_price": "close",
    "trade_volume": "volume",
}


# ==============================================================================
# Examples
# ==============================================================================


def example_1_table_loading():
    """Load from SQLite table using DataContext."""
    print("\n" + "=" * 60)
    print("Example 1: Load from SQLite Table")
    print("=" * 60)

    source = SQLSource(
        connection=sqlite3.connect(DB_PATH),
        table="equity_prices",
        column_mapping=COLUMN_MAPPING,
    )

    ctx = DataContext()
    prices = ctx.load(SQLPricing, source=source, start_date="2020-01-01", end_date="2020-01-31")
    print(f"Loaded {prices.collect().shape[0]:,} rows")
    print(prices.collect().head())


def example_2_custom_query():
    """Load with custom SQL query."""
    print("\n" + "=" * 60)
    print("Example 2: Custom SQL Query")
    print("=" * 60)

    source = SQLSource(
        connection=sqlite3.connect(DB_PATH),
        query="SELECT * FROM equity_prices WHERE symbol = 'AAPL'",
        column_mapping=COLUMN_MAPPING,
    )
    data = source.read(date_col="date", start_date="2020-01-01", end_date="2020-01-31")
    print(f"Loaded {data.collect().shape[0]:,} AAPL rows")
    print(data.collect().head())


def example_3_factor_pipeline():
    """Factor pipeline with SQLite data."""
    print("\n" + "=" * 60)
    print("Example 3: Factor Pipeline with SQLite Data")
    print("=" * 60)

    # Load data using DataContext
    ctx = DataContext()
    ctx.bind(
        SQLPricing,
        SQLSource(
            connection=sqlite3.connect(DB_PATH, check_same_thread=False),
            table="equity_prices",
            column_mapping=COLUMN_MAPPING,
        ),
    )
    prices = ctx.load(SQLPricing, start_date="2020-01-01")

    # Define factors
    close = EquityPricing.close
    volume = EquityPricing.volume

    returns = close.pct_change(1)
    momentum = returns.rolling_sum(20)
    momentum_rank = momentum.rank(pct=True)
    volume_rank = volume.rank(pct=True)

    pipeline = (
        Pipeline(prices)
        .add_factors(
            {
                "returns": returns,
                "momentum": momentum,
                "momentum_rank": momentum_rank,
                "volume_rank": volume_rank,
            }
        )
        .screen(close > 0)
    )

    print("\nExecution Plan:")
    print(pipeline.explain())

    result = pipeline.run(start_date="2020-02-01", collect=True)
    print(f"\nResult: {result.shape[0]:,} rows")
    print(result.select(["date", "asset", "close", "momentum", "momentum_rank"]).head(8))


def main():
    print("\n" + "#" * 60)
    print("# SQLite Integration Examples")
    print("#" * 60)

    create_sample_database()
    example_1_table_loading()
    example_2_custom_query()
    example_3_factor_pipeline()

    print("\n" + "#" * 60)
    print("# Examples Complete!")
    print("#" * 60)

    print("\nKey points:")
    print("  - Use sqlite3.connect() (no extra dependencies)")
    print("  - column_mapping translates schema names")
    print("  - Use check_same_thread=False for multi-threaded loading")
    print("  - SQLSource supports both table= and query= parameters")
    print(f"\nThe {DB_PATH} file is in the current directory.")


if __name__ == "__main__":
    main()
