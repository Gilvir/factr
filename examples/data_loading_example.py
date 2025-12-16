"""Data loading examples: sources, binding, and point-in-time correctness.

This example covers:
1. DataFrameSource - load from in-memory DataFrames
2. ParquetSource / CSVSource - load from files
3. SQLSource - load from databases (see sqlite_example.py for more)
4. Column mapping - translate schema names
5. DataContext - manage multiple datasets
6. Custom sources - implement your own (protocol-based)
7. Point-in-time correctness - handle reporting delays

Run with: python examples/data_loading_example.py
"""

import polars as pl

from factr import Pipeline
from factr.data import (
    CSVSource,
    DataContext,
    DataFrameSource,
    ParquetSource,
    combine_sources,
)
from factr.datasets import Column, DataSet, EquityPricing

# ==============================================================================
# Dataset Definitions with Config
# ==============================================================================


class PricingData(DataSet):
    """Dataset with direct source in Config."""

    close = Column(pl.Float64)
    volume = Column(pl.Int64)

    class Config:
        source = ParquetSource("data/prices.parquet")
        date_column = "date"
        entity_column = "asset"


class FundamentalsData(DataSet):
    """Dataset without source - bind at runtime."""

    market_cap = Column(pl.Float64)
    pe_ratio = Column(pl.Float64)


class QuarterlyData(DataSet):
    """Dataset with reporting delay for point-in-time correctness."""

    pe_ratio = Column(pl.Float64)
    market_cap = Column(pl.Float64)

    class Config:
        source = ParquetSource("data/fundamentals.parquet")
        reporting_delay = 45  # Earnings reported 45 days after quarter end
        forward_fill_columns = ["pe_ratio", "market_cap"]


# ==============================================================================
# Examples
# ==============================================================================


def example_1_dataframe():
    """Load from existing DataFrame with column mapping."""
    print("\n" + "=" * 60)
    print("Example 1: DataFrameSource with Column Mapping")
    print("=" * 60)

    # Your data with non-standard column names
    df = pl.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "price": [150.0, 152.0, 151.0],
            "vol": [1000000, 1100000, 1050000],
        }
    )

    # Map to standard columns
    source = DataFrameSource(
        df,
        column_mapping={
            "trade_date": "date",
            "ticker": "asset",
            "price": "close",
            "vol": "volume",
        },
    )
    data = source.read()

    close = EquityPricing.close
    pipeline = Pipeline(data).add_factors({"returns": close.pct_change(1)})
    result = pipeline.run(collect=True)

    print(result)


def example_2_parquet():
    """Load from Parquet with date filtering."""
    print("\n" + "=" * 60)
    print("Example 2: ParquetSource")
    print("=" * 60)

    # Predicate pushdown - only reads needed rows
    source = ParquetSource("data/prices.parquet")

    try:
        data = source.read(start_date="2020-01-01", end_date="2020-12-31")
        print(f"Loaded {data.collect().shape[0]:,} rows")
    except Exception as e:
        print(f"Skipping (file not found): {e}")


def example_3_csv():
    """Load from CSV."""
    print("\n" + "=" * 60)
    print("Example 3: CSVSource")
    print("=" * 60)

    source = CSVSource("data/prices.csv")

    try:
        data = source.read()
        print(f"Loaded {data.collect().shape[0]:,} rows")
    except Exception as e:
        print(f"Skipping (file not found): {e}")


def example_4_data_context():
    """DataContext for managing multiple datasets."""
    print("\n" + "=" * 60)
    print("Example 4: DataContext for Multi-Dataset Workflows")
    print("=" * 60)

    ctx = DataContext()

    # Bind sources explicitly
    ctx.bind(
        PricingData,
        DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                    "asset": ["AAPL", "AAPL", "AAPL"],
                    "close": [150.0, 151.0, 152.0],
                    "volume": [1000000, 1100000, 1200000],
                }
            )
        ),
    )

    ctx.bind(
        FundamentalsData,
        DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                    "asset": ["AAPL", "AAPL", "AAPL"],
                    "market_cap": [1e12, 1e12, 1e12],
                    "pe_ratio": [25.0, 25.1, 25.2],
                }
            )
        ),
    )

    # Load many at once
    data = ctx.load_many(PricingData, FundamentalsData, start_date="2024-01-01")
    print(f"Loaded {data[PricingData].collect().shape[0]:,} pricing rows")
    print(f"Loaded {data[FundamentalsData].collect().shape[0]:,} fundamentals rows")

    # Clone context for testing
    test_ctx = ctx.clone()
    test_ctx.bind(
        PricingData,
        DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2024-01-01"],
                    "asset": ["TEST"],
                    "close": [100.0],
                    "volume": [1000],
                }
            )
        ),
    )
    test_data = test_ctx.load(PricingData)
    print(f"Test context: {test_data.collect().shape[0]} rows")


def example_5_simple_load():
    """Simple dataset loading with source override."""
    print("\n" + "=" * 60)
    print("Example 5: Simple Load with Source Override")
    print("=" * 60)

    # Override dataset's Config.source using DataContext
    ctx = DataContext()
    data = ctx.load(
        PricingData,
        source=DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "asset": ["OVERRIDE", "OVERRIDE"],
                    "close": [100.0, 101.0],
                    "volume": [1000000, 1100000],
                }
            )
        ),
    )
    print(f"Loaded {data.collect().shape[0]:,} rows with custom source")


def example_6_custom_source():
    """Custom source using protocol (no inheritance needed)."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Source (Protocol-Based)")
    print("=" * 60)

    class APISource:
        """Custom API source - just implement read() method."""

        def __init__(self, api_key: str, endpoint: str):
            self.api_key = api_key
            self.endpoint = endpoint

        def read(
            self,
            date_col: str = "date",
            asset_col: str = "asset",
            start_date: str | None = None,
            end_date: str | None = None,
        ) -> pl.LazyFrame:
            # Your API logic here (mocked)
            df = pl.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "asset": ["API_DATA", "API_DATA"],
                    "close": [150.0, 152.0],
                }
            ).with_columns(pl.col("date").str.to_date())

            lf = df.lazy()
            if start_date:
                lf = lf.filter(pl.col(date_col) >= pl.lit(start_date).cast(pl.Date))
            if end_date:
                lf = lf.filter(pl.col(date_col) <= pl.lit(end_date).cast(pl.Date))
            return lf

    source = APISource(api_key="secret", endpoint="https://api.example.com")
    data = source.read(start_date="2024-01-01")

    pipeline = Pipeline(data).add_factors({"returns": EquityPricing.close.pct_change(1)})
    result = pipeline.run(collect=True)
    print(result)


def example_7_point_in_time():
    """Point-in-time correctness with reporting delays."""
    print("\n" + "=" * 60)
    print("Example 7: Point-in-Time Correctness")
    print("=" * 60)

    ctx = DataContext()

    # Daily prices
    ctx.bind(
        PricingData,
        DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2024-01-01", "2024-02-15", "2024-03-01"],
                    "asset": ["A", "A", "A"],
                    "close": [100.0, 105.0, 108.0],
                    "volume": [1000, 1100, 1200],
                }
            )
        ),
    )

    # Quarterly fundamentals (Q4 2023)
    ctx.bind(
        QuarterlyData,
        DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2023-12-31"],  # Q4 end
                    "asset": ["A"],
                    "pe_ratio": [25.0],
                    "market_cap": [1e9],
                }
            )
        ),
    )

    price_factor = PricingData.close
    pe_factor = QuarterlyData.pe_ratio

    # Q4 2023 (2023-12-31) + 45 days = available 2024-02-14
    combined = ctx.load_and_combine([price_factor, pe_factor])
    result = combined.collect()

    print("Point-in-time: Q4 data available 45 days after 2023-12-31")
    print(result)


def example_8_combine_sources():
    """Combine multiple sources with offsets."""
    print("\n" + "=" * 60)
    print("Example 8: combine_sources()")
    print("=" * 60)

    prices = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "asset": ["A", "A", "A"],
                "close": [100.0, 101.0, 102.0],
            }
        )
    )

    fundamentals = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["A"],
                "pe_ratio": [25.0],
            }
        )
    )

    try:
        data = combine_sources(
            prices,  # Primary (daily)
            (fundamentals, {"offset": 1, "forward_fill": ["pe_ratio"]}),  # 1-day lag
        )
        print(f"Combined: {data.collect().shape[0]} rows")
    except Exception as e:
        print(f"combine_sources: {e}")


def main():
    print("\n" + "#" * 60)
    print("# Data Loading Examples")
    print("#" * 60)

    example_1_dataframe()
    example_2_parquet()
    example_3_csv()
    example_4_data_context()
    example_5_simple_load()
    example_6_custom_source()
    example_7_point_in_time()
    example_8_combine_sources()

    print("\n" + "#" * 60)
    print("# Examples Complete!")
    print("#" * 60)

    print("\nKey patterns:")
    print("  - DataFrameSource: in-memory data with column mapping")
    print("  - ParquetSource/CSVSource: file-based with date filtering")
    print("  - DataContext: bind multiple datasets, clone for testing")
    print("  - Custom sources: just implement read() method (protocol-based)")
    print("  - Point-in-time: use reporting_delay and forward_fill")
    print("\nSee sqlite_example.py for SQL database patterns.")


if __name__ == "__main__":
    main()
