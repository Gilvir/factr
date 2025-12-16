"""Tests for data binding - sources and loaders."""

import polars as pl
import pytest

from factr import Pipeline
from factr.data import DataFrameSource, combine_sources
from factr.data.binding import DataSource
from factr.datasets import EquityPricing

# === Test Data ===


def sample_pricing_data() -> pl.DataFrame:
    """Create sample pricing data for testing."""
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "asset": ["AAPL"] * 3 + ["MSFT"] * 3,
            "open": [150.0, 152.0, 151.0, 300.0, 305.0, 303.0],
            "high": [155.0, 156.0, 154.0, 310.0, 312.0, 308.0],
            "low": [149.0, 151.0, 150.0, 298.0, 303.0, 301.0],
            "close": [154.0, 153.0, 152.0, 308.0, 307.0, 306.0],
            "volume": [1000000, 1100000, 1050000, 2000000, 2100000, 2050000],
        }
    )


def sample_fundamentals_data() -> pl.DataFrame:
    """Create sample fundamentals data (quarterly)."""
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "asset": ["AAPL", "MSFT"],
            "market_cap": [3000000000000.0, 2500000000000.0],
            "pe_ratio": [25.0, 30.0],
        }
    )


# === Test DataFrameSource ===


def test_dataframe_source_basic():
    """Test basic DataFrame source."""
    df = sample_pricing_data()
    source = DataFrameSource(df)

    lf = source.read()
    result = lf.collect()

    assert len(result) == 6
    assert "close" in result.columns
    assert "asset" in result.columns


def test_dataframe_source_with_mapping():
    """Test DataFrame source with column mapping."""
    df = pl.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAPL", "AAPL"],
            "price": [150.0, 152.0],
        }
    )

    source = DataFrameSource(
        df,
        column_mapping={
            "trade_date": "date",
            "ticker": "asset",
            "price": "close",
        },
    )

    lf = source.read()
    result = lf.collect()

    assert "date" in result.columns
    assert "asset" in result.columns
    assert "close" in result.columns
    assert "trade_date" not in result.columns


def test_dataframe_source_date_filtering():
    """Test date filtering in DataFrame source."""
    df = sample_pricing_data()
    source = DataFrameSource(df)

    lf = source.read(start_date="2024-01-02", end_date="2024-01-02")
    result = lf.collect()

    assert len(result) == 2  # Only Jan 2 for both assets
    # Date column is cast to Date during filtering
    assert str(result["date"][0]) == "2024-01-02"


# === Test DataFrameSource helper ===


def test_load_dataframe():
    """Test DataFrameSource loading."""
    df = sample_pricing_data()
    source = DataFrameSource(df)
    lf = source.read(start_date="2024-01-01", end_date="2024-01-01")
    result = lf.collect()

    assert len(result) == 2  # Jan 1 for AAPL and MSFT
    assert "close" in result.columns


def test_load_dataframe_with_mapping():
    """Test load_dataframe with column mapping."""
    df = pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "price": [150.0, 300.0],
            "date": ["2024-01-01", "2024-01-01"],
        }
    )

    source = DataFrameSource(df, column_mapping={"ticker": "asset", "price": "close"})
    lf = source.read()
    result = lf.collect()

    assert "asset" in result.columns
    assert "close" in result.columns


# === Test Protocol Compliance ===


def test_custom_source_protocol():
    """Test that custom sources work via protocol."""

    class CustomSource:
        """Custom source without inheriting from anything."""

        def read(
            self, date_col="date", asset_col="asset", start_date=None, end_date=None
        ) -> pl.LazyFrame:
            df = sample_pricing_data()
            lf = df.lazy()
            if start_date:
                lf = lf.filter(pl.col(date_col).cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
            return lf

    # Should work via protocol - no inheritance needed!
    custom = CustomSource()
    assert isinstance(custom, DataSource)  # Protocol check

    lf = custom.read(start_date="2024-01-02")
    result = lf.collect()

    assert len(result) == 4  # Jan 2 and 3 for both assets


# === Test Pipeline Integration ===


def test_pipeline_with_dataframe_source():
    """Test that sources work seamlessly with Pipeline."""
    df = sample_pricing_data()
    source = DataFrameSource(df)

    data = source.read()

    # Use dataset to create factors
    close = EquityPricing.close
    volume = EquityPricing.volume

    pipeline = Pipeline(data).add_factors(
        {
            "returns": close.pct_change(1),
            "dollar_volume": close * volume,
        }
    )

    result = pipeline.run()

    assert "returns" in result.columns
    assert "dollar_volume" in result.columns
    assert len(result) == 6


def test_pipeline_with_loader():
    """Test pipeline with loader function."""
    df = sample_pricing_data()
    source = DataFrameSource(df)
    data = source.read()

    close = EquityPricing.close
    momentum = close.pct_change(1).rolling_mean(2)

    pipeline = Pipeline(data).add_factors({"momentum": momentum})
    result = pipeline.run()

    assert "momentum" in result.columns


# === Test combine_sources ===


def test_combine_sources_basic():
    """Test combining multiple data sources."""
    pricing = sample_pricing_data()
    funds = sample_fundamentals_data()

    pricing_source = DataFrameSource(pricing)
    funds_source = DataFrameSource(funds)

    combined = combine_sources(
        pricing_source,
        (funds_source, {"offset": 0, "forward_fill": ["market_cap", "pe_ratio"]}),
    )

    result = combined.collect()

    # Should have pricing + fundamentals columns
    assert "close" in result.columns
    assert "market_cap" in result.columns
    assert "pe_ratio" in result.columns
    assert len(result) == 6  # All pricing rows


def test_combine_sources_with_offset():
    """Test combining with reporting offset."""
    pricing = sample_pricing_data()

    # Fundamentals dated 2023-12-20, available 2024-01-01 with 12-day offset
    funds = pl.DataFrame(
        {
            "date": ["2023-12-20", "2023-12-20"],
            "asset": ["AAPL", "MSFT"],
            "market_cap": [3000000000000.0, 2500000000000.0],
        }
    )

    pricing_source = DataFrameSource(pricing)
    funds_source = DataFrameSource(funds)

    combined = combine_sources(
        pricing_source,
        (funds_source, {"offset": 12, "forward_fill": ["market_cap"]}),
        start_date="2024-01-01",
    )

    result = combined.collect()

    # Fundamentals should be available starting 2024-01-01
    assert "market_cap" in result.columns
    # Should be forward-filled for all dates
    assert result.filter(pl.col("asset") == "AAPL")["market_cap"].null_count() == 0


# === Test Edge Cases ===


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pl.DataFrame(
        {
            "date": [],
            "asset": [],
            "close": [],
        },
        schema={"date": pl.Utf8, "asset": pl.Utf8, "close": pl.Float64},
    )

    source = DataFrameSource(df)
    lf = source.read()
    result = lf.collect()

    assert len(result) == 0


def test_lazy_frame_input():
    """Test that LazyFrame input works."""
    df = sample_pricing_data()
    lf = df.lazy()

    source = DataFrameSource(lf)
    result = source.read().collect()

    assert len(result) == 6


def test_date_filtering_edge_cases():
    """Test date filtering edge cases."""
    df = sample_pricing_data()
    source = DataFrameSource(df)

    # Start date after all data
    lf = source.read(start_date="2024-12-31")
    assert len(lf.collect()) == 0

    # End date before all data
    lf = source.read(end_date="2023-01-01")
    assert len(lf.collect()) == 0

    # Both filters exclude everything
    lf = source.read(start_date="2024-02-01", end_date="2024-02-02")
    assert len(lf.collect()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
