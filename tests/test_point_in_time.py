"""Tests for point-in-time correctness when combining datasets."""

import polars as pl
import pytest

from factr.data import DataContext, DataFrameSource
from factr.datasets import Column, DataSet


# Test datasets with PIT configuration
class DailyPricing(DataSet):
    """Daily pricing data - primary dataset."""

    close = Column(pl.Float64)
    volume = Column(pl.Int64)

    class Config:
        is_primary = True  # This is the base dataset


class QuarterlyFundamentals(DataSet):
    """Quarterly fundamentals with reporting delay."""

    pe_ratio = Column(pl.Float64)
    market_cap = Column(pl.Float64)

    class Config:
        reporting_delay = 45  # Reported 45 days after quarter end
        forward_fill_columns = ["pe_ratio", "market_cap"]  # Fill to daily frequency


class WeeklySentiment(DataSet):
    """Weekly sentiment data."""

    sentiment_score = Column(pl.Float64)

    class Config:
        reporting_delay = 0
        forward_fill_columns = ["sentiment_score"]


def test_simple_combine_two_datasets():
    """Test combining pricing and fundamentals with PIT correctness."""
    # Daily prices: Jan 1 - Mar 31, 2024
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-15", "2024-02-15", "2024-03-01", "2024-03-15"],
                "asset": ["A"] * 5,
                "close": [100.0, 102.0, 105.0, 108.0, 110.0],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Quarterly fundamentals: Q4 2023 (reported Feb 14, 2024 = Dec 31 + 45 days)
    fundamentals = (
        pl.DataFrame(
            {
                "date": ["2023-12-31"],  # Quarter end date
                "asset": ["A"],
                "pe_ratio": [25.0],
                "market_cap": [1e9],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Setup context
    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(QuarterlyFundamentals, DataFrameSource(fundamentals))

    # Define factors
    momentum = DailyPricing.close.pct_change(1)
    value = QuarterlyFundamentals.pe_ratio

    result = ctx.load_and_combine([momentum, value]).collect()

    pe_col = "quarterly_fundamentals__pe_ratio"

    # Before reporting date (2023-12-31 + 45 days = 2024-02-14), no data
    assert result.filter(pl.col("date") == pl.date(2024, 1, 1))[pe_col][0] is None
    assert result.filter(pl.col("date") == pl.date(2024, 1, 15))[pe_col][0] is None

    # On and after reporting date, data available (forward-filled)
    assert result.filter(pl.col("date") == pl.date(2024, 2, 15))[pe_col][0] == 25.0
    assert result.filter(pl.col("date") == pl.date(2024, 3, 1))[pe_col][0] == 25.0
    assert result.filter(pl.col("date") == pl.date(2024, 3, 15))[pe_col][0] == 25.0


def test_combine_three_datasets():
    """Test combining pricing, fundamentals, and sentiment."""
    # Daily prices
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
                "asset": ["A"] * 4,
                "close": [100.0, 102.0, 105.0, 108.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Quarterly fundamentals (Q4 2023, reported late)
    fundamentals = (
        pl.DataFrame(
            {
                "date": ["2023-12-31"],
                "asset": ["A"],
                "pe_ratio": [20.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Weekly sentiment (every Monday)
    sentiment = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-08", "2024-01-15"],
                "asset": ["A"] * 3,
                "sentiment_score": [0.5, 0.3, 0.7],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Setup
    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(QuarterlyFundamentals, DataFrameSource(fundamentals))
    ctx.bind(WeeklySentiment, DataFrameSource(sentiment))

    # Factors from all three datasets
    close = DailyPricing.close
    pe = QuarterlyFundamentals.pe_ratio
    sentiment_factor = WeeklySentiment.sentiment_score

    result = ctx.load_and_combine([close, pe, sentiment_factor]).collect()

    sentiment_col = "weekly_sentiment__sentiment_score"

    # Should have all dates from primary (prices)
    assert len(result) == 4

    # Check sentiment forward-filling
    assert result.filter(pl.col("date") == pl.date(2024, 1, 1))[sentiment_col][0] == 0.5
    assert result.filter(pl.col("date") == pl.date(2024, 1, 8))[sentiment_col][0] == 0.3
    assert result.filter(pl.col("date") == pl.date(2024, 1, 15))[sentiment_col][0] == 0.7
    assert (
        result.filter(pl.col("date") == pl.date(2024, 1, 22))[sentiment_col][0] == 0.7
    )  # Forward-filled from 2024-01-15


def test_explicit_primary_dataset():
    """Test specifying primary dataset explicitly."""
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "asset": ["A"] * 2,
                "close": [100.0, 101.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    fundamentals = (
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["A"],
                "pe_ratio": [20.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(QuarterlyFundamentals, DataFrameSource(fundamentals))

    factors = [DailyPricing.close, QuarterlyFundamentals.pe_ratio]

    # Specify primary explicitly
    result = ctx.load_and_combine(factors, primary_dataset=DailyPricing).collect()

    # Should have both dates from DailyPricing
    assert len(result) == 2


def test_multiple_assets():
    """Test PIT correctness works across multiple assets."""
    # Daily prices for 2 assets
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-15", "2024-01-01", "2024-01-15"],
                "asset": ["A", "A", "B", "B"],
                "close": [100.0, 105.0, 200.0, 210.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Fundamentals for both assets (different quarter ends)
    fundamentals = (
        pl.DataFrame(
            {
                "date": ["2023-12-31", "2023-12-31"],  # Both Q4 2023
                "asset": ["A", "B"],
                "pe_ratio": [20.0, 30.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(QuarterlyFundamentals, DataFrameSource(fundamentals))

    result = ctx.load_and_combine([DailyPricing.close, QuarterlyFundamentals.pe_ratio]).collect()

    pe_col = "quarterly_fundamentals__pe_ratio"

    # Should have 4 rows (2 assets × 2 dates)
    assert len(result) == 4

    # Check PIT correctness per asset
    # Before reporting (2023-12-31 + 45 = 2024-02-14), no data
    assert (
        result.filter((pl.col("asset") == "A") & (pl.col("date") == pl.date(2024, 1, 1)))[pe_col][0]
        is None
    )
    assert (
        result.filter((pl.col("asset") == "B") & (pl.col("date") == pl.date(2024, 1, 1)))[pe_col][0]
        is None
    )

    # After reporting date (Jan 15 < Feb 14), still no data
    assert (
        result.filter((pl.col("asset") == "A") & (pl.col("date") == pl.date(2024, 1, 15)))[pe_col][
            0
        ]
        is None
    )


def test_no_reporting_delay():
    """Test dataset with no reporting delay."""

    # Create dataset without reporting delay
    class ImmediateData(DataSet):
        value = Column(pl.Float64)

        class Config:
            reporting_delay = 0  # No delay

    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "asset": ["A"] * 2,
                "close": [100.0, 101.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    immediate = (
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["A"],
                "value": [50.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(ImmediateData, DataFrameSource(immediate))

    result = ctx.load_and_combine([DailyPricing.close, ImmediateData.value]).collect()

    value_col = "immediate_data__value"

    # Data available immediately (no delay)
    assert result.filter(pl.col("date") == pl.date(2024, 1, 1))[value_col][0] == 50.0
    assert (
        result.filter(pl.col("date") == pl.date(2024, 1, 2))[value_col][0] == 50.0
    )  # Forward-filled


def test_config_documentation_example():
    """Test the exact example from the docstring."""
    # Setup datasets
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-15", "2024-02-15", "2024-03-01"],
                "asset": ["A"] * 4,
                "close": [100.0, 102.0, 105.0, 108.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    fundamentals = (
        pl.DataFrame(
            {
                "date": ["2023-12-31"],  # Q4 2023
                "asset": ["A"],
                "pe_ratio": [25.0],
                "market_cap": [1e9],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Bind sources
    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(QuarterlyFundamentals, DataFrameSource(fundamentals))

    # Define factors
    momentum = DailyPricing.close.pct_change(20)
    value = QuarterlyFundamentals.pe_ratio

    data = ctx.load_and_combine([momentum, value], start_date="2024-01-01")
    result = data.collect()

    assert "date" in result.columns
    assert "asset" in result.columns
    assert "daily_pricing__close" in result.columns
    assert "quarterly_fundamentals__pe_ratio" in result.columns

    pe_col = "quarterly_fundamentals__pe_ratio"

    # Verify PIT correctness (45-day delay)
    # 2023-12-31 + 45 days = 2024-02-14
    assert result.filter(pl.col("date") == pl.date(2024, 1, 1))[pe_col][0] is None  # Before
    assert result.filter(pl.col("date") == pl.date(2024, 2, 15))[pe_col][0] == 25.0  # After
    assert result.filter(pl.col("date") == pl.date(2024, 3, 1))[pe_col][0] == 25.0  # Forward-filled


def test_no_factors_raises():
    """Test that empty factors raises error."""
    ctx = DataContext()

    with pytest.raises(ValueError, match="No datasets found"):
        ctx.load_and_combine([])


def test_pipeline_integration():
    """Test using load_and_combine with Pipeline."""
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "asset": ["A"] * 2,
                "close": [100.0, 101.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    fundamentals = (
        pl.DataFrame(
            {
                "date": ["2023-12-15"],  # Early enough to be available
                "asset": ["A"],
                "pe_ratio": [20.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    ctx = DataContext()
    ctx.bind(DailyPricing, DataFrameSource(prices))
    ctx.bind(QuarterlyFundamentals, DataFrameSource(fundamentals))

    # Define factors and pipeline
    momentum = DailyPricing.close.pct_change(1)
    value = QuarterlyFundamentals.pe_ratio

    data = ctx.load_and_combine([momentum, value])
    result_df = data.collect()

    assert "daily_pricing__close" in result_df.columns
    assert "quarterly_fundamentals__pe_ratio" in result_df.columns
