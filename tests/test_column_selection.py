"""Tests for automatic column selection optimization.

Tests that only referenced columns are loaded from sources.
"""

import polars as pl
import pytest

from factr.core import Factor
from factr.data import DataFrameSource
from factr.datasets import Column, DataSet


def test_factor_tracks_source_columns():
    """Test that Factors track their source columns."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64)

    close_factor = TestDataSet.close
    volume_factor = TestDataSet.volume

    # Should track which column it references
    assert close_factor.source_columns == frozenset({"close"})
    assert volume_factor.source_columns == frozenset({"volume"})


def test_factor_composition_merges_columns():
    """Test that factor composition merges source columns."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64)

    # Combine factors
    combined = TestDataSet.close + TestDataSet.volume

    # Should track both columns
    assert combined.source_columns == frozenset({"close", "volume"})


def test_factor_operations_preserve_columns():
    """Test that operations preserve source column tracking."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)

    # Various operations
    pct_change = TestDataSet.close.pct_change(1)
    rolling = TestDataSet.close.rolling_mean(10)
    ranked = TestDataSet.close.rank()

    # All should track 'close' column
    assert pct_change.source_columns == frozenset({"close"})
    assert rolling.source_columns == frozenset({"close"})
    assert ranked.source_columns == frozenset({"close"})


def test_dataset_load_with_column_selection():
    """Test loading only specific columns."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64)
        open = Column(pl.Float64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02"],
                        "asset": ["AAPL", "AAPL"],
                        "close": [150.0, 152.0],
                        "volume": [1000000, 1100000],
                        "open": [149.0, 151.0],
                    }
                )
            )

    # Load only close and volume
    lf = TestDataSet.load(columns=["close", "volume"])
    df = lf.collect()

    # Should only have date, asset, close, volume
    assert set(df.columns) == {"date", "asset", "close", "volume"}
    assert "open" not in df.columns  # Excluded


def test_dataset_load_with_aliases():
    """Test column selection works with aliases."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64, alias="price")
        volume = Column(pl.Int64, alias="trading_volume")

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "price": [150.0],  # Source name
                        "trading_volume": [1000000],  # Source name
                    }
                )
            )

    # Request by field name, not source name
    lf = TestDataSet.load(columns=["close"])
    df = lf.collect()

    # Should load 'price' and rename to 'close'
    assert "close" in df.columns
    assert "price" not in df.columns
    assert "volume" not in df.columns  # Not requested


def test_column_selection_always_includes_date_asset():
    """Test that date and entity columns are always included."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "close": [150.0],
                        "volume": [1000000],
                    }
                )
            )

    # Request only 'close'
    lf = TestDataSet.load(columns=["close"])
    df = lf.collect()

    # Should include date and asset automatically
    assert "date" in df.columns
    assert "asset" in df.columns
    assert "close" in df.columns
    assert "volume" not in df.columns


def test_multi_factor_column_tracking():
    """Test tracking columns across multiple factors."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64)
        open = Column(pl.Float64)

    # Create multiple factors
    momentum = TestDataSet.close.pct_change(20)
    volume_avg = TestDataSet.volume.rolling_mean(10)
    spread = TestDataSet.close - TestDataSet.open

    # Check individual tracking
    assert momentum.source_columns == frozenset({"close"})
    assert volume_avg.source_columns == frozenset({"volume"})
    assert spread.source_columns == frozenset({"close", "open"})

    # Combine all factors
    all_factors = [momentum, volume_avg, spread]
    all_columns = set()
    for factor in all_factors:
        all_columns.update(factor.source_columns)

    # Should have tracked all 3 columns
    assert all_columns == {"close", "volume", "open"}


def test_empty_source_columns_for_manual_factors():
    """Test that manually created factors have empty source_columns."""

    # Create a factor without using Column descriptor
    manual_factor = Factor(pl.col("some_col"))

    # Should have empty source_columns
    assert manual_factor.source_columns == frozenset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
