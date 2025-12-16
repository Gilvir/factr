"""Tests for dataset loading with Config pattern.

Tests the simplified configuration system for datasets.
"""

import polars as pl
import pytest

from factr.data import (
    DataContext,
    DataFrameSource,
)
from factr.data.config import DataSetConfig
from factr.datasets import Column, DataSet


# Sample datasets for testing
class SimpleDataSet(DataSet):
    """Dataset with inline Config."""

    close = Column(pl.Float64)
    volume = Column(pl.Int64)

    class Config:
        source = DataFrameSource(
            pl.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "asset": ["AAPL", "AAPL"],
                    "close": [150.0, 152.0],
                    "volume": [1000000, 1100000],
                }
            )
        )
        date_column = "date"
        entity_column = "asset"


class NoConfigDataSet(DataSet):
    """Dataset without Config."""

    close = Column(pl.Float64)


# Tests


def test_dataset_get_config():
    """Test getting config from dataset."""
    config = SimpleDataSet.get_config()
    assert config is not None
    assert isinstance(config, DataSetConfig)
    assert config.date_column == "date"
    assert config.entity_column == "asset"


def test_dataset_no_config():
    """Test dataset without Config class."""
    config = NoConfigDataSet.get_config()
    assert config is None


def test_dataset_load_simple():
    """Test loading dataset with direct source."""
    lf = SimpleDataSet.load()

    assert isinstance(lf, pl.LazyFrame)
    df = lf.collect()
    assert len(df) == 2
    assert "close" in df.columns
    assert "volume" in df.columns


def test_dataset_load_with_filters():
    """Test loading with date filters."""
    lf = SimpleDataSet.load(start_date="2024-01-02")

    df = lf.collect()
    assert len(df) == 1
    assert df["close"][0] == 152.0


def test_dataset_load_no_config_raises():
    """Test that loading without config raises error."""
    with pytest.raises(ValueError, match="has no Config class"):
        NoConfigDataSet.load()


def test_simple_dataset_load():
    """Test SimpleDataSet.load() using Config."""
    lf = SimpleDataSet.load()

    df = lf.collect()
    assert len(df) == 2
    assert "close" in df.columns


def test_context_load_with_override():
    """Test DataContext.load() with explicit source override."""
    override_source = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-02-01"],
                "asset": ["GOOGL"],
                "close": [2800.0],
                "volume": [2000000],
            }
        )
    )

    ctx = DataContext()
    lf = ctx.load(SimpleDataSet, source=override_source)

    df = lf.collect()
    assert len(df) == 1
    assert df["close"][0] == 2800.0
    assert df["asset"][0] == "GOOGL"


def test_data_context_bind():
    """Test binding datasets to sources in context."""
    ctx = DataContext()

    test_source = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["MSFT"],
                "close": [380.0],
            }
        )
    )

    ctx.bind(NoConfigDataSet, test_source)

    lf = ctx.load(NoConfigDataSet)
    df = lf.collect()
    assert df["close"][0] == 380.0


def test_data_context_load_with_config():
    """Test context loads from dataset Config if no binding."""
    ctx = DataContext()

    lf = ctx.load(SimpleDataSet)
    df = lf.collect()
    assert len(df) == 2


def test_data_context_override_priority():
    """Test that explicit source > bound > config."""
    ctx = DataContext()

    # Bind a source
    bound_source = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["BOUND"],
                "close": [111.0],
            }
        )
    )
    ctx.bind(SimpleDataSet, bound_source)

    # Explicit source should override
    explicit_source = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["EXPLICIT"],
                "close": [222.0],
            }
        )
    )

    lf = ctx.load(SimpleDataSet, source=explicit_source)
    df = lf.collect()
    assert df["asset"][0] == "EXPLICIT"
    assert df["close"][0] == 222.0


def test_data_context_bound_overrides_config():
    """Test that bound source overrides dataset Config."""
    ctx = DataContext()

    bound_source = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["BOUND"],
                "close": [111.0],
            }
        )
    )
    ctx.bind(SimpleDataSet, bound_source)

    lf = ctx.load(SimpleDataSet)
    df = lf.collect()
    assert df["asset"][0] == "BOUND"  # Not from Config source


def test_data_context_no_source_raises():
    """Test that loading with no source raises error."""
    ctx = DataContext()

    with pytest.raises(ValueError, match="No source configured"):
        ctx.load(NoConfigDataSet)


def test_data_context_load_many():
    """Test loading multiple datasets."""
    ctx = DataContext()

    data = ctx.load_many(SimpleDataSet)

    assert SimpleDataSet in data
    assert isinstance(data[SimpleDataSet], pl.LazyFrame)


def test_data_context_clone():
    """Test cloning context."""
    ctx = DataContext()

    test_source = DataFrameSource(
        pl.DataFrame(
            {
                "date": ["2024-01-01"],
                "asset": ["TEST"],
                "close": [100.0],
            }
        )
    )
    ctx.bind(SimpleDataSet, test_source)

    # Clone
    cloned = ctx.clone()

    # Cloned context should have same bindings
    lf = cloned.load(SimpleDataSet)
    df = lf.collect()
    assert df["asset"][0] == "TEST"


def test_column_mapping_in_direct_source():
    """Test column mapping when using direct source object."""

    class MappedDataSet(DataSet):
        close = Column(pl.Float64)

        class Config:
            # When using direct source, pass column_mapping to the source
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "trade_date": ["2024-01-01"],
                        "ticker": ["AAPL"],
                        "price": [150.0],
                    }
                ),
                column_mapping={
                    "trade_date": "date",
                    "ticker": "asset",
                    "price": "close",
                },
            )

    lf = MappedDataSet.load()
    df = lf.collect()

    # Should have mapped column names
    assert "date" in df.columns
    assert "asset" in df.columns
    assert "close" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
