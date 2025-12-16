"""Tests for Column field features (Pydantic-inspired).

Tests alias, default, fill_null, fill_strategy, and validation features.
"""

import polars as pl
import pytest

from factr.data import DataFrameSource
from factr.datasets import Column, DataSet


def test_column_with_alias():
    """Test column name aliasing."""

    class PricingWithAlias(DataSet):
        close = Column(pl.Float64, alias="price")
        volume = Column(pl.Int64, alias="trading_volume")

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02"],
                        "asset": ["AAPL", "AAPL"],
                        "price": [150.0, 152.0],  # Source column name
                        "trading_volume": [1000000, 1100000],  # Source column name
                    }
                )
            )

    lf = PricingWithAlias.load()
    df = lf.collect()

    # Should be renamed to field names
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "price" not in df.columns  # Original name should be dropped
    assert "trading_volume" not in df.columns

    assert df["close"][0] == 150.0
    assert df["volume"][0] == 1000000


def test_column_with_default():
    """Test default values for missing optional columns."""

    class DataWithDefaults(DataSet):
        close = Column(pl.Float64)
        sector = Column(pl.Utf8, default="Unknown", required=False)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "close": [150.0],
                        # 'sector' is missing from source
                    }
                )
            )

    lf = DataWithDefaults.load()
    df = lf.collect()

    # Should have default value
    assert "sector" in df.columns
    assert df["sector"][0] == "Unknown"


def test_column_required_missing_raises():
    """Test that missing required columns raise error."""

    class DataMissingRequired(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64, required=True)  # Explicitly required

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "close": [150.0],
                        # 'volume' is missing!
                    }
                )
            )

    with pytest.raises(ValueError, match="Required column 'volume'"):
        DataMissingRequired.load().collect()


def test_column_fill_null_value():
    """Test filling nulls with specific value."""

    class DataWithNullFill(DataSet):
        close = Column(pl.Float64, fill_null=0.0)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "asset": ["AAPL", "AAPL", "AAPL"],
                        "close": [150.0, None, 152.0],
                    }
                )
            )

    lf = DataWithNullFill.load()
    df = lf.collect()

    # Nulls should be filled with 0.0
    assert df["close"][1] == 0.0


def test_column_fill_strategy_forward():
    """Test forward fill strategy."""

    class DataWithForwardFill(DataSet):
        close = Column(pl.Float64, fill_strategy="forward")

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "asset": ["AAPL", "AAPL", "AAPL"],
                        "close": [150.0, None, None],
                    }
                )
            )

    lf = DataWithForwardFill.load()
    df = lf.collect()

    # Should forward fill
    assert df["close"][1] == 150.0
    assert df["close"][2] == 150.0


def test_column_fill_strategy_zero():
    """Test zero fill strategy."""

    class DataWithZeroFill(DataSet):
        volume = Column(pl.Int64, fill_strategy="zero")

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02"],
                        "asset": ["AAPL", "AAPL"],
                        "volume": [1000000, None],
                    }
                )
            )

    lf = DataWithZeroFill.load()
    df = lf.collect()

    assert df["volume"][1] == 0


def test_column_get_column_mapping():
    """Test get_column_mapping() method."""

    class DataSetWithAliases(DataSet):
        close = Column(pl.Float64, alias="price")
        volume = Column(pl.Int64, alias="trading_volume")
        sector = Column(pl.Utf8)  # No alias

    mapping = DataSetWithAliases.get_column_mapping()

    assert mapping == {
        "price": "close",
        "trading_volume": "volume",
    }


def test_column_get_column_descriptors():
    """Test get_column_descriptors() method."""

    class TestDataSet(DataSet):
        close = Column(pl.Float64)
        volume = Column(pl.Int64)

    descriptors = TestDataSet.get_column_descriptors()

    assert "close" in descriptors
    assert "volume" in descriptors
    assert isinstance(descriptors["close"], Column)
    assert descriptors["close"].dtype == pl.Float64


def test_column_apply_transforms_without_loading():
    """Test apply_transforms can be used standalone."""
    col = Column(pl.Float64, fill_null=99.0)

    # Create expression
    expr = col.apply_transforms(pl.col("test"))

    # Test with data
    df = pl.DataFrame({"test": [10.0, None, 50.0, 150.0]})
    result = df.select(expr.alias("transformed"))

    assert result["transformed"][0] == 10.0  # Unchanged
    assert result["transformed"][1] == 99.0  # Filled null
    assert result["transformed"][2] == 50.0  # Unchanged
    assert result["transformed"][3] == 150.0  # Unchanged


def test_column_source_name_property():
    """Test source_name property."""
    col_with_alias = Column(pl.Float64, alias="price")
    col_with_alias.name = "close"

    col_without_alias = Column(pl.Float64)
    col_without_alias.name = "volume"

    assert col_with_alias.source_name == "price"
    assert col_without_alias.source_name == "volume"


def test_column_repr():
    """Test Column __repr__."""
    col1 = Column(pl.Float64)
    col1.name = "close"

    col2 = Column(pl.Int64, alias="trading_volume", required=False, fill_strategy="forward")
    col2.name = "volume"

    assert "'close'" in repr(col1)
    assert "dtype" in repr(col1)

    assert "'volume'" in repr(col2)
    assert "alias='trading_volume'" in repr(col2)
    assert "required=False" in repr(col2)
    assert "fill_strategy='forward'" in repr(col2)


def test_mixed_features():
    """Test combining multiple features."""

    class ComplexDataSet(DataSet):
        # Aliased with default and null filling
        sector = Column(
            pl.Utf8,
            alias="gics_sector",
            default="Unknown",
            required=False,
        )

        # Aliased with forward fill
        sentiment = Column(
            pl.Float64,
            alias="news_sentiment",
            fill_strategy="forward",
        )

        # Simple required column
        close = Column(pl.Float64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "asset": ["AAPL", "AAPL", "AAPL"],
                        "close": [150.0, 151.0, 152.0],
                        "news_sentiment": [
                            0.5,
                            None,
                            0.8,
                        ],  # Has null
                        # gics_sector is missing entirely
                    }
                )
            )

    lf = ComplexDataSet.load()
    df = lf.collect()

    # Check all features work together
    assert "sector" in df.columns
    assert df["sector"][0] == "Unknown"  # Default

    assert "sentiment" in df.columns
    assert df["sentiment"][0] == 0.5  # Normal
    assert df["sentiment"][1] == 0.5  # Forward filled
    assert df["sentiment"][2] == 0.8  # Normal

    assert "close" in df.columns
    assert df["close"][0] == 150.0


def test_load_with_apply_transforms_false():
    """Test that apply_transforms=False skips transformations."""

    class DataWithTransforms(DataSet):
        price = Column(pl.Float64, alias="close", fill_null=0.0)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "close": [None],
                    }
                )
            )

    # With transforms (default)
    lf_with = DataWithTransforms.load(apply_transforms=True)
    df_with = lf_with.collect()
    assert "price" in df_with.columns  # Renamed
    assert df_with["price"][0] == 0.0  # Filled

    # Without transforms
    lf_without = DataWithTransforms.load(apply_transforms=False)
    df_without = lf_without.collect()
    assert "close" in df_without.columns  # Original name
    assert df_without["close"][0] is None  # Not filled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
