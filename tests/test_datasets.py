"""Tests for Dataset definitions."""

import polars as pl
import pytest

from factr.core import Factor
from factr.datasets import (
    Column,
    DataSet,
    EquityPricing,
    Fundamentals,
    ReferenceData,
    Sentiment,
    dataset,
)


def test_column_descriptor():
    """Test Column descriptor creates Factors."""

    class TestData(DataSet):
        price = Column(pl.Float64)

    # Accessing column returns a Factor
    factor = TestData.price
    assert isinstance(factor, Factor)
    assert factor.name == "price"


def test_equity_pricing_columns():
    """Test EquityPricing dataset has expected columns."""
    # Check columns exist
    assert hasattr(EquityPricing, "open")
    assert hasattr(EquityPricing, "high")
    assert hasattr(EquityPricing, "low")
    assert hasattr(EquityPricing, "close")
    assert hasattr(EquityPricing, "volume")

    # Check they return Factors
    close = EquityPricing.close
    assert isinstance(close, Factor)
    assert close.name == "close"


def test_equity_pricing_usage():
    """Test using EquityPricing in actual computation."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 2000, 3000],
        }
    ).lazy()

    # Use dataset columns
    close = EquityPricing.close
    volume = EquityPricing.volume

    # Compose factors
    dollar_volume = close * volume

    result = df.with_columns([dollar_volume.expr.alias("dollar_vol")]).collect()

    assert "dollar_vol" in result.columns
    assert result["dollar_vol"][0] == 100000.0
    assert result["dollar_vol"][1] == 400000.0
    assert result["dollar_vol"][2] == 900000.0


def test_dataset_columns_method():
    """Test DataSet.columns() method."""
    columns = EquityPricing.columns()
    assert "open" in columns
    assert "high" in columns
    assert "low" in columns
    assert "close" in columns
    assert "volume" in columns
    assert len(columns) == 5


def test_dataset_get_column():
    """Test DataSet.get_column() method."""
    close = EquityPricing.get_column("close")
    assert isinstance(close, Factor)
    assert close.name == "close"

    with pytest.raises(AttributeError):
        EquityPricing.get_column("nonexistent")


def test_fundamentals_dataset():
    """Test Fundamentals dataset."""
    columns = Fundamentals.columns()
    assert "market_cap" in columns
    assert "pe_ratio" in columns
    assert "pb_ratio" in columns

    # Check accessing column
    pe = Fundamentals.pe_ratio
    assert isinstance(pe, Factor)


def test_reference_data_dataset():
    """Test ReferenceData dataset."""
    sector = ReferenceData.sector
    assert isinstance(sector, Factor)
    assert sector.name == "sector"


def test_sentiment_dataset():
    """Test Sentiment dataset."""
    news_sent = Sentiment.news_sentiment
    assert isinstance(news_sent, Factor)


def test_custom_dataset():
    """Test creating custom dataset with dataset() helper."""
    MyData = dataset(
        "MyData",
        price=Column(pl.Float64),
        volume=Column(pl.Int64),
    )

    assert hasattr(MyData, "price")
    assert hasattr(MyData, "volume")

    price = MyData.price
    assert isinstance(price, Factor)
    assert price.name == "price"


def test_custom_dataset_class():
    """Test creating custom dataset as class."""

    class CustomData(DataSet):
        metric1 = Column(pl.Float64)
        metric2 = Column(pl.Float64)

    assert "metric1" in CustomData.columns()
    assert "metric2" in CustomData.columns()

    m1 = CustomData.metric1
    assert isinstance(m1, Factor)


def test_dataset_repr():
    """Test DataSet string representation."""
    repr_str = repr(EquityPricing)
    assert "EquityPricing" in repr_str
    assert "columns" in repr_str


def test_factor_composition_with_datasets():
    """Test composing factors from datasets."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "open": [100.0, 200.0, 300.0],
            "close": [105.0, 205.0, 305.0],
        }
    ).lazy()

    # Compose factors
    open_price = EquityPricing.open
    close_price = EquityPricing.close
    intraday_return = (close_price / open_price) - 1

    result = df.with_columns([intraday_return.expr.alias("intraday_ret")]).collect()

    assert "intraday_ret" in result.columns
    # All should be approximately 5% return
    assert abs(result["intraday_ret"][0] - 0.05) < 1e-10


def test_cross_sectional_with_datasets():
    """Test cross-sectional operations with dataset columns."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 300.0],
        }
    ).lazy()

    # Use dataset column with cross-sectional operation
    close = EquityPricing.close
    ranked = close.rank(pct=True)

    result = df.with_columns([ranked.expr.alias("rank")]).collect()

    assert "rank" in result.columns
    # Should be ranked 0, 0.5, 1.0
    assert result["rank"][0] == 0.0
    assert result["rank"][1] == 0.5
    assert result["rank"][2] == 1.0


def test_pipeline_with_datasets():
    """Test using datasets in Pipeline."""
    from factr import Pipeline

    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 2000, 3000],
        }
    ).lazy()

    # Use dataset columns in pipeline
    close = EquityPricing.close
    volume = EquityPricing.volume
    returns = close.pct_change()
    dollar_vol = close * volume

    pipeline = Pipeline(df).add_factors({"returns": returns, "dollar_vol": dollar_vol})

    result = pipeline.run(collect=True)

    assert "returns" in result.columns
    assert "dollar_vol" in result.columns
