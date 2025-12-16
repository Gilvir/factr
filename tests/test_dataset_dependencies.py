"""Tests for automatic dataset dependency tracking and smart loading."""

import polars as pl

from factr.core import extract_datasets
from factr.data import DataContext, DataFrameSource
from factr.datasets import Column, DataSet
from factr.pipeline import Pipeline


# Define test datasets
class TestPricing(DataSet):
    """Test pricing dataset."""

    close = Column(pl.Float64)
    open = Column(pl.Float64)
    volume = Column(pl.Int64)


class TestFundamentals(DataSet):
    """Test fundamentals dataset."""

    pe_ratio = Column(pl.Float64)
    market_cap = Column(pl.Float64)


class TestSentiment(DataSet):
    """Test sentiment dataset."""

    sentiment_score = Column(pl.Float64)


class TestReferenceData(DataSet):
    """Test reference dataset."""

    sector = Column(pl.Utf8)
    industry = Column(pl.Utf8)


# Test dataset dependency tracking
def test_factor_tracks_dataset():
    """Test that Factor tracks which dataset it came from."""
    close = TestPricing.close
    assert TestPricing in close.source_datasets
    assert len(close.source_datasets) == 1


def test_factor_composition_merges_datasets():
    """Test that composing factors merges dataset dependencies."""
    close = TestPricing.close
    pe = TestFundamentals.pe_ratio

    # Combine factors from different datasets
    combined = close / pe

    # Should track both datasets
    assert TestPricing in combined.source_datasets
    assert TestFundamentals in combined.source_datasets
    assert len(combined.source_datasets) == 2


def test_factor_operations_preserve_datasets():
    """Test that operations preserve dataset tracking."""
    close = TestPricing.close

    # Various operations
    returns = close.pct_change(1)
    momentum = returns.rolling_sum(20)
    ranked = momentum.rank()

    # All should preserve dataset tracking
    assert TestPricing in returns.source_datasets
    assert TestPricing in momentum.source_datasets
    assert TestPricing in ranked.source_datasets


def test_cross_sectional_ops_preserve_datasets():
    """Test that cross-sectional ops preserve datasets."""
    close = TestPricing.close
    momentum = close.pct_change(20)

    # Cross-sectional operations
    ranked = momentum.rank()
    demeaned = momentum.demean()
    zscored = momentum.zscore()
    winsorized = momentum.winsorize()

    # All should track TestPricing
    assert TestPricing in ranked.source_datasets
    assert TestPricing in demeaned.source_datasets
    assert TestPricing in zscored.source_datasets
    assert TestPricing in winsorized.source_datasets


def test_extract_datasets_from_list():
    """Test extracting datasets from a list of factors."""
    close = TestPricing.close
    pe = TestFundamentals.pe_ratio
    sentiment = TestSentiment.sentiment_score

    factors = [close, pe, sentiment]
    datasets = extract_datasets(factors)

    assert TestPricing in datasets
    assert TestFundamentals in datasets
    assert TestSentiment in datasets
    assert len(datasets) == 3


def test_extract_datasets_from_dict():
    """Test extracting datasets from a dict of factors."""
    factors = {
        "close": TestPricing.close,
        "pe": TestFundamentals.pe_ratio,
        "sentiment": TestSentiment.sentiment_score,
    }

    datasets = extract_datasets(factors.values())

    assert TestPricing in datasets
    assert TestFundamentals in datasets
    assert TestSentiment in datasets
    assert len(datasets) == 3


def test_extract_datasets_with_duplicates():
    """Test that extract_datasets deduplicates."""
    close = TestPricing.close
    open = TestPricing.open
    volume = TestPricing.volume

    factors = [close, open, volume]
    datasets = extract_datasets(factors)

    # Should only have TestPricing once
    assert TestPricing in datasets
    assert len(datasets) == 1


def test_multi_dataset_factor():
    """Test factor using columns from multiple datasets."""
    close = TestPricing.close
    pe = TestFundamentals.pe_ratio

    # Factor combining multiple datasets
    value_momentum = (close.pct_change(20) / pe).rank()

    # Should track both datasets
    assert TestPricing in value_momentum.source_datasets
    assert TestFundamentals in value_momentum.source_datasets
    assert len(value_momentum.source_datasets) == 2


# Test DataContext.load_for_factors
def test_context_load_for_factors_basic():
    """Test loading only needed datasets from context."""
    # Create test data
    pricing_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 150.0],
            "open": [99.0, 199.0, 149.0],
            "volume": [1000, 2000, 1500],
        }
    ).lazy()

    fundamentals_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "pe_ratio": [15.0, 20.0, 18.0],
            "market_cap": [1e9, 2e9, 1.5e9],
        }
    ).lazy()

    sentiment_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "sentiment_score": [0.5, -0.3, 0.1],
        }
    ).lazy()

    # Setup context with all datasets
    ctx = DataContext()
    ctx.bind(TestPricing, DataFrameSource(pricing_data))
    ctx.bind(TestFundamentals, DataFrameSource(fundamentals_data))
    ctx.bind(TestSentiment, DataFrameSource(sentiment_data))

    # Only use pricing and fundamentals in factors
    close = TestPricing.close
    pe = TestFundamentals.pe_ratio
    factors = [close, pe]

    # Load for factors (should only load TestPricing and TestFundamentals)
    loaded = ctx.load_for_factors(factors)

    # Should only have 2 datasets loaded
    assert len(loaded) == 2
    assert TestPricing in loaded
    assert TestFundamentals in loaded
    assert TestSentiment not in loaded


def test_context_load_for_factors_with_dict():
    """Test loading with dict of factors."""
    pricing_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 150.0],
        }
    ).lazy()

    fundamentals_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "pe_ratio": [15.0, 20.0, 18.0],
        }
    ).lazy()

    ctx = DataContext()
    ctx.bind(TestPricing, DataFrameSource(pricing_data))
    ctx.bind(TestFundamentals, DataFrameSource(fundamentals_data))
    ctx.bind(TestSentiment, DataFrameSource(pl.DataFrame().lazy()))

    # Use dict of factors
    factors = {
        "momentum": TestPricing.close.pct_change(1),
        "value": TestFundamentals.pe_ratio,
    }

    loaded = ctx.load_for_factors(factors)

    assert len(loaded) == 2
    assert TestPricing in loaded
    assert TestFundamentals in loaded
    assert TestSentiment not in loaded


def test_context_load_for_factors_collect():
    """Test collecting datasets concurrently."""
    pricing_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 150.0],
        }
    ).lazy()

    ctx = DataContext()
    ctx.bind(TestPricing, DataFrameSource(pricing_data))

    factors = [TestPricing.close]
    loaded = ctx.load_for_factors(factors, collect=True)

    # Should return DataFrames when collect=True
    assert isinstance(loaded[TestPricing], pl.DataFrame)


def test_context_load_for_factors_with_date_filters():
    """Test loading with date filters."""
    pricing_data = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"] * 2,
            "asset": ["A", "A", "A", "B", "B", "B"],
            "close": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
        }
    ).lazy()

    ctx = DataContext()
    ctx.bind(TestPricing, DataFrameSource(pricing_data))

    factors = [TestPricing.close]
    loaded = ctx.load_for_factors(
        factors, start_date="2020-01-02", end_date="2020-01-02", collect=True
    )

    result = loaded[TestPricing]
    # Should only have data from 2020-01-02
    assert len(result) == 2
    assert result.filter(pl.col("date") == pl.date(2020, 1, 2)).height == 2


# Test Pipeline.get_dataset_dependencies
def test_pipeline_get_dataset_dependencies():
    """Test extracting dataset dependencies from pipeline."""
    momentum = TestPricing.close.pct_change(20)
    pe = TestFundamentals.pe_ratio

    pipeline = Pipeline().add_factors({"momentum": momentum, "pe": pe})

    datasets = pipeline.get_dataset_dependencies()

    assert TestPricing in datasets
    assert TestFundamentals in datasets
    assert len(datasets) == 2


def test_pipeline_get_dataset_dependencies_empty():
    """Test pipeline with no factors."""
    pipeline = Pipeline()
    datasets = pipeline.get_dataset_dependencies()
    assert len(datasets) == 0


def test_pipeline_get_dataset_dependencies_single():
    """Test pipeline with single dataset."""
    close = TestPricing.close
    open = TestPricing.open

    pipeline = Pipeline().add_factors({"close": close, "open": open})

    datasets = pipeline.get_dataset_dependencies()

    assert TestPricing in datasets
    assert len(datasets) == 1


# Integration tests
def test_full_workflow():
    """Test complete workflow: define factors -> extract datasets -> load -> pipeline."""
    # Create test data
    pricing_data = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"] * 3,
            "asset": ["A", "A", "B", "B", "C", "C"],
            "close": [100.0, 101.0, 200.0, 202.0, 150.0, 151.0],
            "open": [99.0, 100.0, 199.0, 201.0, 149.0, 150.0],
            "volume": [1000, 1100, 2000, 2100, 1500, 1600],
        }
    ).lazy()

    fundamentals_data = pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"] * 3,
            "asset": ["A", "A", "B", "B", "C", "C"],
            "pe_ratio": [15.0, 15.1, 20.0, 20.1, 18.0, 18.1],
        }
    ).lazy()

    # Setup context with multiple datasets (including one we won't use)
    ctx = DataContext()
    ctx.bind(TestPricing, DataFrameSource(pricing_data))
    ctx.bind(TestFundamentals, DataFrameSource(fundamentals_data))
    ctx.bind(TestSentiment, DataFrameSource(pl.DataFrame().lazy()))
    ctx.bind(TestReferenceData, DataFrameSource(pl.DataFrame().lazy()))

    # Define factors (only using Pricing and Fundamentals)
    momentum = TestPricing.close.pct_change(1)
    value = TestFundamentals.pe_ratio

    factors = {"momentum": momentum, "value": value}

    # Load only needed datasets
    loaded = ctx.load_for_factors(factors, collect=True)

    # Should only load 2 datasets (not all 4)
    assert len(loaded) == 2
    assert TestPricing in loaded
    assert TestFundamentals in loaded

    # Verify data is correct
    pricing = loaded[TestPricing]
    assert "test_pricing__close" in pricing.columns
    assert len(pricing) == 6

    fundamentals = loaded[TestFundamentals]
    assert "test_fundamentals__pe_ratio" in fundamentals.columns
    assert len(fundamentals) == 6


def test_workflow_with_pipeline():
    """Test workflow using pipeline to get dependencies."""
    pricing_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 150.0],
        }
    ).lazy()

    fundamentals_data = pl.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "pe_ratio": [15.0, 20.0, 18.0],
        }
    ).lazy()

    # Setup context
    ctx = DataContext()
    ctx.bind(TestPricing, DataFrameSource(pricing_data))
    ctx.bind(TestFundamentals, DataFrameSource(fundamentals_data))
    ctx.bind(TestSentiment, DataFrameSource(pl.DataFrame().lazy()))

    # Create pipeline with factors
    momentum = TestPricing.close.pct_change(1)
    value = TestFundamentals.pe_ratio

    pipeline = Pipeline().add_factors({"momentum": momentum, "value": value})

    # Get datasets from pipeline
    datasets = pipeline.get_dataset_dependencies()

    # Load using those datasets
    loaded = ctx.load_many(*datasets, collect=True)

    # Should only load what pipeline needs
    assert len(loaded) == 2
    assert TestPricing in loaded
    assert TestFundamentals in loaded
    assert TestSentiment not in loaded
