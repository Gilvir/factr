"""Tests for Universe classes."""

import polars as pl

from factr.core import Filter
from factr.universe import (
    Q500US,
    Q1500US,
    AllAssets,
    LiquidUniverse,
    Universe,
    custom_universe,
)


def test_universe_creation():
    """Test basic Universe creation."""
    filter = Filter(pl.col("close") > 5, name="price_filter")
    universe = Universe(filter=filter, name="TestUniverse", description="Test")

    assert universe.name == "TestUniverse"
    assert universe.description == "Test"
    assert isinstance(universe.filter, Filter)


def test_q500us():
    """Test Q500US universe."""
    universe = Q500US(window=20, min_price=5.0)

    assert universe.name == "Top500"  # Now uses TopNUniverse
    assert isinstance(universe.filter, Filter)
    assert "$5" in universe.description


def test_q1500us():
    """Test Q1500US universe."""
    universe = Q1500US(window=20, min_price=5.0)

    assert universe.name == "Top1500"  # Now uses TopNUniverse
    assert isinstance(universe.filter, Filter)


def test_liquid_universe():
    """Test LiquidUniverse."""
    universe = LiquidUniverse(min_price=5.0, min_volume=1e6)

    assert universe.name == "LiquidUniverse"
    assert isinstance(universe.filter, Filter)
    assert "price >=" in universe.description
    assert "volume >=" in universe.description


def test_liquid_universe_with_dollar_volume():
    """Test LiquidUniverse with dollar volume constraint."""
    universe = LiquidUniverse(min_price=5.0, min_volume=1e6, min_dollar_volume=5e6)

    assert "dollar_volume >=" in universe.description


def test_all_assets():
    """Test AllAssets universe."""
    universe = AllAssets()

    assert universe.name == "AllAssets"
    assert isinstance(universe.filter, Filter)


def test_custom_universe():
    """Test custom_universe helper."""
    filter = Filter(pl.col("market_cap") > 1e9, name="large_cap")
    universe = custom_universe(filter=filter, name="LargeCap", description="Market cap > $1B")

    assert universe.name == "LargeCap"
    assert universe.description == "Market cap > $1B"
    assert universe.filter == filter


def test_q500us_filtering():
    """Test Q500US actually filters correctly."""
    # Create sample data with varying dollar volumes
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 1000,
            "asset": [f"STOCK_{i}" for i in range(1000)],
            "close": [10.0 + i * 0.1 for i in range(1000)],
            "volume": [1000000 - i * 100 for i in range(1000)],  # Decreasing volume
        }
    ).lazy()

    universe = Q500US(window=1, min_price=5.0)

    # Apply the filter
    result = df.filter(universe.filter.expr).collect()

    # Should have approximately 500 stocks (those with highest dollar volume)
    # Plus all stocks meeting min price (which is all in this case)
    # The top 500 by dollar volume should be selected
    assert len(result) == 500


def test_liquid_universe_filtering():
    """Test LiquidUniverse filtering."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 10,
            "asset": [f"STOCK_{i}" for i in range(10)],
            "close": [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
            "volume": [5e5, 5e5, 2e6, 2e6, 2e6, 2e6, 2e6, 2e6, 2e6, 2e6],
        }
    ).lazy()

    universe = LiquidUniverse(min_price=5.0, min_volume=1e6)

    result = df.filter(universe.filter.expr).collect()

    # Should filter out stocks with price < 5 or volume < 1e6
    # First stock: price 3.0 < 5.0 ❌
    # Second stock: price 5.0 >= 5.0, but volume 5e5 < 1e6 ❌
    # Rest: price >= 5.0 and volume >= 1e6 ✓
    assert len(result) == 8


def test_universe_repr():
    """Test Universe string representation."""
    universe = Q500US()
    repr_str = repr(universe)
    assert "Top500" in repr_str  # Now uses TopNUniverse


def test_all_assets_returns_everything():
    """Test AllAssets doesn't filter anything."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 5,
            "asset": ["A", "B", "C", "D", "E"],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).lazy()

    universe = AllAssets()
    result = df.filter(universe.filter.expr).collect()

    # All rows should pass through
    assert len(result) == 5
