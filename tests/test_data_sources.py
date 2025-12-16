"""Tests for data alignment functions."""

from datetime import date

import polars as pl

from factr.data.alignment import apply_offset, asof_join, forward_fill


def test_apply_offset():
    """Test offset application."""
    df = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-15", "2024-02-01"],
                "asset": ["A"] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    result = apply_offset(df, offset_days=10).collect()

    # Check offset was applied
    assert result["available_date"][0] == date(2024, 1, 11)
    assert result["available_date"][1] == date(2024, 1, 25)


def test_forward_fill():
    """Test forward fill."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "asset": ["A"] * 4,
            "value": [1.0, None, None, 2.0],
        }
    ).lazy()

    result = forward_fill(df, ["value"]).collect()

    # Check forward fill worked
    assert result["value"][0] == 1.0
    assert result["value"][1] == 1.0
    assert result["value"][2] == 1.0
    assert result["value"][3] == 2.0


def test_asof_join():
    """Test point-in-time asof join."""
    # Daily prices
    prices = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "asset": ["A"] * 3,
                "close": [100.0, 101.0, 102.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Quarterly fundamentals
    funds = (
        pl.DataFrame(
            {
                "date": ["2023-12-31", "2024-01-02"],
                "asset": ["A"] * 2,
                "pe_ratio": [20.0, 22.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    result = asof_join(prices, funds, on="date", by="asset").collect()

    # Check point-in-time correctness
    assert result["pe_ratio"][0] == 20.0  # 2024-01-01 should use 2023-12-31 data
    assert result["pe_ratio"][1] == 22.0  # 2024-01-02 should use 2024-01-02 data
    assert result["pe_ratio"][2] == 22.0  # 2024-01-03 should use 2024-01-02 data


def test_combine_sources_alignment():
    """Test combining multiple sources with offsets using alignment module."""
    # Daily prices
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

    # Quarterly fundamentals (with 45-day lag)
    funds = (
        pl.DataFrame({"date": ["2023-12-31"], "asset": ["A"], "pe_ratio": [25.0]})
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    # Apply offset and join manually using alignment functions
    funds_with_offset = apply_offset(funds, offset_days=45)
    funds_with_offset = funds_with_offset.drop("date").rename({"available_date": "date"})

    result = asof_join(prices, funds_with_offset, on="date", by="asset")
    result = forward_fill(result, ["pe_ratio"], by="asset", order_by="date")
    result = result.collect()

    # Before offset date (2023-12-31 + 45 days = 2024-02-14), no data
    assert result["pe_ratio"][0] is None  # 2024-01-01
    assert result["pe_ratio"][1] is None  # 2024-01-15

    # After offset date, data available and forward-filled
    assert result["pe_ratio"][2] == 25.0  # 2024-02-15
    assert result["pe_ratio"][3] == 25.0  # 2024-03-01
