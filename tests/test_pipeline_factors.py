"""Tests for Pipeline with Factor objects."""

import polars as pl

from factr import Factor, Filter, Pipeline
from factr import factors as F
from factr.universe import LiquidUniverse


def create_sample_data():
    return pl.DataFrame(
        {
            "date": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
            "asset": ["A", "B", "C", "D", "E"] * 2,
            "close": [
                100.0,
                200.0,
                150.0,
                180.0,
                220.0,
                105.0,
                205.0,
                148.0,
                185.0,
                225.0,
            ],
            "volume": [
                1e6,
                2e6,
                1.5e6,
                1.8e6,
                2.2e6,
                1.1e6,
                2.1e6,
                1.4e6,
                1.9e6,
                2.3e6,
            ],
        }
    ).lazy()


def test_pipeline_with_factors():
    df = create_sample_data()

    returns = F.returns(window=1)
    momentum = returns.rolling_sum(5)

    pipeline = Pipeline(df)
    pipeline.add_factors({"ret": returns, "mom": momentum})

    result = pipeline.run(collect=True)

    # Check that columns were added
    assert "ret" in result.columns
    assert "mom" in result.columns


def test_pipeline_with_universe():
    df = create_sample_data()

    pipeline = Pipeline(df)
    universe = LiquidUniverse(min_price=150.0, min_volume=1.5e6)
    pipeline.screen(universe)

    result = pipeline.run(collect=True)

    assert len(result) < len(df.collect())
    assert "B" in result["asset"].unique().to_list()


def test_pipeline_with_filter():
    df = create_sample_data()

    price_filter = Filter(pl.col("close") > 150, name="high_price")

    pipeline = Pipeline(df).screen(price_filter)
    result = pipeline.run(collect=True)

    assert (result["close"] > 150).all()


def test_pipeline_with_composed_filters():
    df = create_sample_data()

    price_filter = Filter(pl.col("close") > 150, name="high_price")
    volume_filter = Filter(pl.col("volume") > 1.5e6, name="high_volume")
    combined = price_filter & volume_filter

    pipeline = Pipeline(df).screen(combined)
    result = pipeline.run(collect=True)

    assert (result["close"] > 150).all()
    assert (result["volume"] > 1.5e6).all()


def test_pipeline_factors_and_universe():
    df = create_sample_data()

    returns = F.returns(window=1)
    universe = LiquidUniverse(min_price=150, min_volume=1.5e6)

    pipeline = Pipeline(df).add_factors({"returns": returns}).screen(universe)

    result = pipeline.run(collect=True)

    assert "returns" in result.columns
    assert len(result) < len(df.collect())


def test_pipeline_date_filters_with_factors():
    df = (
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
                "asset": ["A"] * 3 + ["B"] * 3,
                "close": [100.0, 105.0, 110.0, 200.0, 205.0, 210.0],
            }
        )
        .with_columns([pl.col("date").str.to_date()])
        .lazy()
    )

    returns = F.returns(window=1)

    pipeline = Pipeline(df).add_factors({"returns": returns})

    result = pipeline.run(start_date="2024-01-02", end_date="2024-01-02", collect=True)

    from datetime import date

    expected_date = date(2024, 1, 2)
    assert result["date"].unique().to_list() == [expected_date]


def test_pipeline_composed_factors():
    """Test that pipeline can compute composed factors (expression dependencies handled by Polars)."""
    df = create_sample_data()

    from factr.core import Scope

    close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
    returns = close.pct_change()  # TIME_SERIES scope
    volatility = returns.rolling_std(5)  # TIME_SERIES scope

    pipeline = Pipeline(df).add_factors({"vol": volatility})

    result = pipeline.run(collect=True)

    assert "vol" in result.columns


def test_pipeline_multiple_add_factors():
    """Test calling add_factors multiple times."""
    df = create_sample_data()

    returns = F.returns(window=1)
    volume = Factor(pl.col("volume"), name="volume")
    dollar_vol = F.dollar_volume(1)

    pipeline = (
        Pipeline(df)
        .add_factors({"returns": returns})
        .add_factors({"volume": volume})
        .add_factors({"dollar_vol": dollar_vol})
    )

    result = pipeline.run(collect=True)

    assert "returns" in result.columns
    assert "volume" in result.columns
    assert "dollar_vol" in result.columns


def test_pipeline_chaining():
    df = create_sample_data()

    df = df.with_columns([pl.col("date").str.to_date()])

    result = (
        Pipeline(df)
        .add_factors({"returns": F.returns(window=1), "momentum": F.momentum(window=5, skip=1)})
        .screen(LiquidUniverse(min_price=150, min_volume=1.5e6))
        .run(start_date="2024-01-01", collect=True)
    )

    assert "returns" in result.columns
    assert "momentum" in result.columns
    assert len(result) > 0


def test_pipeline_empty_factors():
    df = create_sample_data()

    pipeline = Pipeline(df)
    result = pipeline.run(collect=True)

    assert len(result) == len(df.collect())


def test_pipeline_cross_sectional_in_factors():
    """Test pipeline with cross-sectional factors (rank computed per-date via .over())."""
    df = create_sample_data()

    from factr.core import Scope

    close_factor = Factor(pl.col("close"), name="close", scope=Scope.RAW)
    ranked_close = close_factor.rank(pct=True)  # CROSS_SECTION scope

    pipeline = Pipeline(df).add_factors({"ranked_close": ranked_close})
    result = pipeline.run(collect=True)

    assert "ranked_close" in result.columns

    ranks = result["ranked_close"].drop_nulls()
    if len(ranks) > 0:
        assert ranks.min() >= 0
        assert ranks.max() <= 1


def test_screen_with_ts_then_cs_filter():
    """Test screening with a filter derived from TS → CS factor chain.

    Pattern: compute returns (TIME_SERIES), then rank them (CROSS_SECTION),
    then filter to top half of returns.

    Note: This test uses a simpler pattern because filters with CS intermediates
    have the same materialization limitation as CS→TS transitions.
    """
    from datetime import date

    # Create panel data with multiple dates for meaningful returns
    df = pl.DataFrame(
        {
            "date": [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 2),
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 1, 3),
                date(2020, 1, 3),
            ],
            "asset": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            # A: grows 10%, B: flat, C: drops 5%
            "close": [100.0, 100.0, 100.0, 110.0, 100.0, 95.0, 121.0, 100.0, 90.25],
        }
    ).lazy()

    from factr.core import Scope

    close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

    # Simple CS filter that doesn't require TS intermediate
    # Filter: keep only assets with above-median close price
    ranked_close = close.rank(pct=True)  # CROSS_SECTION
    top_half_filter = ranked_close > 0.3  # Keep top ~70%

    pipeline = Pipeline(df).add_factors({"ranked": ranked_close})
    pipeline.screen(top_half_filter)

    result = pipeline.run(collect=True)

    # Should filter out some rows
    assert len(result) < 9  # 3 assets * 3 dates = 9 original rows

    # The filtered rows should have higher ranks
    assert all(r > 0.3 for r in result["ranked"].to_list())


def test_screen_with_combined_filters():
    """Test screening with combined TS and CS filters."""
    from datetime import date

    df = pl.DataFrame(
        {
            "date": [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 2),
                date(2020, 1, 2),
            ],
            "asset": ["A", "B", "C", "A", "B", "C"],
            "close": [100.0, 200.0, 300.0, 110.0, 190.0, 310.0],
        }
    ).lazy()

    from factr.core import Scope

    close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

    # TS filter: price increasing (returns > 0)
    returns = close.pct_change(1)
    positive_returns = returns > 0

    # CS filter: not the lowest rank
    ranked = close.rank()
    not_lowest = ranked > 1

    # Combined filter (AND)
    combined_filter = positive_returns & not_lowest

    # The combined filter should be CROSS_SECTION (CS dominates)
    assert combined_filter.scope == Scope.CROSS_SECTION

    pipeline = Pipeline(df).screen(combined_filter)
    result = pipeline.run(collect=True)

    # Should have fewer rows than original
    assert len(result) < 6
