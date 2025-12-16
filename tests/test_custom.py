"""Tests for custom factor decorators."""

import polars as pl

from factr.core import Factor, Scope
from factr.custom import (
    cross_section,
    expression_factor,
    factor_func,
    make_factor,
    rolling_factor,
    time_series,
)


def test_factor_func_decorator():
    """Test factor_func decorator."""

    @factor_func
    def custom_momentum(window: int = 20, price_col: str = "close") -> Factor:
        """Custom momentum factor."""
        close = Factor(pl.col(price_col), name=price_col)
        return close.pct_change().rolling_sum(window)

    # Create factor
    factor = custom_momentum(window=10)

    assert isinstance(factor, Factor)
    assert "custom_momentum" in factor.name or "rolling_sum" in factor.name


def test_rolling_factor_with_inputs():
    """Test rolling_factor decorator with inputs list."""

    @rolling_factor(window=20, inputs=["close", "volume"])
    def vol_weighted_return(close, volume):
        """Return weighted by volume."""
        ret = close.pct_change()
        weight = volume / volume.rolling_sum(window_size=20)
        return (ret * weight).rolling_sum(window_size=20)

    factor = vol_weighted_return()

    assert isinstance(factor, Factor)
    assert factor.name == "vol_weighted_return"


def test_rolling_factor_with_kwargs():
    """Test rolling_factor decorator with keyword column mappings."""

    @rolling_factor(window=20, price="close", vol="volume")
    def vol_weighted_return(price, vol):
        """Return weighted by volume."""
        ret = price.pct_change()
        weight = vol / vol.rolling_sum(window_size=20)
        return (ret * weight).rolling_sum(window_size=20)

    factor = vol_weighted_return()

    assert isinstance(factor, Factor)


def test_rolling_factor_column_override():
    """Test rolling_factor with column name overrides."""

    @rolling_factor(window=20, inputs=["close", "volume"])
    def vol_weighted_return(close, volume):
        ret = close.pct_change()
        weight = volume / volume.rolling_sum(window_size=20)
        return (ret * weight).rolling_sum(window_size=20)

    # Override column names
    factor = vol_weighted_return(close="adj_close", volume="adj_volume")

    assert isinstance(factor, Factor)


def test_expression_factor():
    """Test expression_factor decorator."""

    @expression_factor()
    def price_volume_ratio(price_col="close", volume_col="volume"):
        """Custom price-volume ratio."""
        return pl.col(price_col) / pl.col(volume_col).log()

    factor = price_volume_ratio()

    assert isinstance(factor, Factor)
    assert factor.name == "price_volume_ratio"


def test_expression_factor_with_name():
    """Test expression_factor with custom name."""

    @expression_factor(name="custom_ratio")
    def some_ratio(col1="a", col2="b"):
        return pl.col(col1) / pl.col(col2)

    factor = some_ratio()

    assert factor.name == "custom_ratio"


def test_make_factor_simple_expr():
    """Test make_factor with simple expression."""
    expr = pl.col("close") / pl.col("open")
    factor = make_factor(expr, name="price_ratio")

    assert isinstance(factor, Factor)
    assert factor.name == "price_ratio"


def test_make_factor_callable():
    """Test make_factor with callable expression."""

    def momentum_expr(price_col, window):
        return pl.col(price_col).pct_change(window)

    factory = make_factor(momentum_expr, name="momentum")

    # Factory should be callable
    assert callable(factory)

    # Call it to get a factor
    factor = factory(price_col="close", window=20)

    assert isinstance(factor, Factor)
    assert factor.name == "momentum"


def test_custom_factor_in_pipeline():
    """Test using custom factors in a pipeline."""
    from factr import Pipeline

    @factor_func
    def weighted_momentum(window: int = 20) -> Factor:
        close = Factor(pl.col("close"), name="close")
        volume = Factor(pl.col("volume"), name="volume")
        returns = close.pct_change()
        weights = volume / volume.rolling_sum(window)
        return (returns * weights).rolling_sum(window)

    # Create sample data
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000.0, 2000.0, 3000.0],
        }
    ).lazy()

    # Use in pipeline
    pipeline = Pipeline(df)
    factor = weighted_momentum(window=5)

    # This should work without errors
    pipeline.add_factors({"wm": factor})
    # Don't actually run since we don't have enough data for window


def test_factor_func_metadata():
    """Test that factor_func preserves function metadata."""

    @factor_func
    def my_factor(window: int = 20) -> Factor:
        """This is a custom factor.

        Args:
            window: Window size

        Returns:
            Factor object
        """
        close = Factor(pl.col("close"), name="close")
        return close.rolling_mean(window)

    # Check metadata is preserved
    assert my_factor.__name__ == "my_factor"
    assert "This is a custom factor" in my_factor.__doc__
    assert hasattr(my_factor, "__factor_func__")


def test_rolling_factor_metadata():
    """Test that rolling_factor adds metadata."""

    @rolling_factor(window=20, inputs=["close"])
    def sma(close):
        return close.rolling_mean(window_size=20)

    assert hasattr(sma, "__rolling_factor__")
    assert sma.__window__ == 20
    assert sma.__inputs__ == ["close"]


def test_complex_custom_factor():
    """Test a complex custom factor with multiple operations."""

    @factor_func
    def momentum_reversal_signal(
        mom_window: int = 252, rev_window: int = 5, price_col: str = "close"
    ) -> Factor:
        """Combines momentum and short-term reversal."""
        close = Factor(pl.col(price_col), name=price_col)

        # Long-term momentum
        momentum = close.pct_change(mom_window)

        # Short-term reversal
        reversal = -close.pct_change(rev_window)

        # Combine (simple average)
        signal = (momentum + reversal) / 2

        return signal

    factor = momentum_reversal_signal(mom_window=60, rev_window=3)

    assert isinstance(factor, Factor)


def test_expression_factor_with_params():
    """Test expression_factor with parameters."""

    @expression_factor()
    def scaled_value(col="value", scale=1.0):
        return pl.col(col) * scale

    factor = scaled_value(col="price", scale=100.0)

    # Test it actually works in a dataframe
    df = pl.DataFrame({"price": [1.0, 2.0, 3.0]}).lazy()

    result = df.with_columns([factor.expr.alias("scaled")]).collect()
    assert result["scaled"].to_list() == [100.0, 200.0, 300.0]


# ===== Scope-aware decorator tests (v2.0) =====


def test_time_series_decorator():
    """Test @time_series decorator sets TIME_SERIES scope."""

    @time_series
    def custom_momentum(window: int = 20) -> Factor:
        """Custom momentum calculation."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        return close.pct_change().rolling_sum(window)

    factor = custom_momentum(window=10)

    assert isinstance(factor, Factor)
    assert factor.scope == Scope.TIME_SERIES
    assert hasattr(custom_momentum, "__time_series__")


def test_time_series_decorator_overrides_scope():
    """Test @time_series decorator overrides incorrect scope."""

    @time_series
    def should_be_time_series(window: int = 20) -> Factor:
        # Return a factor with wrong scope
        close = Factor(pl.col("close"), name="close", scope=Scope.CROSS_SECTION)
        return close  # Wrong scope!

    factor = should_be_time_series(window=10)

    # Decorator should override to TIME_SERIES
    assert factor.scope == Scope.TIME_SERIES


def test_cross_section_decorator():
    """Test @cross_section decorator sets CROSS_SECTION scope."""

    @cross_section()
    def momentum_rank(window: int = 20) -> Factor:
        """Ranked momentum."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        momentum = close.pct_change().rolling_sum(window)
        return momentum.rank(pct=True)

    factor = momentum_rank(window=10)

    assert isinstance(factor, Factor)
    assert factor.scope == Scope.CROSS_SECTION
    assert hasattr(momentum_rank, "__cross_section__")
    assert factor.groupby is None


def test_cross_section_decorator_with_groupby():
    """Test @cross_section decorator with groupby parameter."""

    @cross_section(by="sector")
    def sector_neutral_rank(window: int = 20) -> Factor:
        """Sector-neutral ranked momentum."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        momentum = close.pct_change().rolling_sum(window)
        return momentum.rank(pct=True)

    factor = sector_neutral_rank(window=10)

    assert factor.scope == Scope.CROSS_SECTION
    assert factor.groupby == ["sector"]


def test_cross_section_decorator_with_multiple_groupby():
    """Test @cross_section decorator with multiple groupby columns."""

    @cross_section(by=["sector", "country"])
    def grouped_rank() -> Factor:
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        return close.rank(pct=True)

    factor = grouped_rank()

    assert factor.scope == Scope.CROSS_SECTION
    assert factor.groupby == ["sector", "country"]


def test_rolling_factor_has_time_series_scope():
    """Test @rolling_factor creates factors with TIME_SERIES scope."""

    @rolling_factor(window=20, inputs=["close", "volume"])
    def vol_weighted_return(close, volume):
        """Return weighted by volume."""
        ret = close.pct_change()
        weight = volume / volume.rolling_sum(window_size=20)
        return (ret * weight).rolling_sum(window_size=20)

    factor = vol_weighted_return()

    assert factor.scope == Scope.TIME_SERIES


def test_expression_factor_default_scope():
    """Test @expression_factor defaults to TIME_SERIES scope."""

    @expression_factor()
    def simple_ratio(price_col="close", volume_col="volume"):
        return pl.col(price_col) / pl.col(volume_col)

    factor = simple_ratio()

    assert factor.scope == Scope.TIME_SERIES


def test_expression_factor_custom_scope():
    """Test @expression_factor with custom scope parameter."""

    @expression_factor(scope=Scope.CROSS_SECTION)
    def ranked_ratio(price_col="close", volume_col="volume"):
        ratio = pl.col(price_col) / pl.col(volume_col)
        return ratio.rank()

    factor = ranked_ratio()

    assert factor.scope == Scope.CROSS_SECTION


def test_factor_func_scope_inference_from_inputs():
    """Test @factor_func infers scope from Factor inputs."""

    @factor_func
    def combine_factors(mom: Factor, vol: Factor) -> Factor:
        """Combine momentum and volatility."""
        return mom / vol

    # Create inputs with different scopes
    momentum = Factor(pl.col("close").pct_change(), name="mom", scope=Scope.TIME_SERIES)
    volatility = Factor(pl.col("close").rolling_std(20), name="vol", scope=Scope.TIME_SERIES)

    factor = combine_factors(momentum, volatility)

    # Should preserve TIME_SERIES
    assert factor.scope == Scope.TIME_SERIES


def test_factor_func_scope_inference_cross_section():
    """Test @factor_func infers CROSS_SECTION when any input is CROSS_SECTION."""

    @factor_func
    def combine_ts_and_cs(ts_factor: Factor, cs_factor: Factor) -> Factor:
        """Combine time-series and cross-sectional factors."""
        return ts_factor + cs_factor

    # Create inputs with different scopes
    ts = Factor(pl.col("close").pct_change(), name="ts", scope=Scope.TIME_SERIES)
    cs = Factor(pl.col("close").rank(), name="cs", scope=Scope.CROSS_SECTION)

    factor = combine_ts_and_cs(ts, cs)

    # Should infer CROSS_SECTION (conservative rule)
    assert factor.scope == Scope.CROSS_SECTION


def test_factor_func_preserves_explicit_scope():
    """Test @factor_func preserves explicitly set CROSS_SECTION scope."""

    @factor_func
    def ranked_momentum(window: int = 20) -> Factor:
        """Momentum with explicit rank."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        mom = close.pct_change().rolling_sum(window)
        # Explicitly ranked - should be CROSS_SECTION
        return mom.rank(pct=True)

    factor = ranked_momentum(window=10)

    # .rank() already set CROSS_SECTION, should be preserved
    assert factor.scope == Scope.CROSS_SECTION


def test_decorator_metadata_preservation():
    """Test that scope-aware decorators preserve function metadata."""

    @time_series
    def my_ts_factor(window: int = 20) -> Factor:
        """This is a time-series factor.

        Args:
            window: Window size

        Returns:
            Factor object
        """
        close = Factor(pl.col("close"), name="close")
        return close.rolling_mean(window)

    assert my_ts_factor.__name__ == "my_ts_factor"
    assert "time-series factor" in my_ts_factor.__doc__

    @cross_section(by="sector")
    def my_cs_factor() -> Factor:
        """Cross-sectional factor."""
        close = Factor(pl.col("close"), name="close")
        return close.rank()

    assert my_cs_factor.__name__ == "my_cs_factor"
    assert hasattr(my_cs_factor, "__groupby__")
    assert my_cs_factor.__groupby__ == "sector"
