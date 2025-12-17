"""Tests for custom_factor decorator."""

import polars as pl
import pytest

from factr import Pipeline, custom_factor
from factr.core import Factor, Scope

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.fixture
def sample_data():
    """Sample panel data for testing."""
    return pl.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"] * 2,
            "asset": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "close": [100.0, 200.0, 101.0, 202.0, 102.0, 204.0, 103.0, 206.0],
            "volume": [1000, 2000, 1100, 2100, 1200, 2200, 1300, 2300],
            "sector": ["tech", "finance", "tech", "finance"] * 2,
        }
    ).with_columns(pl.col("date").str.to_date())


class TestCustomFactorBasic:
    """Test basic custom_factor functionality."""

    def test_time_series_custom_factor(self, sample_data):
        """Test TIME_SERIES custom factor."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def simple_custom(df: pl.DataFrame) -> pl.Series:
            """Multiply close by 2."""
            return df["close"] * 2

        factor = simple_custom()
        assert isinstance(factor, Factor)
        assert factor.scope == Scope.TIME_SERIES
        assert factor.name == "simple_custom"
        assert factor.source_columns == frozenset(["close"])

        # Run through pipeline
        pipeline = Pipeline(sample_data.lazy()).add_factors({"custom": factor})
        result = pipeline.run()

        # Verify results - compare sorted since pipeline sorts by [asset, date]
        sorted_data = sample_data.sort(["asset", "date"])
        expected = sorted_data["close"] * 2
        assert result["custom"].to_list() == expected.to_list()

    def test_cross_section_custom_factor(self, sample_data):
        """Test CROSS_SECTION custom factor."""

        @custom_factor(scope=Scope.CROSS_SECTION, inputs=["close"])
        def rank_custom(df: pl.DataFrame) -> pl.Series:
            """Custom ranking logic using pure Polars."""
            # Use Polars for ranking
            return df["close"].rank()

        factor = rank_custom()
        assert factor.scope == Scope.CROSS_SECTION
        assert factor.groupby is None

        # Run through pipeline
        pipeline = Pipeline(sample_data.lazy()).add_factors({"custom_rank": factor})
        result = pipeline.run()

        # Verify it executed (exact values depend on cross-sectional grouping)
        assert "custom_rank" in result.columns

    def test_multiple_inputs(self, sample_data):
        """Test custom factor with multiple input columns."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close", "volume"])
        def vwap_proxy(df: pl.DataFrame) -> pl.Series:
            """Simple VWAP-like calculation."""
            return df["close"] * df["volume"]

        factor = vwap_proxy()
        assert factor.source_columns == frozenset(["close", "volume"])

        pipeline = Pipeline(sample_data.lazy()).add_factors({"vwap": factor})
        result = pipeline.run()

        # Compare with sorted data - pipeline sorts by [asset, date]
        sorted_data = sample_data.sort(["asset", "date"])
        expected = sorted_data["close"] * sorted_data["volume"]
        assert result["vwap"].to_list() == expected.to_list()

    def test_custom_name(self, sample_data):
        """Test custom output name."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"], output_name="custom_name")
        def my_func(df: pl.DataFrame) -> pl.Series:
            return df["close"]

        factor = my_func()
        assert factor.name == "custom_name"


class TestCustomFactorGroupby:
    """Test custom_factor with groupby."""

    def test_cross_section_with_groupby(self, sample_data):
        """Test CROSS_SECTION with groupby parameter."""

        @custom_factor(scope=Scope.CROSS_SECTION, inputs=["close"], groupby="sector")
        def sector_rank(df: pl.DataFrame) -> pl.Series:
            """Rank within sector using Polars."""
            return df["close"].rank()

        factor = sector_rank()
        assert factor.scope == Scope.CROSS_SECTION
        assert factor.groupby == ["sector"]

        pipeline = Pipeline(sample_data.lazy()).add_factors({"sector_rank": factor})
        result = pipeline.run()
        assert "sector_rank" in result.columns

    def test_cross_section_with_multiple_groupby(self, sample_data):
        """Test CROSS_SECTION with multiple groupby columns."""

        @custom_factor(scope=Scope.CROSS_SECTION, inputs=["close"], groupby=["sector", "asset"])
        def multi_group(df: pl.DataFrame) -> pl.Series:
            return df["close"]

        factor = multi_group()
        assert factor.groupby == ["sector", "asset"]

    def test_time_series_rejects_groupby(self):
        """Test that TIME_SERIES scope rejects groupby parameter."""
        with pytest.raises(ValueError, match="TIME_SERIES scope doesn't support groupby"):

            @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"], groupby="sector")
            def invalid(df: pl.DataFrame) -> pl.Series:
                return df["close"]


class TestCustomFactorValidation:
    """Test validation and error handling."""

    def test_invalid_scope(self):
        """Test that RAW scope is rejected."""
        with pytest.raises(ValueError, match="must have TIME_SERIES or CROSS_SECTION scope"):

            @custom_factor(scope=Scope.RAW, inputs=["close"])
            def invalid(df: pl.DataFrame) -> pl.Series:
                return df["close"]

    def test_empty_inputs(self):
        """Test that empty inputs are rejected."""
        with pytest.raises(ValueError, match="Must specify at least one input column"):

            @custom_factor(scope=Scope.TIME_SERIES, inputs=[])
            def invalid(df: pl.DataFrame) -> pl.Series:
                return pl.Series([])

    @pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
    def test_return_type_validation(self, sample_data):
        """Test that non-Series return types are handled."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def returns_numpy(df: pl.DataFrame):
            """Returns numpy array instead of Series."""
            return df["close"].to_numpy()

        factor = returns_numpy()
        pipeline = Pipeline(sample_data.lazy()).add_factors({"np_result": factor})
        result = pipeline.run()

        # Should auto-convert numpy array to Series
        assert "np_result" in result.columns

    def test_invalid_return_type(self, sample_data):
        """Test that invalid return types raise error."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def returns_invalid(df: pl.DataFrame) -> str:
            """Returns invalid type."""
            return "invalid"

        factor = returns_invalid()
        pipeline = Pipeline(sample_data.lazy()).add_factors({"invalid": factor})

        with pytest.raises(TypeError, match="must return pl.Series or array-like"):
            pipeline.run()


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
class TestCustomFactorWithNumpy:
    """Test integration with numpy."""

    def test_numpy_computation(self, sample_data):
        """Test custom factor using numpy."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def log_returns(df: pl.DataFrame) -> pl.Series:
            """Calculate log returns using numpy."""
            prices = df["close"].to_numpy()
            returns = np.diff(np.log(prices))
            # Pad with NaN to match length
            result = np.concatenate([[np.nan], returns])
            return pl.Series(result)

        factor = log_returns()
        pipeline = Pipeline(sample_data.lazy()).add_factors({"log_ret": factor})
        result = pipeline.run()

        assert "log_ret" in result.columns
        # Check that we have log returns computed
        log_ret_values = result["log_ret"]
        # Should have some NaN values (for first obs per entity) and some non-NaN values
        assert log_ret_values.is_nan().any()  # Has some NaN values
        assert (~log_ret_values.is_nan()).any()  # Has some non-NaN values

    def test_numpy_cross_sectional(self, sample_data):
        """Test cross-sectional custom factor with numpy."""

        @custom_factor(scope=Scope.CROSS_SECTION, inputs=["close"])
        def percentile_rank(df: pl.DataFrame) -> pl.Series:
            """Calculate percentile rank using numpy."""
            values = df["close"].to_numpy()
            ranks = np.searchsorted(np.sort(values), values)
            pct_ranks = ranks / (len(values) - 1) if len(values) > 1 else ranks
            return pl.Series(pct_ranks)

        factor = percentile_rank()
        pipeline = Pipeline(sample_data.lazy()).add_factors({"pct_rank": factor})
        result = pipeline.run()

        assert "pct_rank" in result.columns


class TestCustomFactorIntegration:
    """Test integration with existing Factor operations."""

    def test_composition_with_regular_factors(self, sample_data):
        """Test that custom factors compose with regular factors."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def custom_indicator(df: pl.DataFrame) -> pl.Series:
            return df["close"] * 1.1

        custom = custom_indicator()

        # Use custom factor in pipeline with regular factors
        from factr.datasets import EquityPricing

        close = EquityPricing.close
        returns = close.pct_change(1)

        pipeline = Pipeline(sample_data.lazy()).add_factors(
            {
                "custom": custom,
                "returns": returns,
            }
        )

        result = pipeline.run()
        assert "custom" in result.columns
        assert "returns" in result.columns

    def test_factor_objects_as_inputs(self, sample_data):
        """Test using Factor objects as inputs instead of strings."""
        from factr.datasets import EquityPricing

        # Use Factor objects as inputs
        @custom_factor(scope=Scope.TIME_SERIES, inputs=[EquityPricing.close, EquityPricing.volume])
        def vwap_custom(df: pl.DataFrame) -> pl.Series:
            """Calculate using Factor object inputs."""
            return df["close"] * df["volume"]

        factor = vwap_custom()

        # Check that source columns and datasets were tracked
        assert factor.source_columns == frozenset(["close", "volume"])
        assert len(factor.source_datasets) > 0

        # Run through pipeline
        pipeline = Pipeline(sample_data.lazy()).add_factors({"vwap": factor})
        result = pipeline.run()

        # Verify results - pipeline sorts by [asset, date]
        sorted_data = sample_data.sort(["asset", "date"])
        expected = sorted_data["close"] * sorted_data["volume"]
        assert result["vwap"].to_list() == expected.to_list()

    def test_mixed_factor_and_string_inputs(self, sample_data):
        """Test mixing Factor objects and string column names."""
        from factr.datasets import EquityPricing

        # Mix Factor objects and strings
        @custom_factor(
            scope=Scope.TIME_SERIES,
            inputs=[EquityPricing.close, "volume"],  # One Factor, one string
        )
        def mixed_inputs(df: pl.DataFrame) -> pl.Series:
            return df["close"] + df["volume"]

        factor = mixed_inputs()
        assert factor.source_columns == frozenset(["close", "volume"])

        pipeline = Pipeline(sample_data.lazy()).add_factors({"mixed": factor})
        result = pipeline.run()
        assert "mixed" in result.columns

    def test_source_columns_tracking(self):
        """Test that source columns are tracked correctly."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close", "volume", "open"])
        def multi_input(df: pl.DataFrame) -> pl.Series:
            return df["close"]

        factor = multi_input()
        assert factor.source_columns == frozenset(["close", "volume", "open"])


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
class TestCustomFactorRealWorld:
    """Test real-world use cases."""

    def test_talib_style_indicator(self, sample_data):
        """Simulate TA-Lib style indicator."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def simple_rsi(df: pl.DataFrame) -> pl.Series:
            """Simple RSI-like calculation (not real RSI)."""
            prices = df["close"].to_numpy()
            if len(prices) < 2:
                return pl.Series([50.0] * len(prices))

            changes = np.diff(prices)
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)

            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0

            if avg_loss == 0:
                rsi_value = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))

            # Return constant RSI for this entity
            return pl.Series([rsi_value] * len(prices))

        factor = simple_rsi()
        pipeline = Pipeline(sample_data.lazy()).add_factors({"rsi": factor})
        result = pipeline.run()

        assert "rsi" in result.columns
        # RSI should be between 0 and 100
        assert result["rsi"].min() >= 0
        assert result["rsi"].max() <= 100

    def test_scipy_style_computation(self, sample_data):
        """Simulate scipy-style statistical computation."""

        @custom_factor(scope=Scope.TIME_SERIES, inputs=["close"])
        def rolling_zscore(df: pl.DataFrame) -> pl.Series:
            """Calculate rolling z-score using numpy."""
            values = df["close"].to_numpy()
            if len(values) < 2:
                return pl.Series([0.0] * len(values))

            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return pl.Series([0.0] * len(values))

            zscore = (values[-1] - mean) / std
            # Return constant z-score for this entity
            return pl.Series([zscore] * len(values))

        factor = rolling_zscore()
        pipeline = Pipeline(sample_data.lazy()).add_factors({"zscore": factor})
        result = pipeline.run()

        assert "zscore" in result.columns
