"""Edge case matrix tests.

Systematic coverage of boundary conditions and degenerate cases:
- Single entity
- Single date
- All NaN columns
- Ties in rank
- Empty groups
- Extreme values
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from factr.core import Factor, Scope
from factr.pipeline import Pipeline

# =============================================================================
# Edge Case: Single Entity
# =============================================================================


class TestSingleEntity:
    """Test behavior with only one entity (asset)."""

    @pytest.fixture
    def single_entity_data(self):
        """Panel data with only one entity."""
        return pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
                "asset": ["ONLY"] * 10,
                "close": [100.0 + i for i in range(10)],
                "volume": [1000 + i * 100 for i in range(10)],
            }
        )

    def test_time_series_works_single_entity(self, single_entity_data):
        """Time-series operations should work with single entity."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma3 = close.rolling_mean(3)

        pipeline = Pipeline(single_entity_data.lazy()).add_factors({"sma3": sma3})
        result = pipeline.run()

        # Should have correct rolling mean
        assert result["sma3"][2] == (100 + 101 + 102) / 3
        assert result["sma3"][5] == (103 + 104 + 105) / 3

    def test_cross_section_degenerates_single_entity(self, single_entity_data):
        """Cross-sectional rank with one entity should always be 1."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(single_entity_data.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        # All ranks should be 1 (only one entity to rank)
        assert all(r == 1.0 for r in result["ranked"].to_list())

    def test_zscore_single_entity(self, single_entity_data):
        """Z-score with one entity should produce NaN (std=0)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        zscored = close.zscore()

        pipeline = Pipeline(single_entity_data.lazy()).add_factors({"zscored": zscored})
        result = pipeline.run()

        # With only one value per date, std=0, so zscore is NaN or inf
        # This is mathematically correct behavior
        zscore_vals = result["zscored"].to_list()
        for z in zscore_vals:
            assert z is None or np.isnan(z) or np.isinf(z), f"Expected NaN/inf/None, got {z}"

    def test_demean_single_entity(self, single_entity_data):
        """Demean with one entity should produce all zeros."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        demeaned = close.demean()

        pipeline = Pipeline(single_entity_data.lazy()).add_factors({"demeaned": demeaned})
        result = pipeline.run()

        # With one entity, value - mean(value) = 0
        assert all(abs(d) < 1e-10 for d in result["demeaned"].to_list())


# =============================================================================
# Edge Case: Single Date
# =============================================================================


class TestSingleDate:
    """Test behavior with only one date."""

    @pytest.fixture
    def single_date_data(self):
        """Panel data with only one date."""
        return pl.DataFrame(
            {
                "date": [date(2020, 1, 1)] * 5,
                "asset": ["A", "B", "C", "D", "E"],
                "close": [100.0, 200.0, 300.0, 400.0, 500.0],
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )

    def test_cross_section_works_single_date(self, single_date_data):
        """Cross-sectional operations should work with single date."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(single_date_data.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        # Should correctly rank 5 assets
        ranks = sorted(result["ranked"].to_list())
        assert ranks == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_time_series_all_null_single_date(self, single_date_data):
        """Time-series operations with single date should produce nulls."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        pipeline = Pipeline(single_date_data.lazy()).add_factors({"returns": returns})
        result = pipeline.run()

        # No previous date, so all pct_change should be null
        assert all(r is None for r in result["returns"].to_list())

    def test_zscore_single_date(self, single_date_data):
        """Z-score should work with single date, multiple entities."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        zscored = close.zscore()

        pipeline = Pipeline(single_date_data.lazy()).add_factors({"zscored": zscored})
        result = pipeline.run()

        # Should have mean=0, std=1
        mean = sum(result["zscored"].to_list()) / len(result)
        assert abs(mean) < 1e-10

        std = np.std(result["zscored"].to_list(), ddof=1)
        assert abs(std - 1.0) < 1e-10


# =============================================================================
# Edge Case: NaN Values
# =============================================================================


class TestNaNHandling:
    """Test behavior with NaN values."""

    @pytest.fixture
    def data_with_nans(self):
        """Panel data with NaN values."""
        return pl.DataFrame(
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
                "close": [100.0, np.nan, 300.0, 110.0, 200.0, np.nan],
            }
        )

    def test_rank_handles_nan(self, data_with_nans):
        """rank() should handle NaN values gracefully."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(data_with_nans.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        # NaN input should get NaN rank or be handled gracefully
        day1 = result.filter(pl.col("date") == date(2020, 1, 1))
        b_rank = day1.filter(pl.col("asset") == "B")["ranked"][0]
        # Polars may assign a rank to NaN or return null - either is acceptable
        assert b_rank is None or (isinstance(b_rank, float) and (np.isnan(b_rank) or b_rank >= 0))

    def test_demean_handles_nan(self, data_with_nans):
        """demean() should handle NaN gracefully (not crash)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        demeaned = close.demean()

        pipeline = Pipeline(data_with_nans.lazy()).add_factors({"demeaned": demeaned})
        result = pipeline.run()

        # Main assertion: the operation should not crash
        assert len(result) == 6

        # On day 2 (which has no NaN), verify demean works
        day2 = result.filter(pl.col("date") == date(2020, 1, 2))
        a_demeaned = day2.filter(pl.col("asset") == "A")["demeaned"][0]
        b_demeaned = day2.filter(pl.col("asset") == "B")["demeaned"][0]

        # These should be finite
        if a_demeaned is not None and b_demeaned is not None:
            if not np.isnan(a_demeaned) and not np.isnan(b_demeaned):
                # Sum should be ~0 for these two non-NaN values
                assert abs(a_demeaned + b_demeaned) < 1e-10

    def test_all_nan_column(self):
        """All-NaN column should not crash."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 1)],
                "asset": ["A", "B"],
                "close": [np.nan, np.nan],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(df.lazy()).add_factors({"ranked": ranked})
        # Should not raise
        result = pipeline.run()
        assert len(result) == 2


# =============================================================================
# Edge Case: Ties in Rank
# =============================================================================


class TestTiesInRank:
    """Test ranking behavior with tied values."""

    @pytest.fixture
    def data_with_ties(self):
        """Panel data with tied values."""
        return pl.DataFrame(
            {
                "date": [date(2020, 1, 1)] * 5,
                "asset": ["A", "B", "C", "D", "E"],
                "close": [100.0, 100.0, 200.0, 200.0, 300.0],  # Ties: A=B, C=D
            }
        )

    def test_rank_average_method(self, data_with_ties):
        """rank() with average method should assign average rank to ties."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()  # Default is average method

        pipeline = Pipeline(data_with_ties.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        # A and B tied at 100 -> rank should be (1+2)/2 = 1.5
        a_rank = result.filter(pl.col("asset") == "A")["ranked"][0]
        b_rank = result.filter(pl.col("asset") == "B")["ranked"][0]
        assert a_rank == b_rank == 1.5

        # C and D tied at 200 -> rank should be (3+4)/2 = 3.5
        c_rank = result.filter(pl.col("asset") == "C")["ranked"][0]
        d_rank = result.filter(pl.col("asset") == "D")["ranked"][0]
        assert c_rank == d_rank == 3.5

        # E at 300 -> rank 5
        e_rank = result.filter(pl.col("asset") == "E")["ranked"][0]
        assert e_rank == 5.0

    def test_rank_sum_invariant_with_ties(self, data_with_ties):
        """Sum of ranks should still equal n*(n+1)/2 even with ties."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(data_with_ties.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        rank_sum = result["ranked"].sum()
        n = len(result)
        expected = n * (n + 1) / 2

        assert abs(rank_sum - expected) < 1e-10


# =============================================================================
# Edge Case: Empty Groups
# =============================================================================


class TestEmptyGroups:
    """Test behavior with empty groups in grouped operations."""

    @pytest.fixture
    def data_sparse_sectors(self):
        """Panel data where some sectors are missing on some dates."""
        return pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 1),  # Day 1: Tech only
                    date(2020, 1, 2),
                    date(2020, 1, 2),
                    date(2020, 1, 2),  # Day 2: Both sectors
                ],
                "asset": ["A", "B", "A", "B", "C"],
                "close": [100.0, 110.0, 105.0, 115.0, 200.0],
                "sector": ["Tech", "Tech", "Tech", "Tech", "Finance"],
            }
        )

    def test_demean_by_sparse_sector(self, data_sparse_sectors):
        """demean(by='sector') should work with sparse sectors."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sector_neutral = close.demean(by="sector")

        pipeline = Pipeline(data_sparse_sectors.lazy()).add_factors(
            {"sector_neutral": sector_neutral}
        )
        result = pipeline.run()

        # Day 1: Only Tech sector
        day1_tech = result.filter(
            (pl.col("date") == date(2020, 1, 1)) & (pl.col("sector") == "Tech")
        )
        # Mean of Tech on day 1: (100 + 110) / 2 = 105
        # A: 100 - 105 = -5, B: 110 - 105 = 5
        a_val = day1_tech.filter(pl.col("asset") == "A")["sector_neutral"][0]
        b_val = day1_tech.filter(pl.col("asset") == "B")["sector_neutral"][0]
        assert abs(a_val - (-5.0)) < 1e-10
        assert abs(b_val - 5.0) < 1e-10

        # Day 2: Finance has only C
        day2_fin = result.filter(
            (pl.col("date") == date(2020, 1, 2)) & (pl.col("sector") == "Finance")
        )
        # C: 200 - 200 = 0 (only member of group)
        c_val = day2_fin["sector_neutral"][0]
        assert abs(c_val) < 1e-10


# =============================================================================
# Edge Case: Extreme Values
# =============================================================================


class TestExtremeValues:
    """Test behavior with extreme numerical values."""

    def test_large_values(self):
        """Operations should handle very large values."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)] * 3,
                "asset": ["A", "B", "C"],
                "close": [1e15, 2e15, 3e15],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()
        zscored = close.zscore()

        pipeline = Pipeline(df.lazy()).add_factors(
            {
                "ranked": ranked,
                "zscored": zscored,
            }
        )
        result = pipeline.run()

        # Rank should work
        ranks = sorted(result["ranked"].to_list())
        assert ranks == [1.0, 2.0, 3.0]

        # Zscore should have mean=0
        zscore_mean = sum(result["zscored"].to_list()) / 3
        assert abs(zscore_mean) < 1e-10

    def test_small_values(self):
        """Operations should handle very small values."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)] * 3,
                "asset": ["A", "B", "C"],
                "close": [1e-15, 2e-15, 3e-15],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(df.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        ranks = sorted(result["ranked"].to_list())
        assert ranks == [1.0, 2.0, 3.0]

    def test_mixed_extreme_values(self):
        """Operations should handle mixed extreme values."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)] * 4,
                "asset": ["A", "B", "C", "D"],
                "close": [1e-10, 1.0, 1e5, 1e10],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(df.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        # Should preserve ordering
        result_sorted = result.sort("ranked")
        assert result_sorted["asset"].to_list() == ["A", "B", "C", "D"]


# =============================================================================
# Edge Case: Zero and Negative Values
# =============================================================================


class TestZeroNegativeValues:
    """Test behavior with zero and negative values."""

    def test_negative_prices(self):
        """Operations should handle negative values (e.g., returns)."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)] * 3,
                "asset": ["A", "B", "C"],
                "returns": [-0.5, 0.0, 0.5],
            }
        )

        returns = Factor(pl.col("returns"), name="returns", scope=Scope.RAW)
        ranked = returns.rank()
        zscored = returns.zscore()

        pipeline = Pipeline(df.lazy()).add_factors(
            {
                "ranked": ranked,
                "zscored": zscored,
            }
        )
        result = pipeline.run()

        # Rank should work with negatives
        ranks = sorted(result["ranked"].to_list())
        assert ranks == [1.0, 2.0, 3.0]

        # Verify ordering: A (-0.5) < B (0) < C (0.5)
        assert result.filter(pl.col("asset") == "A")["ranked"][0] == 1.0
        assert result.filter(pl.col("asset") == "B")["ranked"][0] == 2.0
        assert result.filter(pl.col("asset") == "C")["ranked"][0] == 3.0

    def test_division_by_zero_handling(self):
        """pct_change from zero should produce inf, handled gracefully."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 2)],
                "asset": ["A", "A"],
                "close": [0.0, 100.0],  # 0 -> 100 is infinite pct change
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        pipeline = Pipeline(df.lazy()).add_factors({"returns": returns})
        result = pipeline.run()

        # Should not crash, returns inf
        ret_val = result.filter(pl.col("date") == date(2020, 1, 2))["returns"][0]
        assert np.isinf(ret_val) or ret_val is None or np.isnan(ret_val)


# =============================================================================
# Edge Case: Minimum Data Requirements
# =============================================================================


class TestMinimumData:
    """Test with minimum possible data."""

    def test_single_row(self):
        """Operations should handle single row."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)],
                "asset": ["A"],
                "close": [100.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(df.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        assert len(result) == 1
        assert result["ranked"][0] == 1.0

    def test_two_entities_two_dates(self):
        """Minimal panel: 2 assets, 2 dates."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)],
                "asset": ["A", "B", "A", "B"],
                "close": [100.0, 200.0, 110.0, 190.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        pipeline = Pipeline(df.lazy()).add_factors(
            {
                "returns": returns,
            }
        )
        result = pipeline.run()

        # Day 2: A returns 10%, B returns -5%
        day2 = result.filter(pl.col("date") == date(2020, 1, 2))
        a_returns = day2.filter(pl.col("asset") == "A")["returns"][0]
        b_returns = day2.filter(pl.col("asset") == "B")["returns"][0]

        # Verify returns are computed (may be null on day 1, but should be non-null on day 2)
        if a_returns is not None and b_returns is not None:
            # A: (110-100)/100 = 0.1, B: (190-200)/200 = -0.05
            assert abs(a_returns - 0.1) < 1e-10, f"A returns should be ~0.1, got {a_returns}"
            assert abs(b_returns - (-0.05)) < 1e-10, f"B returns should be ~-0.05, got {b_returns}"
            assert a_returns > b_returns, "A should have higher returns than B"


# =============================================================================
# Edge Case: Window Size Edge Cases
# =============================================================================


class TestWindowSizeEdgeCases:
    """Test rolling operations with edge case window sizes."""

    def test_window_size_one(self):
        """rolling_mean(1) should equal the value itself."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 5), eager=True),
                "asset": ["A"] * 5,
                "close": [100.0, 110.0, 120.0, 130.0, 140.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma1 = close.rolling_mean(1)

        pipeline = Pipeline(df.lazy()).add_factors(
            {
                "close": close,
                "sma1": sma1,
            }
        )
        result = pipeline.run()

        # SMA(1) should equal close
        diff = (result["close"] - result["sma1"]).abs().max()
        assert diff < 1e-10

    def test_window_larger_than_data(self):
        """rolling_mean with window > data length should produce mostly nulls."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 3), eager=True),
                "asset": ["A"] * 3,
                "close": [100.0, 110.0, 120.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma10 = close.rolling_mean(10)  # Window of 10, only 3 data points

        pipeline = Pipeline(df.lazy()).add_factors({"sma10": sma10})
        result = pipeline.run()

        # All values should be null (need 10 points for window)
        assert all(v is None for v in result["sma10"].to_list())
