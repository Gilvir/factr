"""Point-in-time correctness tests.

Critical for quant applications: ensure no look-ahead bias.
At date T, factor values should only use data from <= T.
"""

from datetime import date

import polars as pl
import pytest

from factr.core import Factor, Scope
from factr.pipeline import Pipeline

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sequential_panel():
    """Panel data with sequential, predictable values for easy verification."""
    dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True)
    assets = ["A", "B", "C"]

    rows = []
    for asset_idx, asset in enumerate(assets):
        for day_idx, dt in enumerate(dates):
            # Prices are sequential: A starts at 100, B at 200, C at 300
            # Each day increases by 1
            base = (asset_idx + 1) * 100
            price = base + day_idx
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "close": float(price),
                    "volume": 1000 * (day_idx + 1),
                }
            )

    return pl.DataFrame(rows)


@pytest.fixture
def panel_with_future_data():
    """Panel data where we'll test that future data doesn't leak into past calculations."""
    # First 5 days: normal prices
    # Day 6 onwards: extreme spike to detect look-ahead
    dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True)
    assets = ["A", "B"]

    rows = []
    for asset in assets:
        for day_idx, dt in enumerate(dates):
            if day_idx < 5:
                price = 100.0 + day_idx  # Normal: 100, 101, 102, 103, 104
            else:
                price = 1000.0 + day_idx  # Spike: 1005, 1006, ...

            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "close": price,
                }
            )

    return pl.DataFrame(rows)


# =============================================================================
# Test: Shift operations only access past data
# =============================================================================


class TestShiftPointInTime:
    """Verify shift() only accesses past data."""

    def test_shift_uses_only_past_data(self, sequential_panel):
        """shift(n) at date T should return value from T-n, not T+n."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        lagged = close.shift(2)

        pipeline = Pipeline(sequential_panel.lazy()).add_factors(
            {
                "close": close,
                "lagged": lagged,
            }
        )
        result = pipeline.run()

        for asset in result["asset"].unique().to_list():
            asset_data = result.filter(pl.col("asset") == asset).sort("date")
            closes = asset_data["close"].to_list()
            lagged_vals = asset_data["lagged"].to_list()

            # First 2 should be null
            assert lagged_vals[0] is None
            assert lagged_vals[1] is None

            # From index 2 onwards, lagged[i] should equal close[i-2]
            for i in range(2, len(closes)):
                assert lagged_vals[i] == closes[i - 2], (
                    f"Asset {asset}, index {i}: lagged={lagged_vals[i]}, expected={closes[i - 2]}"
                )

    def test_shift_never_sees_future(self, panel_with_future_data):
        """Verify shift at day 4 doesn't see the spike starting day 5."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        # Negative shift would look forward - but we don't support it
        # Positive shift looks backward
        lagged = close.shift(1)

        pipeline = Pipeline(panel_with_future_data.lazy()).add_factors({"lagged": lagged})
        result = pipeline.run()

        # On day 5 (index 4), lagged should be day 4's price (104), not day 6's
        day5_data = result.filter(pl.col("date") == date(2020, 1, 5))
        for row in day5_data.iter_rows(named=True):
            # Day 5's lagged value should be day 4's close = 103
            assert row["lagged"] == 103.0, f"Expected 103, got {row['lagged']}"


# =============================================================================
# Test: Rolling operations only access past data
# =============================================================================


class TestRollingPointInTime:
    """Verify rolling operations only use past and current data."""

    def test_rolling_mean_uses_only_past_data(self, sequential_panel):
        """rolling_mean(3) at T should use T, T-1, T-2 only."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma3 = close.rolling_mean(3)

        pipeline = Pipeline(sequential_panel.lazy()).add_factors(
            {
                "close": close,
                "sma3": sma3,
            }
        )
        result = pipeline.run()

        for asset in result["asset"].unique().to_list():
            asset_data = result.filter(pl.col("asset") == asset).sort("date")
            closes = asset_data["close"].to_list()
            sma_vals = asset_data["sma3"].to_list()

            # Check non-null values
            for i in range(2, len(closes)):
                expected = (closes[i] + closes[i - 1] + closes[i - 2]) / 3
                actual = sma_vals[i]
                assert abs(actual - expected) < 1e-10, (
                    f"Asset {asset}, index {i}: sma3={actual}, expected={expected}"
                )

    def test_rolling_window_never_includes_future(self, panel_with_future_data):
        """Rolling window at day 4 should not include spike data from day 5+."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma3 = close.rolling_mean(3)

        pipeline = Pipeline(panel_with_future_data.lazy()).add_factors({"sma3": sma3})
        result = pipeline.run()

        # Day 5 (index 4) SMA should be average of days 3, 4, 5 = (102 + 103 + 104) / 3 = 103
        day5_data = result.filter(pl.col("date") == date(2020, 1, 5))
        for row in day5_data.iter_rows(named=True):
            # If there was look-ahead, the SMA would be much higher due to spike
            expected = (102 + 103 + 104) / 3  # = 103
            assert abs(row["sma3"] - expected) < 1e-10, (
                f"Expected ~103, got {row['sma3']} - possible look-ahead detected!"
            )


# =============================================================================
# Test: pct_change only uses past data
# =============================================================================


class TestPctChangePointInTime:
    """Verify pct_change uses only past data."""

    def test_pct_change_uses_past(self, sequential_panel):
        """pct_change(n) at T computes (T - T-n) / T-n."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(2)

        pipeline = Pipeline(sequential_panel.lazy()).add_factors(
            {
                "close": close,
                "returns": returns,
            }
        )
        result = pipeline.run()

        for asset in result["asset"].unique().to_list():
            asset_data = result.filter(pl.col("asset") == asset).sort("date")
            closes = asset_data["close"].to_list()
            returns_vals = asset_data["returns"].to_list()

            for i in range(2, len(closes)):
                expected = (closes[i] - closes[i - 2]) / closes[i - 2]
                actual = returns_vals[i]
                assert abs(actual - expected) < 1e-10, (
                    f"Asset {asset}, index {i}: returns={actual}, expected={expected}"
                )


# =============================================================================
# Test: Cross-sectional ops are point-in-time correct
# =============================================================================


class TestCrossSectionPointInTime:
    """Verify cross-sectional operations are point-in-time correct."""

    def test_rank_uses_only_same_date(self, sequential_panel):
        """rank() at date T should only rank assets from date T."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        pipeline = Pipeline(sequential_panel.lazy()).add_factors(
            {
                "close": close,
                "ranked": ranked,
            }
        )
        result = pipeline.run()

        # For each date, verify ranking is based only on that date's values
        for dt in result["date"].unique().to_list():
            date_data = result.filter(pl.col("date") == dt).sort("close")

            # Asset C should always rank highest (300+), B middle (200+), A lowest (100+)
            rankings = date_data.select(["asset", "ranked"]).sort("ranked")
            assets_by_rank = rankings["asset"].to_list()

            assert assets_by_rank == ["A", "B", "C"], f"Wrong ranking on {dt}: {assets_by_rank}"

    def test_demean_uses_only_same_date(self, sequential_panel):
        """demean() at date T should use mean from date T only."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        demeaned = close.demean()

        pipeline = Pipeline(sequential_panel.lazy()).add_factors(
            {
                "close": close,
                "demeaned": demeaned,
            }
        )
        result = pipeline.run()

        # Verify demeaning is done per-date correctly
        for dt in result["date"].unique().to_list():
            date_data = result.filter(pl.col("date") == dt)
            closes = date_data["close"].to_list()
            demeaned_vals = date_data["demeaned"].to_list()

            date_mean = sum(closes) / len(closes)
            expected_demeaned = [c - date_mean for c in closes]

            for actual, expected in zip(demeaned_vals, expected_demeaned):
                assert abs(actual - expected) < 1e-10, (
                    f"Demean error on {date}: {actual} != {expected}"
                )


# =============================================================================
# Test: Adding future data doesn't change past calculations
# =============================================================================


class TestAddingFutureDataInvariant:
    """Verify that adding future data doesn't change past factor values."""

    def test_rolling_invariant_to_future(self):
        """Rolling calculations at T should be identical with or without T+1 data."""
        # Dataset without future
        df_short = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                    date(2020, 1, 4),
                    date(2020, 1, 5),
                ],
                "asset": ["A"] * 5,
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        # Dataset with future (extreme value)
        df_long = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                    date(2020, 1, 4),
                    date(2020, 1, 5),
                    date(2020, 1, 6),
                ],
                "asset": ["A"] * 6,
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 99999.0],  # Extreme future value
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma3 = close.rolling_mean(3)

        # Compute on short dataset
        pipeline_short = Pipeline(df_short.lazy()).add_factors({"sma3": sma3})
        result_short = pipeline_short.run()

        # Compute on long dataset
        pipeline_long = Pipeline(df_long.lazy()).add_factors({"sma3": sma3})
        result_long = pipeline_long.run()

        # Filter long to same dates as short
        result_long_filtered = result_long.filter(pl.col("date") <= date(2020, 1, 5))

        # Compare - should be identical
        short_sma = result_short["sma3"].to_list()
        long_sma = result_long_filtered["sma3"].to_list()

        for i, (s, long_val) in enumerate(zip(short_sma, long_sma)):
            if s is None and long_val is None:
                continue
            assert abs(s - long_val) < 1e-10, (
                f"Index {i}: short={s}, long={long_val} - future data affected past!"
            )

    def test_pct_change_invariant_to_future(self):
        """pct_change at T should be identical with or without T+1 data."""
        df_short = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
                "asset": ["A"] * 3,
                "close": [100.0, 110.0, 121.0],
            }
        )

        df_long = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
                "asset": ["A"] * 4,
                "close": [100.0, 110.0, 121.0, 0.001],  # Extreme future drop
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        pipeline_short = Pipeline(df_short.lazy()).add_factors({"returns": returns})
        result_short = pipeline_short.run()

        pipeline_long = Pipeline(df_long.lazy()).add_factors({"returns": returns})
        result_long = pipeline_long.run()

        result_long_filtered = result_long.filter(pl.col("date") <= date(2020, 1, 3))

        short_ret = result_short["returns"].to_list()
        long_ret = result_long_filtered["returns"].to_list()

        for i, (s, long_val) in enumerate(zip(short_ret, long_ret)):
            if s is None and long_val is None:
                continue
            assert abs(s - long_val) < 1e-10, (
                f"Index {i}: short={s}, long={long_val} - future data affected past!"
            )


# =============================================================================
# Test: Chained operations maintain point-in-time correctness
# =============================================================================


class TestChainedPointInTime:
    """Verify that chained time-series + cross-section operations are PIT correct."""

    def test_returns_then_rank_pit(self):
        """pct_change().rank() should be point-in-time correct."""
        df = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                    date(2020, 1, 3),
                ],
                "asset": ["A", "B", "A", "B", "A", "B"],
                "close": [100.0, 100.0, 110.0, 105.0, 115.0, 120.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked_returns = returns.rank()

        pipeline = Pipeline(df.lazy()).add_factors(
            {
                "returns": returns,
                "ranked_returns": ranked_returns,
            }
        )
        result = pipeline.run()

        # Day 2:
        # A returns: (110-100)/100 = 0.10
        # B returns: (105-100)/100 = 0.05
        # A should rank higher (2), B lower (1)
        day2 = result.filter(pl.col("date") == date(2020, 1, 2))
        a_rank = day2.filter(pl.col("asset") == "A")["ranked_returns"][0]
        b_rank = day2.filter(pl.col("asset") == "B")["ranked_returns"][0]

        # Skip if ranks are null (first day has no returns)
        if a_rank is not None and b_rank is not None:
            assert a_rank > b_rank, "A should rank higher than B on day 2"

        # Day 3:
        # A returns: (115-110)/110 = 0.0454...
        # B returns: (120-105)/105 = 0.1428...
        # B should rank higher
        day3 = result.filter(pl.col("date") == date(2020, 1, 3))
        a_rank = day3.filter(pl.col("asset") == "A")["ranked_returns"][0]
        b_rank = day3.filter(pl.col("asset") == "B")["ranked_returns"][0]

        if a_rank is not None and b_rank is not None:
            assert b_rank > a_rank, "B should rank higher than A on day 3"


# =============================================================================
# Test: Entity isolation (no cross-contamination between assets)
# =============================================================================


class TestEntityIsolation:
    """Verify time-series operations don't leak between entities."""

    def test_shift_isolated_per_entity(self):
        """shift() should not use data from other entities."""
        df = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 2),
                ],
                "asset": ["A", "B", "A", "B"],
                "close": [100.0, 999.0, 110.0, 888.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        lagged = close.shift(1)

        pipeline = Pipeline(df.lazy()).add_factors({"lagged": lagged})
        result = pipeline.run()

        # Day 2, Asset A should have lagged = 100 (not 999 from B)
        day2_a = result.filter((pl.col("date") == date(2020, 1, 2)) & (pl.col("asset") == "A"))
        assert day2_a["lagged"][0] == 100.0, "Asset A's lagged should be 100, not B's 999"

        # Day 2, Asset B should have lagged = 999 (not 100 from A)
        day2_b = result.filter((pl.col("date") == date(2020, 1, 2)) & (pl.col("asset") == "B"))
        assert day2_b["lagged"][0] == 999.0, "Asset B's lagged should be 999, not A's 100"

    def test_rolling_isolated_per_entity(self):
        """rolling operations should not mix data between entities."""
        df = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 2),
                    date(2020, 1, 3),
                    date(2020, 1, 3),
                ],
                "asset": ["A", "B", "A", "B", "A", "B"],
                "close": [10.0, 1000.0, 20.0, 2000.0, 30.0, 3000.0],
            }
        )

        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma2 = close.rolling_mean(2)

        pipeline = Pipeline(df.lazy()).add_factors({"sma2": sma2})
        result = pipeline.run()

        # Day 3, Asset A: SMA should be (20+30)/2 = 25
        day3_a = result.filter((pl.col("date") == date(2020, 1, 3)) & (pl.col("asset") == "A"))
        assert abs(day3_a["sma2"][0] - 25.0) < 1e-10, (
            f"Asset A SMA should be 25, got {day3_a['sma2'][0]}"
        )

        # Day 3, Asset B: SMA should be (2000+3000)/2 = 2500
        day3_b = result.filter((pl.col("date") == date(2020, 1, 3)) & (pl.col("asset") == "B"))
        assert abs(day3_b["sma2"][0] - 2500.0) < 1e-10, (
            f"Asset B SMA should be 2500, got {day3_b['sma2'][0]}"
        )
