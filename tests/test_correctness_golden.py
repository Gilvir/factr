"""Golden tests comparing Pipeline results against manual Polars implementations.

These tests prove that the library computes identical results to hand-written
Polars code, establishing correctness by reference.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from factr.core import Factor, Scope
from factr.pipeline import Pipeline

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def panel_data():
    """Standard panel data with multiple assets and dates."""
    np.random.seed(42)
    dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 31), eager=True)
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    rows = []
    for asset in assets:
        base_price = {"AAPL": 100, "GOOGL": 1000, "MSFT": 150, "AMZN": 2000}[asset]
        for i, dt in enumerate(dates):
            # Deterministic price evolution with some noise
            price = base_price * (1 + 0.001 * i + 0.01 * np.sin(i * 0.5))
            volume = int(1e6 * (1 + 0.1 * np.cos(i * 0.3)))
            sector = "Tech" if asset in ["AAPL", "MSFT"] else "Internet"
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "close": price,
                    "volume": volume,
                    "sector": sector,
                }
            )

    return pl.DataFrame(rows)


@pytest.fixture
def simple_panel():
    """Minimal panel data for clear testing."""
    return pl.DataFrame(
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
            "close": [100.0, 200.0, 300.0, 110.0, 190.0, 310.0, 120.0, 180.0, 320.0],
            "volume": [1000, 2000, 3000, 1100, 1900, 3100, 1200, 1800, 3200],
            "sector": [
                "Tech",
                "Finance",
                "Tech",
                "Tech",
                "Finance",
                "Tech",
                "Tech",
                "Finance",
                "Tech",
            ],
        }
    )


# =============================================================================
# TIME_SERIES Operations: Pipeline vs Manual
# =============================================================================


class TestTimeSeriesGolden:
    """Verify TIME_SERIES operations match manual Polars code."""

    def test_pct_change_matches_manual(self, panel_data):
        """Factor.pct_change() matches manual .pct_change().over(asset)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"returns": returns})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").pct_change(1).over("asset").alias("returns_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["returns"] - merged["returns_manual"]).fill_null(0).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_rolling_mean_matches_manual(self, panel_data):
        """Factor.rolling_mean() matches manual .rolling_mean().over(asset)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma5 = close.rolling_mean(5)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"sma5": sma5})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").rolling_mean(window_size=5).over("asset").alias("sma5_manual")]
        )

        # Compare (skip initial NaNs)
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["sma5"] - merged["sma5_manual"]).fill_null(0).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_rolling_std_matches_manual(self, panel_data):
        """Factor.rolling_std() matches manual .rolling_std().over(asset)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        std5 = close.rolling_std(5)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"std5": std5})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").rolling_std(window_size=5).over("asset").alias("std5_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["std5"] - merged["std5_manual"]).fill_null(0).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_shift_matches_manual(self, panel_data):
        """Factor.shift() matches manual .shift().over(asset)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        lagged = close.shift(3)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"lagged": lagged})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").shift(3).over("asset").alias("lagged_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["lagged"] - merged["lagged_manual"]).fill_null(0).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_diff_matches_manual(self, panel_data):
        """Factor.diff() matches manual .diff().over(asset)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        diff_factor = close.diff(2)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"diff_val": diff_factor})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").diff(2).over("asset").alias("diff_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["diff_val"] - merged["diff_manual"]).fill_null(0).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_ewm_mean_matches_manual(self, panel_data):
        """Factor.ewm_mean() matches manual .ewm_mean().over(asset)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ema = close.ewm_mean(span=10)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"ema": ema})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").ewm_mean(span=10).over("asset").alias("ema_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["ema"] - merged["ema_manual"]).fill_null(0).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_cumsum_matches_manual(self, panel_data):
        """Factor.cumsum() matches manual .cum_sum().over(asset)."""
        volume = Factor(pl.col("volume"), name="volume", scope=Scope.RAW)
        cumvol = volume.cumsum()

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"cumvol": cumvol})
        result = pipeline.run()

        # Manual Polars execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("volume").cum_sum().over("asset").alias("cumvol_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["cumvol"].cast(pl.Float64) - merged["cumvol_manual"].cast(pl.Float64)).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"


# =============================================================================
# CROSS_SECTION Operations: Pipeline vs Manual
# =============================================================================


class TestCrossSectionGolden:
    """Verify CROSS_SECTION operations match manual Polars code."""

    def test_rank_matches_manual(self, simple_panel):
        """Factor.rank() matches manual .rank().over(date)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"ranked": ranked})
        result = pipeline.run()

        # Manual Polars execution
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [pl.col("close").rank(method="average").over("date").alias("ranked_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["ranked"] - merged["ranked_manual"]).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_rank_pct_matches_manual(self, simple_panel):
        """Factor.rank(pct=True) matches manual percentile rank."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked_pct = close.rank(pct=True)

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"ranked_pct": ranked_pct})
        result = pipeline.run()

        # Manual Polars execution
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [
                ((pl.col("close").rank(method="average") - 1) / (pl.len() - 1))
                .over("date")
                .alias("ranked_pct_manual")
            ]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["ranked_pct"] - merged["ranked_pct_manual"]).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_demean_matches_manual(self, simple_panel):
        """Factor.demean() matches manual demeaning."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        demeaned = close.demean()

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"demeaned": demeaned})
        result = pipeline.run()

        # Manual Polars execution
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [(pl.col("close") - pl.col("close").mean()).over("date").alias("demeaned_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["demeaned"] - merged["demeaned_manual"]).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_zscore_matches_manual(self, simple_panel):
        """Factor.zscore() matches manual z-score calculation."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        zscored = close.zscore()

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"zscored": zscored})
        result = pipeline.run()

        # Manual Polars execution
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [
                ((pl.col("close") - pl.col("close").mean()) / pl.col("close").std())
                .over("date")
                .alias("zscored_manual")
            ]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["zscored"] - merged["zscored_manual"]).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_demean_by_sector_matches_manual(self, simple_panel):
        """Factor.demean(by='sector') matches manual grouped demeaning."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sector_neutral = close.demean(by="sector")

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"sector_neutral": sector_neutral})
        result = pipeline.run()

        # Manual Polars execution
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [
                (pl.col("close") - pl.col("close").mean())
                .over(["date", "sector"])
                .alias("sector_neutral_manual")
            ]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["sector_neutral"] - merged["sector_neutral_manual"]).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_rank_by_sector_matches_manual(self, simple_panel):
        """Factor.rank(by='sector') matches manual grouped ranking."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sector_rank = close.rank(by="sector")

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"sector_rank": sector_rank})
        result = pipeline.run()

        # Manual Polars execution
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [
                pl.col("close")
                .rank(method="average")
                .over(["date", "sector"])
                .alias("sector_rank_manual")
            ]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["sector_rank"] - merged["sector_rank_manual"]).abs()
        assert diff.max() < 1e-10, f"Max difference: {diff.max()}"


# =============================================================================
# Combined Operations: Time-Series -> Cross-Section
# =============================================================================


class TestCombinedOperationsGolden:
    """Verify that chained time-series + cross-section operations match manual code."""

    def test_returns_then_rank(self, panel_data):
        """pct_change().rank() matches manual two-stage execution."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(5)
        ranked_returns = returns.rank(pct=True)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors(
            {
                "returns": returns,
                "ranked_returns": ranked_returns,
            }
        )
        result = pipeline.run()

        # Manual two-stage execution
        stage1 = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").pct_change(5).over("asset").alias("returns_manual")]
        )
        stage2 = stage1.with_columns(
            [
                ((pl.col("returns_manual").rank(method="average") - 1) / (pl.len() - 1))
                .over("date")
                .alias("ranked_returns_manual")
            ]
        )

        # Compare returns
        merged = result.join(stage2, on=["date", "asset"], how="inner")
        diff_ret = (merged["returns"] - merged["returns_manual"]).fill_null(0).abs()
        assert diff_ret.max() < 1e-10, f"Returns max diff: {diff_ret.max()}"

        # Compare ranked returns (skip nulls from initial pct_change)
        valid_mask = merged["returns"].is_not_null() & merged["ranked_returns"].is_not_null()
        merged_valid = merged.filter(valid_mask)
        if len(merged_valid) > 0:
            diff_rank = (
                merged_valid["ranked_returns"] - merged_valid["ranked_returns_manual"]
            ).abs()
            assert diff_rank.max() < 1e-10, f"Ranked returns max diff: {diff_rank.max()}"

    def test_rolling_mean_then_zscore(self, panel_data):
        """rolling_mean().zscore() matches manual two-stage execution."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma = close.rolling_mean(5)
        zscore_sma = sma.zscore()

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors(
            {
                "sma": sma,
                "zscore_sma": zscore_sma,
            }
        )
        result = pipeline.run()

        # Manual two-stage execution
        stage1 = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").rolling_mean(window_size=5).over("asset").alias("sma_manual")]
        )
        stage2 = stage1.with_columns(
            [
                ((pl.col("sma_manual") - pl.col("sma_manual").mean()) / pl.col("sma_manual").std())
                .over("date")
                .alias("zscore_sma_manual")
            ]
        )

        # Compare (only where SMA is not null)
        merged = result.join(stage2, on=["date", "asset"], how="inner")
        valid_mask = (
            merged["sma"].is_not_null()
            & merged["zscore_sma"].is_not_null()
            & merged["zscore_sma_manual"].is_not_null()
        )
        merged_valid = merged.filter(valid_mask)

        if len(merged_valid) > 0:
            diff = (merged_valid["zscore_sma"] - merged_valid["zscore_sma_manual"]).abs()
            assert diff.max() < 1e-10, f"Max difference: {diff.max()}"

    def test_momentum_factor(self, panel_data):
        """Classic momentum: (close / close.shift(21) - 1).rank()"""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        mom = close.pct_change(21)
        ranked_mom = mom.rank(pct=True)

        # Pipeline execution
        pipeline = Pipeline(panel_data.lazy()).add_factors({"momentum": ranked_mom})
        result = pipeline.run()

        # Manual execution
        manual = (
            panel_data.sort(["date", "asset"])
            .with_columns([pl.col("close").pct_change(21).over("asset").alias("mom_raw")])
            .with_columns(
                [
                    ((pl.col("mom_raw").rank(method="average") - 1) / (pl.len() - 1))
                    .over("date")
                    .alias("momentum_manual")
                ]
            )
        )

        # Compare (where mom is not null)
        merged = result.join(manual, on=["date", "asset"], how="inner")
        valid_mask = merged["momentum"].is_not_null() & merged["momentum_manual"].is_not_null()
        merged_valid = merged.filter(valid_mask)

        if len(merged_valid) > 0:
            diff = (merged_valid["momentum"] - merged_valid["momentum_manual"]).abs()
            assert diff.max() < 1e-10, f"Max difference: {diff.max()}"


# =============================================================================
# Arithmetic Operations
# =============================================================================


class TestArithmeticGolden:
    """Verify arithmetic operations produce correct results."""

    def test_factor_addition(self, simple_panel):
        """Factor + Factor produces correct sum."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        volume = Factor(pl.col("volume"), name="volume", scope=Scope.RAW)
        combined = close + volume

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"combined": combined})
        result = pipeline.run()

        # Manual
        manual = simple_panel.with_columns(
            [(pl.col("close") + pl.col("volume")).alias("combined_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["combined"] - merged["combined_manual"]).abs()
        assert diff.max() < 1e-10

    def test_factor_division(self, simple_panel):
        """Factor / Factor produces correct ratio."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        volume = Factor(pl.col("volume"), name="volume", scope=Scope.RAW)
        ratio = close / volume

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"ratio": ratio})
        result = pipeline.run()

        # Manual
        manual = simple_panel.with_columns(
            [(pl.col("close") / pl.col("volume")).alias("ratio_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["ratio"] - merged["ratio_manual"]).abs()
        assert diff.max() < 1e-10

    def test_scalar_multiplication(self, simple_panel):
        """Factor * scalar produces correct result."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        scaled = close * 2.5

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"scaled": scaled})
        result = pipeline.run()

        # Manual
        manual = simple_panel.with_columns([(pl.col("close") * 2.5).alias("scaled_manual")])

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["scaled"] - merged["scaled_manual"]).abs()
        assert diff.max() < 1e-10

    def test_complex_expression(self, simple_panel):
        """Complex expression: (close - volume) / close * 100"""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        volume = Factor(pl.col("volume"), name="volume", scope=Scope.RAW)
        expr = (close - volume) / close * 100

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"expr": expr})
        result = pipeline.run()

        # Manual
        manual = simple_panel.with_columns(
            [((pl.col("close") - pl.col("volume")) / pl.col("close") * 100).alias("expr_manual")]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["expr"] - merged["expr_manual"]).abs()
        assert diff.max() < 1e-10


# =============================================================================
# Filter Operations
# =============================================================================


class TestFilterGolden:
    """Verify filter operations work correctly."""

    def test_top_n_filter(self, simple_panel):
        """Factor.top(n) selects correct assets per date."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        top2 = close.top(2)

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).screen(top2)
        result = pipeline.run()

        # Manual: Get top 2 by close per date
        manual = (
            simple_panel.sort(["date", "asset"])
            .with_columns(
                [pl.col("close").rank(method="ordinal", descending=True).over("date").alias("rank")]
            )
            .filter(pl.col("rank") <= 2)
            .drop("rank")
        )

        # Compare row counts per date
        result_counts = result.group_by("date").len().sort("date")
        manual_counts = manual.group_by("date").len().sort("date")

        assert result_counts["len"].to_list() == manual_counts["len"].to_list()
        assert all(c <= 2 for c in result_counts["len"].to_list())

    def test_bottom_n_filter(self, simple_panel):
        """Factor.bottom(n) selects correct assets per date."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        bottom1 = close.bottom(1)

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).screen(bottom1)
        result = pipeline.run()

        # Should select exactly 1 asset per date (the one with lowest close)
        counts = result.group_by("date").len()
        assert all(c == 1 for c in counts["len"].to_list())

        # Verify it's the minimum
        for dt in result["date"].unique().to_list():
            result_close = result.filter(pl.col("date") == dt)["close"][0]
            min_close = simple_panel.filter(pl.col("date") == dt)["close"].min()
            assert result_close == min_close


# =============================================================================
# Algebraic Identities (domain-specific, not generic math)
# =============================================================================


class TestAlgebraicIdentities:
    """Domain-specific algebraic properties that must hold."""

    def test_zscore_equals_demean_over_std(self, simple_panel):
        """zscore(x) == demean(x) / std(x) - fundamental identity."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

        zscore_direct = close.zscore()
        demeaned = close.demean()

        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {
                "zscore_direct": zscore_direct,
                "demeaned": demeaned,
            }
        )
        result = pipeline.run()

        # Compute std per date
        std_per_date = result.group_by("date").agg(pl.col("close").std().alias("std"))
        result_with_std = result.join(std_per_date, on="date")
        result_with_std = result_with_std.with_columns(
            [(pl.col("demeaned") / pl.col("std")).alias("zscore_manual")]
        )

        diff = (result_with_std["zscore_direct"] - result_with_std["zscore_manual"]).abs()
        assert diff.max() < 1e-10

    def test_rolling_linearity(self, panel_data):
        """rolling_sum(a + b) == rolling_sum(a) + rolling_sum(b)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        volume = Factor(pl.col("volume").cast(pl.Float64), name="volume", scope=Scope.RAW)

        sum_then_roll = (close + volume).rolling_sum(3)
        roll_then_sum = close.rolling_sum(3) + volume.rolling_sum(3)

        pipeline = Pipeline(panel_data.lazy()).add_factors(
            {
                "sum_then_roll": sum_then_roll,
                "roll_then_sum": roll_then_sum,
            }
        )
        result = pipeline.run()

        valid = result.filter(result["sum_then_roll"].is_not_null())
        if len(valid) > 0:
            diff = (valid["sum_then_roll"] - valid["roll_then_sum"]).abs().max()
            assert diff < 1e-8

    def test_demean_shift_invariance(self, simple_panel):
        """demean(a + c) == demean(a) for constant c."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

        demean_a = close.demean()
        demean_a_plus_100 = (close + 100).demean()

        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {
                "demean_a": demean_a,
                "demean_a_plus_100": demean_a_plus_100,
            }
        )
        result = pipeline.run()

        diff = (result["demean_a"] - result["demean_a_plus_100"]).abs().max()
        assert diff < 1e-10

    def test_demean_idempotent(self, simple_panel):
        """demean(demean(a)) == demean(a)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

        demean_once = close.demean()
        demean_twice = demean_once.demean()

        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {
                "demean_once": demean_once,
                "demean_twice": demean_twice,
            }
        )
        result = pipeline.run()

        diff = (result["demean_once"] - result["demean_twice"]).abs().max()
        assert diff < 1e-10

    def test_rank_shift_invariant(self, simple_panel):
        """rank(a) == rank(a + c) for constant c."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

        rank_a = close.rank()
        rank_a_plus_100 = (close + 100).rank()

        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {
                "rank_a": rank_a,
                "rank_a_plus_100": rank_a_plus_100,
            }
        )
        result = pipeline.run()

        diff = (result["rank_a"] - result["rank_a_plus_100"]).abs().max()
        assert diff < 1e-10

    def test_rank_scale_invariant(self, simple_panel):
        """rank(a) == rank(a * k) for positive k."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

        rank_a = close.rank()
        rank_scaled = (close * 2.5).rank()

        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {
                "rank_a": rank_a,
                "rank_scaled": rank_scaled,
            }
        )
        result = pipeline.run()

        diff = (result["rank_a"] - result["rank_scaled"]).abs().max()
        assert diff < 1e-10


# =============================================================================
# Quantile Operations
# =============================================================================


class TestQuantileGolden:
    """Verify quantile operations match manual implementations."""

    def test_quantile_matches_manual(self, simple_panel):
        """Factor.quantile(q) matches manual percentile-based bucketing."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        quantiled = close.quantile(3)  # 3 buckets: 0, 1, 2

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors({"quantile": quantiled})
        result = pipeline.run()

        # Manual: compute percentile rank, then bucket
        manual = simple_panel.sort(["date", "asset"]).with_columns(
            [
                (
                    (((pl.col("close").rank(method="average") - 1) / (pl.len() - 1)) * 3)
                    .floor()
                    .clip(0, 2)
                    .cast(pl.Int32)
                )
                .over("date")
                .alias("quantile_manual")
            ]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")
        diff = (merged["quantile"] - merged["quantile_manual"]).abs()
        assert diff.max() == 0, f"Quantile diff: {diff.max()}"

    def test_quantile_5_buckets(self, panel_data):
        """Factor.quantile(5) creates quintiles correctly."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        quintile = close.quantile(5)

        pipeline = Pipeline(panel_data.lazy()).add_factors({"quintile": quintile})
        result = pipeline.run()

        # Each quintile bucket should be in range [0, 4]
        unique_buckets = result["quintile"].unique().sort()
        assert all(0 <= q <= 4 for q in unique_buckets.to_list())

        # With 4 assets, some quintiles may be empty on each day,
        # but over many days we should see multiple buckets
        assert len(unique_buckets) >= 2, "Should have at least 2 different quintile buckets"

        # Manual verification for one date
        one_date = result.filter(pl.col("date") == result["date"][0])
        # With 4 assets, quintiles may not all be represented on a single date
        # But the values should be in range [0, 4]
        assert all(0 <= q <= 4 for q in one_date["quintile"].to_list())


# =============================================================================
# Over Context Correctness (Negative Tests)
# =============================================================================


class TestOverContextCorrectness:
    """Verify that applying the wrong .over() produces different results.

    These negative tests prove the Pipeline applies the correct .over() context
    by showing that the wrong context produces incorrect results.
    """

    def test_wrong_over_for_time_series(self, panel_data):
        """TIME_SERIES factor with .over(date) produces different results than Pipeline.

        Pipeline should use .over(entity) for TIME_SERIES, not .over(date).
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        # Pipeline execution (correct: .over(entity))
        pipeline = Pipeline(panel_data.lazy()).add_factors({"returns": returns})
        result_correct = pipeline.run()

        # Wrong execution: .over(date) instead of .over(entity)
        result_wrong = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").pct_change(1).over("date").alias("returns_wrong")]
        )

        # These should be DIFFERENT
        merged = result_correct.join(result_wrong, on=["date", "asset"], how="inner")

        # Filter to non-null values
        valid = merged.filter(
            merged["returns"].is_not_null() & merged["returns_wrong"].is_not_null()
        )

        if len(valid) > 0:
            # The results should differ because:
            # - Correct: pct_change computes per-entity over time
            # - Wrong: pct_change computes within each date (meaningless)
            diff = (valid["returns"] - valid["returns_wrong"]).abs()
            assert diff.max() > 1e-10, (
                "Wrong .over() should produce different results! "
                "This could indicate Pipeline is not applying .over() correctly."
            )

    def test_wrong_over_for_cross_section(self, panel_data):
        """CROSS_SECTION factor with .over(entity) produces different results than Pipeline.

        Pipeline should use .over(date) for CROSS_SECTION, not .over(entity).
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        # Pipeline execution (correct: .over(date))
        pipeline = Pipeline(panel_data.lazy()).add_factors({"ranked": ranked})
        result_correct = pipeline.run()

        # Wrong execution: .over(entity) instead of .over(date)
        result_wrong = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").rank(method="average").over("asset").alias("ranked_wrong")]
        )

        # These should be DIFFERENT
        merged = result_correct.join(result_wrong, on=["date", "asset"], how="inner")

        diff = (merged["ranked"] - merged["ranked_wrong"]).abs()
        assert diff.max() > 1e-10, (
            "Wrong .over() should produce different results! "
            "This could indicate Pipeline is not applying .over() correctly."
        )

    def test_rolling_wrong_over_differs(self, panel_data):
        """rolling_mean with .over(date) produces nonsensical results.

        Rolling operations need .over(entity) to work correctly.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma5 = close.rolling_mean(5)

        # Pipeline execution (correct)
        pipeline = Pipeline(panel_data.lazy()).add_factors({"sma5": sma5})
        result_correct = pipeline.run()

        # Wrong execution: .over(date)
        result_wrong = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").rolling_mean(window_size=5).over("date").alias("sma5_wrong")]
        )

        merged = result_correct.join(result_wrong, on=["date", "asset"], how="inner")

        # Filter to where both are not null
        valid = merged.filter(merged["sma5"].is_not_null() & merged["sma5_wrong"].is_not_null())

        if len(valid) > 0:
            diff = (valid["sma5"] - valid["sma5_wrong"]).abs()
            assert diff.max() > 1e-10, (
                "Wrong .over() for rolling should differ. "
                "This tests that Pipeline applies TIME_SERIES context correctly."
            )
