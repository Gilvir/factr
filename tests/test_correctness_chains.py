"""Correctness tests for complex scope chains.

These tests verify that the Pipeline correctly handles:
1. CS → CS chains (e.g., rank().rank(), demean().rank())
2. TS after CS (e.g., rank().shift(), demean().rolling_mean())

These are secondary scope patterns that are less common but could cause
subtle bugs in advanced usage.
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
def simple_panel():
    """Simple panel data for clear testing."""
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
        }
    )


@pytest.fixture
def panel_with_more_dates():
    """Panel data with more dates for time-series operations after CS."""
    dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True)
    assets = ["A", "B", "C"]

    rows = []
    for asset_idx, asset in enumerate(assets):
        base = (asset_idx + 1) * 100  # A=100, B=200, C=300
        for day_idx, dt in enumerate(dates):
            # Add some variation so ranks change over time
            price = base + day_idx * (3 - asset_idx)  # A grows fastest
            rows.append({"date": dt, "asset": asset, "close": float(price)})

    return pl.DataFrame(rows)


# =============================================================================
# Test: Cross-Section → Cross-Section Chains
# =============================================================================


class TestCsToCsChains:
    """Verify CS → CS operation chains produce correct results."""

    def test_rank_then_rank(self, simple_panel):
        """rank().rank() - ranking the ranks.

        For 3 assets with distinct values, ranks are [1, 2, 3].
        Ranking [1, 2, 3] again should give [1, 2, 3] (idempotent for distinct values).
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()
        double_ranked = ranked.rank()

        # Verify scope propagation
        assert ranked.scope == Scope.CROSS_SECTION
        assert double_ranked.scope == Scope.CROSS_SECTION

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {"ranked": ranked, "double_ranked": double_ranked}
        )
        result = pipeline.run()

        # Manual two-stage Polars execution
        stage1 = simple_panel.sort(["date", "asset"]).with_columns(
            [pl.col("close").rank(method="average").over("date").alias("ranked_manual")]
        )
        stage2 = stage1.with_columns(
            [
                pl.col("ranked_manual")
                .rank(method="average")
                .over("date")
                .alias("double_ranked_manual")
            ]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        # First rank should match
        diff_ranked = (merged["ranked"] - merged["ranked_manual"]).abs()
        assert diff_ranked.max() < 1e-10, f"First rank diff: {diff_ranked.max()}"

        # Double rank should match
        diff_double = (merged["double_ranked"] - merged["double_ranked_manual"]).abs()
        assert diff_double.max() < 1e-10, f"Double rank diff: {diff_double.max()}"

        # For distinct values, rank of ranks equals original ranks
        diff_idempotent = (merged["ranked"] - merged["double_ranked"]).abs()
        assert diff_idempotent.max() < 1e-10, (
            "rank().rank() should be idempotent for distinct values"
        )

    def test_demean_then_rank(self, simple_panel):
        """demean().rank() - ranking demeaned values.

        Demeaning shifts values but preserves relative ordering,
        so rank(demean(x)) == rank(x).
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        demeaned = close.demean()
        ranked_demeaned = demeaned.rank()

        # Verify scopes
        assert demeaned.scope == Scope.CROSS_SECTION
        assert ranked_demeaned.scope == Scope.CROSS_SECTION

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {"demeaned": demeaned, "ranked_demeaned": ranked_demeaned}
        )
        result = pipeline.run()

        # Manual execution
        stage1 = simple_panel.sort(["date", "asset"]).with_columns(
            [(pl.col("close") - pl.col("close").mean()).over("date").alias("demeaned_manual")]
        )
        stage2 = stage1.with_columns(
            [
                pl.col("demeaned_manual")
                .rank(method="average")
                .over("date")
                .alias("ranked_demeaned_manual")
            ]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        diff_demeaned = (merged["demeaned"] - merged["demeaned_manual"]).abs()
        assert diff_demeaned.max() < 1e-10, f"Demean diff: {diff_demeaned.max()}"

        diff_ranked = (merged["ranked_demeaned"] - merged["ranked_demeaned_manual"]).abs()
        assert diff_ranked.max() < 1e-10, f"Ranked demeaned diff: {diff_ranked.max()}"

        # rank(demean(x)) == rank(x) because demean is shift-invariant
        direct_rank = simple_panel.sort(["date", "asset"]).with_columns(
            [pl.col("close").rank(method="average").over("date").alias("direct_rank")]
        )
        merged2 = result.join(direct_rank, on=["date", "asset"], how="inner")
        diff_invariant = (merged2["ranked_demeaned"] - merged2["direct_rank"]).abs()
        assert diff_invariant.max() < 1e-10, "rank(demean(x)) should equal rank(x)"

    def test_zscore_then_zscore(self, simple_panel):
        """zscore().zscore() - double standardization.

        zscore(zscore(x)) == zscore(x) because zscore already has mean=0, std=1.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        zscored = close.zscore()
        double_zscored = zscored.zscore()

        # Verify scopes
        assert zscored.scope == Scope.CROSS_SECTION
        assert double_zscored.scope == Scope.CROSS_SECTION

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {"zscored": zscored, "double_zscored": double_zscored}
        )
        result = pipeline.run()

        # Manual execution
        stage1 = simple_panel.sort(["date", "asset"]).with_columns(
            [
                ((pl.col("close") - pl.col("close").mean()) / pl.col("close").std())
                .over("date")
                .alias("zscored_manual")
            ]
        )
        stage2 = stage1.with_columns(
            [
                (
                    (pl.col("zscored_manual") - pl.col("zscored_manual").mean())
                    / pl.col("zscored_manual").std()
                )
                .over("date")
                .alias("double_zscored_manual")
            ]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        diff_zscored = (merged["zscored"] - merged["zscored_manual"]).abs()
        assert diff_zscored.max() < 1e-10, f"Zscore diff: {diff_zscored.max()}"

        diff_double = (merged["double_zscored"] - merged["double_zscored_manual"]).abs()
        assert diff_double.max() < 1e-10, f"Double zscore diff: {diff_double.max()}"

        # zscore is idempotent (already normalized)
        diff_idempotent = (merged["zscored"] - merged["double_zscored"]).abs()
        assert diff_idempotent.max() < 1e-10, "zscore().zscore() should be idempotent"

    def test_rank_pct_then_demean(self, simple_panel):
        """rank(pct=True).demean() - demeaning percentile ranks.

        Percentile ranks are in [0, 1], demeaning should center them around 0.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        pct_rank = close.rank(pct=True)
        demeaned_rank = pct_rank.demean()

        # Verify scopes
        assert pct_rank.scope == Scope.CROSS_SECTION
        assert demeaned_rank.scope == Scope.CROSS_SECTION

        # Pipeline execution
        pipeline = Pipeline(simple_panel.lazy()).add_factors(
            {"pct_rank": pct_rank, "demeaned_rank": demeaned_rank}
        )
        result = pipeline.run()

        # Manual execution
        stage1 = simple_panel.sort(["date", "asset"]).with_columns(
            [
                ((pl.col("close").rank(method="average") - 1) / (pl.len() - 1))
                .over("date")
                .alias("pct_rank_manual")
            ]
        )
        stage2 = stage1.with_columns(
            [
                (pl.col("pct_rank_manual") - pl.col("pct_rank_manual").mean())
                .over("date")
                .alias("demeaned_rank_manual")
            ]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        diff_pct = (merged["pct_rank"] - merged["pct_rank_manual"]).abs()
        assert diff_pct.max() < 1e-10, f"Pct rank diff: {diff_pct.max()}"

        diff_demeaned = (merged["demeaned_rank"] - merged["demeaned_rank_manual"]).abs()
        assert diff_demeaned.max() < 1e-10, f"Demeaned rank diff: {diff_demeaned.max()}"

        # Demeaned values should sum to zero per date
        date_sums = result.group_by("date").agg(pl.col("demeaned_rank").sum())
        for s in date_sums["demeaned_rank"]:
            assert abs(s) < 1e-10, "Demeaned ranks should sum to zero per date"


# =============================================================================
# Test: Time-Series After Cross-Section
# =============================================================================


class TestTsAfterCs:
    """Verify TS operations after CS operations produce correct results.

    This tests the pattern: rank() -> shift(), demean() -> rolling_mean(), etc.
    The CS operation must be computed first (with .over(date)), then the TS
    operation must be applied to the result (with .over(entity)).
    """

    def test_shift_after_rank(self, panel_with_more_dates):
        """rank().shift(1) - lagging cross-sectional ranks.

        This is a common pattern: compute today's rank, then look at yesterday's rank.
        The CS operation (rank) must be computed first, then the TS operation (shift)
        is applied to the materialized result.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()
        lagged_rank = ranked.shift(1)

        # Verify scopes - shift should produce TIME_SERIES
        assert ranked.scope == Scope.CROSS_SECTION
        assert lagged_rank.scope == Scope.TIME_SERIES

        # Pipeline execution
        pipeline = Pipeline(panel_with_more_dates.lazy()).add_factors(
            {"ranked": ranked, "lagged_rank": lagged_rank}
        )
        result = pipeline.run()

        # Manual two-stage execution:
        # Stage 1: Compute rank per date
        stage1 = panel_with_more_dates.sort(["date", "asset"]).with_columns(
            [pl.col("close").rank(method="average").over("date").alias("ranked_manual")]
        )
        # Stage 2: Lag the rank per entity
        stage2 = stage1.with_columns(
            [pl.col("ranked_manual").shift(1).over("asset").alias("lagged_rank_manual")]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        # Ranked should match
        diff_ranked = (merged["ranked"] - merged["ranked_manual"]).abs()
        assert diff_ranked.max() < 1e-10, f"Rank diff: {diff_ranked.max()}"

        # Lagged rank should match (accounting for nulls)
        valid_mask = (
            merged["lagged_rank"].is_not_null() & merged["lagged_rank_manual"].is_not_null()
        )
        merged_valid = merged.filter(valid_mask)

        if len(merged_valid) > 0:
            diff_lagged = (merged_valid["lagged_rank"] - merged_valid["lagged_rank_manual"]).abs()
            assert diff_lagged.max() < 1e-10, f"Lagged rank diff: {diff_lagged.max()}"

        # Verify lagged is actually shifted (not same-day value)
        for asset in result["asset"].unique().to_list():
            asset_data = result.filter(pl.col("asset") == asset).sort("date")
            ranks = asset_data["ranked"].to_list()
            lagged = asset_data["lagged_rank"].to_list()

            # First day should be null
            assert lagged[0] is None, f"First lagged_rank for {asset} should be null"

            # Subsequent days should be previous day's rank
            for i in range(1, len(ranks)):
                if lagged[i] is not None:
                    assert lagged[i] == ranks[i - 1], (
                        f"Asset {asset}, day {i}: lagged={lagged[i]}, expected={ranks[i - 1]}"
                    )

    def test_rolling_mean_after_demean(self, panel_with_more_dates):
        """demean().rolling_mean(3) - smoothing demeaned values over time.

        This pattern: demean cross-sectionally, then smooth time-series wise.
        The CS operation (demean) must be computed first, then the TS operation
        (rolling_mean) is applied to the materialized result.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        demeaned = close.demean()
        smoothed_demeaned = demeaned.rolling_mean(3)

        # Verify scopes
        assert demeaned.scope == Scope.CROSS_SECTION
        assert smoothed_demeaned.scope == Scope.TIME_SERIES

        # Pipeline execution
        pipeline = Pipeline(panel_with_more_dates.lazy()).add_factors(
            {"demeaned": demeaned, "smoothed_demeaned": smoothed_demeaned}
        )
        result = pipeline.run()

        # Manual two-stage execution:
        # Stage 1: Demean per date
        stage1 = panel_with_more_dates.sort(["date", "asset"]).with_columns(
            [(pl.col("close") - pl.col("close").mean()).over("date").alias("demeaned_manual")]
        )
        # Stage 2: Rolling mean per entity
        stage2 = stage1.with_columns(
            [
                pl.col("demeaned_manual")
                .rolling_mean(window_size=3)
                .over("asset")
                .alias("smoothed_demeaned_manual")
            ]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        # Demeaned should match
        diff_demeaned = (merged["demeaned"] - merged["demeaned_manual"]).abs()
        assert diff_demeaned.max() < 1e-10, f"Demean diff: {diff_demeaned.max()}"

        # Smoothed should match (accounting for nulls from rolling window)
        valid_mask = (
            merged["smoothed_demeaned"].is_not_null()
            & merged["smoothed_demeaned_manual"].is_not_null()
        )
        merged_valid = merged.filter(valid_mask)

        if len(merged_valid) > 0:
            diff_smoothed = (
                merged_valid["smoothed_demeaned"] - merged_valid["smoothed_demeaned_manual"]
            ).abs()
            assert diff_smoothed.max() < 1e-10, f"Smoothed diff: {diff_smoothed.max()}"

    def test_pct_change_after_rank(self, panel_with_more_dates):
        """rank().pct_change(1) - change in rank over time.

        Useful for detecting rank momentum (improving/declining position).
        The CS operation (rank) must be computed first, then the TS operation
        (pct_change) is applied to the materialized result.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()
        rank_change = ranked.pct_change(1)

        # Verify scopes
        assert ranked.scope == Scope.CROSS_SECTION
        assert rank_change.scope == Scope.TIME_SERIES

        # Pipeline execution
        pipeline = Pipeline(panel_with_more_dates.lazy()).add_factors(
            {"ranked": ranked, "rank_change": rank_change}
        )
        result = pipeline.run()

        # Manual two-stage execution
        stage1 = panel_with_more_dates.sort(["date", "asset"]).with_columns(
            [pl.col("close").rank(method="average").over("date").alias("ranked_manual")]
        )
        stage2 = stage1.with_columns(
            [pl.col("ranked_manual").pct_change(1).over("asset").alias("rank_change_manual")]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        valid_mask = (
            merged["rank_change"].is_not_null() & merged["rank_change_manual"].is_not_null()
        )
        merged_valid = merged.filter(valid_mask)

        if len(merged_valid) > 0:
            diff = (merged_valid["rank_change"] - merged_valid["rank_change_manual"]).abs()
            assert diff.max() < 1e-10, f"Rank change diff: {diff.max()}"

    def test_diff_after_zscore(self, panel_with_more_dates):
        """zscore().diff(1) - change in z-score over time.

        Measures how an asset's relative position changes.
        The CS operation (zscore) must be computed first, then the TS operation
        (diff) is applied to the materialized result.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        zscored = close.zscore()
        zscore_diff = zscored.diff(1)

        # Verify scopes
        assert zscored.scope == Scope.CROSS_SECTION
        assert zscore_diff.scope == Scope.TIME_SERIES

        # Pipeline execution
        pipeline = Pipeline(panel_with_more_dates.lazy()).add_factors(
            {"zscored": zscored, "zscore_diff": zscore_diff}
        )
        result = pipeline.run()

        # Manual execution
        stage1 = panel_with_more_dates.sort(["date", "asset"]).with_columns(
            [
                ((pl.col("close") - pl.col("close").mean()) / pl.col("close").std())
                .over("date")
                .alias("zscored_manual")
            ]
        )
        stage2 = stage1.with_columns(
            [pl.col("zscored_manual").diff(1).over("asset").alias("zscore_diff_manual")]
        )

        # Compare
        merged = result.join(stage2, on=["date", "asset"], how="inner")

        valid_mask = (
            merged["zscore_diff"].is_not_null() & merged["zscore_diff_manual"].is_not_null()
        )
        merged_valid = merged.filter(valid_mask)

        if len(merged_valid) > 0:
            diff = (merged_valid["zscore_diff"] - merged_valid["zscore_diff_manual"]).abs()
            assert diff.max() < 1e-10, f"Zscore diff diff: {diff.max()}"


# =============================================================================
# Test: Complex Multi-Stage Chains
# =============================================================================


class TestComplexChains:
    """Test complex multi-stage chains involving multiple scope transitions."""

    def test_ts_cs_ts_chain(self, panel_with_more_dates):
        """pct_change().rank().shift() - TS → CS → TS chain.

        A realistic pattern: compute returns, rank them, then lag the rank.
        The Pipeline materializes intermediates in the correct order:
        1. TS operation (pct_change) with .over(entity)
        2. CS operation (rank) with .over(date) on materialized returns
        3. TS operation (shift) with .over(entity) on materialized ranks
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked_returns = returns.rank()
        lagged_ranked_returns = ranked_returns.shift(1)

        # Verify scope chain
        assert returns.scope == Scope.TIME_SERIES
        assert ranked_returns.scope == Scope.CROSS_SECTION
        assert lagged_ranked_returns.scope == Scope.TIME_SERIES

        # Pipeline execution
        pipeline = Pipeline(panel_with_more_dates.lazy()).add_factors(
            {
                "returns": returns,
                "ranked_returns": ranked_returns,
                "lagged_ranked_returns": lagged_ranked_returns,
            }
        )
        result = pipeline.run()

        # Manual three-stage execution
        stage1 = panel_with_more_dates.sort(["date", "asset"]).with_columns(
            [pl.col("close").pct_change(1).over("asset").alias("returns_manual")]
        )
        stage2 = stage1.with_columns(
            [
                pl.col("returns_manual")
                .rank(method="average")
                .over("date")
                .alias("ranked_returns_manual")
            ]
        )
        stage3 = stage2.with_columns(
            [
                pl.col("ranked_returns_manual")
                .shift(1)
                .over("asset")
                .alias("lagged_ranked_returns_manual")
            ]
        )

        # Compare all stages
        merged = result.join(stage3, on=["date", "asset"], how="inner")

        # Returns
        valid_ret = merged.filter(
            merged["returns"].is_not_null() & merged["returns_manual"].is_not_null()
        )
        if len(valid_ret) > 0:
            diff_ret = (valid_ret["returns"] - valid_ret["returns_manual"]).abs()
            assert diff_ret.max() < 1e-10, f"Returns diff: {diff_ret.max()}"

        # Ranked returns
        valid_rank = merged.filter(
            merged["ranked_returns"].is_not_null() & merged["ranked_returns_manual"].is_not_null()
        )
        if len(valid_rank) > 0:
            diff_rank = (valid_rank["ranked_returns"] - valid_rank["ranked_returns_manual"]).abs()
            assert diff_rank.max() < 1e-10, f"Ranked returns diff: {diff_rank.max()}"

        # Lagged ranked returns
        valid_lagged = merged.filter(
            merged["lagged_ranked_returns"].is_not_null()
            & merged["lagged_ranked_returns_manual"].is_not_null()
        )
        if len(valid_lagged) > 0:
            diff_lagged = (
                valid_lagged["lagged_ranked_returns"] - valid_lagged["lagged_ranked_returns_manual"]
            ).abs()
            assert diff_lagged.max() < 1e-10, f"Lagged ranked returns diff: {diff_lagged.max()}"

    def test_rolling_demean_rank_chain(self, panel_with_more_dates):
        """rolling_mean().demean().rank() - TS → CS → CS chain.

        Smooth prices, then demean, then rank the demeaned values.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        smoothed = close.rolling_mean(3)
        demeaned_smoothed = smoothed.demean()
        ranked_demeaned = demeaned_smoothed.rank()

        # Verify scope chain
        assert smoothed.scope == Scope.TIME_SERIES
        assert demeaned_smoothed.scope == Scope.CROSS_SECTION
        assert ranked_demeaned.scope == Scope.CROSS_SECTION

        # Pipeline execution
        pipeline = Pipeline(panel_with_more_dates.lazy()).add_factors(
            {
                "smoothed": smoothed,
                "demeaned_smoothed": demeaned_smoothed,
                "ranked_demeaned": ranked_demeaned,
            }
        )
        result = pipeline.run()

        # Manual execution
        stage1 = panel_with_more_dates.sort(["date", "asset"]).with_columns(
            [pl.col("close").rolling_mean(window_size=3).over("asset").alias("smoothed_manual")]
        )
        stage2 = stage1.with_columns(
            [
                (pl.col("smoothed_manual") - pl.col("smoothed_manual").mean())
                .over("date")
                .alias("demeaned_smoothed_manual")
            ]
        )
        stage3 = stage2.with_columns(
            [
                pl.col("demeaned_smoothed_manual")
                .rank(method="average")
                .over("date")
                .alias("ranked_demeaned_manual")
            ]
        )

        # Compare
        merged = result.join(stage3, on=["date", "asset"], how="inner")

        valid = merged.filter(
            merged["ranked_demeaned"].is_not_null() & merged["ranked_demeaned_manual"].is_not_null()
        )
        if len(valid) > 0:
            diff = (valid["ranked_demeaned"] - valid["ranked_demeaned_manual"]).abs()
            assert diff.max() < 1e-10, f"Ranked demeaned diff: {diff.max()}"
