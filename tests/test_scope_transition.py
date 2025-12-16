"""Tests for scope-transition materialization.

These tests verify that the Pipeline correctly handles factors where
TIME_SERIES operations are followed by CROSS_SECTION operations.
The TIME_SERIES intermediate must be materialized with .over(entity)
before the CROSS_SECTION operation is applied with .over(date).
"""

import polars as pl
import pytest

from factr import Pipeline
from factr.core import Factor, Scope, collect_dependencies


@pytest.fixture
def panel_data():
    """Create simple panel data for testing."""
    data = {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"] * 3,
        "asset": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        "close": [100.0, 102.0, 101.0, 50.0, 51.0, 52.0, 200.0, 198.0, 202.0],
        "sector": ["Tech"] * 3 + ["Finance"] * 3 + ["Tech"] * 3,
    }
    return pl.DataFrame(data).with_columns(pl.col("date").str.to_date())


class TestScopeTransitionMaterialization:
    """Test that TIME_SERIES -> CROSS_SECTION transitions work correctly."""

    def test_pct_change_then_rank(self, panel_data):
        """pct_change (TIME_SERIES) followed by rank (CROSS_SECTION)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked = returns.rank(pct=True)

        # Verify scope inference
        assert returns.scope == Scope.TIME_SERIES
        assert ranked.scope == Scope.CROSS_SECTION
        assert ranked._parent is returns  # Parent should be set

        # Run through Pipeline
        result = Pipeline(panel_data.lazy()).add_factors({"ranked_returns": ranked}).run()

        # Verify computation is correct - returns computed per-entity, rank per-date
        # Asset A: returns [null, 0.02, -0.0098]
        # Asset B: returns [null, 0.02, 0.0196]
        # Asset C: returns [null, -0.01, 0.0202]
        # On 2020-01-02: ranks should be based on [0.02, 0.02, -0.01]
        # On 2020-01-03: ranks should be based on [-0.0098, 0.0196, 0.0202]

        assert "ranked_returns" in result.columns
        # Should have results (nulls for first date due to pct_change)
        non_null = result.filter(pl.col("ranked_returns").is_not_null())
        assert len(non_null) == 6  # 2 dates * 3 assets

    def test_rolling_mean_then_demean(self, panel_data):
        """rolling_mean (TIME_SERIES) followed by demean (CROSS_SECTION)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        sma = close.rolling_mean(2)
        demeaned = sma.demean()

        # Verify scope inference
        assert sma.scope == Scope.TIME_SERIES
        assert demeaned.scope == Scope.CROSS_SECTION
        assert demeaned._parent is sma

        result = Pipeline(panel_data.lazy()).add_factors({"demeaned_sma": demeaned}).run()

        assert "demeaned_sma" in result.columns
        # Demean should sum to zero per date (within numerical precision)
        date_sums = result.group_by("date").agg(pl.col("demeaned_sma").sum())
        for s in date_sums["demeaned_sma"]:
            if s is not None:
                assert abs(s) < 1e-10

    def test_multiple_ts_then_cs(self, panel_data):
        """Multiple TIME_SERIES ops followed by CROSS_SECTION."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        smoothed = returns.rolling_mean(2)
        ranked = smoothed.rank()

        # Verify chain
        assert returns.scope == Scope.TIME_SERIES
        assert smoothed.scope == Scope.TIME_SERIES
        assert ranked.scope == Scope.CROSS_SECTION
        assert ranked._parent is smoothed

        result = Pipeline(panel_data.lazy()).add_factors({"ranked_smoothed_returns": ranked}).run()

        assert "ranked_smoothed_returns" in result.columns

    def test_output_includes_ts_intermediate(self, panel_data):
        """When TS intermediate is also an output, both are computed correctly."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked = returns.rank(pct=True)

        # Add BOTH returns and ranked to output
        result = (
            Pipeline(panel_data.lazy())
            .add_factors({"returns": returns, "ranked_returns": ranked})
            .run()
        )

        assert "returns" in result.columns
        assert "ranked_returns" in result.columns

        # Verify returns are correct (per-entity)
        a_data = result.filter(pl.col("asset") == "A").sort("date")
        # A: [100, 102, 101] -> returns [null, 0.02, -0.0098...]
        assert a_data["returns"][0] is None
        assert abs(a_data["returns"][1] - 0.02) < 1e-10

    def test_no_parent_for_raw_to_cs(self, panel_data):
        """RAW -> CROSS_SECTION should NOT set parent (column already exists)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        ranked = close.rank()

        # RAW factors don't need materialization, so no parent
        assert ranked._parent is None
        assert ranked.scope == Scope.CROSS_SECTION

        result = Pipeline(panel_data.lazy()).add_factors({"ranked_close": ranked}).run()

        # Original close column should still exist
        assert "close" in result.columns
        assert "ranked_close" in result.columns


class TestCollectDependencies:
    """Test the collect_dependencies function."""

    def test_single_factor_no_deps(self):
        """Factor without parent has no dependencies."""
        f = Factor(pl.col("x"), name="x", scope=Scope.RAW)
        deps = collect_dependencies([f])
        assert len(deps) == 1
        assert deps[0] is f

    def test_factor_with_parent(self):
        """Factor with parent includes parent in deps."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked = returns.rank()

        deps = collect_dependencies([ranked])

        # Should include: returns (parent), ranked
        assert len(deps) == 2
        assert deps[0] is returns  # Parent first
        assert deps[1] is ranked

    def test_multiple_factors_shared_parent(self):
        """Multiple factors sharing a parent only include it once."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked = returns.rank()
        demeaned = returns.demean()

        deps = collect_dependencies([ranked, demeaned])

        # returns should appear only once
        returns_count = sum(1 for d in deps if d is returns)
        assert returns_count == 1

        # All three should be present
        assert len(deps) == 3

    def test_topological_order(self):
        """Dependencies are returned in topological order (parents first)."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        smoothed = returns.rolling_mean(2)
        ranked = smoothed.rank()

        deps = collect_dependencies([ranked])

        # Only smoothed and ranked should be in deps
        # TIME_SERIES -> TIME_SERIES (returns -> smoothed) doesn't create parent
        # TIME_SERIES -> CROSS_SECTION (smoothed -> ranked) creates parent
        assert len(deps) == 2
        assert deps[0] is smoothed  # Parent of ranked
        assert deps[1] is ranked


class TestDiamondDependencyExecution:
    """Test execution correctness when multiple factors share a common parent."""

    def test_diamond_dependency_values_correct(self, panel_data):
        """When two CS factors share a TS parent, both should compute correctly.

        Pattern:
            close -> returns (TS)
                  -> ranked (CS, parent=returns)
                  -> demeaned (CS, parent=returns)

        This tests that:
        1. returns is computed once
        2. Both ranked and demeaned reference the same intermediate
        3. Both final values match manual computation
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked = returns.rank(pct=True)
        demeaned = returns.demean()

        # Verify structure
        assert returns.scope == Scope.TIME_SERIES
        assert ranked.scope == Scope.CROSS_SECTION
        assert demeaned.scope == Scope.CROSS_SECTION
        assert ranked._parent is returns
        assert demeaned._parent is returns

        # Execute through Pipeline
        result = (
            Pipeline(panel_data.lazy())
            .add_factors({"returns": returns, "ranked": ranked, "demeaned": demeaned})
            .run()
        )

        # Manual execution
        manual = panel_data.sort(["date", "asset"]).with_columns(
            [pl.col("close").pct_change(1).over("asset").alias("returns_manual")]
        )
        manual = manual.with_columns(
            [
                ((pl.col("returns_manual").rank(method="average") - 1) / (pl.len() - 1))
                .over("date")
                .alias("ranked_manual"),
                (pl.col("returns_manual") - pl.col("returns_manual").mean())
                .over("date")
                .alias("demeaned_manual"),
            ]
        )

        # Compare
        merged = result.join(manual, on=["date", "asset"], how="inner")

        # Check returns match
        valid_ret = merged.filter(
            merged["returns"].is_not_null() & merged["returns_manual"].is_not_null()
        )
        if len(valid_ret) > 0:
            diff_ret = (valid_ret["returns"] - valid_ret["returns_manual"]).abs()
            assert diff_ret.max() < 1e-10, f"Returns diff: {diff_ret.max()}"

        # Check ranked match
        valid_rank = merged.filter(
            merged["ranked"].is_not_null() & merged["ranked_manual"].is_not_null()
        )
        if len(valid_rank) > 0:
            diff_rank = (valid_rank["ranked"] - valid_rank["ranked_manual"]).abs()
            assert diff_rank.max() < 1e-10, f"Ranked diff: {diff_rank.max()}"

        # Check demeaned match
        valid_demean = merged.filter(
            merged["demeaned"].is_not_null() & merged["demeaned_manual"].is_not_null()
        )
        if len(valid_demean) > 0:
            diff_demean = (valid_demean["demeaned"] - valid_demean["demeaned_manual"]).abs()
            assert diff_demean.max() < 1e-10, f"Demeaned diff: {diff_demean.max()}"

        # Demeaned should sum to zero per date
        date_sums = result.group_by("date").agg(pl.col("demeaned").sum())
        for s in date_sums["demeaned"]:
            if s is not None:
                assert abs(s) < 1e-10, "Demeaned returns should sum to zero"

    def test_diamond_with_different_cs_ops(self, panel_data):
        """Multiple different CS operations on the same TS parent.

        Note: This test doesn't use screen() with top() because filter parent
        tracking has a known limitation similar to CS→TS transitions.
        """
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)

        # Two different CS operations on returns
        ranked = returns.rank()
        zscored = returns.zscore()

        # Execute without screen
        pipeline = Pipeline(panel_data.lazy())
        pipeline.add_factors({"returns": returns, "ranked": ranked, "zscored": zscored})
        result = pipeline.run()

        assert "ranked" in result.columns
        assert "zscored" in result.columns
        assert "returns" in result.columns

        # Verify zscores have mean ~0 per date (for non-null values)
        for dt in result["date"].unique().to_list():
            date_data = result.filter(pl.col("date") == dt)
            zscores = date_data["zscored"].drop_nulls().to_list()
            if len(zscores) > 1:
                mean = sum(zscores) / len(zscores)
                assert abs(mean) < 1e-10, f"Zscores should have mean 0, got {mean}"


class TestExplainShowsIntermediates:
    """Test that explain() shows intermediate factors."""

    def test_explain_shows_intermediate_marker(self, panel_data):
        """Explain output should mark intermediates."""
        close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
        returns = close.pct_change(1)
        ranked = returns.rank()

        pipeline = Pipeline(panel_data.lazy()).add_factors({"ranked": ranked})
        explanation = pipeline.explain()

        # Should show [intermediate] marker for returns
        assert "[intermediate]" in explanation
        assert "TIME_SERIES" in explanation
        assert "CROSS_SECTION" in explanation
