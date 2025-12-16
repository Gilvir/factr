"""Property-based tests for scope inference correctness.

Uses Hypothesis to generate random factor compositions and verify
that scope inference rules always hold.
"""

from hypothesis import given, strategies as st, settings, assume

import polars as pl

from factr.core import Factor, Filter, Scope


# =============================================================================
# Strategies for generating random Factors
# =============================================================================


@st.composite
def factor_with_scope(draw, scope: Scope | None = None):
    """Generate a Factor with a specific or random scope."""
    if scope is None:
        scope = draw(st.sampled_from(list(Scope)))

    col_name = draw(st.sampled_from(["close", "volume", "returns", "price"]))
    return Factor(
        expr=pl.col(col_name),
        name=f"factor_{col_name}",
        scope=scope,
    )


@st.composite
def raw_factor(draw):
    """Generate a RAW scope factor."""
    return draw(factor_with_scope(scope=Scope.RAW))


@st.composite
def time_series_factor(draw):
    """Generate a TIME_SERIES scope factor."""
    return draw(factor_with_scope(scope=Scope.TIME_SERIES))


@st.composite
def cross_section_factor(draw):
    """Generate a CROSS_SECTION scope factor."""
    return draw(factor_with_scope(scope=Scope.CROSS_SECTION))


@st.composite
def any_factor(draw):
    """Generate a factor with any scope."""
    return draw(factor_with_scope())


# =============================================================================
# Property: CROSS_SECTION is absorbing (dominates all operations)
# =============================================================================


class TestCrossSectionAbsorbing:
    """CROSS_SECTION scope should dominate in all binary operations."""

    @given(cross_section_factor(), any_factor())
    @settings(max_examples=50)
    def test_cross_section_plus_any_is_cross_section(self, cs_factor, other):
        """CS + any = CS"""
        result = cs_factor + other
        assert result.scope == Scope.CROSS_SECTION

    @given(any_factor(), cross_section_factor())
    @settings(max_examples=50)
    def test_any_plus_cross_section_is_cross_section(self, other, cs_factor):
        """any + CS = CS"""
        result = other + cs_factor
        assert result.scope == Scope.CROSS_SECTION

    @given(cross_section_factor(), any_factor())
    @settings(max_examples=50)
    def test_cross_section_mul_any_is_cross_section(self, cs_factor, other):
        """CS * any = CS"""
        result = cs_factor * other
        assert result.scope == Scope.CROSS_SECTION

    @given(cross_section_factor(), any_factor())
    @settings(max_examples=50)
    def test_cross_section_sub_any_is_cross_section(self, cs_factor, other):
        """CS - any = CS"""
        result = cs_factor - other
        assert result.scope == Scope.CROSS_SECTION

    @given(cross_section_factor(), any_factor())
    @settings(max_examples=50)
    def test_cross_section_div_any_is_cross_section(self, cs_factor, other):
        """CS / any = CS"""
        result = cs_factor / other
        assert result.scope == Scope.CROSS_SECTION


# =============================================================================
# Property: TIME_SERIES + TIME_SERIES = TIME_SERIES
# =============================================================================


class TestTimeSeriesPreservation:
    """TIME_SERIES operations with TIME_SERIES should stay TIME_SERIES."""

    @given(time_series_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_ts_plus_ts_is_ts(self, f1, f2):
        """TS + TS = TS"""
        result = f1 + f2
        assert result.scope == Scope.TIME_SERIES

    @given(time_series_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_ts_mul_ts_is_ts(self, f1, f2):
        """TS * TS = TS"""
        result = f1 * f2
        assert result.scope == Scope.TIME_SERIES

    @given(time_series_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_ts_sub_ts_is_ts(self, f1, f2):
        """TS - TS = TS"""
        result = f1 - f2
        assert result.scope == Scope.TIME_SERIES

    @given(time_series_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_ts_div_ts_is_ts(self, f1, f2):
        """TS / TS = TS"""
        result = f1 / f2
        assert result.scope == Scope.TIME_SERIES


# =============================================================================
# Property: RAW + RAW = RAW
# =============================================================================


class TestRawPreservation:
    """RAW operations with RAW should stay RAW."""

    @given(raw_factor(), raw_factor())
    @settings(max_examples=50)
    def test_raw_plus_raw_is_raw(self, f1, f2):
        """RAW + RAW = RAW"""
        result = f1 + f2
        assert result.scope == Scope.RAW

    @given(raw_factor(), raw_factor())
    @settings(max_examples=50)
    def test_raw_mul_raw_is_raw(self, f1, f2):
        """RAW * RAW = RAW"""
        result = f1 * f2
        assert result.scope == Scope.RAW


# =============================================================================
# Property: TIME_SERIES + RAW = TIME_SERIES (promotion)
# =============================================================================


class TestScopePromotion:
    """RAW should promote to TIME_SERIES when combined."""

    @given(time_series_factor(), raw_factor())
    @settings(max_examples=50)
    def test_ts_plus_raw_is_ts(self, ts_factor, raw_factor):
        """TS + RAW = TS"""
        result = ts_factor + raw_factor
        assert result.scope == Scope.TIME_SERIES

    @given(raw_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_raw_plus_ts_is_ts(self, raw_factor, ts_factor):
        """RAW + TS = TS"""
        result = raw_factor + ts_factor
        assert result.scope == Scope.TIME_SERIES


# =============================================================================
# Property: Operations that force scope
# =============================================================================


class TestScopeForcingOperations:
    """Certain operations force a specific scope."""

    @given(any_factor())
    @settings(max_examples=50)
    def test_rank_forces_cross_section(self, factor):
        """rank() always produces CROSS_SECTION."""
        result = factor.rank()
        assert result.scope == Scope.CROSS_SECTION

    @given(any_factor())
    @settings(max_examples=50)
    def test_demean_forces_cross_section(self, factor):
        """demean() always produces CROSS_SECTION."""
        result = factor.demean()
        assert result.scope == Scope.CROSS_SECTION

    @given(any_factor())
    @settings(max_examples=50)
    def test_zscore_forces_cross_section(self, factor):
        """zscore() always produces CROSS_SECTION."""
        result = factor.zscore()
        assert result.scope == Scope.CROSS_SECTION

    @given(any_factor())
    @settings(max_examples=50)
    def test_winsorize_forces_cross_section(self, factor):
        """winsorize() always produces CROSS_SECTION."""
        result = factor.winsorize()
        assert result.scope == Scope.CROSS_SECTION

    @given(any_factor())
    @settings(max_examples=50)
    def test_top_forces_cross_section(self, factor):
        """top() always produces CROSS_SECTION Filter."""
        result = factor.top(5)
        assert result.scope == Scope.CROSS_SECTION
        assert isinstance(result, Filter)

    @given(any_factor())
    @settings(max_examples=50)
    def test_bottom_forces_cross_section(self, factor):
        """bottom() always produces CROSS_SECTION Filter."""
        result = factor.bottom(5)
        assert result.scope == Scope.CROSS_SECTION
        assert isinstance(result, Filter)

    @given(any_factor())
    @settings(max_examples=50)
    def test_quantile_forces_cross_section(self, factor):
        """quantile() always produces CROSS_SECTION."""
        result = factor.quantile(5)
        assert result.scope == Scope.CROSS_SECTION


class TestTimeSeriesForcingOperations:
    """Time-series operations force TIME_SERIES scope."""

    @given(any_factor())
    @settings(max_examples=50)
    def test_shift_forces_time_series(self, factor):
        """shift() always produces TIME_SERIES."""
        result = factor.shift(1)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_pct_change_forces_time_series(self, factor):
        """pct_change() always produces TIME_SERIES."""
        result = factor.pct_change(1)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_diff_forces_time_series(self, factor):
        """diff() always produces TIME_SERIES."""
        result = factor.diff(1)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_rolling_sum_forces_time_series(self, factor):
        """rolling_sum() always produces TIME_SERIES."""
        result = factor.rolling_sum(5)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_rolling_mean_forces_time_series(self, factor):
        """rolling_mean() always produces TIME_SERIES."""
        result = factor.rolling_mean(5)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_rolling_std_forces_time_series(self, factor):
        """rolling_std() always produces TIME_SERIES."""
        result = factor.rolling_std(5)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_ewm_mean_forces_time_series(self, factor):
        """ewm_mean() always produces TIME_SERIES."""
        result = factor.ewm_mean(10)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_cumsum_forces_time_series(self, factor):
        """cumsum() always produces TIME_SERIES."""
        result = factor.cumsum()
        assert result.scope == Scope.TIME_SERIES


# =============================================================================
# Property: Scope-preserving operations
# =============================================================================


class TestScopePreservingOperations:
    """Operations that preserve the input scope."""

    @given(any_factor())
    @settings(max_examples=50)
    def test_log_preserves_scope(self, factor):
        """log() preserves scope."""
        result = factor.log()
        assert result.scope == factor.scope

    @given(any_factor())
    @settings(max_examples=50)
    def test_sqrt_preserves_scope(self, factor):
        """sqrt() preserves scope."""
        result = factor.sqrt()
        assert result.scope == factor.scope

    @given(any_factor())
    @settings(max_examples=50)
    def test_sign_preserves_scope(self, factor):
        """sign() preserves scope."""
        result = factor.sign()
        assert result.scope == factor.scope

    @given(any_factor())
    @settings(max_examples=50)
    def test_clip_preserves_scope(self, factor):
        """clip() preserves scope."""
        result = factor.clip(0, 100)
        assert result.scope == factor.scope

    @given(any_factor())
    @settings(max_examples=50)
    def test_fill_null_preserves_scope(self, factor):
        """fill_null() preserves scope."""
        result = factor.fill_null(0)
        assert result.scope == factor.scope

    @given(any_factor())
    @settings(max_examples=50)
    def test_neg_preserves_scope(self, factor):
        """Negation preserves scope."""
        result = -factor
        assert result.scope == factor.scope

    @given(any_factor())
    @settings(max_examples=50)
    def test_abs_preserves_scope(self, factor):
        """abs() preserves scope."""
        result = abs(factor)
        assert result.scope == factor.scope


# =============================================================================
# Property: Scalar operations preserve scope
# =============================================================================


class TestScalarOperations:
    """Operations with scalars should preserve factor scope."""

    @given(
        any_factor(),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
    )
    @settings(max_examples=50)
    def test_add_scalar_preserves_scope(self, factor, scalar):
        """factor + scalar preserves scope."""
        result = factor + scalar
        assert result.scope == factor.scope

    @given(
        any_factor(),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
    )
    @settings(max_examples=50)
    def test_mul_scalar_preserves_scope(self, factor, scalar):
        """factor * scalar preserves scope."""
        result = factor * scalar
        assert result.scope == factor.scope

    @given(
        any_factor(),
        st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_div_scalar_preserves_scope(self, factor, scalar):
        """factor / scalar preserves scope."""
        assume(scalar != 0)
        result = factor / scalar
        assert result.scope == factor.scope

    @given(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
        any_factor(),
    )
    @settings(max_examples=50)
    def test_scalar_add_preserves_scope(self, scalar, factor):
        """scalar + factor preserves scope."""
        result = scalar + factor
        assert result.scope == factor.scope


# =============================================================================
# Property: Comparison operations infer scope correctly
# =============================================================================


class TestComparisonScopeInference:
    """Comparison operations should produce Filter with correct scope."""

    @given(
        any_factor(),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
    )
    @settings(max_examples=50)
    def test_gt_scalar_preserves_scope(self, factor, scalar):
        """factor > scalar produces Filter with same scope."""
        result = factor > scalar
        assert isinstance(result, Filter)
        assert result.scope == factor.scope

    @given(cross_section_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_comparison_with_cross_section_is_cross_section(self, cs_factor, ts_factor):
        """Comparing CS and TS factors produces CS Filter."""
        result = cs_factor > ts_factor
        assert isinstance(result, Filter)
        assert result.scope == Scope.CROSS_SECTION

    @given(time_series_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_comparison_ts_ts_is_ts(self, f1, f2):
        """Comparing two TS factors produces TS Filter."""
        result = f1 > f2
        assert isinstance(result, Filter)
        assert result.scope == Scope.TIME_SERIES


# =============================================================================
# Property: Filter boolean operations infer scope correctly
# =============================================================================


class TestFilterBooleanOperations:
    """Filter boolean operations should infer scope correctly."""

    @given(cross_section_factor(), any_factor())
    @settings(max_examples=50)
    def test_and_with_cross_section_is_cross_section(self, cs_factor, other):
        """(CS > 0) & (any > 0) = CS Filter."""
        filter1 = cs_factor > 0
        filter2 = other > 0
        result = filter1 & filter2
        assert isinstance(result, Filter)
        assert result.scope == Scope.CROSS_SECTION

    @given(time_series_factor(), time_series_factor())
    @settings(max_examples=50)
    def test_and_ts_ts_is_ts(self, f1, f2):
        """(TS > 0) & (TS > 0) = TS Filter."""
        filter1 = f1 > 0
        filter2 = f2 > 0
        result = filter1 & filter2
        assert isinstance(result, Filter)
        assert result.scope == Scope.TIME_SERIES

    @given(any_factor())
    @settings(max_examples=50)
    def test_invert_preserves_scope(self, factor):
        """~Filter preserves scope."""
        filter_obj = factor > 0
        result = ~filter_obj
        assert isinstance(result, Filter)
        assert result.scope == filter_obj.scope


# =============================================================================
# Property: Deep composition chains
# =============================================================================


class TestDeepComposition:
    """Test scope inference in deeply nested factor compositions."""

    @given(raw_factor())
    @settings(max_examples=50)
    def test_long_time_series_chain(self, factor):
        """Chain of time-series ops stays TIME_SERIES."""
        result = factor.pct_change(1).rolling_mean(5).shift(1).diff(1)
        assert result.scope == Scope.TIME_SERIES

    @given(raw_factor())
    @settings(max_examples=50)
    def test_time_series_then_cross_section(self, factor):
        """Time-series ops followed by cross-section ops → CROSS_SECTION."""
        result = factor.pct_change(1).rolling_mean(5).rank()
        assert result.scope == Scope.CROSS_SECTION

    @given(raw_factor(), raw_factor())
    @settings(max_examples=50)
    def test_complex_composition(self, f1, f2):
        """Complex composition: (f1.returns + f2.returns).rank()"""
        returns1 = f1.pct_change(1)
        returns2 = f2.pct_change(1)
        combined = returns1 + returns2
        ranked = combined.rank()

        # Intermediate: TIME_SERIES + TIME_SERIES = TIME_SERIES
        assert combined.scope == Scope.TIME_SERIES
        # Final: rank() → CROSS_SECTION
        assert ranked.scope == Scope.CROSS_SECTION


# =============================================================================
# Property: Groupby propagation
# =============================================================================


class TestGroupbyPropagation:
    """Test that groupby is correctly propagated through operations."""

    @given(any_factor())
    @settings(max_examples=30)
    def test_rank_by_sets_groupby(self, factor):
        """rank(by='sector') sets groupby=['sector']."""
        result = factor.rank(by="sector")
        assert result.groupby == ["sector"]

    @given(any_factor())
    @settings(max_examples=30)
    def test_demean_by_sets_groupby(self, factor):
        """demean(by='sector') sets groupby=['sector']."""
        result = factor.demean(by="sector")
        assert result.groupby == ["sector"]

    @given(any_factor())
    @settings(max_examples=30)
    def test_zscore_by_sets_groupby(self, factor):
        """zscore(by='sector') sets groupby=['sector']."""
        result = factor.zscore(by="sector")
        assert result.groupby == ["sector"]

    def test_groupby_preserved_in_binop(self):
        """Groupby is preserved when one operand has it."""
        f1 = Factor(pl.col("a"), scope=Scope.CROSS_SECTION, groupby=["sector"])
        f2 = Factor(pl.col("b"), scope=Scope.TIME_SERIES)
        result = f1 + f2
        assert result.groupby == ["sector"]


# =============================================================================
# Scope lattice properties
# =============================================================================


class TestScopeLattice:
    """Test that scope forms a proper lattice: RAW < TIME_SERIES < CROSS_SECTION."""

    def test_scope_ordering_is_consistent(self):
        """Scope ordering should be consistent with _infer_scope."""
        raw = Factor(pl.col("a"), scope=Scope.RAW)
        ts = Factor(pl.col("b"), scope=Scope.TIME_SERIES)
        cs = Factor(pl.col("c"), scope=Scope.CROSS_SECTION)

        # RAW + TS = TS
        assert (raw + ts).scope == Scope.TIME_SERIES
        # TS + CS = CS
        assert (ts + cs).scope == Scope.CROSS_SECTION
        # RAW + CS = CS
        assert (raw + cs).scope == Scope.CROSS_SECTION

    @given(any_factor(), any_factor())
    @settings(max_examples=50)
    def test_binop_commutativity_for_scope(self, f1, f2):
        """Scope inference is commutative for addition."""
        r1 = f1 + f2
        r2 = f2 + f1
        assert r1.scope == r2.scope

    @given(any_factor(), any_factor(), any_factor())
    @settings(max_examples=50)
    def test_binop_associativity_for_scope(self, f1, f2, f3):
        """Scope inference is associative for addition."""
        r1 = (f1 + f2) + f3
        r2 = f1 + (f2 + f3)
        assert r1.scope == r2.scope
