"""Tests for core Factor types and scope-based execution."""

import polars as pl

from factr.core import Classifier, Factor, Filter, Scope


def test_factor_creation():
    factor = Factor(pl.col("close"), name="close", scope=Scope.RAW)
    assert factor.name == "close"
    assert isinstance(factor.expr, pl.Expr)
    assert factor.scope == Scope.RAW


def test_factor_auto_naming():
    factor = Factor(pl.col("close"))
    assert factor.name.startswith("factor_")


def test_factor_arithmetic():
    f1 = Factor(pl.col("a"), name="a", scope=Scope.RAW)
    f2 = Factor(pl.col("b"), name="b", scope=Scope.RAW)

    result = f1 + f2
    assert isinstance(result, Factor)
    assert result.scope == Scope.RAW  # RAW + RAW = RAW
    assert "(a + b)" in result.name

    result = f1 - f2
    assert "(a - b)" in result.name
    assert result.scope == Scope.RAW

    result = f1 * f2
    assert "(a * b)" in result.name
    assert result.scope == Scope.RAW

    result = f1 / f2
    assert "(a / b)" in result.name
    assert result.scope == Scope.RAW

    result = f1**2
    assert "(a ** 2)" in result.name
    assert result.scope == Scope.RAW


def test_factor_comparison():
    factor = Factor(pl.col("value"), name="value")

    result = factor > 0
    assert isinstance(result, Filter)
    assert "(value > 0)" in result.name

    result = factor >= 10
    assert isinstance(result, Filter)

    result = factor < 100
    assert isinstance(result, Filter)

    result = factor == 50
    assert isinstance(result, Filter)


def test_filter_logical_ops():
    f1 = Filter(pl.col("a") > 0, name="a_positive")
    f2 = Filter(pl.col("b") > 0, name="b_positive")

    result = f1 & f2
    assert isinstance(result, Filter)
    assert "a_positive" in result.name
    assert "b_positive" in result.name

    result = f1 | f2
    assert isinstance(result, Filter)

    result = ~f1
    assert isinstance(result, Filter)
    assert "a_positive" in result.name


def test_factor_cross_sectional_ops():
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "value": [10.0, 20.0, 30.0],
        }
    ).lazy()

    factor = Factor(pl.col("value"), name="value")

    ranked = factor.rank()
    assert isinstance(ranked, Factor)
    assert "rank" in ranked.name

    demeaned = factor.demean()
    assert isinstance(demeaned, Factor)
    result = df.with_columns([demeaned.expr.alias("demeaned")]).collect()
    assert abs(result["demeaned"].mean()) < 1e-10

    zscored = factor.zscore()
    assert isinstance(zscored, Factor)

    top_filter = factor.top(2)
    assert isinstance(top_filter, Filter)


def test_factor_transformations():
    factor = Factor(pl.col("value"), name="value")

    log_factor = factor.log()
    assert "log(value)" in log_factor.name

    sqrt_factor = factor.sqrt()
    assert "sqrt(value)" in sqrt_factor.name

    clipped = factor.clip(0, 100)
    assert "clip" in clipped.name

    filled = factor.fill_null(0)
    assert "fill_null" in filled.name


def test_factor_time_series():
    factor = Factor(pl.col("close"), name="close")

    shifted = factor.shift(1)
    assert "shift" in shifted.name

    returns = factor.pct_change()
    assert "pct_change" in returns.name

    sma = factor.rolling_mean(20)
    assert "rolling_mean" in sma.name

    std = factor.rolling_std(20)
    assert "rolling_std" in std.name


def test_factor_quantile():
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 5,
            "asset": ["A", "B", "C", "D", "E"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).lazy()

    factor = Factor(pl.col("value"), name="value")
    quintiles = factor.quantile(5, labels=True)

    assert isinstance(quintiles, Classifier)
    result = df.with_columns([quintiles.expr.alias("quintile")]).collect()
    assert result["quintile"].min() >= 0


def test_factor_in_dataframe():
    """Test cross-sectional rank operation applied per-date using .over()."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "asset": ["A", "B", "C"] * 2,
            "value": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
        }
    ).lazy()

    factor = Factor(pl.col("value"), name="value", scope=Scope.RAW)
    ranked = factor.rank()  # CROSS_SECTION scope - needs .over("date")

    # Apply .over("date") for cross-sectional operation
    result = df.with_columns([ranked.expr.over("date").alias("rank")]).collect()

    date1 = result.filter(pl.col("date") == "2024-01-01")
    assert date1["rank"].to_list() == [1.0, 2.0, 3.0]


def test_classifier_creation():
    classifier = Classifier(pl.col("sector"), name="sector")
    assert classifier.name == "sector"
    assert isinstance(classifier, Factor)


def test_factor_demean_by_group():
    """Test cross-sectional demean grouped by sector using .over()."""
    df = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 4,
            "asset": ["A", "B", "C", "D"],
            "sector": ["Tech", "Tech", "Finance", "Finance"],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    ).lazy()

    factor = Factor(pl.col("value"), name="value", scope=Scope.RAW)
    sector_neutral = factor.demean(by="sector")  # CROSS_SECTION scope with groupby=["sector"]

    # Apply .over("date", "sector") for cross-sectional grouped operation
    result = df.with_columns(
        [sector_neutral.expr.over("date", "sector").alias("neutral")]
    ).collect()

    tech_mean = result.filter(pl.col("sector") == "Tech")["neutral"].mean()
    finance_mean = result.filter(pl.col("sector") == "Finance")["neutral"].mean()

    assert abs(tech_mean) < 1e-10
    assert abs(finance_mean) < 1e-10
