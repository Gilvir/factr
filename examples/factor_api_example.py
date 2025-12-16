"""Comprehensive Factor API example with scope-based execution.

This example covers:
1. Basic factor creation and composition
2. Cross-sectional operations (rank, demean, zscore)
3. Universes and screening
4. Factor arithmetic and filters
5. Custom factors with decorators (@time_series, @cross_section)
6. Classifiers for grouping
7. Scope-based execution and Pipeline.explain()

Run with: python examples/factor_api_example.py
"""

import polars as pl

from factr import (
    Q500US,
    Factor,
    Pipeline,
    cross_section,
    custom,
    time_series,
)
from factr import classifiers as C
from factr import factors as F
from factr.core import Scope
from factr.datasets import EquityPricing


def create_sample_data():
    """Create sample panel data."""
    import random

    random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    assets = [f"STOCK_{i:03d}" for i in range(50)]
    sectors = ["Tech", "Finance", "Healthcare", "Energy"]

    data = []
    for date in dates:
        for i, asset in enumerate(assets):
            price = 100 + i * 2 + random.gauss(0, 5)
            data.append(
                {
                    "date": date,
                    "asset": asset,
                    "close": max(price, 1.0),
                    "volume": max(1e6 + random.gauss(0, 2e5), 0),
                    "sector": sectors[i % len(sectors)],
                }
            )

    return pl.DataFrame(data).with_columns(pl.col("date").str.to_date()).lazy()


def example_1_basic_factors():
    """Basic factor creation and composition."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Factors")
    print("=" * 60)

    df = create_sample_data()

    pipeline = Pipeline(df).add_factors(
        {
            "returns": F.returns(window=1),
            "momentum": F.momentum(window=5, skip=1),
        }
    )

    result = pipeline.run(collect=True)
    print(result.select(["date", "asset", "returns", "momentum"]).head())


def example_2_cross_sectional():
    """Cross-sectional operations computed per-date across entities."""
    print("\n" + "=" * 60)
    print("Example 2: Cross-Sectional Operations")
    print("=" * 60)

    df = create_sample_data()

    close = Factor(pl.col("close"), name="close", scope=Scope.RAW)

    # These operations set CROSS_SECTION scope
    ranked = close.rank(pct=True)
    demeaned = close.demean()
    sector_neutral = close.demean(by="sector")  # Grouped by sector

    print(f"Scopes: rank={ranked.scope.name}, demean={demeaned.scope.name}")
    print(f"Sector neutral groupby: {sector_neutral.groupby}")

    pipeline = Pipeline(df).add_factors(
        {
            "rank_pct": ranked,
            "demeaned": demeaned,
            "sector_neutral": sector_neutral,
        }
    )

    result = pipeline.run(collect=True)
    print(result.select(["date", "asset", "sector", "close", "rank_pct", "sector_neutral"]).head(8))
    print(f"\nMean of demeaned (should be ~0): {result['demeaned'].mean():.2e}")


def example_3_universes():
    """Using universes for asset selection."""
    print("\n" + "=" * 60)
    print("Example 3: Universes and Screening")
    print("=" * 60)

    df = create_sample_data()

    universe = Q500US(window=1, min_price=100)

    pipeline = Pipeline(df).add_factors({"returns": F.returns(window=1)}).screen(universe)

    result = pipeline.run(collect=True)
    print(f"Before screening: {len(df.collect())} rows")
    print(f"After Q500US: {len(result)} rows")


def example_4_composition():
    """Composing factors with operators and filters."""
    print("\n" + "=" * 60)
    print("Example 4: Factor Composition")
    print("=" * 60)

    df = create_sample_data()

    close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
    volume = Factor(pl.col("volume"), name="volume", scope=Scope.RAW)

    # Arithmetic preserves scope
    dollar_vol = close * volume
    price_vol_ratio = close / volume.log()

    # Filters from comparisons
    high_price = close > 150
    high_volume = volume > 1e6
    liquid_stocks = high_price & high_volume

    pipeline = (
        Pipeline(df)
        .add_factors({"dollar_vol": dollar_vol, "price_vol_ratio": price_vol_ratio})
        .screen(liquid_stocks)
    )

    result = pipeline.run(collect=True)
    print(result.select(["date", "asset", "close", "volume", "dollar_vol"]).head())
    print(f"\nFiltered to {len(result)} rows (high price AND high volume)")


def example_5_custom_decorators():
    """Custom factors with @time_series and @cross_section decorators."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Factors with Decorators")
    print("=" * 60)

    df = create_sample_data()

    # @time_series: rolling operations per entity
    @time_series
    def sharpe_ratio(window: int = 5) -> Factor:
        """Rolling Sharpe-like ratio."""
        close = EquityPricing.close
        returns = close.pct_change(1)
        return returns.rolling_mean(window) / returns.rolling_std(window)

    # @cross_section(): rank across all assets per date
    @cross_section()
    def sharpe_rank() -> Factor:
        """Rank Sharpe ratios globally."""
        return sharpe_ratio(window=5).rank(pct=True)

    # @cross_section(by='sector'): rank within each sector
    @cross_section(by="sector")
    def sector_sharpe_rank() -> Factor:
        """Rank Sharpe ratios within sector."""
        return sharpe_ratio(window=5).rank(pct=True)

    # Alternative: @custom.time_series for inline use
    @custom.time_series
    def momentum_quality(window: int = 5) -> Factor:
        close = EquityPricing.close
        momentum = close.pct_change(window)
        trend = close > close.shift(1)
        return trend * momentum

    pipeline = Pipeline(df).add_factors(
        {
            "sharpe": sharpe_ratio(window=5),
            "sharpe_rank": sharpe_rank(),
            "sector_sharpe_rank": sector_sharpe_rank(),
            "mom_quality": momentum_quality(window=3),
        }
    )

    print("\nExecution plan:")
    print(pipeline.explain())

    result = pipeline.run(collect=True)
    print(
        result.select(
            ["date", "asset", "sector", "sharpe", "sharpe_rank", "sector_sharpe_rank"]
        ).tail(8)
    )


def example_6_classifiers():
    """Classifiers for categorical grouping."""
    print("\n" + "=" * 60)
    print("Example 6: Classifiers")
    print("=" * 60)

    df = create_sample_data()

    sector = C.Sector()
    close = Factor(pl.col("close"), name="close", scope=Scope.RAW)
    price_quintiles = close.quantile(5, labels=True)

    print(f"Classifier scope: {price_quintiles.scope.name}")

    pipeline = Pipeline(df).add_factors(
        {
            "price_quintile": price_quintiles,
            "sector": sector,
        }
    )

    result = pipeline.run(collect=True)
    print(result.select(["date", "asset", "close", "price_quintile", "sector"]).head(12))


def example_7_scope_execution():
    """Scope-based multi-stage execution with Pipeline.explain()."""
    print("\n" + "=" * 60)
    print("Example 7: Scope-Based Execution")
    print("=" * 60)

    df = create_sample_data()

    # Build factors with different scopes
    close = EquityPricing.close  # RAW
    returns = close.pct_change()  # TIME_SERIES
    volatility = returns.rolling_std(5)  # TIME_SERIES
    ranked_vol = volatility.rank(pct=True)  # CROSS_SECTION
    momentum = close.pct_change(3).rolling_sum(5)
    sector_neutral = momentum.demean(by="sector")  # CROSS_SECTION with groupby

    print("Factor scopes:")
    print(f"  close: {close.scope.name}")
    print(f"  returns: {returns.scope.name}")
    print(f"  ranked_vol: {ranked_vol.scope.name}")
    print(f"  sector_neutral: {sector_neutral.scope.name} (groupby={sector_neutral.groupby})")

    pipeline = Pipeline(df).add_factors(
        {
            "returns": returns,
            "volatility": volatility,
            "ranked_vol": ranked_vol,
            "sector_neutral": sector_neutral,
        }
    )

    print(pipeline.explain())

    result = pipeline.run(collect=True)
    print(
        result.select(["date", "asset", "sector", "returns", "ranked_vol", "sector_neutral"]).tail(
            8
        )
    )

    print("\nNote: Pipeline applies .over() based on scope:")
    print("  - TIME_SERIES: .over('asset')")
    print("  - CROSS_SECTION: .over('date', *groupby)")


def main():
    print("\n" + "#" * 60)
    print("# Factor API Examples")
    print("#" * 60)

    example_1_basic_factors()
    example_2_cross_sectional()
    example_3_universes()
    example_4_composition()
    example_5_custom_decorators()
    example_6_classifiers()
    example_7_scope_execution()

    print("\n" + "#" * 60)
    print("# Examples Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
