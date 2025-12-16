"""Performance benchmarks with large datasets.

This example demonstrates:
1. Large-scale data handling (1000 assets x 2520 days = 2.5M rows)
2. Time-series and cross-sectional pipelines
3. Sector-neutral factors with grouped operations
4. Multi-factor pipeline performance

Run with: python examples/performance_example.py
"""

import time
from datetime import date, timedelta

import polars as pl

from factr import Pipeline, cross_section, time_series
from factr import factors as F
from factr.datasets import EquityPricing


def create_large_dataset(n_assets: int = 1000, n_days: int = 252, seed: int = 42) -> pl.LazyFrame:
    """Create large synthetic panel data."""
    import random

    random.seed(seed)

    sectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Industrial"]
    start_date = date(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    print(f"\nGenerating: {n_assets:,} assets x {n_days:,} days = {n_assets * n_days:,} rows")

    data = []
    for i in range(n_assets):
        sector = sectors[i % len(sectors)]
        base_price = 10 + (i % 500)
        base_volume = 1e6 + (i % 100) * 5e5

        for day_idx, date_val in enumerate(dates):
            price = base_price * (1 + 0.0001 * day_idx + random.gauss(0, 0.02))
            volume = base_volume * (1 + random.gauss(0, 0.1))

            data.append(
                {
                    "date": date_val,
                    "asset": f"ASSET_{i:04d}",
                    "sector": sector,
                    "close": max(price, 0.01),
                    "volume": max(volume, 1000),
                }
            )

    return pl.DataFrame(data).lazy()


# Custom factors for benchmarks
@time_series
def momentum_quality(window: int = 60, vol_window: int = 20):
    """Momentum / volatility (Sharpe-like)."""
    close = EquityPricing.close
    returns = close.pct_change(1)
    return returns.rolling_sum(window) / returns.rolling_std(vol_window)


@cross_section(by="sector")
def sector_neutral_momentum(window: int = 60):
    """Sector-neutral momentum."""
    return momentum_quality(window=window, vol_window=20).demean()


def benchmark_1_time_series(df: pl.LazyFrame) -> dict:
    """Time-series factors only."""
    print("\n" + "=" * 60)
    print("Benchmark 1: Time-Series Pipeline (4 factors)")
    print("=" * 60)

    start = time.time()
    pipeline = Pipeline(df).add_factors(
        {
            "returns": F.returns(window=1),
            "momentum": F.momentum(window=60, skip=5),
            "sma_20": F.sma(window=20),
            "sma_60": F.sma(window=60),
        }
    )

    result = pipeline.run(collect=True)
    elapsed = time.time() - start

    print(
        f"Result: {result.shape[0]:,} rows, {elapsed:.3f}s, {result.shape[0] / elapsed:,.0f} rows/sec"
    )
    return {"name": "Time-Series", "factors": 4, "rows": result.shape[0], "time": elapsed}


def benchmark_2_cross_sectional(df: pl.LazyFrame) -> dict:
    """Time-series + cross-sectional factors."""
    print("\n" + "=" * 60)
    print("Benchmark 2: Cross-Sectional Pipeline (6 factors)")
    print("=" * 60)

    start = time.time()
    close = EquityPricing.close
    returns = close.pct_change(1)

    pipeline = Pipeline(df).add_factors(
        {
            "returns": returns,
            "momentum": returns.rolling_sum(60),
            "volatility": returns.rolling_std(20),
            "momentum_rank": returns.rolling_sum(60).rank(pct=True),
            "volatility_rank": returns.rolling_std(20).rank(pct=True),
            "return_rank": returns.rank(pct=True),
        }
    )

    result = pipeline.run(collect=True)
    elapsed = time.time() - start

    print(
        f"Result: {result.shape[0]:,} rows, {elapsed:.3f}s, {result.shape[0] / elapsed:,.0f} rows/sec"
    )
    return {"name": "Cross-Sectional", "factors": 6, "rows": result.shape[0], "time": elapsed}


def benchmark_3_sector_neutral(df: pl.LazyFrame) -> dict:
    """Sector-neutral factors (grouped cross-sectional)."""
    print("\n" + "=" * 60)
    print("Benchmark 3: Sector-Neutral Pipeline (6 factors)")
    print("=" * 60)

    start = time.time()
    close = EquityPricing.close
    returns = close.pct_change(1)
    momentum = returns.rolling_sum(60)

    pipeline = Pipeline(df).add_factors(
        {
            "momentum": momentum,
            "volatility": returns.rolling_std(20),
            "global_rank": momentum.rank(pct=True),
            "sector_rank": momentum.rank(pct=True, by="sector"),
            "sector_neutral": momentum.demean(by="sector"),
            "sector_zscore": momentum.zscore(by="sector"),
        }
    )

    result = pipeline.run(collect=True)
    elapsed = time.time() - start

    print(
        f"Result: {result.shape[0]:,} rows, {elapsed:.3f}s, {result.shape[0] / elapsed:,.0f} rows/sec"
    )
    return {"name": "Sector-Neutral", "factors": 6, "rows": result.shape[0], "time": elapsed}


def benchmark_4_complex(df: pl.LazyFrame) -> dict:
    """Complex multi-factor pipeline."""
    print("\n" + "=" * 60)
    print("Benchmark 4: Complex Pipeline (20 factors)")
    print("=" * 60)

    start = time.time()
    close = EquityPricing.close
    volume = EquityPricing.volume
    returns = close.pct_change(1)

    factors = {}

    # Multiple return windows
    for w in [1, 5, 10, 20, 60]:
        factors[f"ret_{w}d"] = F.returns(window=w)

    # Multiple momentum windows
    for w in [20, 60, 120, 252]:
        factors[f"mom_{w}d"] = F.momentum(window=w, skip=5)

    # Multiple SMAs
    for w in [10, 20, 50, 100, 200]:
        factors[f"sma_{w}"] = F.sma(window=w)

    # Cross-sectional
    factors["return_rank"] = returns.rank(pct=True)
    factors["volume_rank"] = volume.rank(pct=True)
    factors["mom_60_rank"] = returns.rolling_sum(60).rank(pct=True)
    factors["sector_neutral_ret"] = returns.demean(by="sector")
    factors["sector_neutral_mom"] = returns.rolling_sum(60).demean(by="sector")

    print(f"Total factors: {len(factors)}")

    pipeline = Pipeline(df).add_factors(factors)
    result = pipeline.run(collect=True)
    elapsed = time.time() - start

    print(
        f"Result: {result.shape[0]:,} rows, {elapsed:.3f}s, {result.shape[0] / elapsed:,.0f} rows/sec"
    )
    return {"name": "Complex", "factors": len(factors), "rows": result.shape[0], "time": elapsed}


def print_summary(benchmarks: list[dict]):
    """Print summary table."""
    print("\n" + "#" * 60)
    print("# Performance Summary")
    print("#" * 60)

    print(f"\n{'Benchmark':<20} {'Factors':<10} {'Rows':<12} {'Time':<10} {'Throughput':<15}")
    print("-" * 65)

    for b in benchmarks:
        print(
            f"{b['name']:<20} {b['factors']:<10} {b['rows']:<12,} {b['time']:<10.3f} {b['rows'] / b['time']:>12,.0f} r/s"
        )


def main():
    print("\n" + "#" * 60)
    print("# factr Performance Benchmarks")
    print("#" * 60)

    # Configuration
    N_ASSETS = 1000
    N_DAYS = 2520  # ~10 years

    print(f"\nConfiguration: {N_ASSETS:,} assets x {N_DAYS:,} days = {N_ASSETS * N_DAYS:,} rows")

    # Generate dataset
    start = time.time()
    df = create_large_dataset(n_assets=N_ASSETS, n_days=N_DAYS)
    print(f"Generation time: {time.time() - start:.3f}s")

    # Run benchmarks
    benchmarks = [
        benchmark_1_time_series(df),
        benchmark_2_cross_sectional(df),
        benchmark_3_sector_neutral(df),
        benchmark_4_complex(df),
    ]

    print_summary(benchmarks)

    print("\nKey insights:")
    print("  - Multi-stage execution separates TIME_SERIES and CROSS_SECTION")
    print("  - Polars lazy evaluation optimizes the expression graph")
    print("  - Grouped operations use .over('date', 'sector') efficiently")

    print("\n" + "#" * 60)
    print("# Benchmarks Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
