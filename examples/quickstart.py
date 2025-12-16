"""Quickstart: Get started with factr in 5 minutes.

This example shows the basics:
1. Type-safe DataSet columns (no Polars knowledge needed)
2. Factor composition with operators
3. Pipeline execution

Run with: python examples/quickstart.py
"""

import polars as pl

from factr import Pipeline
from factr.datasets import EquityPricing, Fundamentals


def create_sample_data():
    """Create sample panel data."""
    import random

    random.seed(42)

    data = []
    for date in ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]:
        for i, asset in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN"]):
            price = 100 + i * 50 + random.gauss(0, 5)
            data.append(
                {
                    "date": date,
                    "asset": asset,
                    "close": price,
                    "volume": int(1e6 + random.gauss(0, 1e5)),
                    "sector": ["Tech", "Tech", "Tech", "Consumer"][i],
                    "pe_ratio": 20 + i * 5 + random.gauss(0, 2),
                }
            )

    return pl.DataFrame(data).with_columns(pl.col("date").str.to_date()).lazy()


def main():
    print("\n" + "=" * 60)
    print("factr Quickstart")
    print("=" * 60)

    df = create_sample_data()

    # 1. Access columns via DataSets (type-safe, IDE autocomplete)
    close = EquityPricing.close
    volume = EquityPricing.volume
    pe_ratio = Fundamentals.pe_ratio

    # 2. Compose factors with operators
    returns = close.pct_change(1)  # Daily returns
    dollar_volume = close * volume  # Dollar volume
    earnings_yield = 1 / pe_ratio  # Value factor

    # 3. Cross-sectional operations (per-date)
    momentum_rank = returns.rank(pct=True)  # Global rank
    sector_neutral = returns.demean(by="sector")  # Sector-neutral

    # 4. Build and run pipeline
    pipeline = Pipeline(df).add_factors(
        {
            "returns": returns,
            "dollar_volume": dollar_volume,
            "earnings_yield": earnings_yield,
            "momentum_rank": momentum_rank,
            "sector_neutral": sector_neutral,
        }
    )

    result = pipeline.run(collect=True)

    print("\nResult:")
    print(
        result.select(
            ["date", "asset", "sector", "close", "returns", "momentum_rank", "sector_neutral"]
        )
    )

    print("\nKey concepts:")
    print("  - EquityPricing.close returns a Factor (not pl.col)")
    print("  - .pct_change(), .rank() etc. return new Factors")
    print("  - .demean(by='sector') creates sector-neutral factors")
    print("  - Pipeline handles .over() automatically based on scope")

    print("\nNext steps:")
    print("  - See factor_api_example.py for comprehensive API coverage")
    print("  - See data_loading_example.py for loading from files/databases")
    print("  - See performance_example.py for large-scale benchmarks")


if __name__ == "__main__":
    main()
