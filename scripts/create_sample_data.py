"""Script to create sample data files for examples."""

import polars as pl
from datetime import date, timedelta
import random

random.seed(42)


# Generate sample pricing data
def create_pricing_data():
    """Create sample equity pricing data."""
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(252 * 2)]  # 2 years
    tickers = [f"STOCK_{i:03d}" for i in range(100)]
    sectors = ["Tech", "Finance", "Healthcare", "Energy", "Consumer"]

    data = []
    for ticker_idx, ticker in enumerate(tickers):
        base_price = 50 + ticker_idx * 5
        sector = sectors[ticker_idx % len(sectors)]

        for day_idx, date_val in enumerate(dates):
            # Simulate price movements
            price_change = random.gauss(0, 0.02)
            price = base_price * (1 + price_change * day_idx / 100)
            volume = int(1e6 * (1 + random.gauss(0, 0.3)))

            data.append(
                {
                    "date": date_val,
                    "ticker": ticker,
                    "asset": ticker,  # Add alias
                    "open": max(price * 0.99, 1.0),
                    "high": max(price * 1.01, 1.0),
                    "low": max(price * 0.98, 1.0),
                    "close": max(price, 1.0),
                    "volume": max(volume, 1000),
                    "sector": sector,
                }
            )

    df = pl.DataFrame(data)

    # Save to parquet
    df.write_parquet("data/prices.parquet")
    df.write_parquet("examples/data/prices.parquet")
    print(f"Created prices.parquet: {df.shape[0]:,} rows")

    # Also save as CSV for CSV examples
    df.write_csv("data/prices.csv")
    df.write_csv("examples/data/prices.csv")
    print(f"Created prices.csv: {df.shape[0]:,} rows")

    return df


# Generate fundamentals data (quarterly)
def create_fundamentals_data():
    """Create sample fundamentals data (quarterly)."""
    # Quarterly dates for 2 years
    dates = []
    for year in [2020, 2021]:
        for quarter in ["03-31", "06-30", "09-30", "12-31"]:
            dates.append(date.fromisoformat(f"{year}-{quarter}"))

    tickers = [f"STOCK_{i:03d}" for i in range(100)]

    data = []
    for ticker in tickers:
        for date_val in dates:
            data.append(
                {
                    "date": date_val,
                    "ticker": ticker,
                    "asset": ticker,
                    "market_cap": random.uniform(1e9, 100e9),
                    "pe_ratio": random.uniform(5, 50),
                    "earnings": random.uniform(1e6, 1e8),
                }
            )

    df = pl.DataFrame(data)
    df.write_parquet("data/fundamentals.parquet")
    df.write_parquet("examples/data/fundamentals.parquet")
    print(f"Created fundamentals.parquet: {df.shape[0]:,} rows")

    return df


# Generate reference data
def create_reference_data():
    """Create sample reference data."""
    tickers = [f"STOCK_{i:03d}" for i in range(100)]
    sectors = ["Tech", "Finance", "Healthcare", "Energy", "Consumer"]
    industries = {
        "Tech": ["Software", "Hardware", "Semiconductors"],
        "Finance": ["Banking", "Insurance", "Asset Management"],
        "Healthcare": ["Pharmaceuticals", "Biotech", "Medical Devices"],
        "Energy": ["Oil & Gas", "Renewables", "Utilities"],
        "Consumer": ["Retail", "Food & Beverage", "E-commerce"],
    }

    data = []
    for idx, ticker in enumerate(tickers):
        sector = sectors[idx % len(sectors)]
        industry = random.choice(industries[sector])

        data.append(
            {
                "ticker": ticker,
                "asset": ticker,
                "gics_sector": sector,  # Source column
                "gics_industry": industry,  # Source column
                "name": f"Company {ticker}",
            }
        )

    df = pl.DataFrame(data)
    df.write_parquet("data/reference.parquet")
    df.write_parquet("examples/data/reference.parquet")
    print(f"Created reference.parquet: {df.shape[0]:,} rows")

    return df


# Generate sentiment data (daily)
def create_sentiment_data():
    """Create sample sentiment data."""
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(252 * 2)]
    tickers = [f"STOCK_{i:03d}" for i in range(100)]

    data = []
    for ticker in tickers:
        for date_val in dates:
            data.append(
                {
                    "date": date_val,
                    "ticker": ticker,
                    "asset": ticker,
                    "news_sentiment": random.gauss(0, 1),
                    "social_volume": int(random.uniform(100, 10000)),
                }
            )

    df = pl.DataFrame(data)
    df.write_parquet("data/sentiment.parquet")
    df.write_parquet("examples/data/sentiment.parquet")
    print(f"Created sentiment.parquet: {df.shape[0]:,} rows")

    return df


# Create simple dataset
def create_simple_data():
    """Create simple dataset for minimal examples."""
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    tickers = ["AAPL", "GOOGL", "MSFT"]

    data = []
    for ticker in tickers:
        base_price = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300}[ticker]
        for date_val in dates:
            price = base_price * (1 + random.gauss(0, 0.02))
            data.append(
                {
                    "date": date_val,
                    "ticker": ticker,
                    "asset": ticker,
                    "close": max(price, 1.0),
                    "volume": int(1e6 * (1 + random.gauss(0, 0.3))),
                }
            )

    df = pl.DataFrame(data)
    df.write_parquet("data/simple.parquet")
    df.write_parquet("examples/data/simple.parquet")
    print(f"Created simple.parquet: {df.shape[0]:,} rows")

    return df


# Create combined dataset
def create_combined_data(prices_df, fundamentals_df, reference_df):
    """Create combined dataset for multi-source examples."""
    # Join pricing with fundamentals and reference
    # Note: Prices already has sector column, so we'll keep that
    combined = prices_df.join(
        fundamentals_df.select(["date", "ticker", "market_cap", "pe_ratio"]),
        on=["date", "ticker"],
        how="left",
    )

    combined.write_parquet("data/combined.parquet")
    combined.write_parquet("examples/data/combined.parquet")
    print(f"Created combined.parquet: {combined.shape[0]:,} rows")

    return combined


if __name__ == "__main__":
    print("Creating sample data files for examples...\n")

    # Create all data files
    prices = create_pricing_data()
    fundamentals = create_fundamentals_data()
    reference = create_reference_data()
    sentiment = create_sentiment_data()
    simple = create_simple_data()
    combined = create_combined_data(prices, fundamentals, reference)

    print("\n✓ All sample data files created successfully!")
    print("\nFiles created in both data/ and examples/data/:")
    print("  • prices.parquet")
    print("  • prices.csv")
    print("  • fundamentals.parquet")
    print("  • reference.parquet")
    print("  • sentiment.parquet")
    print("  • simple.parquet")
    print("  • combined.parquet")
