# factr

Factor library for quantitative finance built on Polars.

## Overview

Composable factor definitions with automatic scope-based execution. Factors wrap Polars expressions with metadata to handle time-series vs cross-sectional operations correctly.

## Installation

```bash
pip install factr
```

For development (using [uv](https://github.com/astral-sh/uv)):
```bash
git clone https://github.com/gilvir/factr.git
cd factr
uv pip install -e ".[dev]"
```

Or with pip:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Factor Composition

```python
import polars as pl
from factr.datasets import EquityPricing
from factr.pipeline import Pipeline

# Load sample data
data = pl.DataFrame({
    'date': ['2024-01-01'] * 3 + ['2024-01-02'] * 3,
    'asset': ['AAPL', 'GOOGL', 'MSFT'] * 2,
    'close': [150.0, 2800.0, 380.0, 152.0, 2820.0, 382.0],
    'volume': [1e6, 2e6, 1.5e6, 1.1e6, 1.9e6, 1.6e6],
})

# Define factors using natural composition
close = EquityPricing.close
returns = close.pct_change(1)
momentum = returns.rolling_sum(252)
ranked = momentum.rank(pct=True)

# Build and run pipeline
pipeline = Pipeline(data).add_factors({
    'momentum': momentum,
    'rank': ranked,
})

result = pipeline.run()
print(result)
```

### Sector-Neutral Strategy

```python
from factr.datasets import EquityPricing
from factr.universe import Q500US
from factr.pipeline import Pipeline
from factr import factors as F

# Define factors
close = EquityPricing.close
momentum = F.momentum(252, 21)  # 252-day momentum, skip last 21 days
volatility = F.volatility(60)

# Sector-neutral ranking
risk_adjusted = momentum / volatility
sector_neutral = risk_adjusted.demean(by='sector')
ranked = sector_neutral.rank(pct=True)

# Build pipeline with universe filter
# Assumes you have a LazyFrame 'prices_lf' with columns: date, asset, close, volume, sector
pipeline = (
    Pipeline(prices_lf)
    .add_factors({
        'sector_neutral_mom': sector_neutral,
        'rank': ranked,
    })
    .screen(Q500US())
)

# Show execution plan
print(pipeline.explain())

# Run pipeline
result = pipeline.run(start_date='2020-01-01')
```

### Dataset Loading

Configure data sources and field transforms using Pydantic-inspired patterns:

```python
from factr.datasets import DataSet, Column
from factr.data import ParquetSource, SQLSource, DataContext
import polars as pl

# Define dataset with field-level configuration
class EquityPricing(DataSet):
    # Simple columns
    close = Column(pl.Float64)
    volume = Column(pl.Int64)

    # With alias (source has different name)
    market_cap = Column(pl.Float64, alias='mkt_cap', fill_strategy='forward')

    # With null filling
    sentiment = Column(pl.Float64, fill_null=0.0)

    sector = Column(pl.Utf8, default='Unknown', required=False)

    class Config:
        source = ParquetSource('data/prices.parquet')
        date_column = 'date'
        entity_column = 'ticker'

# Load data - automatically applies:
# - Column name mappings (aliases)
# - Null filling strategies
# - Defaults for missing columns
prices = EquityPricing.load(start_date='2020-01-01')

# Explicit source binding with DataContext
class Fundamentals(DataSet):
    market_cap = Column(pl.Float64)
    pe_ratio = Column(pl.Float64)
    # No Config - bind source explicitly at runtime

# Bind source and load
ctx = DataContext()
ctx.bind(Fundamentals, ParquetSource('data/fundamentals.parquet'))
funds = ctx.load(Fundamentals, start_date='2020-01-01')

# Use DataContext for complex workflows (with concurrent loading)
ctx = DataContext()
ctx.bind(EquityPricing, ParquetSource('prices.parquet'))
ctx.bind(Fundamentals, SQLSource('db', table='fundamentals'))

# Concurrent collection for better performance
data = ctx.load_many(
    EquityPricing,
    Fundamentals,
    start_date='2020-01-01',
    collect=True  # Collect all datasets concurrently
)
```

**Key Features:**

- **Pydantic-inspired Column fields** - alias, default, fill_null, validation bounds
- **Composition over inheritance** - datasets compose columns and sources
- **Field-level transformations** - automatic null filling, bounds enforcement
- **Flexible source configuration** - direct source in Config or explicit binding via DataContext
- **Concurrent loading** - use `collect=True` for parallel execution
- **No global state** - DataContext is explicit and composable
- **Testing-friendly** - clone contexts, swap sources easily

See `examples/data_loading_example.py` for comprehensive examples.

## Core Concepts

### Factors = Expressions + Scope

```python
from factr.core import Factor, Scope

close = EquityPricing.close  # RAW scope
returns = close.pct_change(1)  # TIME_SERIES scope
ranked = returns.rank(pct=True)  # CROSS_SECTION scope
```

Scopes:

- `RAW` - raw column data
- `TIME_SERIES` - per-entity operations (rolling windows, shifts)
- `CROSS_SECTION` - per-date operations (rank, demean, zscore)

### Pipeline

```python
pipeline = Pipeline(data).add_factors({'momentum': momentum, 'rank': ranked})
result = pipeline.run()
```

Pipeline handles `.over()` application based on scope automatically.

### Datasets

```python
from factr.datasets import EquityPricing, Fundamentals

close = EquityPricing.close
volume = EquityPricing.volume
market_cap = Fundamentals.market_cap
```

### Built-in Factors

```python
from factr import factors as F

momentum = F.momentum(252, 21)
returns = F.returns(1)
sma_50 = F.sma(50)
rsi = F.rsi(14)
```

## Factor Library

The library includes 26+ production-ready financial indicators across multiple categories:

### Price & Returns

- **`returns(window=1)`** - Simple returns
- **`log_returns(window=1)`** - Logarithmic returns
- **`momentum(window=252, skip=21)`** - Price momentum
- **`reversal(window=21)`** - Short-term mean reversion

### Technical Indicators

- **`sma(window=20)`** - Simple moving average
- **`ema(window=20)`** - Exponential moving average
- **`macd(fast=12, slow=26, signal=9)`** - MACD with signal line and histogram
- **`rsi(window=14)`** - Relative Strength Index
- **`bollinger_bands(window=20, num_std=2.0)`** - Bollinger Bands (lower, middle, upper)
- **`stochastic(window=14, smooth_k=3, smooth_d=3)`** - Stochastic Oscillator (%K, %D)
- **`atr(window=14)`** - Average True Range (volatility measure)
- **`parabolic_sar()`** - Parabolic Stop and Reverse

### Momentum & Trend

- **`acceleration(window=21)`** - Price acceleration (2nd derivative)
- **`trend_strength(window=63)`** - Linear regression R-squared

### Risk Indicators

- **`volatility(window=21, annualize=True)`** - Rolling volatility
- **`max_drawdown(window=252)`** - Maximum peak-to-trough decline
- **`downside_deviation(window=21, annualize=True)`** - Semi-deviation (downside only)

### Volume Indicators

- **`dollar_volume(window=1)`** - Price × Volume
- **`vwap(window=20)`** - Volume-Weighted Average Price
- **`vwap_bands(window=20, num_std=2.0)`** - VWAP with std bands
- **`obv()`** - On-Balance Volume
- **`chaikin_money_flow(window=21)`** - CMF indicator
- **`volume_profile(window=21)`** - Normalized volume

### Statistical

- **`autocorrelation(window=21, lag=1)`** - Rolling autocorrelation

### Value Factors

- **`earnings_yield()`** - 1 / P/E ratio
- **`book_to_market()`** - 1 / P/B ratio
- **`profit_margin()`** - Earnings / Revenue

### Growth Factors

- **`revenue_growth(window=252)`** - YoY revenue growth
- **`earnings_growth(window=252)`** - YoY earnings growth

### Usage Examples

```python
from factr import factors as F
from factr.datasets import EquityPricing, Fundamentals

# Technical indicators
macd_line, signal, histogram = F.macd(fast=12, slow=26)
percent_k, percent_d = F.stochastic(window=14)
atr_value = F.atr(window=14)

# Risk-adjusted momentum
mom = F.momentum(252, 21)
vol = F.volatility(60)
sharpe = mom / vol

# Volume analysis
obv_factor = F.obv()
cmf = F.chaikin_money_flow(window=21)
lower, vwap_mid, upper = F.vwap_bands(window=20)

# Value investing
earnings_yield = F.earnings_yield(Fundamentals.pe_ratio)
margin = F.profit_margin()
growth = F.earnings_growth(window=252)

# Mean reversion
reversal = F.reversal(window=21)  # Negative of recent returns
acf = F.autocorrelation(window=21, lag=1)

# Combine factors
alpha = (
    F.momentum(252, 21).zscore() +
    F.earnings_yield().zscore() +
    F.volume_profile().zscore()
) / 3
```

### Custom Factors

#### Composing Polars Expressions

```python
from factr import custom

@custom.time_series
def momentum_quality(window: int = 5):
    close = EquityPricing.close
    momentum = close.pct_change(window)
    trend = close > close.shift(1)
    return trend * momentum

@custom.cross_section(by='sector')
def sector_neutral_momentum():
    mom = F.momentum(252, 21)
    return mom.demean()
```

#### Custom Python Functions

For calculations that can't be expressed in Polars (e.g., using numpy, scipy, ta-lib):

```python
from factr import custom_factor
from factr.core import Scope
from factr.datasets import EquityPricing
import polars as pl

# Time-series custom factor (per-entity)
# Can use Factor objects or string column names as inputs
@custom_factor(
    scope=Scope.TIME_SERIES,
    inputs=[EquityPricing.close, EquityPricing.volume]  # Factor objects for type safety
)
def custom_indicator(df: pl.DataFrame) -> pl.Series:
    """Uses numpy/scipy for complex calculations."""
    import numpy as np
    close = df['close'].to_numpy()
    volume = df['volume'].to_numpy()

    # Your custom logic here
    result = np.some_complex_calculation(close, volume)
    return pl.Series(result)

# Or use string column names
@custom_factor(scope=Scope.TIME_SERIES, inputs=['close', 'volume'])
def custom_indicator_v2(df: pl.DataFrame) -> pl.Series:
    return df['close'] * df['volume']

# Cross-sectional custom factor (per-date)
@custom_factor(scope=Scope.CROSS_SECTION, inputs=['returns'], groupby='sector')
def sector_adjusted(df: pl.DataFrame) -> pl.Series:
    """Custom sector-neutral calculation."""
    import numpy as np
    returns = df['returns'].to_numpy()

    # Apply custom transformation
    adjusted = custom_logic(returns)
    return pl.Series(adjusted)

# Use in pipeline like any other factor
factor = custom_indicator()
pipeline = Pipeline(data).add_factors({'custom': factor})
result = pipeline.run()
```

**Note:** Custom factors use `map_batches` which breaks Polars' query optimization. Use only when necessary - prefer pure Polars expressions when possible.

### Universe Filters

```python
from factr.universe import Q500US, LiquidUniverse

q500 = Q500US()
pipeline.screen(q500)
```

## Architecture

```
Factor = pl.Expr + Scope metadata
```

Polars handles expression dependencies via lazy evaluation. We track scope to apply `.over()` correctly.

```
factr/
├── core/          # Factor, Filter, Classifier, Scope
├── pipeline.py    # Multi-stage orchestration
├── factors.py     # Built-in factors
├── datasets.py    # Type-safe column access
├── universe.py    # Universe filters
└── custom.py      # Decorators
```

## Examples

See the [examples/](examples/) directory for complete runnable examples:
- `quickstart.py` - Get started in 5 minutes
- `factor_api_example.py` - Comprehensive API coverage
- `data_loading_example.py` - Data loading patterns
- `sqlite_example.py` - SQLite integration
- `performance_example.py` - Large-scale benchmarking

```python
from factr import factors as F
from factr.pipeline import Pipeline
from factr.universe import Q500US

# Multi-factor alpha combining momentum and value
mom = F.momentum(252, 21)
value = F.earnings_yield()
alpha = (mom + value) / 2

# Build pipeline (assumes you have data loaded)
pipeline = Pipeline(prices_lf).add_factors({'alpha': alpha}).screen(Q500US())
result = pipeline.run(start_date='2020-01-01')
```

## Testing

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=factr --cov-report=term-missing

# Format code
uv run ruff format factr tests examples

# Lint and auto-fix
uv run ruff check factr tests examples --fix
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## License

MIT
