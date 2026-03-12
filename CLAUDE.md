# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**factr** is a quantitative finance factor library built on Polars. It enables composable factor definitions with automatic scope-based execution. Factors wrap Polars expressions with metadata to handle time-series vs cross-sectional operations correctly.

## Development Commands

### Installation
```bash
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_factor.py

# Run with coverage
pytest --cov=factr --cov-report=term-missing
```

### Code Quality
```bash
# Format code (not installed by default - use if available)
black factr tests examples

# Lint and auto-fix
ruff check factr tests examples --fix
```

## Architecture

### Core Design Principle: Factor = Expression + Scope

The entire library is built around wrapping Polars expressions with scope metadata. This enables:
1. Automatic `.over()` application based on operation type
2. Lazy evaluation with Polars query optimization
3. Type-safe factor composition
4. Dependency tracking for multi-stage execution

### Scope System (Critical Concept)

The `Scope` enum determines how expressions are executed:

- **`Scope.RAW`**: Raw column data, no `.over()` needed
  - Example: `EquityPricing.close`, `ReferenceData.sector`

- **`Scope.TIME_SERIES`**: Per-entity operations using `.over(entity_column)`
  - Example: `close.pct_change(1)`, `close.rolling_mean(20)`
  - These operate independently for each asset/entity

- **`Scope.CROSS_SECTION`**: Per-date operations using `.over(date_column, *groupby)`
  - Example: `returns.rank()`, `momentum.zscore(by='sector')`
  - These operate across assets within each date

**Key insight**: When a TIME_SERIES factor is input to a CROSS_SECTION operation, the Pipeline automatically materializes it as an intermediate column before computing the cross-section factor.

### Factor Class Structure

```python
@dataclass(frozen=True, eq=False)
class Factor:
    name: str                           # Unique identifier
    expr: pl.Expr                       # Lazy Polars expression
    scope: Scope                        # RAW | TIME_SERIES | CROSS_SECTION
    groupby: list[str] | None          # For CROSS_SECTION grouping
    source_columns: frozenset[str]     # Tracks column dependencies
    source_datasets: frozenset[type]   # Tracks dataset dependencies
    _parent: Factor | None             # Tracks intermediate dependencies

Filter(Factor)       # Boolean-valued factors (for universe filtering)
Classifier(Factor)   # Categorical factors (for grouping operations)
```

### Dependency Resolution and Execution Flow

The Pipeline executes factors in dependency order:

1. **Dependency Collection**: Traverse `_parent` chain to find all required intermediate factors
2. **Topological Sort**: Order factors so dependencies are computed first
3. **Lazy Expression Building**: Build `.with_columns()` chain with correct `.over()` application
4. **Materialization**: Collect LazyFrame only when needed

Example:
```python
close = EquityPricing.close              # RAW
returns = close.pct_change(1)            # TIME_SERIES, expr references pl.col('close')
ranked = returns.rank()                   # CROSS_SECTION, _parent=returns

# Pipeline execution order:
# 1. Compute returns: pl.col('close').pct_change(1).over('asset')
# 2. Materialize as intermediate column 'factor_xyz'
# 3. Compute ranked: pl.col('factor_xyz').rank().over('date')
```

### Data Loading System

The dataset system uses a Pydantic-inspired pattern with field descriptors:

**Column Descriptor Pattern**:
```python
class EquityPricing(DataSet):
    close = Column(pl.Float64)                                    # Simple column
    market_cap = Column(pl.Float64, alias='mkt_cap')            # Map from source name
    sentiment = Column(pl.Float64, fill_null=0.0)               # Null handling
    sector = Column(pl.Utf8, default='Unknown', required=False) # Optional with default

    class Config:
        source = ParquetSource('data/prices.parquet')
        date_column = 'date'
        entity_column = 'ticker'
```

**Three-tier source binding priority**:
1. Explicit source parameter: `EquityPricing.load(source=custom_source)`
2. DataContext binding: `ctx.bind(EquityPricing, source)`
3. Dataset Config: `EquityPricing.Config.source`

**DataContext** provides dependency injection for multi-dataset workflows:
```python
ctx = DataContext()
ctx.bind(EquityPricing, ParquetSource('prices.parquet'))
ctx.bind(Fundamentals, SQLSource('db', table='fundamentals'))

# Concurrent loading for better performance
data = ctx.load_many(EquityPricing, Fundamentals, collect=True)
```

### Module Structure

```
factr/
├── core/
│   ├── factor.py      # Factor/Filter/Classifier classes + dependency helpers
│   └── scope.py       # Scope enum definition
├── data/
│   ├── alignment.py   # asof_join, apply_offset, forward_fill
│   ├── binding.py     # DataSource/ColumnMapper protocols
│   ├── config.py      # DataSetConfig for source configuration
│   ├── context.py     # DataContext for multi-dataset management
│   ├── loaders.py     # combine_sources helper
│   └── sources.py     # ParquetSource, SQLSource, DataFrameSource, etc.
├── classifiers.py     # Sector, Industry, Quantiles, MarketCapBuckets
├── custom.py          # @factor_func, @time_series, @cross_section decorators
├── datasets.py        # Column descriptor + DataSet base class
├── factors.py         # 26+ built-in financial indicators
├── pipeline.py        # Multi-stage orchestrator
└── universe.py        # Q500US, LiquidUniverse, etc.
```

## Key Implementation Patterns

### Scope Transition Handling

When creating derived factors, use the `_new()` helper method to properly handle scope transitions:

```python
def rank(self, pct: bool = False, by: list[str] | None = None):
    base_expr, parent = self._cs_context(by)  # Get context for cross-section
    expr = base_expr.rank(descending=True, pct=pct)
    return self._new(
        expr=expr,
        scope=Scope.CROSS_SECTION,
        groupby=by,
        _parent=parent  # Critical: tracks dependency for materialization
    )
```

### Custom Factor Creation

**For Polars expressions (preferred)**:
```python
from factr import custom

@custom.time_series
def momentum_quality(window: int = 5):
    close = EquityPricing.close
    momentum = close.pct_change(window)
    trend = close > close.shift(1)
    return trend * momentum
```

**For Python functions (use sparingly - breaks optimization)**:
```python
from factr import custom_factor
from factr.core import Scope

@custom_factor(
    scope=Scope.TIME_SERIES,
    inputs=[EquityPricing.close, EquityPricing.volume]
)
def custom_indicator(df: pl.DataFrame) -> pl.Series:
    import numpy as np
    close = df['close'].to_numpy()
    volume = df['volume'].to_numpy()
    result = np.some_complex_calculation(close, volume)
    return pl.Series(result)
```

### Point-in-Time Correctness

Use data alignment helpers to maintain point-in-time correctness:

```python
from factr.data import asof_join, apply_offset, forward_fill

# Apply reporting lag (e.g., quarterly data reported 45 days later)
fundamentals_lf = apply_offset(fundamentals_lf, offset_days=45)

# Join with last-available semantics
joined = asof_join(prices_lf, fundamentals_lf, on='date', by='asset')

# Forward-fill sparse data within entities
filled = forward_fill(joined, columns=['pe_ratio', 'market_cap'], by='asset')
```

## Important Notes

### Factor Immutability

Factors are frozen dataclasses - they cannot be mutated after creation. This enables:
- Safe caching and memoization
- Identity-based hashing (`__hash__` uses `id(self)`)
- Prevention of accidental modifications

### Expression vs Column References

When a factor transitions scope (e.g., TIME_SERIES → CROSS_SECTION):
- The new factor's `expr` references the **column name** (via `pl.col(parent.name)`)
- The `_parent` field tracks the **factor dependency**
- Pipeline materializes the parent factor before executing the child

### Lazy Evaluation

All factor operations are lazy until `Pipeline.run()` is called:
- Factor composition creates expression DAGs, not data
- Pipeline builds a single Polars LazyFrame with chained `.with_columns()`
- Optimization happens at `.collect()` time (predicate pushdown, column selection)

### No Global State

The library avoids global state:
- DataContext is explicit dependency injection
- No singletons or global configuration
- Easy to test with different source configurations
- Safe for concurrent pipelines with different data sources

## Testing Patterns

When writing tests:
- Use `DataFrameSource` for in-memory test data
- Create test DataContext for source isolation
- Test factor composition separately from execution
- Use small synthetic datasets for fast tests

Example:
```python
import polars as pl
from factr.data import DataFrameSource, DataContext
from factr.datasets import EquityPricing

# Create test data
test_data = pl.DataFrame({
    'date': ['2024-01-01'] * 2 + ['2024-01-02'] * 2,
    'asset': ['A', 'B'] * 2,
    'close': [100.0, 200.0, 102.0, 198.0],
})

# Bind to test context
ctx = DataContext()
ctx.bind(EquityPricing, DataFrameSource(test_data))

# Test factor
close = EquityPricing.close
returns = close.pct_change(1)
assert returns.scope == Scope.TIME_SERIES
```

## Common Pitfalls

1. **Don't mix scopes incorrectly**: A CROSS_SECTION factor cannot be input to a TIME_SERIES operation (would violate temporal ordering)

2. **Custom factors break optimization**: Use `@custom_factor` only when Polars expressions are insufficient. They use `map_batches` which prevents query optimization.

3. **Column aliasing**: If source data has different column names, use `Column(alias='source_name')` rather than renaming in the source

4. **Groupby in cross-section**: When using `by` parameter, the grouping columns must exist in the data. Use Classifiers for categorical grouping.

5. **Data sorting**: Pipeline automatically sorts by `[entity_column, date_column]`. Don't pre-sort data unless required for specific operations.
- always us us uv to run stuff