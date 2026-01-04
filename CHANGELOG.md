# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-04

### Added
- Initial release of factr quantitative finance library
- Core Factor/Filter/Classifier system with scope-based execution (RAW, TIME_SERIES, CROSS_SECTION)
- Pipeline orchestration with automatic dependency resolution and topological sorting
- 26+ built-in financial indicators organized by category:
  - Returns: simple returns, log returns, cumulative returns
  - Moving averages: SMA, EMA, VWAP
  - Momentum: RSI, MACD, Stochastic
  - Volatility: standard deviation, ATR, Bollinger Bands
  - Volume indicators: volume ratio, OBV
  - Technical indicators: drawdown, rolling correlation
- DataSet system with Pydantic-inspired Column descriptors for schema definition
- Flexible data source system:
  - ParquetSource for Parquet files
  - SQLSource with SQLAlchemy integration
  - CSVSource for CSV files
  - DataFrameSource for in-memory testing
  - Custom source protocol support
- DataContext for multi-dataset dependency injection and concurrent loading
- Point-in-time data alignment utilities (asof_join, apply_offset, forward_fill)
- Custom factor creation via decorators:
  - `@time_series` for per-entity operations
  - `@cross_section` for per-date operations
  - `@custom_factor` for Python function-based factors
- Built-in universe definitions (Q500US, Q1500US, LiquidUniverse, AllAssets)
- Built-in classifiers (Sector, Industry, Quantiles, MarketCapBuckets)
- Comprehensive test suite with 24+ test files covering core functionality
- Example scripts demonstrating:
  - Quick start guide
  - Factor API usage
  - Data loading patterns
  - SQLite integration
  - Performance benchmarking

### Technical Details
- Built on Polars for high-performance lazy evaluation
- Type-safe factor composition with frozen dataclasses
- No global state - explicit dependency injection pattern
- Automatic `.over()` application based on factor scope
- Query optimization through Polars LazyFrame execution

[Unreleased]: https://github.com/gilvir/factr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gilvir/factr/releases/tag/v0.1.0
