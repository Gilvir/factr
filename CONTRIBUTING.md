# Contributing to factr

Thank you for your interest in contributing to factr! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.11 or higher (3.11-3.14 supported)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/gilvir/factr.git
cd factr

# Install with development dependencies
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=factr --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_factor.py

# Run tests with verbose output
uv run pytest -v

# Run tests for a specific function
uv run pytest tests/test_factor.py::test_factor_composition
```

## Code Quality

### Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check code with ruff
uv run ruff check factr tests examples

# Auto-fix issues
uv run ruff check factr tests examples --fix

# Format code
uv run ruff format factr tests examples

# Check formatting without changes
uv run ruff format --check factr tests examples
```

### Type Checking

This project uses type hints throughout. Ensure your contributions include proper type annotations for all public APIs.

## Coding Standards

- **Follow PEP 8**: Style guidelines are enforced by ruff
- **Maximum line length**: 100 characters
- **Use type hints**: For all public functions and classes
- **Write docstrings**: For public functions and classes using Google-style docstrings
- **Maintain immutability**: Factor objects must be frozen dataclasses
- **Avoid global state**: Use explicit dependency injection via DataContext

### Code Style Example

```python
from dataclasses import dataclass
import polars as pl
from factr.core import Factor, Scope

@dataclass(frozen=True)
class MyFactor(Factor):
    """Brief description of the factor.

    Args:
        window: Lookback window in periods
        min_periods: Minimum periods required for calculation

    Returns:
        Factor with TIME_SERIES scope
    """

    def rolling_custom(self, window: int, min_periods: int | None = None) -> Factor:
        """Calculate rolling custom metric.

        Args:
            window: Lookback window in periods
            min_periods: Minimum periods required (defaults to window)

        Returns:
            New Factor with TIME_SERIES scope
        """
        min_periods = min_periods or window
        expr = self.expr.rolling_mean(window, min_periods=min_periods)
        return self._new(expr=expr, scope=Scope.TIME_SERIES)
```

## Testing Guidelines

- **Write tests for all new features**: Ensure comprehensive coverage
- **Maintain or improve coverage**: Current coverage should not decrease
- **Use `DataFrameSource`**: For in-memory test data to keep tests fast
- **Use `hypothesis`**: For property-based testing when appropriate
- **Test across Python versions**: Ensure tests pass on Python 3.11-3.14
- **Keep tests fast**: Use small synthetic datasets

### Test Example

```python
import polars as pl
from factr.data import DataFrameSource, DataContext
from factr.datasets import EquityPricing
from factr.pipeline import Pipeline

def test_my_feature():
    # Create test data
    test_data = pl.DataFrame({
        'date': ['2024-01-01'] * 2 + ['2024-01-02'] * 2,
        'asset': ['A', 'B'] * 2,
        'close': [100.0, 200.0, 102.0, 198.0],
        'volume': [1000, 2000, 1100, 1900],
    })

    # Setup context
    ctx = DataContext()
    ctx.bind(EquityPricing, DataFrameSource(test_data))

    # Test factor
    close = EquityPricing.close
    returns = close.pct_change(1)

    # Run pipeline
    pipeline = Pipeline(test_data).add_factors({'returns': returns})
    result = pipeline.run()

    # Assertions
    assert 'returns' in result.columns
    assert result.filter(pl.col('date') == '2024-01-02')['returns'].to_list() == [0.02, -0.01]
```

## Design Principles

When contributing, please adhere to these core design principles:

### 1. Scope-Based Execution
Maintain the Factor = Expression + Scope paradigm. Every factor must have a clear scope (RAW, TIME_SERIES, or CROSS_SECTION) that determines how it's executed.

### 2. Lazy Evaluation
Prefer Polars expressions over custom Python functions. Polars expressions enable query optimization and predicate pushdown.

### 3. Immutability
Factors must be frozen dataclasses. This enables safe caching, identity-based hashing, and prevents accidental modifications.

### 4. No Global State
Use explicit dependency injection via DataContext. No singletons or global configuration.

### 5. Point-in-Time Correctness
Ensure no lookahead bias in factor calculations. Use data alignment utilities (asof_join, apply_offset) appropriately.

## Pull Request Process

1. **Fork the repository** and create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following the coding standards above

3. **Run tests and linting**:
   ```bash
   uv run pytest
   uv run ruff check factr tests examples --fix
   ```

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add amazing feature for cross-sectional ranking"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update", "Refactor")
- Keep first line under 72 characters
- Reference issues when applicable (e.g., "Fix #123: Correct scope transition bug")

### Examples

```
Add momentum quality factor combining trend and returns

Fix scope transition bug in cross-sectional operations

Update documentation for custom factor decorators

Refactor Pipeline dependency resolution for clarity
```

## Documentation

When adding features, please update documentation:

- **README.md**: For user-facing features
- **CLAUDE.md**: For architectural changes or internal patterns
- **Examples**: Add example scripts for significant new features
- **Docstrings**: Include usage examples in docstrings

## Reporting Bugs

When reporting bugs, please include:

1. **Minimal reproducible example**: Code that demonstrates the bug
2. **Expected behavior**: What you expected to happen
3. **Actual behavior**: What actually happened
4. **Environment details**:
   - factr version
   - Python version
   - OS
   - Installation method (pip, uv)

## Requesting Features

When requesting features:

1. **Use case**: Explain the problem this solves
2. **Proposed API**: Show how you'd like the API to look
3. **Alternatives**: Other approaches you've considered
4. **Context**: Any additional relevant information

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
