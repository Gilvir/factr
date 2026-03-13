# CLAUDE.md

## Development Commands

```bash
# Install
uv pip install -e ".[dev]"

# Test
uv run pytest
uv run pytest --cov=factr --cov-report=term-missing

# Lint
uv run ruff check factr tests examples --fix
uv run ruff format factr tests examples
```

Always use `uv` to run things.

## Architecture

**Core concept: Factor = Polars Expression + Scope metadata**

Scopes determine how `.over()` is applied:
- `RAW` — raw column data
- `TIME_SERIES` — per-entity operations (`.over(entity_column)`)
- `CROSS_SECTION` — per-date operations (`.over(date_column)`)

Pipeline resolves dependencies via `_parent` chain, materializes intermediates, and applies `.over()` automatically.

## Key Patterns

- Factors are frozen dataclasses (immutable, identity-hashed)
- Scope transitions (e.g. TIME_SERIES → CROSS_SECTION) create a `_parent` dependency that Pipeline materializes as an intermediate column
- Prefer pure Polars expressions over `@custom_factor` (which uses `map_batches` and breaks optimization)
- No global state — DataContext is explicit dependency injection
- Dataset system uses Pydantic-inspired Column descriptors with alias, fill_null, default support
