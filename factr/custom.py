"""CustomFactor decorators and helpers for user-defined factors."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Sequence
import warnings

import polars as pl

from .core import Factor, Scope


def factor_func(func: Callable[..., Factor]) -> Callable[..., Factor]:
    """Decorator for reusable factor functions with automatic scope inference."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Factor:
        factor_inputs = []
        for arg in args:
            if isinstance(arg, Factor):
                factor_inputs.append(arg)
        for value in kwargs.values():
            if isinstance(value, Factor):
                factor_inputs.append(value)

        factor = func(*args, **kwargs)

        if factor_inputs:
            has_cross_section = any(f.scope == Scope.CROSS_SECTION for f in factor_inputs)
            if has_cross_section and factor.scope != Scope.CROSS_SECTION:
                object.__setattr__(factor, "scope", Scope.CROSS_SECTION)

        if not factor.name or factor.name.startswith("factor_"):
            object.__setattr__(factor, "name", func.__name__)
        return factor

    wrapper.__factor_func__ = True
    return wrapper


def rolling_factor(window: int, inputs: list[str] | None = None, **input_cols: str) -> Callable:
    """Decorator for custom rolling window factors."""

    def decorator(func: Callable) -> Callable[..., Factor]:
        @wraps(func)
        def wrapper(**override_cols) -> Factor:
            if inputs:
                col_map = {name: override_cols.get(name, name) for name in inputs}
            else:
                col_map = {
                    param: override_cols.get(param, col) for param, col in input_cols.items()
                }

            exprs = {param: pl.col(col) for param, col in col_map.items()}
            result_expr = func(**exprs)
            return Factor(expr=result_expr, name=func.__name__, scope=Scope.TIME_SERIES)

        wrapper.__rolling_factor__ = True
        wrapper.__window__ = window
        wrapper.__inputs__ = inputs or list(input_cols.keys())
        return wrapper

    return decorator


def expression_factor(name: str | None = None, scope: Scope = Scope.TIME_SERIES) -> Callable:
    """Wraps a function returning pl.Expr into a Factor."""

    def decorator(func: Callable) -> Callable[..., Factor]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Factor:
            expr = func(*args, **kwargs)
            factor_name = name or func.__name__
            return Factor(expr=expr, name=factor_name, scope=scope)

        return wrapper

    return decorator


def time_series(func: Callable[..., Factor]) -> Callable[..., Factor]:
    """Marks a factor as TIME_SERIES scope."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Factor:
        factor = func(*args, **kwargs)
        if factor.scope != Scope.TIME_SERIES:
            object.__setattr__(factor, "scope", Scope.TIME_SERIES)
        if not factor.name or factor.name.startswith("factor_"):
            object.__setattr__(factor, "name", func.__name__)
        return factor

    wrapper.__time_series__ = True
    return wrapper


def cross_section(by: list[str] | str | None = None) -> Callable:
    """Marks a factor as CROSS_SECTION scope."""

    def decorator(func: Callable[..., Factor]) -> Callable[..., Factor]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Factor:
            factor = func(*args, **kwargs)
            if factor.scope != Scope.CROSS_SECTION:
                object.__setattr__(factor, "scope", Scope.CROSS_SECTION)
            if by is not None:
                groupby = [by] if isinstance(by, str) else by
                if factor.groupby != groupby:
                    by_str = ",".join(groupby) if isinstance(groupby, list) else groupby
                    object.__setattr__(factor, "name", f"{factor.name}[by={by_str}]")
                object.__setattr__(factor, "groupby", groupby)
            if not factor.name or factor.name.startswith("factor_"):
                object.__setattr__(factor, "name", func.__name__)
            return factor

        wrapper.__cross_section__ = True
        wrapper.__groupby__ = by
        return wrapper

    return decorator


def make_factor(
    expr: pl.Expr | Callable[..., pl.Expr], name: str | None = None, **col_mappings
) -> Factor | Callable[..., Factor]:
    """Create a Factor from an expression or expression factory."""
    if isinstance(expr, pl.Expr):
        return Factor(expr=expr, name=name or "custom")
    else:

        def factory(**kwargs) -> Factor:
            params = {**col_mappings, **kwargs}
            result_expr = expr(**params)
            return Factor(expr=result_expr, name=name or expr.__name__)

        return factory


def custom_factor(
    scope: Scope,
    inputs: Sequence[str | Factor],
    groupby: str | list[str] | None = None,
    output_name: str | None = None,
    return_dtype: pl.DataType = pl.Float64,
) -> Callable:
    """Decorator for custom Python functions that can't be expressed as Polars expressions.

    This allows you to write arbitrary Python logic (using numpy, scipy, etc.) while
    still integrating with the Factor/Pipeline system. The decorated function will be
    executed using Polars' map_batches, which breaks optimization but enables flexibility.

    Args:
        scope: Execution scope (TIME_SERIES or CROSS_SECTION).
               - TIME_SERIES: Function receives data for one entity across all dates
               - CROSS_SECTION: Function receives data for one date across all entities
        inputs: Column names or Factor objects the function depends on.
                If Factor objects are provided, their source columns will be extracted.
                These will be passed to your function as a DataFrame.
        groupby: Optional grouping columns for CROSS_SECTION scope (e.g., 'sector').
        output_name: Optional name for the resulting factor. Defaults to function name.
        return_dtype: Polars data type for the returned series. Defaults to pl.Float64.

    Returns:
        A decorator that wraps your function and returns a Factor-producing callable.

    Performance Note:
        Custom factors use map_batches which breaks Polars' query optimization and
        parallelization. Use only when necessary - prefer pure Polars expressions when possible.

    Examples:
        Basic time-series custom factor:
        >>> @custom_factor(scope=Scope.TIME_SERIES, inputs=['close', 'volume'])
        ... def custom_indicator(df: pl.DataFrame) -> pl.Series:
        ...     import numpy as np
        ...     close = df['close'].to_numpy()
        ...     volume = df['volume'].to_numpy()
        ...     result = np.some_complex_calculation(close, volume)
        ...     return pl.Series(result)
        ...
        >>> factor = custom_indicator()  # Returns a Factor
        >>> pipeline = Pipeline(data).add_factors({'custom': factor})

        Cross-sectional with grouping:
        >>> @custom_factor(
        ...     scope=Scope.CROSS_SECTION,
        ...     inputs=['returns'],
        ...     groupby='sector'
        ... )
        ... def sector_adjusted(df: pl.DataFrame) -> pl.Series:
        ...     # Receives one (date, sector) group at a time
        ...     returns = df['returns'].to_numpy()
        ...     adjusted = custom_sector_logic(returns)
        ...     return pl.Series(adjusted)

        Using external libraries:
        >>> @custom_factor(scope=Scope.TIME_SERIES, inputs=['close'])
        ... def ta_lib_indicator(df: pl.DataFrame) -> pl.Series:
        ...     import talib
        ...     close = df['close'].to_numpy()
        ...     result = talib.RSI(close, timeperiod=14)
        ...     return pl.Series(result)

        Using Factor objects as inputs:
        >>> from factr.datasets import EquityPricing
        >>> @custom_factor(scope=Scope.TIME_SERIES, inputs=[EquityPricing.close, EquityPricing.volume])
        ... def vwap_indicator(df: pl.DataFrame) -> pl.Series:
        ...     return df['close'] * df['volume']
    """
    if scope not in (Scope.TIME_SERIES, Scope.CROSS_SECTION):
        raise ValueError(
            f"Custom factors must have TIME_SERIES or CROSS_SECTION scope, got {scope}"
        )

    if scope == Scope.TIME_SERIES and groupby is not None:
        raise ValueError("TIME_SERIES scope doesn't support groupby parameter")

    if not inputs:
        raise ValueError("Must specify at least one input column")

    column_names = []
    all_source_columns = set()
    all_source_datasets = set()

    for inp in inputs:
        if isinstance(inp, Factor):
            if inp.source_columns:
                column_names.extend(inp.source_columns)
                all_source_columns.update(inp.source_columns)
            else:
                raise ValueError(
                    f"Factor {inp.name} has no source_columns. "
                    "Please use column names directly or ensure Factors track source columns."
                )
            if inp.source_datasets:
                all_source_datasets.update(inp.source_datasets)
        elif isinstance(inp, str):
            column_names.append(inp)
            all_source_columns.add(inp)
        else:
            raise TypeError(f"inputs must be strings or Factor objects, got {type(inp).__name__}")

    seen = set()
    unique_column_names = []
    for col in column_names:
        if col not in seen:
            seen.add(col)
            unique_column_names.append(col)

    column_names = unique_column_names
    source_cols_frozen = frozenset(all_source_columns)
    source_datasets_frozen = frozenset(all_source_datasets)

    def decorator(func: Callable[[pl.DataFrame], pl.Series]) -> Callable[[], Factor]:
        """Decorator that wraps the user function."""
        name = output_name or func.__name__

        @wraps(func)
        def factor_builder() -> Factor:
            """Returns a Factor with the custom function wrapped in map_batches."""

            warnings.warn(
                f"Custom factor '{name}' uses map_batches which disables Polars optimization. "
                "Use only when necessary.",
                UserWarning,
                stacklevel=2,
            )

            def _wrapper(s: pl.Series) -> pl.Series:
                """Wrapper that calls user function on struct series."""
                df = s.struct.unnest()

                result = func(df)

                if not isinstance(result, pl.Series):
                    if hasattr(result, "__array__"):
                        result = pl.Series(result)
                    else:
                        raise TypeError(
                            f"Custom factor '{name}' must return pl.Series or array-like, "
                            f"got {type(result)}"
                        )

                return result

            expr = pl.struct(*column_names).map_batches(_wrapper, return_dtype=return_dtype)

            groupby_list = None
            if groupby is not None:
                groupby_list = [groupby] if isinstance(groupby, str) else list(groupby)

            return Factor(
                expr=expr,
                name=name,
                scope=scope,
                groupby=groupby_list,
                source_columns=source_cols_frozen,
                source_datasets=source_datasets_frozen,
            )

        return factor_builder

    return decorator
