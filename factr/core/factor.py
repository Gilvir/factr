"""Core Factor types for composable factor computation."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, overload

import polars as pl

from .scope import Scope


@dataclass(frozen=True, eq=False)
class Factor:
    """Polars expression + scope metadata.

    Scope determines execution context (RAW, TIME_SERIES, CROSS_SECTION).

    Tracks source columns for automatic column selection optimization.
    Tracks source datasets for automatic dataset loading optimization.

    When a cross-sectional operation is applied to a time-series factor,
    the _parent field tracks the dependency so the Pipeline can materialize
    intermediates in the correct order with proper .over() contexts.
    """

    expr: pl.Expr = field(hash=False, compare=False)
    name: str = ""
    scope: Scope = Scope.TIME_SERIES
    groupby: list[str] | None = field(default=None, hash=False, compare=False)
    source_columns: frozenset[str] = field(default_factory=frozenset, hash=False, compare=False)
    source_datasets: frozenset[type] = field(default_factory=frozenset, hash=False, compare=False)
    _parent: Factor | None = field(default=None, hash=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"factor_{id(self)}")

    def __hash__(self) -> int:
        return id(self)

    def _new(
        self,
        expr: pl.Expr,
        name: str = "",
        scope: Scope | None = None,
        groupby: list[str] | None = None,
        source_columns: frozenset[str] | None = None,
        source_datasets: frozenset[type] | None = None,
        _parent: Factor | None = None,
    ) -> Factor:
        """Create new Factor with same scope/groupby unless overridden."""
        return Factor(
            expr=expr,
            name=name,
            scope=scope if scope is not None else self.scope,
            groupby=groupby if groupby is not None else self.groupby,
            source_columns=source_columns if source_columns is not None else self.source_columns,
            source_datasets=source_datasets
            if source_datasets is not None
            else self.source_datasets,
            _parent=_parent,
        )

    def _ts_context(self) -> tuple[pl.Expr, Factor | None]:
        """Return (base_expr, parent) for a TIME_SERIES op.

        If the input is CROSS_SECTION, the TS op must reference the materialized
        intermediate column (via ``pl.col(self.name)``) and track ``_parent=self``
        so the Pipeline materializes the CS intermediate first.
        """
        needs_parent = self.scope == Scope.CROSS_SECTION
        base_expr = pl.col(self.name) if needs_parent else self.expr
        parent = self if needs_parent else self._parent
        return base_expr, parent

    def _wrap_ts(self, expr: pl.Expr, name: str, parent: Factor | None) -> Factor:
        return Factor(
            expr=expr,
            name=name,
            scope=Scope.TIME_SERIES,
            groupby=self.groupby,
            source_columns=self.source_columns,
            source_datasets=self.source_datasets,
            _parent=parent,
        )

    def _cs_context(
        self, by: Classifier | str | None
    ) -> tuple[pl.Expr, Factor | None, list[str] | None]:
        """Return (base_expr, parent, groupby) for a CROSS_SECTION op.

        If the input is TIME_SERIES, the CS op must reference the materialized
        intermediate column (via ``pl.col(self.name)``) and track ``_parent=self``
        so the Pipeline materializes the TS intermediate first.

        RAW factors reference existing columns, so no materialization is needed.

        Args:
            by: Grouping classifier - can be a Classifier object (preferred) or column name string
        """
        from . import Classifier  # Import here to avoid circular dependency

        needs_parent = self.scope == Scope.TIME_SERIES
        base_expr = pl.col(self.name) if needs_parent else self.expr
        parent = self if needs_parent else self._parent

        # Extract column name from Classifier or use string directly
        if by is None:
            groupby = None
        elif isinstance(by, Classifier):
            groupby = [by.name]
        else:
            groupby = [by]

        return base_expr, parent, groupby

    @overload
    def _wrap_cs(
        self,
        expr: pl.Expr,
        name: str,
        groupby: list[str] | None,
        parent: Factor | None,
        result_type: type[Filter],
    ) -> Filter: ...

    @overload
    def _wrap_cs(
        self,
        expr: pl.Expr,
        name: str,
        groupby: list[str] | None,
        parent: Factor | None,
        result_type: type[Classifier],
    ) -> Classifier: ...

    @overload
    def _wrap_cs(
        self,
        expr: pl.Expr,
        name: str,
        groupby: list[str] | None,
        parent: Factor | None,
        result_type: None = None,
    ) -> Factor: ...

    def _wrap_cs(
        self,
        expr: pl.Expr,
        name: str,
        groupby: list[str] | None,
        parent: Factor | None,
        result_type: type[Factor] | None = None,
    ) -> Factor | Filter | Classifier:
        result_cls = result_type or Factor
        return result_cls(
            expr=expr,
            name=name,
            scope=Scope.CROSS_SECTION,
            groupby=groupby,
            source_columns=self.source_columns,
            source_datasets=self.source_datasets,
            _parent=parent,
        )

    def _infer_scope(self, other: Factor | None = None) -> Scope:
        if other is None:
            return self.scope
        if self.scope == Scope.CROSS_SECTION or other.scope == Scope.CROSS_SECTION:
            return Scope.CROSS_SECTION
        if self.scope == Scope.RAW and other.scope == Scope.RAW:
            return Scope.RAW
        return Scope.TIME_SERIES

    def _binop(
        self,
        other: Factor | float | int,
        op: Callable,
        sym: str,
        reflected: bool = False,
    ) -> Factor:
        if reflected:
            return Factor(
                expr=op(other, self.expr),
                name=f"({other} {sym} {self.name})",
                scope=self.scope,
                groupby=self.groupby,
                source_columns=self.source_columns,
                source_datasets=self.source_datasets,
            )

        if isinstance(other, Factor):
            return Factor(
                expr=op(self.expr, other.expr),
                name=f"({self.name} {sym} {other.name})",
                scope=self._infer_scope(other),
                groupby=self.groupby or other.groupby,
                source_columns=self.source_columns | other.source_columns,
                source_datasets=self.source_datasets | other.source_datasets,
            )

        return Factor(
            expr=op(self.expr, other),
            name=f"({self.name} {sym} {other})",
            scope=self.scope,
            groupby=self.groupby,
            source_columns=self.source_columns,
            source_datasets=self.source_datasets,
        )

    def _unaryop(self, op: Callable, name_template: str) -> Factor:
        return Factor(
            expr=op(self.expr),
            name=name_template.format(self.name),
            scope=self.scope,
            groupby=self.groupby,
            source_columns=self.source_columns,
            source_datasets=self.source_datasets,
        )

    def __add__(self, other):
        return self._binop(other, operator.add, "+")

    def __radd__(self, other):
        return self._binop(other, operator.add, "+", reflected=True)

    def __sub__(self, other):
        return self._binop(other, operator.sub, "-")

    def __rsub__(self, other):
        return self._binop(other, operator.sub, "-", reflected=True)

    def __mul__(self, other):
        return self._binop(other, operator.mul, "*")

    def __rmul__(self, other):
        return self._binop(other, operator.mul, "*", reflected=True)

    def __truediv__(self, other):
        return self._binop(other, operator.truediv, "/")

    def __rtruediv__(self, other):
        return self._binop(other, operator.truediv, "/", reflected=True)

    def __pow__(self, other):
        return self._binop(other, operator.pow, "**")

    def __rpow__(self, other):
        return self._binop(other, operator.pow, "**", reflected=True)

    def __mod__(self, other):
        return self._binop(other, operator.mod, "%")

    def __rmod__(self, other):
        return self._binop(other, operator.mod, "%", reflected=True)

    def __floordiv__(self, other):
        return self._binop(other, operator.floordiv, "//")

    def __rfloordiv__(self, other):
        return self._binop(other, operator.floordiv, "//", reflected=True)

    def __neg__(self):
        return self._unaryop(operator.neg, "(-{})")

    def __abs__(self):
        return self._unaryop(lambda e: e.abs(), "abs({})")

    def _compop(self, other: Factor | float | int | Any, op: Callable, sym: str) -> Filter:
        if isinstance(other, Factor):
            return Filter(
                expr=op(self.expr, other.expr),
                name=f"({self.name} {sym} {other.name})",
                scope=self._infer_scope(other),
                groupby=self.groupby or other.groupby,
                source_columns=self.source_columns | other.source_columns,
                source_datasets=self.source_datasets | other.source_datasets,
            )

        return Filter(
            expr=op(self.expr, other),
            name=f"({self.name} {sym} {other})",
            scope=self.scope,
            groupby=self.groupby,
            source_columns=self.source_columns,
            source_datasets=self.source_datasets,
        )

    def __eq__(self, other: Any) -> Filter:  # type: ignore[override]
        return self._compop(other, operator.eq, "==")

    def __ne__(self, other: Any) -> Filter:  # type: ignore[override]
        return self._compop(other, operator.ne, "!=")

    def __lt__(self, other):
        return self._compop(other, operator.lt, "<")

    def __le__(self, other):
        return self._compop(other, operator.le, "<=")

    def __gt__(self, other):
        return self._compop(other, operator.gt, ">")

    def __ge__(self, other):
        return self._compop(other, operator.ge, ">=")

    def rank(self, pct: bool = False, by: Classifier | str | None = None) -> Factor:
        """Rank within each date.

        Args:
            pct: If True, return percentile ranks (0-1)
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)
        expr = base_expr.rank(method="average")
        if pct:
            count_expr = pl.len()
            expr = (expr - 1) / (count_expr - 1)
        return self._wrap_cs(expr, f"{self.name}.rank(pct={pct}, by={by})", groupby, parent)

    def demean(self, by: Classifier | str | None = None) -> Factor:
        """Demean within each date.

        Args:
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)
        expr = base_expr - base_expr.mean()
        return self._wrap_cs(expr, f"{self.name}.demean(by={by})", groupby, parent)

    def zscore(self, by: Classifier | str | None = None) -> Factor:
        """Z-score within each date.

        Args:
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)
        expr = (base_expr - base_expr.mean()) / base_expr.std()
        return self._wrap_cs(expr, f"{self.name}.zscore(by={by})", groupby, parent)

    def winsorize(
        self, lower: float = 0.01, upper: float = 0.99, by: Classifier | str | None = None
    ) -> Factor:
        """Winsorize within each date.

        Args:
            lower: Lower quantile threshold
            upper: Upper quantile threshold
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)
        lower_bound = base_expr.quantile(lower)
        upper_bound = base_expr.quantile(upper)
        expr = base_expr.clip(lower_bound, upper_bound)
        return self._wrap_cs(
            expr, f"{self.name}.winsorize({lower}, {upper}, by={by})", groupby, parent
        )

    def top(self, n: int, by: Classifier | str | None = None) -> Filter:
        """Select top N assets.

        Args:
            n: Number of top assets to select
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)
        rank_expr = base_expr.rank(method="ordinal", descending=True)
        expr = rank_expr <= n
        return self._wrap_cs(
            expr,
            f"{self.name}.top({n}, by={by})",
            groupby,
            parent,
            result_type=Filter,
        )

    def bottom(self, n: int, by: Classifier | str | None = None) -> Filter:
        """Select bottom N assets.

        Args:
            n: Number of bottom assets to select
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)
        rank_expr = base_expr.rank(method="ordinal", descending=False)
        expr = rank_expr <= n
        return self._wrap_cs(
            expr,
            f"{self.name}.bottom({n}, by={by})",
            groupby,
            parent,
            result_type=Filter,
        )

    def quantile(
        self, q: int, labels: bool = False, by: Classifier | str | None = None
    ) -> Classifier:
        """Bin into quantiles.

        Args:
            q: Number of quantiles
            labels: If True, label bins starting from 1 instead of 0
            by: Optional grouping - pass a Classifier for type-safe grouping or column name string
        """
        base_expr, parent, groupby = self._cs_context(by)

        rank_expr = base_expr.rank(method="average")
        count_expr = pl.len()
        pct = (rank_expr - 1) / (count_expr - 1)
        expr = (pct * q).floor().clip(0, q - 1).cast(pl.Int32)
        if labels:
            expr = expr + 1

        return self._wrap_cs(
            expr,
            f"{self.name}.quantile({q}, by={by})",
            groupby,
            parent,
            result_type=Classifier,
        )

    def log(self) -> Factor:
        return self._unaryop(lambda e: e.log(), "log({})")

    def sqrt(self) -> Factor:
        return self._unaryop(lambda e: e.sqrt(), "sqrt({})")

    def sign(self) -> Factor:
        """Return the sign of the factor (-1, 0, or 1)."""
        return self._unaryop(lambda e: e.sign(), "sign({})")

    def cumsum(self) -> Factor:
        """Cumulative sum."""
        base_expr, parent = self._ts_context()
        expr = base_expr.cum_sum()
        return self._wrap_ts(expr, f"{self.name}.cumsum()", parent)

    def clip(self, lower: float | None = None, upper: float | None = None) -> Factor:
        expr = self.expr.clip(lower, upper)
        name = f"{self.name}.clip({lower}, {upper})"
        return self._new(expr=expr, name=name)

    def fill_null(
        self,
        value: float | None = None,
        strategy: Literal["forward", "backward", "mean", "zero"] | None = None,
    ) -> Factor:
        if strategy == "forward":
            expr = self.expr.forward_fill()
        elif strategy == "backward":
            expr = self.expr.backward_fill()
        elif strategy == "mean":
            expr = self.expr.fill_null(self.expr.mean())
        elif strategy == "zero" or value == 0:
            expr = self.expr.fill_null(0)
        elif value is not None:
            expr = self.expr.fill_null(value)
        else:
            expr = self.expr

        name = f"{self.name}.fill_null({value or strategy})"
        return self._new(expr=expr, name=name)

    def shift(self, periods: int = 1) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.shift(periods)
        return self._wrap_ts(expr, f"{self.name}.shift({periods})", parent)

    def pct_change(self, periods: int = 1) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.pct_change(periods)
        return self._wrap_ts(expr, f"{self.name}.pct_change({periods})", parent)

    def diff(self, periods: int = 1) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.diff(periods)
        return self._wrap_ts(expr, f"{self.name}.diff({periods})", parent)

    def rolling_sum(self, window: int) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.rolling_sum(window_size=window)
        return self._wrap_ts(expr, f"{self.name}.rolling_sum({window})", parent)

    def rolling_mean(self, window: int) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.rolling_mean(window_size=window)
        return self._wrap_ts(expr, f"{self.name}.rolling_mean({window})", parent)

    def rolling_std(self, window: int) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.rolling_std(window_size=window)
        return self._wrap_ts(expr, f"{self.name}.rolling_std({window})", parent)

    def rolling_min(self, window: int) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.rolling_min(window_size=window)
        return self._wrap_ts(expr, f"{self.name}.rolling_min({window})", parent)

    def rolling_max(self, window: int) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.rolling_max(window_size=window)
        return self._wrap_ts(expr, f"{self.name}.rolling_max({window})", parent)

    def ewm_mean(self, span: int) -> Factor:
        base_expr, parent = self._ts_context()
        expr = base_expr.ewm_mean(span=span)
        return self._wrap_ts(expr, f"{self.name}.ewm_mean({span})", parent)


@dataclass(frozen=True)
class Filter(Factor):
    """Boolean factor for filtering."""

    def __and__(self, other: Filter) -> Filter:
        expr = self.expr & other.expr
        name = f"({self.name} & {other.name})"
        return Filter(
            expr=expr,
            name=name,
            scope=self._infer_scope(other),
            groupby=self.groupby or other.groupby,
            source_columns=self.source_columns | other.source_columns,
            source_datasets=self.source_datasets | other.source_datasets,
        )

    def __or__(self, other: Filter) -> Filter:
        expr = self.expr | other.expr
        name = f"({self.name} | {other.name})"
        return Filter(
            expr=expr,
            name=name,
            scope=self._infer_scope(other),
            groupby=self.groupby or other.groupby,
            source_columns=self.source_columns | other.source_columns,
            source_datasets=self.source_datasets | other.source_datasets,
        )

    def __invert__(self) -> Filter:
        expr = ~self.expr
        name = f"(~{self.name})"
        return Filter(
            expr=expr,
            name=name,
            scope=self.scope,
            groupby=self.groupby,
            source_columns=self.source_columns,
            source_datasets=self.source_datasets,
        )


@dataclass(frozen=True)
class Classifier(Factor):
    """Categorical factor for grouping."""

    pass


def extract_datasets(factors: Iterable[Factor]) -> frozenset[type]:
    """Extract all dataset dependencies from a collection of factors.

    Args:
        factors: Iterable of Factor objects (can be dict values, list, etc.)

    Returns:
        Frozenset of DataSet classes that are referenced by the factors

    Example:
        >>> momentum = EquityPricing.close.pct_change(20)
        >>> pe = Fundamentals.pe_ratio
        >>> datasets = extract_datasets([momentum, pe])
        >>> assert EquityPricing in datasets
        >>> assert Fundamentals in datasets
    """
    all_datasets: set[type] = set()
    for factor in factors:
        all_datasets.update(factor.source_datasets)
    return frozenset(all_datasets)


def collect_dependencies(factors: Iterable[Factor]) -> list[Factor]:
    """Collect all factor dependencies in topological order.

    Traverses the _parent chain of each factor to find all intermediate
    factors that need to be materialized before the final factors.

    Args:
        factors: Iterable of Factor objects (the "output" factors)

    Returns:
        List of all factors in dependency order (parents before children).
        Factors without parents come first.

    Example:
        >>> close = EquityPricing.close
        >>> returns = close.pct_change(1)  # TIME_SERIES
        >>> ranked = returns.rank()  # CROSS_SECTION, _parent=returns
        >>> deps = collect_dependencies([ranked])
        >>> # Returns [returns, ranked] - returns must be computed first
    """
    all_factors: dict[int, Factor] = {}

    def collect(factor: Factor) -> None:
        factor_id = id(factor)
        if factor_id in all_factors:
            return
        if factor._parent is not None:
            collect(factor._parent)
        all_factors[factor_id] = factor

    for factor in factors:
        collect(factor)

    return list(all_factors.values())
