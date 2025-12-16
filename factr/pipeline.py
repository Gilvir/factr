"""Pipeline for composing multiple operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import polars as pl

from .core import Factor, Filter, Scope, collect_dependencies, extract_datasets
from .universe import Universe

if TYPE_CHECKING:
    from .datasets import DataSet


@dataclass
class PipelineConfig:
    """Configuration for factor computation Pipeline.

    Args:
        date_column: Column name for dates (default: "date").
        entity_column: Column name for entities/assets (default: "asset").
    """

    date_column: str = "date"
    entity_column: str = "asset"


class Pipeline:
    """Orchestrate factor computations with automatic scope-based execution."""

    def __init__(self, data: pl.LazyFrame | None = None, config: PipelineConfig | None = None):
        self._factors: dict[str, Factor] = {}
        self._screen_filter: Filter | None = None
        self._screen_fn: Callable | None = None
        self._initial_data = data
        self.config = config or PipelineConfig()

    def add(self, name: str, factor: Factor) -> Pipeline:
        self._factors[name] = factor
        return self

    def add_factors(self, factors: dict[str, Factor]) -> Pipeline:
        self._factors.update(factors)
        return self

    def screen(
        self, filter: Filter | Universe | Callable[[pl.LazyFrame], pl.LazyFrame]
    ) -> Pipeline:
        if isinstance(filter, Universe):
            self._screen_filter = filter.filter
        elif isinstance(filter, Filter):
            self._screen_filter = filter
        else:
            self._screen_fn = filter
        return self

    def run(
        self,
        data: pl.LazyFrame | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        collect: bool = True,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        lf = data if data is not None else self._initial_data

        if lf is None:
            return None

        # Sort by [entity, date] so each entity's time-series is contiguous
        # This is critical for rolling window operations to work correctly
        lf = lf.sort([self.config.entity_column, self.config.date_column])

        if self._factors:
            all_factors = collect_dependencies(self._factors.values())

            output_factor_ids = {id(f) for f in self._factors.values()}

            factor_to_output_name: dict[int, str] = {}
            for name, factor in self._factors.items():
                factor_to_output_name[id(factor)] = name

            output_factor_names = {f.name for f in self._factors.values()}

            intermediate_names: list[str] = []

            for factor in all_factors:
                factor_id = id(factor)
                is_output = factor_id in output_factor_ids

                col_name = factor.name

                if not is_output and col_name not in output_factor_names:
                    intermediate_names.append(col_name)

                # Build expression with appropriate .over() based on scope
                if factor.scope == Scope.RAW:
                    # RAW factors are already columns - no .over() needed
                    expr = factor.expr.alias(col_name)
                elif factor.scope == Scope.TIME_SERIES:
                    # TIME_SERIES: per-entity operations
                    expr = factor.expr.over(self.config.entity_column).alias(col_name)
                else:  # CROSS_SECTION
                    # CROSS_SECTION: per-date operations, optionally grouped
                    over_cols = [self.config.date_column]
                    if factor.groupby:
                        if isinstance(factor.groupby, list):
                            over_cols.extend(factor.groupby)
                        else:
                            over_cols.append(factor.groupby)
                    expr = factor.expr.over(*over_cols).alias(col_name)

                lf = lf.with_columns(expr)

            rename_mapping = {
                factor.name: output_name
                for factor_id, output_name in factor_to_output_name.items()
                for factor in all_factors
                if id(factor) == factor_id and factor.name != output_name
            }
            if rename_mapping:
                lf = lf.rename(rename_mapping)

            if intermediate_names:
                lf = lf.drop(intermediate_names)

        if start_date:
            start_val = (
                pl.lit(start_date).cast(pl.Date) if isinstance(start_date, str) else start_date
            )
            lf = lf.filter(pl.col(self.config.date_column) >= start_val)

        if end_date:
            end_val = pl.lit(end_date).cast(pl.Date) if isinstance(end_date, str) else end_date
            lf = lf.filter(pl.col(self.config.date_column) <= end_val)

        if self._screen_filter:
            if self._screen_filter.scope == Scope.CROSS_SECTION:
                over_cols = [self.config.date_column]
                if self._screen_filter.groupby:
                    if isinstance(self._screen_filter.groupby, list):
                        over_cols.extend(self._screen_filter.groupby)
                    else:
                        over_cols.append(self._screen_filter.groupby)
                filter_expr = self._screen_filter.expr.over(*over_cols)
            else:
                filter_expr = self._screen_filter.expr

            lf = lf.filter(filter_expr)
        elif self._screen_fn:
            lf = self._screen_fn(lf)

        return lf.collect() if collect else lf

    def get_dataset_dependencies(self) -> frozenset[type[DataSet]]:
        """Extract dataset dependencies from all factors in this Pipeline.

        Returns which DataSet classes are referenced by the factors in this Pipeline.
        Useful for loading only the datasets you need.

        Returns:
            Frozenset of DataSet classes referenced by factors

        Example:
            >>> # Define factors
            >>> momentum = EquityPricing.close.pct_change(20)
            >>> pe = Fundamentals.pe_ratio
            >>>
            >>> # Add to Pipeline
            >>> Pipeline = Pipeline().add_factors({'momentum': momentum, 'pe': pe})
            >>>
            >>> # Get dataset dependencies
            >>> datasets = Pipeline.get_dataset_dependencies()
            >>> assert EquityPricing in datasets
            >>> assert Fundamentals in datasets
            >>>
            >>> # Use with DataContext to load only needed datasets
            >>> ctx = DataContext()
            >>> # ... bind many datasets ...
            >>> data = ctx.load_many(*datasets, start_date='2020-01-01')
        """
        return extract_datasets(self._factors.values())

    def explain(self) -> str:
        lines = ["", "Execution Plan:", "=" * 50, ""]

        lines.append(f"Sort Order: [{self.config.entity_column}, {self.config.date_column}]")
        lines.append("")

        if self._factors:
            all_factors = collect_dependencies(self._factors.values())
            output_factor_ids = {id(f) for f in self._factors.values()}

            factor_to_output_name: dict[int, str] = {}
            for name, factor in self._factors.items():
                factor_to_output_name[id(factor)] = name

            time_series_factors: list[tuple[str, Factor, bool]] = []
            cross_section_factors: list[tuple[str, Factor, bool]] = []

            for factor in all_factors:
                factor_id = id(factor)
                is_output = factor_id in output_factor_ids
                col_name = factor_to_output_name.get(factor_id, factor.name)

                if factor.scope in (Scope.TIME_SERIES, Scope.RAW):
                    time_series_factors.append((col_name, factor, is_output))
                else:
                    cross_section_factors.append((col_name, factor, is_output))

            if time_series_factors:
                lines.append("Stage 1: Time-Series & Raw Factors")
                lines.append("-" * 50)
                lines.append(f"  Computed per-entity using .over('{self.config.entity_column}')")
                lines.append("")
                for col_name, factor, is_output in time_series_factors:
                    intermediate_tag = "" if is_output else " [intermediate]"
                    lines.append(f"  • {col_name} ({factor.scope.name}){intermediate_tag}")
                lines.append("")

            if cross_section_factors:
                lines.append("Stage 2: Cross-Sectional Factors")
                lines.append("-" * 50)
                lines.append(f"  Computed per-date using .over('{self.config.date_column}', ...)")
                lines.append("")
                for col_name, factor, is_output in cross_section_factors:
                    groupby_info = ""
                    if factor.groupby:
                        groups = (
                            factor.groupby if isinstance(factor.groupby, list) else [factor.groupby]
                        )
                        groupby_info = (
                            f" - grouped by: {self.config.date_column}, {', '.join(groups)}"
                        )
                    else:
                        groupby_info = f" - grouped by: {self.config.date_column}"
                    intermediate_tag = "" if is_output else " [intermediate]"
                    lines.append(
                        f"  • {col_name} ({factor.scope.name}){groupby_info}{intermediate_tag}"
                    )
                lines.append("")

        if self._screen_filter:
            lines.append("Universe Filter:")
            lines.append("-" * 50)
            scope_info = f"({self._screen_filter.scope.name})"
            if self._screen_filter.scope == Scope.CROSS_SECTION:
                groupby_info = ""
                if self._screen_filter.groupby:
                    groups = (
                        self._screen_filter.groupby
                        if isinstance(self._screen_filter.groupby, list)
                        else [self._screen_filter.groupby]
                    )
                    groupby_info = f" - grouped by: {self.config.date_column}, {', '.join(groups)}"
                else:
                    groupby_info = f" - grouped by: {self.config.date_column}"
                lines.append(f"  {self._screen_filter.name} {scope_info}{groupby_info}")
            else:
                lines.append(f"  {self._screen_filter.name} {scope_info}")
            lines.append("")
        elif self._screen_fn:
            lines.append("Universe Filter:")
            lines.append("-" * 50)
            lines.append("  Custom filter function")
            lines.append("")

        lines.append("Note: Intermediate factors are materialized for scope transitions,")
        lines.append("      then dropped from final output.")
        lines.append("=" * 50)
        lines.append("")

        return "\n".join(lines)
