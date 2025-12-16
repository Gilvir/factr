"""DataContext for managing multiple dataset sources.

Inspired by dependency injection patterns - compose datasets and sources explicitly.
No global state, no magic, just clear configuration.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Iterable

import polars as pl

from ..core import Factor, extract_datasets
from .alignment import apply_offset, asof_join, forward_fill

if TYPE_CHECKING:
    from ..datasets import DataSet
    from .binding import DataSource


class DataContext:
    """Context for managing dataset sources.

    Compose multiple datasets with their sources. No global state - each context
    is explicit and isolated.

    Example:
        >>> from factr.datasets import EquityPricing, Fundamentals
        >>> from factr.data.sources import ParquetSource, SQLSource
        >>>
        >>> # Create context and bind sources
        >>> ctx = DataContext()
        >>> ctx.bind(EquityPricing, ParquetSource('data/prices.parquet'))
        >>> ctx.bind(Fundamentals, SQLSource('postgres://...', table='fundamentals'))
        >>>
        >>> # Load datasets
        >>> prices = ctx.load(EquityPricing, start_date='2020-01-01')
        >>> funds = ctx.load(Fundamentals, start_date='2020-01-01')
        >>>
        >>> # Combine in Pipeline
        >>> from factr.pipeline import Pipeline
        >>> Pipeline = Pipeline(prices).add_factors({...})
    """

    def __init__(self):
        """Initialize context.

        Use .bind() to explicitly associate datasets with sources.
        """
        self._sources: dict[type[DataSet], DataSource] = {}

    def bind(self, dataset: type[DataSet], source: DataSource) -> DataContext:
        """Bind a dataset to a source.

        Args:
            dataset: DataSet class
            source: DataSource to use for loading

        Returns:
            Self (for chaining)

        Example:
            >>> ctx.bind(EquityPricing, ParquetSource('prices.parquet'))
            >>> ctx.bind(Fundamentals, SQLSource('db', table='fundamentals'))
        """
        self._sources[dataset] = source
        return self

    def load(
        self,
        dataset: type[DataSet],
        source: DataSource | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        _namespace: bool = False,
    ) -> pl.LazyFrame:
        """Load a dataset.

        Source resolution priority:
        1. Explicit source parameter (highest priority)
        2. Bound source via .bind()
        3. Dataset.Config source (if exists)

        Args:
            dataset: DataSet class to load
            source: Optional explicit source (overrides bound/config)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            LazyFrame with dataset

        Example:
            >>> # Use bound source
            >>> lf = ctx.load(EquityPricing, start_date='2020-01-01')
            >>>
            >>> # Override with explicit source
            >>> lf = ctx.load(
            ...     EquityPricing,
            ...     source=ParquetSource('other.parquet'),
            ...     start_date='2020-01-01'
            ... )

        Raises:
            ValueError: If no source can be resolved
        """
        resolved_source = self._resolve_source(dataset, source)

        config = dataset.get_config()
        date_col = config.date_column if config else "date"
        entity_col = config.entity_column if config else "asset"

        lf = resolved_source.read(
            date_col=date_col,
            asset_col=entity_col,
            start_date=start_date,
            end_date=end_date,
        )

        lf = dataset._apply_column_transforms(lf, namespace=_namespace)

        return lf

    def _resolve_source(
        self,
        dataset: type[DataSet],
        explicit_source: DataSource | None,
    ) -> DataSource:
        """Resolve source from multiple possible locations.

        Priority:
        1. Explicit source parameter
        2. Bound source
        3. Dataset Config

        Args:
            dataset: DataSet class
            explicit_source: Explicitly provided source

        Returns:
            Resolved DataSource

        Raises:
            ValueError: If no source found
        """
        if explicit_source is not None:
            return explicit_source

        if dataset in self._sources:
            return self._sources[dataset]

        config = dataset.get_config()
        if config is not None:
            return config.resolve_source()

        raise ValueError(
            f"No source configured for {dataset.__name__}. "
            f"Either bind a source via .bind(), add a Config class to {dataset.__name__}, "
            f"or pass source explicitly to .load()"
        )

    def load_many(
        self,
        *datasets: type[DataSet],
        start_date: str | None = None,
        end_date: str | None = None,
        collect: bool = False,
        n_jobs: int | None = None,
    ) -> dict[type[DataSet], pl.LazyFrame | pl.DataFrame]:
        """Load multiple datasets in parallel.

        Returns LazyFrames by default (deferred execution). Dataset loading happens
        in parallel using ThreadPoolExecutor for better performance when loading from
        multiple sources. If collect=True, collects all LazyFrames concurrently.

        Args:
            *datasets: DataSet classes to load
            start_date: Optional start date (applied to all)
            end_date: Optional end date (applied to all)
            collect: If True, collect all LazyFrames concurrently (default: False)
            n_jobs: Number of parallel jobs. None (default) uses ThreadPoolExecutor default,
                   -1 uses all available CPUs, positive integer uses that many workers

        Returns:
            Dict mapping dataset class to LazyFrame (or DataFrame if collect=True)

        Example:
            >>> # Lazy loading with parallel initialization (auto thread count)
            >>> data = ctx.load_many(EquityPricing, Fundamentals)
            >>> prices = data[EquityPricing].collect()  # Execute when needed
            >>>
            >>> # Concurrent collection with specific thread count
            >>> data = ctx.load_many(
            ...     EquityPricing,
            ...     Fundamentals,
            ...     ReferenceData,
            ...     start_date='2020-01-01',
            ...     collect=True,  # Collect all concurrently
            ...     n_jobs=4  # Use 4 worker threads
            ... )
            >>> prices = data[EquityPricing]  # Already a DataFrame
            >>>
            >>> # Use all available CPUs
            >>> data = ctx.load_many(
            ...     EquityPricing,
            ...     Fundamentals,
            ...     n_jobs=-1  # Use all CPUs
            ... )
        """
        import os

        if n_jobs is None:
            max_workers = None
        elif n_jobs == -1:
            max_workers = os.cpu_count()
        elif n_jobs > 0:
            max_workers = n_jobs
        else:
            raise ValueError(f"n_jobs must be -1, None, or positive integer, got {n_jobs}")

        def load_one(dataset: type[DataSet]) -> tuple[type[DataSet], pl.LazyFrame]:
            """Load a single dataset and return with its type for mapping."""
            return dataset, self.load(
                dataset, start_date=start_date, end_date=end_date, _namespace=True
            )

        lazy_frames = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {executor.submit(load_one, ds): ds for ds in datasets}

            for future in as_completed(future_to_dataset):
                dataset, lf = future.result()
                lazy_frames[dataset] = lf

        if collect:
            collected = pl.collect_all(list(lazy_frames.values()))
            return dict(zip(lazy_frames.keys(), collected))

        return lazy_frames

    def load_for_factors(
        self,
        factors: Iterable[Factor] | dict[str, Factor],
        start_date: str | None = None,
        end_date: str | None = None,
        collect: bool = False,
        n_jobs: int | None = None,
    ) -> dict[type[DataSet], pl.LazyFrame | pl.DataFrame]:
        """Load only the datasets needed by the given factors.

        Automatically extracts dataset dependencies from factors and loads only
        those datasets. This is the key optimization - if you have 10 datasets
        defined but only use 2, only those 2 tables are queried.

        Args:
            factors: Factors to analyze (can be dict values, list, etc.)
            start_date: Optional start date (applied to all)
            end_date: Optional end date (applied to all)
            collect: If True, collect all LazyFrames concurrently (default: False)
            n_jobs: Number of parallel jobs for loading datasets (see load_many for details)

        Returns:
            Dict mapping dataset class to LazyFrame (or DataFrame if collect=True)

        Example:
            >>> # Define many datasets in context
            >>> ctx = DataContext()
            >>> ctx.bind(EquityPricing, ParquetSource('prices.parquet'))
            >>> ctx.bind(Fundamentals, SQLSource('db', table='fundamentals'))
            >>> ctx.bind(Sentiment, SQLSource('db', table='sentiment'))
            >>> ctx.bind(ReferenceData, ParquetSource('reference.parquet'))
            >>> # ... 10 more datasets
            >>>
            >>> # Only use 2 datasets in factors
            >>> momentum = EquityPricing.close.pct_change(20)
            >>> pe = Fundamentals.pe_ratio
            >>>
            >>> # Smart load: only queries EquityPricing and Fundamentals tables
            >>> data = ctx.load_for_factors([momentum, pe], start_date='2020-01-01')
            >>> # data only contains EquityPricing and Fundamentals
            >>>
            >>> # Works with dict of factors too
            >>> factors = {'momentum': momentum, 'pe': pe}
            >>> data = ctx.load_for_factors(factors, collect=True, n_jobs=-1)
        """
        if isinstance(factors, dict):
            factors = factors.values()

        needed_datasets = extract_datasets(factors)

        return self.load_many(
            *needed_datasets,
            start_date=start_date,
            end_date=end_date,
            collect=collect,
            n_jobs=n_jobs,
        )

    def load_and_combine(
        self,
        factors: Iterable[Factor] | dict[str, Factor],
        primary_dataset: type[DataSet] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        n_jobs: int | None = None,
    ) -> pl.LazyFrame:
        """Load datasets for factors and combine with point-in-time correctness.

        Automatically handles:
        - Reporting delays (via Config.reporting_delay)
        - Different data frequencies (daily prices + quarterly fundamentals)
        - Forward filling sparse data (via Config.forward_fill_columns)
        - As-of joins to prevent look-ahead bias
        - Parallel dataset loading for better performance

        Args:
            factors: Factors to analyze (can be dict values, list, etc.)
            primary_dataset: Dataset to use as base for joins (if None, uses first with is_primary=True)
            start_date: Optional start date (applied to all)
            end_date: Optional end date (applied to all)
            n_jobs: Number of parallel jobs for loading datasets (see load_many for details)

        Returns:
            Combined LazyFrame with all datasets joined correctly

        Example:
            >>> # Configure datasets with PIT metadata
            >>> class EquityPricing(DataSet):
            ...     close = Column(pl.Float64)
            ...     class Config:
            ...         source = ParquetSource('prices.parquet')
            ...         is_primary = True  # This is the base dataset
            >>>
            >>> class Fundamentals(DataSet):
            ...     pe_ratio = Column(pl.Float64)
            ...     market_cap = Column(pl.Float64)
            ...     class Config:
            ...         source = SQLSource('db', table='fundamentals')
            ...         reporting_delay = 45  # Earnings reported 45 days after quarter end
            ...         forward_fill_columns = ['pe_ratio', 'market_cap']  # Fill to daily
            >>>
            >>> # Define factors
            >>> momentum = EquityPricing.close.pct_change(20)
            >>> value = Fundamentals.pe_ratio
            >>>
            >>> # Load and combine with PIT correctness
            >>> ctx = DataContext()
            >>> data = ctx.load_and_combine([momentum, value], start_date='2020-01-01')
            >>> # Result: daily prices joined with fundamentals using asof join,
            >>> # with 45-day reporting delay applied, and forward-filled
        """
        if isinstance(factors, dict):
            factors = factors.values()

        needed_datasets = extract_datasets(factors)

        if not needed_datasets:
            raise ValueError("No datasets found in factors")

        # Calculate earliest date needed based on reporting delays
        adjusted_start_date = start_date
        if start_date:
            max_delay = 0
            for dataset in needed_datasets:
                config = dataset.get_config()
                if config and config.reporting_delay > 0:
                    max_delay = max(max_delay, config.reporting_delay)

            if max_delay > 0:
                from datetime import datetime, timedelta

                dt = datetime.strptime(start_date, "%Y-%m-%d")
                dt = dt - timedelta(days=max_delay)
                adjusted_start_date = dt.strftime("%Y-%m-%d")

        loaded = self.load_many(
            *needed_datasets,
            start_date=adjusted_start_date,
            end_date=end_date,
            n_jobs=n_jobs,
        )

        if primary_dataset is None:
            for dataset in needed_datasets:
                config = dataset.get_config()
                if config and config.is_primary:
                    primary_dataset = dataset
                    break

            if primary_dataset is None:
                primary_dataset = next(iter(needed_datasets))

        if primary_dataset not in loaded:
            raise ValueError(f"Primary dataset {primary_dataset.__name__} not in loaded datasets")

        result = loaded[primary_dataset]
        primary_config = primary_dataset.get_config()
        date_col = primary_config.date_column if primary_config else "date"
        entity_col = primary_config.entity_column if primary_config else "asset"

        for dataset, lf in loaded.items():
            if dataset == primary_dataset:
                continue

            config = dataset.get_config()
            if not config:
                result = asof_join(result, lf, on=date_col, by=entity_col)
                continue

            if config.reporting_delay > 0:
                lf = apply_offset(lf, config.reporting_delay, date_col=config.date_column)
                lf = lf.drop(config.date_column).rename({"available_date": config.date_column})

            result = asof_join(
                result,
                lf,
                on=date_col,
                by=entity_col,
                strategy="backward",
            )

            if config.forward_fill_columns:
                # Use collect_schema() for efficient schema access (Polars 1.0+)
                result_cols = result.collect_schema().names()
                cols_to_fill = [col for col in config.forward_fill_columns if col in result_cols]
                if cols_to_fill:
                    result = forward_fill(result, cols_to_fill, by=entity_col, order_by=date_col)

        if start_date:
            result = result.filter(pl.col(date_col) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            result = result.filter(pl.col(date_col) <= pl.lit(end_date).cast(pl.Date))

        column_mapping: dict[tuple[type, str], str] = {}
        for dataset in loaded.keys():
            namespace = dataset._get_namespace()
            for col_name in dataset.columns():
                namespaced_name = f"{namespace}__{col_name}"
                column_mapping[(dataset, col_name)] = namespaced_name

        result._factr_column_mapping = column_mapping  # type: ignore

        return result

    def clone(self) -> DataContext:
        """Create a copy of this context.

        Useful for creating test contexts from production config.

        Returns:
            New DataContext with same bindings

        Example:
            >>> prod_ctx = DataContext()
            >>> prod_ctx.bind(EquityPricing, SQLSource('prod_db', table='prices'))
            >>>
            >>> # Clone for testing
            >>> test_ctx = prod_ctx.clone()
            >>> test_ctx.bind(EquityPricing, DataFrameSource(test_data))
        """
        new_ctx = DataContext()
        new_ctx._sources = self._sources.copy()
        return new_ctx
