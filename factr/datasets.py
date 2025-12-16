"""Dataset definitions for type-safe column access.

Inspired by Pydantic's BaseModel pattern - datasets compose columns and configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from .core import Classifier, Factor, Scope

if TYPE_CHECKING:
    from .data.config import DataSetConfig


class Column:
    """Descriptor for dataset columns that creates Factors or Classifiers.

    Inspired by Pydantic's Field - provides field-level configuration for
    data loading and transformation.

    Args:
        dtype: Polars data type for this column
        alias: Source column name (if different from field name)
        required: Whether column must exist in source
        default: Default value if column missing from source
        fill_null: Value to use for null entries
        fill_strategy: Strategy for filling nulls ('forward', 'backward', 'mean', 'zero')
        categorical: If True, column returns Classifier for grouping operations (default: False)

    Example:
        >>> class Pricing(DataSet):
        ...     # Simple numeric column
        ...     close = Column(pl.Float64)
        ...
        ...     # With alias (source has different name)
        ...     volume = Column(pl.Int64, alias='trading_volume')
        ...
        ...     # With null handling
        ...     market_cap = Column(pl.Float64, fill_strategy='forward')
        ...
        ...     # Categorical column for grouping
        ...     sector = Column(pl.Utf8, categorical=True)
        ...
        ...     # With default
        ...     region = Column(pl.Utf8, default='Unknown', required=False, categorical=True)
    """

    _UNSET = object()

    def __init__(
        self,
        dtype: type = pl.Float64,
        *,
        alias: str | None = None,
        required: bool | object = _UNSET,
        default: Any = None,
        fill_null: Any = None,
        fill_strategy: Literal["forward", "backward", "mean", "zero"] | None = None,
        categorical: bool = False,
    ):
        """Initialize a Column descriptor.

        Note on fill_strategy:
            Fill strategies at the Column level operate GLOBALLY (across all entities).
            For entity-specific fill operations (e.g., forward-fill within each asset),
            use Factor methods after loading:

            >>> close = EquityPricing.close
            >>> filled = close.fill_null(strategy='forward')  # Uses .over(entity) in Pipeline
        """
        self.dtype = dtype
        self.alias = alias
        self.required = required
        self.default = default
        self.fill_null = fill_null
        self.fill_strategy = fill_strategy
        self.categorical = categorical
        self.name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = f"{name}"
        self.dataset_class = owner

    def __get__(self, obj: Any, objtype: type | None = None) -> Factor | Classifier:
        if self.name is None:
            raise AttributeError("Column name not set")

        cls = Classifier if self.categorical else Factor
        return cls(
            pl.col(self.name),
            name=self.name,
            scope=Scope.RAW,
            source_columns=frozenset({self.name}),
            source_datasets=frozenset({self.dataset_class}),
        )

    @property
    def source_name(self) -> str:
        """Get the source column name (alias if set, otherwise field name)."""
        return self.alias if self.alias else (self.name or "")

    def apply_transforms(self, expr: pl.Expr) -> pl.Expr:
        """Apply field-level transformations to an expression.

        This includes:
        - Filling null values (GLOBALLY - not per-entity)
        - Type casting

        Important:
            Fill strategies at this level operate across the ENTIRE column,
            not per-entity. For panel data with multiple entities, this means:
            - forward_fill: Fills nulls from ANY previous row (may leak across entities)
            - backward_fill: Fills nulls from ANY following row (may leak across entities)
            - mean: Uses global mean across all entities and dates

            For entity-specific fills, use Factor methods in Pipeline:
            >>> close = EquityPricing.close
            >>> filled = close.fill_null(strategy='forward')  # Respects entity boundaries

        Args:
            expr: Base Polars expression

        Returns:
            Transformed expression
        """
        result = expr

        result = result.cast(self.dtype)

        if self.fill_null is not None:
            result = result.fill_null(self.fill_null)
        elif self.fill_strategy == "forward":
            result = result.forward_fill()
        elif self.fill_strategy == "backward":
            result = result.backward_fill()
        elif self.fill_strategy == "zero":
            result = result.fill_null(0)
        elif self.fill_strategy == "mean":
            if self.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                result = result.fill_null(result.mean())

        return result

    def __repr__(self) -> str:
        parts = [f"'{self.name}'", f"dtype={self.dtype}"]
        if self.alias:
            parts.append(f"alias='{self.alias}'")
        if not self.required:
            parts.append("required=False")
        if self.fill_strategy:
            parts.append(f"fill_strategy='{self.fill_strategy}'")
        if self.categorical:
            parts.append("categorical=True")
        return f"Column({', '.join(parts)})"


class DataSetMeta(type):
    def __repr__(cls) -> str:
        columns = [name for name, val in cls.__dict__.items() if isinstance(val, Column)]
        return f"<DataSet '{cls.__name__}' with columns: {', '.join(columns)}>"

    def _get_namespace(cls) -> str:
        """Get namespace prefix for this dataset.

        Converts class name to lowercase with underscores.
        Example: EquityPricing -> equity_pricing
        """
        import re

        # Convert CamelCase to snake_case
        name = cls.__name__
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.lower()


class DataSet(metaclass=DataSetMeta):
    """Base class for dataset definitions.

    Compose columns and configuration, inspired by Pydantic's pattern.

    Example:
        >>> class EquityPricing(DataSet):
        ...     open = Column(pl.Float64)
        ...     close = Column(pl.Float64)
        ...
        ...     class Config:
        ...         source = ParquetSource('data/prices.parquet')
        ...         date_column = 'date'
        ...         entity_column = 'asset'
    """

    @classmethod
    def columns(cls) -> list[str]:
        """Get list of column names in this dataset."""
        return [name for name, val in cls.__dict__.items() if isinstance(val, Column)]

    @classmethod
    def get_column(cls, name: str) -> Factor:
        """Get a column as a Factor."""
        if not hasattr(cls, name):
            raise AttributeError(f"DataSet '{cls.__name__}' has no column '{name}'")
        return getattr(cls, name)

    @classmethod
    def get_column_descriptors(cls) -> dict[str, Column]:
        """Get all Column descriptors with their field names.

        Returns:
            Dict mapping field name to Column descriptor
        """
        return {name: val for name, val in cls.__dict__.items() if isinstance(val, Column)}

    @classmethod
    def get_column_mapping(cls) -> dict[str, str]:
        """Get column name mappings from field aliases.

        Returns:
            Dict mapping source column name (alias) to field name

        Example:
            >>> class Pricing(DataSet):
            ...     close = Column(pl.Float64, alias='price')
            >>> Pricing.get_column_mapping()
            {'price': 'close'}
        """
        mapping = {}
        for field_name, col in cls.get_column_descriptors().items():
            if col.alias:
                mapping[col.alias] = field_name
        return mapping

    @classmethod
    def get_config(cls) -> DataSetConfig | None:
        """Get dataset configuration if Config class exists."""
        if not hasattr(cls, "Config"):
            return None

        from .data.config import config_from_class

        return config_from_class(cls.Config)

    @classmethod
    def load(
        cls,
        start_date: str | None = None,
        end_date: str | None = None,
        apply_transforms: bool = True,
        columns: list[str] | None = None,
    ) -> pl.LazyFrame:
        """Load data from configured source with field-level transformations.

        This method:
        1. Loads data from the configured source
        2. Optionally selects only needed columns (optimization)
        3. Applies column name mappings (aliases)
        4. Applies field transformations (fill_null, bounds, etc.)
        5. Adds defaults for missing optional columns

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            apply_transforms: Whether to apply field transforms (default: True)
            columns: Optional list of columns to load (None = load all).
                     Automatically handles aliases - pass field names, not source names.

        Returns:
            LazyFrame with dataset

        Example:
            >>> class Pricing(DataSet):
            ...     close = Column(pl.Float64, alias='price', fill_null=0.0)
            ...     volume = Column(pl.Int64)
            ...     class Config:
            ...         source = ParquetSource('data.parquet')
            >>>
            >>> # Load only specific columns (optimization)
            >>> lf = Pricing.load(columns=['close', 'volume'], start_date='2020-01-01')
            >>> # Automatically renames 'price' -> 'close', fills nulls

        Raises:
            ValueError: If no Config class or source configured
        """
        config = cls.get_config()
        if config is None:
            raise ValueError(
                f"{cls.__name__} has no Config class - cannot load without source configuration"
            )

        source = config.resolve_source()
        lf = source.read(
            date_col=config.date_column,
            asset_col=config.entity_column,
            start_date=start_date,
            end_date=end_date,
        )

        if columns is not None:
            descriptors = cls.get_column_descriptors()
            source_cols_needed = set()

            source_cols_needed.add(config.date_column)
            source_cols_needed.add(config.entity_column)

            for col_name in columns:
                if col_name in descriptors:
                    desc = descriptors[col_name]
                    source_cols_needed.add(desc.source_name)
                else:
                    source_cols_needed.add(col_name)

            # Use collect_schema() for efficient schema access (Polars 1.0+)
            schema = lf.collect_schema()
            available_cols = [c for c in source_cols_needed if c in schema.names()]
            if available_cols:
                lf = lf.select(available_cols)

        if not apply_transforms:
            return lf

        lf = cls._apply_column_transforms(lf, columns=columns)

        return lf

    @classmethod
    def _apply_column_transforms(
        cls, lf: pl.LazyFrame, columns: list[str] | None = None, namespace: bool = False
    ) -> pl.LazyFrame:
        """Apply field-level transformations to a LazyFrame.

        Args:
            lf: Input LazyFrame
            columns: Optional list of columns to process (None = all columns)
            namespace: If True, prefix column names with dataset namespace to prevent collisions

        Returns:
            Transformed LazyFrame with:
            - Column aliases applied
            - Null filling strategies applied
            - Defaults added for missing optional columns
            - Namespaced column names (if namespace=True)
        """
        schema = lf.collect_schema()
        source_columns = set(schema.names())

        exprs = []
        columns_to_rename = {}

        descriptors = cls.get_column_descriptors()
        if columns is not None:
            descriptors = {k: v for k, v in descriptors.items() if k in columns}

        dataset_namespace = cls._get_namespace() if namespace else None

        for field_name, col_desc in descriptors.items():
            source_col = col_desc.source_name
            output_name = f"{dataset_namespace}__{field_name}" if dataset_namespace else field_name

            if source_col not in source_columns:
                is_explicitly_required = (
                    col_desc.required is not Column._UNSET and col_desc.required is True
                )

                if columns is not None and field_name in columns:
                    if is_explicitly_required:
                        raise ValueError(
                            f"Required column '{source_col}' (field: '{field_name}') "
                            f"not found in source. Available: {sorted(source_columns)}"
                        )
                    elif col_desc.default is not None:
                        exprs.append(pl.lit(col_desc.default).alias(output_name))
                elif is_explicitly_required and columns is None:
                    raise ValueError(
                        f"Required column '{source_col}' (field: '{field_name}') "
                        f"not found in source. Available: {sorted(source_columns)}"
                    )
                elif col_desc.default is not None:
                    exprs.append(pl.lit(col_desc.default).alias(output_name))
                continue

            expr = col_desc.apply_transforms(pl.col(source_col))

            expr = expr.alias(output_name)
            if source_col != output_name:
                columns_to_rename[source_col] = output_name

            exprs.append(expr)

        if exprs:
            lf = lf.with_columns(exprs)

            cols_to_drop = [src for src in columns_to_rename.keys() if src in source_columns]
            if cols_to_drop:
                lf = lf.drop(cols_to_drop)

        return lf


class EquityPricing(DataSet):
    """Daily equity pricing data."""

    open = Column(pl.Float64)
    high = Column(pl.Float64)
    low = Column(pl.Float64)
    close = Column(pl.Float64)
    volume = Column(pl.Int64)


class Fundamentals(DataSet):
    """Fundamental data for equities."""

    market_cap = Column(pl.Float64)
    pe_ratio = Column(pl.Float64)
    pb_ratio = Column(pl.Float64)
    dividend_yield = Column(pl.Float64)
    revenue = Column(pl.Float64)
    earnings = Column(pl.Float64)
    book_value = Column(pl.Float64)
    shares_outstanding = Column(pl.Int64)


class ReferenceData(DataSet):
    """Reference data for equities."""

    sector = Column(pl.Utf8, categorical=True)
    industry = Column(pl.Utf8, categorical=True)
    exchange = Column(pl.Utf8, categorical=True)
    country = Column(pl.Utf8, categorical=True)


class Sentiment(DataSet):
    """Sentiment and alternative data."""

    news_sentiment = Column(pl.Float64)
    social_sentiment = Column(pl.Float64)
    news_volume = Column(pl.Int64)
    social_volume = Column(pl.Int64)


def dataset(name: str, **columns: Column) -> type[DataSet]:
    """Create a custom dataset."""
    return type(name, (DataSet,), columns)
