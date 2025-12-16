"""Dataset configuration - inspired by Pydantic's Config pattern.

Configuration is data, not code. Compose sources, don't inherit from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .binding import DataSource


@dataclass
class DataSetConfig:
    """Configuration for a dataset's source and loading behavior.

    Keep it simple - define a default source, or use DataContext.bind() to override.

    Example:
        >>> class EquityPricing(DataSet):
        ...     close = Column(pl.Float64)
        ...
        ...     class Config:
        ...         source = ParquetSource('data/prices.parquet')  # Default for local dev
        ...         reporting_delay = 0
        ...         is_primary = True
        >>>
        >>> # For different environments, use explicit config files:
        >>> # config/prod.py
        >>> def get_context():
        ...     ctx = DataContext()
        ...     ctx.bind(EquityPricing, SQLSource('prod_db', table='prices'))
        ...     return ctx
    """

    source: DataSource | None = None

    column_mapping: dict[str, str] = field(default_factory=dict)

    date_column: str = "date"
    entity_column: str = "asset"

    reporting_delay: int = 0
    forward_fill_columns: list[str] = field(
        default_factory=list
    )
    is_primary: bool = False

    def resolve_source(self) -> DataSource:
        """Get the configured source.

        Returns:
            DataSource if configured

        Raises:
            ValueError: If no source configured
        """
        if self.source is not None:
            return self.source

        raise ValueError(
            "No source configured. Either:\n"
            "1. Add 'source = SomeSource(...)' to your Config class, or\n"
            "2. Use DataContext.bind(YourDataSet, source) to bind explicitly"
        )


def config_from_class(config_class: type) -> DataSetConfig:
    """Extract DataSetConfig from a Config class.

    Args:
        config_class: Inner Config class from a DataSet

    Returns:
        DataSetConfig with values from the class

    Example:
        >>> class MyDataSet(DataSet):
        ...     class Config:
        ...         source = ParquetSource('data.parquet')
        ...         date_column = 'trade_date'
        ...         reporting_delay = 45
        >>>
        >>> config = config_from_class(MyDataSet.Config)
    """
    kwargs = {}

    for field_name in [
        "source",
        "column_mapping",
        "date_column",
        "entity_column",
        "reporting_delay",
        "forward_fill_columns",
        "is_primary",
    ]:
        if hasattr(config_class, field_name):
            kwargs[field_name] = getattr(config_class, field_name)

    return DataSetConfig(**kwargs)
