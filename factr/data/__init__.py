"""Data layer - protocols, sources, loaders, config, and alignment.

Inspired by Pydantic and dependency injection patterns:
- DataSet.Config for source configuration
- DataContext for managing multiple datasets
- DataSet.load() for simple use cases
"""

from .alignment import apply_offset, asof_join, forward_fill
from .binding import ColumnMapper, DataSource
from .config import DataSetConfig
from .context import DataContext
from .loaders import combine_sources
from .sources import CSVSource, DataFrameSource, ParquetSource, SQLSource

__all__ = [
    # Alignment helpers
    "asof_join",
    "apply_offset",
    "forward_fill",
    # Protocols
    "DataSource",
    "ColumnMapper",
    # Source implementations
    "DataFrameSource",
    "ParquetSource",
    "CSVSource",
    "SQLSource",
    # Loader functions
    "combine_sources",
    # Config and context
    "DataSetConfig",
    "DataContext",
]
