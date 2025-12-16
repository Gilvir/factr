"""Scope enum for factor execution context."""

from enum import Enum


class Scope(Enum):
    """Execution scope for factors.

    - RAW: Raw column data
    - TIME_SERIES: Per-entity operations (rolling windows, shifts)
    - CROSS_SECTION: Per-date operations (rank, demean, zscore)
    """

    RAW = "raw"
    TIME_SERIES = "time_series"
    CROSS_SECTION = "cross_section"
