"""Polars-based cross-sectional factor library.

A modern, high-performance library for cross-sectional factor analysis built on Polars.
Inspired by Zipline, built for the modern data stack.
"""

from . import classifiers, custom, data, datasets, factors, universe
from .core import Classifier, Factor, Filter
from .custom import (
    cross_section,
    custom_factor,
    expression_factor,
    factor_func,
    make_factor,
    rolling_factor,
    time_series,
)
from .datasets import (
    Column,
    DataSet,
    EquityPricing,
    Fundamentals,
    ReferenceData,
    Sentiment,
    dataset,
)
from .pipeline import Pipeline
from .universe import (
    Q500US,
    Q1500US,
    AllAssets,
    LiquidUniverse,
    Universe,
    custom_universe,
)

__version__ = "0.1.0"

__all__ = [
    "Factor",
    "Filter",
    "Classifier",
    "Pipeline",
    "factors",
    "data",
    "universe",
    "classifiers",
    "custom",
    "datasets",
    "Universe",
    "Q500US",
    "Q1500US",
    "LiquidUniverse",
    "AllAssets",
    "custom_universe",
    "factor_func",
    "rolling_factor",
    "expression_factor",
    "make_factor",
    "time_series",
    "cross_section",
    "custom_factor",
    "DataSet",
    "Column",
    "EquityPricing",
    "Fundamentals",
    "ReferenceData",
    "Sentiment",
    "dataset",
]
