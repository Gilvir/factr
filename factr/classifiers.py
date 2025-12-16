"""Built-in classifiers for grouping assets."""

from __future__ import annotations

import polars as pl

from factr.datasets import Fundamentals

from .core import Classifier, Factor


def Sector(column: str = "sector") -> Classifier:
    """Sector classification from reference data."""
    return Classifier(expr=pl.col(column), name="sector")


def Exchange(column: str = "exchange") -> Classifier:
    """Exchange classification from reference data."""
    return Classifier(expr=pl.col(column), name="exchange")


def Industry(column: str = "industry") -> Classifier:
    """Industry classification from reference data."""
    return Classifier(expr=pl.col(column), name="industry")


def Country(column: str = "country") -> Classifier:
    """Country classification from reference data."""
    return Classifier(expr=pl.col(column), name="country")


def Quantiles(factor: Factor, bins: int, labels: bool = True) -> Classifier:
    """Create quantile classifier from a factor.

    Bins assets into quantiles based on factor values within each date.
    """
    return factor.quantile(bins, labels=labels)


def CustomBins(
    factor: Factor, thresholds: list[float], labels: list[str] | None = None
) -> Classifier:
    """Create classifier by binning factor into custom ranges.

    Args:
        factor: Factor to bin
        thresholds: List of threshold values defining bin edges
        labels: Optional labels for bins (default: 'bin_0', 'bin_1', ...)
    """
    if labels is None:
        labels = [f"bin_{i}" for i in range(len(thresholds) + 1)]

    expr = factor.expr.cut(breaks=thresholds, labels=labels)
    name = f"{factor.name}_binned"

    return Classifier(expr=expr, name=name)


def MarketCapBuckets(
    market_cap: Factor = Fundamentals.market_cap,
    large_cap_threshold: float = 10e9,
    mid_cap_threshold: float = 2e9,
) -> Classifier:
    """Classify stocks by market cap: large/mid/small."""
    expr = (
        pl.when(market_cap.expr >= large_cap_threshold)
        .then(pl.lit("large"))
        .when(market_cap.expr >= mid_cap_threshold)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("small"))
    )
    return Classifier(expr=expr, name="market_cap_bucket")
