"""Universe definitions for asset selection."""

from __future__ import annotations

import polars as pl

from .core import Filter, Scope


class Universe:
    """Named filter representing a tradable universe."""

    def __init__(self, filter: Filter, name: str, description: str = ""):
        self.filter = filter
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        return f"Universe(name='{self.name}')"


class TopNUniverse(Universe):
    """Top N stocks by dollar volume."""

    def __init__(
        self,
        n: int = 500,
        window: int = 1,
        min_price: float = 5.0,
        price_col: str = "close",
        volume_col: str = "volume",
        asset_col: str = "asset",
    ):
        if window == 1:
            dv_expr = pl.col(price_col) * pl.col(volume_col)
        else:
            dv_expr = pl.col(f"dollar_volume_{window}")

        rank_expr = dv_expr.rank(method="ordinal", descending=True)
        filter_expr = (rank_expr <= n) & (pl.col(price_col) >= min_price)

        filter = Filter(expr=filter_expr, name=f"top{n}", scope=Scope.CROSS_SECTION)
        super().__init__(
            filter=filter,
            name=f"Top{n}",
            description=f"Top {n} stocks by {window}-day dollar volume (min price ${min_price})",
        )


# Convenient aliases
def Q500US(**kwargs) -> TopNUniverse:
    """Top 500 US stocks by dollar volume."""
    return TopNUniverse(n=500, **kwargs)


def Q1500US(**kwargs) -> TopNUniverse:
    """Top 1500 US stocks by dollar volume."""
    return TopNUniverse(n=1500, **kwargs)


class LiquidUniverse(Universe):
    """Stocks meeting minimum liquidity criteria."""

    def __init__(
        self,
        min_price: float = 5.0,
        min_volume: float = 1e6,
        min_dollar_volume: float | None = None,
        price_col: str = "close",
        volume_col: str = "volume",
    ):
        filter_expr = (pl.col(price_col) >= min_price) & (pl.col(volume_col) >= min_volume)

        if min_dollar_volume is not None:
            filter_expr = filter_expr & (
                (pl.col(price_col) * pl.col(volume_col)) >= min_dollar_volume
            )

        filter = Filter(expr=filter_expr, name="liquid", scope=Scope.RAW)
        desc_parts = [f"price >= ${min_price}", f"volume >= {min_volume:,.0f}"]
        if min_dollar_volume:
            desc_parts.append(f"dollar_volume >= ${min_dollar_volume:,.0f}")

        super().__init__(
            filter=filter,
            name="LiquidUniverse",
            description="Stocks meeting criteria: " + ", ".join(desc_parts),
        )


class AllAssets(Universe):
    """All available assets."""

    def __init__(self):
        filter = Filter(expr=pl.lit(True), name="all_assets", scope=Scope.RAW)
        super().__init__(filter=filter, name="AllAssets", description="All available assets")


def custom_universe(filter: Filter, name: str, description: str = "") -> Universe:
    """Create a custom universe filter from a boolean Factor."""
    return Universe(filter=filter, name=name, description=description)
