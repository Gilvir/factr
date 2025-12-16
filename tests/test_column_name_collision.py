"""Test for column name collision between datasets."""

import polars as pl

from factr.data.context import DataContext
from factr.data.sources import DataFrameSource
from factr.datasets import Column, DataSet


def test_column_name_collision_in_load_and_combine():
    """Test that datasets with same column name are properly namespaced."""

    # Dataset A with 'close' column
    class DatasetA(DataSet):
        close = Column(pl.Float64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02"],
                        "asset": ["AAPL", "AAPL"],
                        "close": [100.0, 101.0],
                    }
                )
            )
            is_primary = True

    # Dataset B with 'close' column (same name!)
    class DatasetB(DataSet):
        close = Column(pl.Float64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01", "2024-01-02"],
                        "asset": ["AAPL", "AAPL"],
                        "close": [200.0, 201.0],
                    }
                )
            )

    # When datasets are loaded via DataContext.load(), columns are NOT namespaced
    # (single dataset, no collision risk)
    ctx = DataContext()
    data_a = ctx.load(DatasetA)
    data_b = ctx.load(DatasetB)

    df_a = data_a.collect()
    df_b = data_b.collect()

    # Single dataset loads have non-namespaced columns
    assert "close" in df_a.columns
    assert "close" in df_b.columns

    # But when using load_for_factors (multi-dataset joins), namespacing is applied
    factor_a = DatasetA.close
    factor_b = DatasetB.close

    # Load for factors applies namespacing
    loaded = ctx.load_for_factors([factor_a, factor_b])

    df_a_ns = loaded[DatasetA]
    df_b_ns = loaded[DatasetB]

    # These are LazyFrames, collect them
    if hasattr(df_a_ns, "collect"):
        df_a_ns = df_a_ns.collect()
        df_b_ns = df_b_ns.collect()

    # These are namespaced to prevent collisions during joins
    assert "dataset_a__close" in df_a_ns.columns
    assert "dataset_b__close" in df_b_ns.columns

    print("✓ Datasets namespace columns only when loading for multi-dataset joins")


def test_factors_track_source_datasets():
    """Test that factors track which dataset they come from."""

    class PricingA(DataSet):
        close = Column(pl.Float64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "close": [100.0],
                    }
                )
            )

    class PricingB(DataSet):
        close = Column(pl.Float64)

        class Config:
            source = DataFrameSource(
                pl.DataFrame(
                    {
                        "date": ["2024-01-01"],
                        "asset": ["AAPL"],
                        "close": [200.0],
                    }
                )
            )

    # Factors track which dataset they come from
    close_a = PricingA.close
    close_b = PricingB.close

    assert PricingA in close_a.source_datasets
    assert PricingB in close_b.source_datasets
    assert close_a.source_datasets != close_b.source_datasets

    print("✓ Factors correctly track their source datasets")


if __name__ == "__main__":
    test_column_name_collision_in_load_and_combine()
    print("\n" + "=" * 60 + "\n")
    test_factors_track_source_datasets()
