"""Tests for factor computation functions."""

import polars as pl

from factr import factors
from factr.core import Factor


def test_returns():
    df = pl.DataFrame({"asset": ["A"] * 3, "close": [100.0, 110.0, 105.0]})
    close = Factor(pl.col("close"), name="close")

    result = df.with_columns([factors.returns(close, 1).expr.alias("returns")])

    assert result["returns"][0] is None
    assert abs(result["returns"][1] - 0.10) < 1e-6
    assert abs(result["returns"][2] - (-0.045454545)) < 1e-6


def test_log_returns():
    df = pl.DataFrame({"asset": ["A"] * 3, "close": [100.0, 110.0, 105.0]})
    close = Factor(pl.col("close"), name="close")

    result = df.with_columns([factors.log_returns(close, 1).expr.alias("log_returns")])

    assert result["log_returns"][0] is None
    assert abs(result["log_returns"][1] - 0.0953) < 1e-3


def test_momentum():
    df = pl.DataFrame({"asset": ["A"] * 12, "close": [100.0 + i for i in range(12)]})
    close = Factor(pl.col("close"), name="close")

    result = df.with_columns([factors.momentum(close, window=11, skip=1).expr.alias("momentum")])

    assert abs(result["momentum"][11] - 0.1) < 1e-6


def test_sma():
    df = pl.DataFrame({"asset": ["A"] * 5, "close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    close = Factor(pl.col("close"), name="close")

    result = df.with_columns([factors.sma(close, 3).expr.alias("sma_3")])

    assert abs(result["sma_3"][2] - 2.0) < 1e-6
    assert abs(result["sma_3"][4] - 4.0) < 1e-6


def test_dollar_volume():
    df = pl.DataFrame({"asset": ["A"] * 3, "close": [10.0, 11.0, 12.0], "volume": [100, 200, 150]})
    close = Factor(pl.col("close"), name="close")
    volume = Factor(pl.col("volume"), name="volume")

    result = df.with_columns([factors.dollar_volume(close, volume, window=1).expr.alias("dv")])

    assert result["dv"][0] == 1000.0
    assert result["dv"][1] == 2200.0
    assert result["dv"][2] == 1800.0


def test_vwap():
    df = pl.DataFrame({"asset": ["A"] * 3, "close": [10.0, 11.0, 12.0], "volume": [100, 200, 150]})
    close = Factor(pl.col("close"), name="close")
    volume = Factor(pl.col("volume"), name="volume")

    result = df.with_columns([factors.vwap(close, volume, window=2).expr.alias("vwap")])

    assert abs(result["vwap"][1] - 10.666666) < 1e-3


def test_rsi():
    df = pl.DataFrame(
        {
            "asset": ["A"] * 20,
            "close": [100.0 + i if i < 10 else 110.0 - (i - 10) for i in range(20)],
        }
    )
    close = Factor(pl.col("close"), name="close")

    result = df.with_columns([factors.rsi(close, window=14).expr.alias("rsi")])

    rsi_values = result["rsi"].drop_nulls()
    assert all(rsi_values >= 0)
    assert all(rsi_values <= 100)


def test_bollinger_bands():
    df = pl.DataFrame({"asset": ["A"] * 20, "close": [100.0 + i for i in range(20)]})
    close = Factor(pl.col("close"), name="close")

    lower, middle, upper = factors.bollinger_bands(close, window=10, num_std=2.0)
    result = df.with_columns(
        [
            lower.expr.alias("bb_lower"),
            middle.expr.alias("bb_middle"),
            upper.expr.alias("bb_upper"),
        ]
    )

    for i in range(10, 20):
        assert result["bb_lower"][i] < result["bb_middle"][i]
        assert result["bb_middle"][i] < result["bb_upper"][i]


def test_earnings_yield():
    df = pl.DataFrame({"pe_ratio": [10.0, 20.0, 25.0]})
    pe_ratio = Factor(pl.col("pe_ratio"), name="pe_ratio")

    result = df.with_columns([factors.earnings_yield(pe_ratio).expr.alias("ey")])

    assert abs(result["ey"][0] - 0.1) < 1e-6
    assert abs(result["ey"][1] - 0.05) < 1e-6


def test_book_to_market():
    df = pl.DataFrame({"pb_ratio": [2.0, 4.0, 5.0]})
    pb_ratio = Factor(pl.col("pb_ratio"), name="pb_ratio")

    result = df.with_columns([factors.book_to_market(pb_ratio).expr.alias("btm")])

    assert abs(result["btm"][0] - 0.5) < 1e-6
    assert abs(result["btm"][1] - 0.25) < 1e-6


def test_liquid_universe():
    df = pl.DataFrame({"close": [10.0, 3.0, 15.0, 8.0], "volume": [2e6, 5e6, 1e5, 3e6]})
    close = Factor(pl.col("close"), name="close")
    volume = Factor(pl.col("volume"), name="volume")

    result = df.filter(factors.liquid_universe(close, volume, min_price=5.0, min_volume=1e6).expr)

    assert len(result) == 2
    assert all(result["close"] >= 5.0)
    assert all(result["volume"] >= 1e6)


def test_multiple_assets():
    df = pl.DataFrame(
        {
            "asset": ["A"] * 5 + ["B"] * 5,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0] + [200.0, 202.0, 204.0, 206.0, 208.0],
        }
    )
    close = Factor(pl.col("close"), name="close")

    result = df.with_columns([factors.returns(close, 1).expr.alias("returns")])

    a_returns = result.filter(pl.col("asset") == "A")["returns"]
    b_returns = result.filter(pl.col("asset") == "B")["returns"]

    assert abs(a_returns[1] - 0.01) < 1e-6
    assert abs(b_returns[1] - 0.01) < 1e-6
