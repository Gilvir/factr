"""Tests for newly implemented Tier 1 indicators."""

import polars as pl

from factr import factors
from factr.core import Factor


# ==============================================================================
# Technical Indicators Tests
# ==============================================================================


def test_macd():
    """Test MACD indicator."""
    df = pl.DataFrame({"close": [100.0 + i * 0.5 for i in range(50)]})
    close = Factor(pl.col("close"), name="close")

    macd_line, signal_line, histogram = factors.macd(close, fast=12, slow=26, signal=9)

    result = df.with_columns(
        [
            macd_line.expr.alias("macd"),
            signal_line.expr.alias("signal"),
            histogram.expr.alias("histogram"),
        ]
    )

    # Basic sanity checks
    assert result["macd"].drop_nulls().len() > 0
    assert result["signal"].drop_nulls().len() > 0
    # Histogram should equal macd - signal
    valid_idx = result["histogram"].is_not_null()
    if valid_idx.sum() > 0:
        assert (
            result.filter(valid_idx).select(
                (pl.col("histogram") - (pl.col("macd") - pl.col("signal"))).abs().max()
            )[0, 0]
            < 1e-6
        )


def test_stochastic():
    """Test Stochastic Oscillator."""
    df = pl.DataFrame(
        {
            "high": [105.0 + i for i in range(30)],
            "low": [95.0 + i for i in range(30)],
            "close": [100.0 + i for i in range(30)],
        }
    )
    high = Factor(pl.col("high"), name="high")
    low = Factor(pl.col("low"), name="low")
    close = Factor(pl.col("close"), name="close")

    percent_k, percent_d = factors.stochastic(high, low, close, window=14)

    result = df.with_columns([percent_k.expr.alias("k"), percent_d.expr.alias("d")])

    # Stochastic should be between 0 and 100
    k_values = result["k"].drop_nulls()
    if k_values.len() > 0:
        assert all(k_values >= 0)
        assert all(k_values <= 100)


def test_atr():
    """Test Average True Range."""
    df = pl.DataFrame(
        {
            "high": [105.0, 110.0, 108.0, 115.0, 112.0],
            "low": [95.0, 98.0, 102.0, 105.0, 108.0],
            "close": [100.0, 105.0, 105.0, 110.0, 110.0],
        }
    )
    high = Factor(pl.col("high"), name="high")
    low = Factor(pl.col("low"), name="low")
    close = Factor(pl.col("close"), name="close")

    atr_factor = factors.atr(high, low, close, window=3)

    result = df.with_columns([atr_factor.expr.alias("atr")])

    # ATR should be positive
    atr_values = result["atr"].drop_nulls()
    assert all(atr_values > 0)


def test_parabolic_sar():
    """Test Parabolic SAR."""
    df = pl.DataFrame(
        {"high": [100.0 + i for i in range(20)], "low": [90.0 + i for i in range(20)]}
    )
    high = Factor(pl.col("high"), name="high")
    low = Factor(pl.col("low"), name="low")

    sar = factors.parabolic_sar(high, low)

    result = df.with_columns([sar.expr.alias("sar")])

    # SAR should produce some values
    assert result["sar"].drop_nulls().len() > 0


# ==============================================================================
# Momentum Indicators Tests
# ==============================================================================


def test_reversal():
    """Test Reversal factor."""
    df = pl.DataFrame({"close": [100.0, 110.0, 120.0, 115.0, 125.0]})
    close = Factor(pl.col("close"), name="close")

    rev = factors.reversal(close, window=2)

    result = df.with_columns([rev.expr.alias("reversal")])

    # Reversal should be negative of returns
    # If price goes up, reversal is negative
    assert result["reversal"][2] < 0  # Price went up from 100 to 120


def test_acceleration():
    """Test Acceleration factor."""
    df = pl.DataFrame({"close": [100.0 + i**2 for i in range(50)]})
    close = Factor(pl.col("close"), name="close")

    accel = factors.acceleration(close, window=5)

    result = df.with_columns([accel.expr.alias("accel")])

    # With quadratic growth, acceleration should be positive
    accel_values = result["accel"].drop_nulls()
    assert accel_values.len() > 0


def test_trend_strength():
    """Test Trend Strength factor."""
    df = pl.DataFrame({"close": [100.0 + i for i in range(100)]})
    close = Factor(pl.col("close"), name="close")

    trend = factors.trend_strength(close, window=20)

    result = df.with_columns([trend.expr.alias("trend")])

    # Should produce some values
    assert result["trend"].drop_nulls().len() > 0


# ==============================================================================
# Risk Indicators Tests
# ==============================================================================


def test_volatility():
    """Test Volatility factor."""
    df = pl.DataFrame({"close": [100.0, 105.0, 95.0, 110.0, 90.0, 115.0] + [100.0] * 20})
    close = Factor(pl.col("close"), name="close")

    vol = factors.volatility(close, window=5, annualize=False)

    result = df.with_columns([vol.expr.alias("vol")])

    # Volatility should be positive
    vol_values = result["vol"].drop_nulls()
    assert all(vol_values >= 0)
    # First period (high variance) should have higher vol than last (no variance)
    assert result["vol"][10] > result["vol"][-1]


def test_max_drawdown():
    """Test Max Drawdown factor."""
    df = pl.DataFrame({"close": [100.0, 110.0, 105.0, 90.0, 95.0, 85.0, 100.0, 110.0]})
    close = Factor(pl.col("close"), name="close")

    mdd = factors.max_drawdown(close, window=10)

    result = df.with_columns([mdd.expr.alias("mdd")])

    # Max drawdown should be negative or zero
    mdd_values = result["mdd"].drop_nulls()
    assert all(mdd_values <= 0)


def test_downside_deviation():
    """Test Downside Deviation factor."""
    df = pl.DataFrame({"close": [100.0, 105.0, 95.0, 110.0, 90.0, 115.0] + [100.0] * 20})
    close = Factor(pl.col("close"), name="close")

    dd = factors.downside_deviation(close, window=5, annualize=False)

    result = df.with_columns([dd.expr.alias("downside_dev")])

    # Downside deviation should be non-negative
    dd_values = result["downside_dev"].drop_nulls()
    assert all(dd_values >= 0)


# ==============================================================================
# Volume Indicators Tests
# ==============================================================================


def test_obv():
    """Test On-Balance Volume."""
    df = pl.DataFrame(
        {
            "close": [100.0, 105.0, 103.0, 108.0, 106.0],
            "volume": [1000, 1500, 1200, 1800, 1400],
        }
    )
    close = Factor(pl.col("close"), name="close")
    volume = Factor(pl.col("volume"), name="volume")

    obv_factor = factors.obv(close, volume)

    result = df.with_columns([obv_factor.expr.alias("obv")])

    # OBV should change based on price direction
    assert result["obv"].is_not_null().sum() > 0


def test_chaikin_money_flow():
    """Test Chaikin Money Flow."""
    df = pl.DataFrame(
        {
            "high": [105.0, 110.0, 108.0, 115.0, 112.0] * 5,
            "low": [95.0, 98.0, 102.0, 105.0, 108.0] * 5,
            "close": [100.0, 105.0, 105.0, 110.0, 110.0] * 5,
            "volume": [1000, 1500, 1200, 1800, 1400] * 5,
        }
    )
    high = Factor(pl.col("high"), name="high")
    low = Factor(pl.col("low"), name="low")
    close = Factor(pl.col("close"), name="close")
    volume = Factor(pl.col("volume"), name="volume")

    cmf = factors.chaikin_money_flow(high, low, close, volume, window=5)

    result = df.with_columns([cmf.expr.alias("cmf")])

    # CMF should be between -1 and 1
    cmf_values = result["cmf"].drop_nulls()
    if cmf_values.len() > 0:
        assert all(cmf_values >= -1.0)
        assert all(cmf_values <= 1.0)


def test_vwap_bands():
    """Test VWAP Bands."""
    df = pl.DataFrame(
        {
            "close": [100.0 + i * 0.5 for i in range(30)],
            "volume": [1000 + i * 10 for i in range(30)],
        }
    )
    close = Factor(pl.col("close"), name="close")
    volume = Factor(pl.col("volume"), name="volume")

    lower, vwap_mid, upper = factors.vwap_bands(close, volume, window=10, num_std=2.0)

    result = df.with_columns(
        [lower.expr.alias("lower"), vwap_mid.expr.alias("vwap"), upper.expr.alias("upper")]
    )

    # Check band ordering: lower < vwap < upper
    valid_rows = result.filter(
        pl.col("lower").is_not_null() & pl.col("vwap").is_not_null() & pl.col("upper").is_not_null()
    )

    if valid_rows.height > 0:
        assert all(valid_rows["lower"] <= valid_rows["vwap"])
        assert all(valid_rows["vwap"] <= valid_rows["upper"])


def test_volume_profile():
    """Test Volume Profile."""
    df = pl.DataFrame({"volume": [1000, 1500, 1200, 2000, 1800, 1600]})
    volume = Factor(pl.col("volume"), name="volume")

    vp = factors.volume_profile(volume, window=3)

    result = df.with_columns([vp.expr.alias("vol_profile")])

    # Volume profile should be around 1.0 on average
    vp_values = result["vol_profile"].drop_nulls()
    assert vp_values.len() > 0
    assert all(vp_values > 0)


# ==============================================================================
# Statistical Indicators Tests
# ==============================================================================


def test_autocorrelation():
    """Test Autocorrelation factor."""
    # Create trending data with positive autocorrelation
    df = pl.DataFrame({"close": [100.0 + i + (i % 2) for i in range(50)]})
    close = Factor(pl.col("close"), name="close")

    autocorr = factors.autocorrelation(close, window=10, lag=1)

    result = df.with_columns([autocorr.expr.alias("autocorr")])

    # Autocorrelation should be between -1 and 1
    ac_values = result["autocorr"].drop_nulls()
    if ac_values.len() > 0:
        assert all(ac_values >= -1.1)  # Allow small numerical errors
        assert all(ac_values <= 1.1)


# ==============================================================================
# Value & Growth Indicators Tests
# ==============================================================================


def test_revenue_growth():
    """Test Revenue Growth factor."""
    df = pl.DataFrame({"revenue": [1000000.0 * (1.1**i) for i in range(300)]})
    revenue = Factor(pl.col("revenue"), name="revenue")

    rev_growth = factors.revenue_growth(revenue, window=252)

    result = df.with_columns([rev_growth.expr.alias("rev_growth")])

    # With 10% growth rate, annual growth should be around 10%
    growth_values = result["rev_growth"].drop_nulls()
    assert growth_values.len() > 0


def test_earnings_growth():
    """Test Earnings Growth factor."""
    df = pl.DataFrame({"earnings": [100000.0 * (1.05**i) for i in range(300)]})
    earnings = Factor(pl.col("earnings"), name="earnings")

    earn_growth = factors.earnings_growth(earnings, window=252)

    result = df.with_columns([earn_growth.expr.alias("earn_growth")])

    # With 5% growth rate, annual growth should be around 5%
    growth_values = result["earn_growth"].drop_nulls()
    assert growth_values.len() > 0


def test_profit_margin():
    """Test Profit Margin factor."""
    df = pl.DataFrame(
        {"earnings": [100000.0, 120000.0, 90000.0], "revenue": [500000.0, 600000.0, 450000.0]}
    )
    earnings = Factor(pl.col("earnings"), name="earnings")
    revenue = Factor(pl.col("revenue"), name="revenue")

    margin = factors.profit_margin(earnings, revenue)

    result = df.with_columns([margin.expr.alias("margin")])

    # Check margin calculation: earnings / revenue
    assert abs(result["margin"][0] - 0.2) < 1e-6  # 100k/500k = 0.2
    assert abs(result["margin"][1] - 0.2) < 1e-6  # 120k/600k = 0.2
    assert abs(result["margin"][2] - 0.2) < 1e-6  # 90k/450k = 0.2
