"""Common factor computations as pure functions returning Factor objects."""

import polars as pl

from factr.datasets import EquityPricing, Fundamentals

from .core import Factor, Filter, Scope
from .custom import factor_func, time_series


def returns(factor: Factor = EquityPricing.close, window: int = 1) -> Factor:
    """Calculate simple returns over specified window."""
    return factor.pct_change(window)


def log_returns(factor: Factor = EquityPricing.close, window: int = 1) -> Factor:
    """Calculate log returns over specified window."""
    return factor.log() - factor.shift(window).log()


def momentum(price: Factor = EquityPricing.close, window: int = 252, skip: int = 21) -> Factor:
    """Calculate momentum: (price[t-skip] / price[t-window]) - 1."""
    return price.shift(skip) / price.shift(window) - 1


def sma(factor: Factor = EquityPricing.close, window: int = 20) -> Factor:
    """Simple moving average over specified window."""
    return factor.rolling_mean(window)


def ema(factor: Factor = EquityPricing.close, window: int = 20) -> Factor:
    """Exponential moving average with specified span."""
    return factor.ewm_mean(span=window)


@factor_func
def dollar_volume(
    price: Factor = EquityPricing.close,
    volume: Factor = EquityPricing.volume,
    window: int = 1,
) -> Factor:
    dv = price * volume
    return dv.rolling_mean(window)


def vwap(
    price: Factor = EquityPricing.close,
    volume: Factor = EquityPricing.volume,
    window: int = 20,
) -> Factor:
    """Volume-weighted average price over specified window."""
    numerator = (price * volume).rolling_sum(window)
    denominator = volume.rolling_sum(window)
    return numerator / denominator


@time_series
def rsi(price: Factor = EquityPricing.close, window: int = 14) -> Factor:
    delta = price.diff(1)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling_mean(window)
    avg_loss = loss.rolling_mean(window)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    price: Factor = EquityPricing.close, window: int = 20, num_std: float = 2.0
) -> tuple[Factor, Factor, Factor]:
    """Calculate Bollinger Bands: (lower, middle, upper)."""
    middle = price.rolling_mean(window)
    std = price.rolling_std(window)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (lower, middle, upper)


def earnings_yield(pe_ratio: Factor = Fundamentals.pe_ratio) -> Factor:
    """Earnings yield (inverse of P/E ratio)."""
    return 1.0 / pe_ratio


def book_to_market(pb_ratio: Factor = Fundamentals.pb_ratio) -> Factor:
    """Book-to-market ratio (inverse of P/B ratio)."""
    return 1.0 / pb_ratio


def liquid_universe(
    price: Factor = EquityPricing.close,
    volume: Factor = EquityPricing.volume,
    min_price: float = 5.0,
    min_volume: float = 1e6,
) -> Filter:
    """Filter for liquid assets based on price and volume thresholds."""
    return (price >= min_price) & (volume >= min_volume)


# ============================================================================
# Technical Indicators
# ============================================================================


def macd(
    price: Factor = EquityPricing.close,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[Factor, Factor, Factor]:
    """Moving Average Convergence Divergence (MACD).

    Args:
        price: Price factor (default: close)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = price.ewm_mean(span=fast)
    slow_ema = price.ewm_mean(span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm_mean(span=signal)
    histogram = macd_line - signal_line
    return (macd_line, signal_line, histogram)


def stochastic(
    high: Factor = EquityPricing.high,
    low: Factor = EquityPricing.low,
    close: Factor = EquityPricing.close,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[Factor, Factor]:
    """Stochastic Oscillator (%K and %D).

    Args:
        high: High price factor
        low: Low price factor
        close: Close price factor
        window: Lookback period (default: 14)
        smooth_k: %K smoothing period (default: 3)
        smooth_d: %D smoothing period (default: 3)

    Returns:
        Tuple of (%K, %D) factors
    """
    lowest_low = low.rolling_min(window)
    highest_high = high.rolling_max(window)

    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    slow_k = fast_k.rolling_mean(smooth_k)
    percent_d = slow_k.rolling_mean(smooth_d)

    return (slow_k, percent_d)


@time_series
def atr(
    high: Factor = EquityPricing.high,
    low: Factor = EquityPricing.low,
    close: Factor = EquityPricing.close,
    window: int = 14,
) -> Factor:
    """Average True Range (ATR).

    Measures market volatility.

    Args:
        high: High price factor
        low: Low price factor
        close: Close price factor
        window: Lookback period (default: 14)

    Returns:
        ATR factor
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range_expr = pl.max_horizontal(tr1.expr, tr2.expr, tr3.expr)
    true_range = Factor(
        expr=true_range_expr,
        name="true_range",
        scope=Scope.TIME_SERIES,
        source_columns=high.source_columns | low.source_columns | close.source_columns,
        source_datasets=high.source_datasets | low.source_datasets | close.source_datasets,
    )

    return true_range.ewm_mean(span=window)


@time_series
def parabolic_sar(
    high: Factor = EquityPricing.high,
    low: Factor = EquityPricing.low,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> Factor:
    """Parabolic SAR (Stop and Reverse).

    Note: This is a simplified version. Full implementation requires
    stateful tracking of trends. This approximation uses price extremes.

    Args:
        high: High price factor
        low: Low price factor
        af_start: Initial acceleration factor (default: 0.02)
        af_increment: AF increment (default: 0.02)
        af_max: Maximum AF (default: 0.2)

    Returns:
        SAR factor (approximate)
    """
    # Simplified SAR using rolling extremes
    # True SAR requires iterative state tracking which is hard in vectorized context
    window = int(af_max / af_increment) + 2

    highest = high.rolling_max(window)
    lowest = low.rolling_min(window)

    # Approximate SAR as weighted average based on position in range
    return lowest + af_max * (highest - lowest)


# ============================================================================
# Momentum Indicators
# ============================================================================


def reversal(
    price: Factor = EquityPricing.close,
    window: int = 21,
) -> Factor:
    """Short-term reversal factor.

    Negative of recent returns - captures mean reversion.

    Args:
        price: Price factor (default: close)
        window: Lookback period (default: 21 days)

    Returns:
        Reversal factor (negative returns)
    """
    return -1.0 * returns(price, window)


@time_series
def acceleration(
    price: Factor = EquityPricing.close,
    window: int = 21,
) -> Factor:
    """Price acceleration (second derivative).

    Momentum of momentum - rate of change of returns.

    Args:
        price: Price factor (default: close)
        window: Lookback period (default: 21 days)

    Returns:
        Acceleration factor
    """
    ret = returns(price, window)
    return ret.diff(window)


@time_series
def trend_strength(
    price: Factor = EquityPricing.close,
    window: int = 63,
) -> Factor:
    """Trend strength using linear regression R-squared.

    Measures how well price follows a linear trend.

    Args:
        price: Price factor (default: close)
        window: Lookback period (default: 63 days)

    Returns:
        Trend strength factor (0 to 1, higher = stronger trend)
    """
    # R² approximation: correlation between price and time index
    # Use rolling correlation with a linear sequence
    log_price = price.log()

    # Rolling correlation of log price with its index
    # Higher correlation = stronger trend
    rolling_corr = log_price.rolling_mean(window) / log_price.rolling_std(window)

    # Normalize to approximate R²
    return abs(rolling_corr)


# ============================================================================
# Risk Indicators
# ============================================================================


def volatility(
    price: Factor = EquityPricing.close,
    window: int = 21,
    annualize: bool = True,
) -> Factor:
    """Rolling volatility (standard deviation of returns).

    Args:
        price: Price factor (default: close)
        window: Lookback period (default: 21 days)
        annualize: If True, annualize volatility (default: True)

    Returns:
        Volatility factor
    """
    ret = returns(price, 1)
    vol = ret.rolling_std(window)

    if annualize:
        # Annualization factor: sqrt(252) for daily data
        vol = vol * (252**0.5)

    return vol


@time_series
def max_drawdown(
    price: Factor = EquityPricing.close,
    window: int = 252,
) -> Factor:
    """Maximum drawdown over rolling window.

    Measures peak-to-trough decline.

    Args:
        price: Price factor (default: close)
        window: Lookback period (default: 252 days)

    Returns:
        Max drawdown factor (negative values, e.g., -0.25 for 25% drawdown)
    """
    rolling_max = price.rolling_max(window)
    drawdown = (price - rolling_max) / rolling_max
    return drawdown.rolling_min(window)


@time_series
def downside_deviation(
    price: Factor = EquityPricing.close,
    window: int = 21,
    annualize: bool = True,
) -> Factor:
    """Downside deviation (semi-deviation).

    Standard deviation of negative returns only.

    Args:
        price: Price factor (default: close)
        window: Lookback period (default: 21 days)
        annualize: If True, annualize deviation (default: True)

    Returns:
        Downside deviation factor
    """
    ret = returns(price, 1)

    # Only negative returns
    negative_returns = ret.clip(upper=0)

    # Standard deviation of negative returns
    downside_dev = negative_returns.rolling_std(window)

    if annualize:
        downside_dev = downside_dev * (252**0.5)

    return downside_dev


# ============================================================================
# Volume Indicators
# ============================================================================


@time_series
def obv(
    close: Factor = EquityPricing.close,
    volume: Factor = EquityPricing.volume,
) -> Factor:
    """On-Balance Volume (OBV).

    Cumulative volume weighted by price direction.

    Args:
        close: Close price factor
        volume: Volume factor

    Returns:
        OBV factor
    """
    price_change = close.diff(1)
    signed_volume = volume * price_change.sign()
    return signed_volume.cumsum()


@time_series
def chaikin_money_flow(
    high: Factor = EquityPricing.high,
    low: Factor = EquityPricing.low,
    close: Factor = EquityPricing.close,
    volume: Factor = EquityPricing.volume,
    window: int = 21,
) -> Factor:
    """Chaikin Money Flow (CMF).

    Measures buying/selling pressure based on close location in range.

    Args:
        high: High price factor
        low: Low price factor
        close: Close price factor
        volume: Volume factor
        window: Lookback period (default: 21 days)

    Returns:
        CMF factor (ranges from -1 to +1)
    """
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    return mf_volume.rolling_sum(window) / volume.rolling_sum(window)


def vwap_bands(
    price: Factor = EquityPricing.close,
    volume: Factor = EquityPricing.volume,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[Factor, Factor, Factor]:
    """VWAP with standard deviation bands.

    Args:
        price: Price factor (default: close)
        volume: Volume factor
        window: Lookback period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (lower_band, vwap, upper_band)
    """
    vwap_value = vwap(price, volume, window)

    # Calculate price std weighted by volume
    price_variance = ((price - vwap_value) ** 2 * volume).rolling_sum(window) / volume.rolling_sum(
        window
    )
    price_std = price_variance**0.5

    lower_band = vwap_value - num_std * price_std
    upper_band = vwap_value + num_std * price_std

    return (lower_band, vwap_value, upper_band)


def volume_profile(
    volume: Factor = EquityPricing.volume,
    window: int = 21,
) -> Factor:
    """Volume profile (normalized volume).

    Volume as percentage of average volume.

    Args:
        volume: Volume factor
        window: Lookback period for average (default: 21 days)

    Returns:
        Volume profile factor (1.0 = average, >1.0 = above average)
    """
    avg_volume = volume.rolling_mean(window)
    return volume / avg_volume


# ============================================================================
# Statistical Indicators
# ============================================================================


@time_series
def autocorrelation(
    factor: Factor = EquityPricing.close,
    window: int = 21,
    lag: int = 1,
) -> Factor:
    """Rolling autocorrelation.

    Correlation of a series with its lagged self.

    Args:
        factor: Input factor (default: close)
        window: Lookback period (default: 21 days)
        lag: Lag for autocorrelation (default: 1)

    Returns:
        Autocorrelation factor (-1 to +1)
    """
    lagged = factor.shift(lag)

    mean_x = factor.rolling_mean(window)
    mean_y = lagged.rolling_mean(window)

    cov = ((factor - mean_x) * (lagged - mean_y)).rolling_sum(window)

    var_x = ((factor - mean_x) ** 2).rolling_sum(window)
    var_y = ((lagged - mean_y) ** 2).rolling_sum(window)

    return cov / (var_x * var_y).sqrt()


# ============================================================================
# Value & Growth Indicators
# ============================================================================


def revenue_growth(
    revenue: Factor = Fundamentals.revenue,
    window: int = 252,
) -> Factor:
    """Year-over-year revenue growth rate.

    Args:
        revenue: Revenue factor
        window: Lookback period (default: 252 days / 1 year)

    Returns:
        Revenue growth factor
    """
    return revenue.pct_change(window)


def earnings_growth(
    earnings: Factor = Fundamentals.earnings,
    window: int = 252,
) -> Factor:
    """Year-over-year earnings growth rate.

    Args:
        earnings: Earnings factor
        window: Lookback period (default: 252 days / 1 year)

    Returns:
        Earnings growth factor
    """
    return earnings.pct_change(window)


def profit_margin(
    earnings: Factor = Fundamentals.earnings,
    revenue: Factor = Fundamentals.revenue,
) -> Factor:
    """Net profit margin.

    Args:
        earnings: Earnings (net income) factor
        revenue: Revenue factor

    Returns:
        Profit margin factor (ratio)
    """
    return earnings / revenue
