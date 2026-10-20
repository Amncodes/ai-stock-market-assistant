"""Calculate technical indicators from historical closing prices."""

import pandas as pd


def add_technical_indicators(
    data: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Return a copy of OHLCV data with RSI and MACD columns added."""
    periods = (rsi_period, macd_fast, macd_slow, macd_signal)
    if any(period <= 0 for period in periods):
        raise ValueError("Indicator periods must be positive integers.")
    if macd_fast >= macd_slow:
        raise ValueError("macd_fast must be smaller than macd_slow.")
    if "Close" not in data.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    close = pd.to_numeric(data["Close"], errors="coerce")
    if close.isna().any():
        raise ValueError("Close prices must be numeric and contain no missing values.")
    if (close <= 0).any():
        raise ValueError("Close prices must be greater than zero.")

    result = data.copy()

    # RSI: compare smoothed average gains with smoothed average losses.
    price_change = close.diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss
    result[f"RSI_{rsi_period}"] = 100 - (100 / (1 + relative_strength))

    # A flat price series has no gains or losses; assign neutral RSI.
    flat_prices = (average_gain == 0) & (average_loss == 0)
    result.loc[flat_prices, f"RSI_{rsi_period}"] = 50

    # MACD: difference between fast and slow exponential moving averages.
    fast_average = close.ewm(
        span=macd_fast,
        min_periods=macd_fast,
        adjust=False,
    ).mean()
    slow_average = close.ewm(
        span=macd_slow,
        min_periods=macd_slow,
        adjust=False,
    ).mean()

    result["MACD"] = fast_average - slow_average
    result["MACD_Signal"] = result["MACD"].ewm(
        span=macd_signal,
        min_periods=macd_signal,
        adjust=False,
    ).mean()
    result["MACD_Hist"] = result["MACD"] - result["MACD_Signal"]

    return result