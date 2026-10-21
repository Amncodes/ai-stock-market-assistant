"""Calculate basic historical risk metrics from closing prices."""

import numpy as np
import pandas as pd


def _validate_close_prices(close_prices: pd.Series) -> pd.Series:
    """Convert and validate a chronological series of closing prices."""
    if not isinstance(close_prices, pd.Series):
        raise TypeError("close_prices must be a pandas Series.")
    if close_prices.empty:
        raise ValueError("close_prices must not be empty.")

    prices = pd.to_numeric(close_prices, errors="coerce")
    values = prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Close prices must be numeric and finite.")
    if (prices <= 0).any():
        raise ValueError("Close prices must be greater than zero.")
    if len(prices) < 2:
        raise ValueError("At least two closing prices are required.")

    return prices


def calculate_daily_returns(close_prices: pd.Series) -> pd.Series:
    """Calculate simple daily percentage returns from chronological closes.

    The first result is NaN because there is no previous closing price to
    compare with.
    """
    prices = _validate_close_prices(close_prices)
    returns = prices.pct_change(fill_method=None)
    returns.name = "Daily_Return"
    return returns


def calculate_annualized_volatility(
    close_prices: pd.Series,
    trading_days_per_year: int = 252,
) -> float:
    """Estimate annualized volatility from daily returns.

    Returns a decimal: for example, 0.25 represents 25% annualized volatility.
    """
    if (
        not isinstance(trading_days_per_year, int)
        or isinstance(trading_days_per_year, bool)
        or trading_days_per_year <= 0
    ):
        raise ValueError("trading_days_per_year must be a positive integer.")

    daily_returns = calculate_daily_returns(close_prices).dropna()
    if len(daily_returns) < 2:
        raise ValueError("At least three closing prices are needed for volatility.")

    return float(daily_returns.std(ddof=1) * np.sqrt(trading_days_per_year))


def calculate_max_drawdown(close_prices: pd.Series) -> float:
    """Return the largest peak-to-trough decline as a negative decimal.

    For example, -0.20 represents a 20% maximum drawdown.
    """
    prices = _validate_close_prices(close_prices)
    drawdowns = prices / prices.cummax() - 1
    return float(drawdowns.min())