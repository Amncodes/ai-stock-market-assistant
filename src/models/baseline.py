"""Evaluate simple one-day-ahead stock-price forecast baselines."""

import numpy as np
import pandas as pd


def evaluate_baselines(
    close_prices: pd.Series,
    test_fraction: float = 0.2,
    moving_average_window: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare previous-close and moving-average forecasts chronologically.

    Each test-day forecast uses prices available before that day. As the test
    period advances, the previous day's actual close becomes available for the
    next one-day-ahead forecast.

    Returns:
        forecasts: Actual closes and both baseline forecasts for test dates.
        metrics: MAE and RMSE for each baseline over the same test dates.
    """
    if not isinstance(close_prices, pd.Series):
        raise TypeError("close_prices must be a pandas Series.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if (
        not isinstance(moving_average_window, int)
        or isinstance(moving_average_window, bool)
        or moving_average_window <= 0
    ):
        raise ValueError("moving_average_window must be a positive integer.")
    if close_prices.empty:
        raise ValueError("close_prices must not be empty.")

    prices = pd.to_numeric(close_prices, errors="coerce")
    values = prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Close prices must be numeric and finite.")
    if (prices <= 0).any():
        raise ValueError("Close prices must be greater than zero.")

    split_index = int(len(prices) * (1 - test_fraction))
    if split_index <= moving_average_window:
        raise ValueError(
            "The historical portion must contain more rows than "
            "moving_average_window."
        )
    if split_index >= len(prices):
        raise ValueError("The test portion must contain at least one row.")

    actual_values = values[split_index:]
    previous_close_forecasts = []
    moving_average_forecasts = []

    for position in range(split_index, len(values)):
        history = values[position - moving_average_window : position]
        previous_close_forecasts.append(history[-1])
        moving_average_forecasts.append(float(history.mean()))

    forecasts = pd.DataFrame(
        {
            "Actual_Close": actual_values,
            "Previous_Close_Forecast": previous_close_forecasts,
            "Moving_Average_Forecast": moving_average_forecasts,
        },
        index=prices.index[split_index:],
    )
    forecasts.index.name = close_prices.index.name

    metric_rows = []
    for name, forecast_column in (
        ("previous_close", "Previous_Close_Forecast"),
        ("moving_average", "Moving_Average_Forecast"),
    ):
        errors = forecasts["Actual_Close"] - forecasts[forecast_column]
        metric_rows.append(
            {
                "Baseline": name,
                "MAE": float(errors.abs().mean()),
                "RMSE": float(np.sqrt(np.square(errors).mean())),
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("Baseline")
    return forecasts, metrics