"""Prepare chronological price sequences for time-series models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class PriceSequences:
    """Sequence inputs and targets for training, validation, and testing."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.Index
    validation_dates: pd.Index
    test_dates: pd.Index
    scaler: MinMaxScaler


def prepare_price_sequences(
    prices: pd.Series,
    sequence_length: int = 30,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PriceSequences:
    """Create next-day price targets from ordered historical prices.

    Each input contains `sequence_length` earlier prices. The target is the
    next price. The scaler is fitted on training-period prices only.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
        raise ValueError("sequence_length must be a positive integer.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave test data.")
    if prices.empty:
        raise ValueError("prices must not be empty.")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices must have unique dates in chronological order.")

    numeric_prices = pd.to_numeric(prices, errors="coerce")
    values = numeric_prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Prices must be numeric and finite.")
    if (values <= 0).any():
        raise ValueError("Prices must be greater than zero.")

    row_count = len(values)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))

    if train_end <= sequence_length:
        raise ValueError("Training data must be longer than sequence_length.")
    if validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Data is too short for the requested chronological splits.")

    # Fit scaling using training prices only. Validation and test values can
    # fall outside the 0-to-1 range; they must not influence this fit.
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end].reshape(-1, 1))
    scaled_values = scaler.transform(values.reshape(-1, 1)).astype(np.float32)

    target_positions = np.arange(sequence_length, row_count)
    x_all = np.array(
        [
            scaled_values[position - sequence_length : position]
            for position in target_positions
        ],
        dtype=np.float32,
    )
    y_all = scaled_values[target_positions]

    train_mask = target_positions < train_end
    validation_mask = (
        (target_positions >= train_end)
        & (target_positions < validation_end)
    )
    test_mask = target_positions >= validation_end

    all_dates = prices.index.take(target_positions)

    return PriceSequences(
        x_train=x_all[train_mask],
        y_train=y_all[train_mask],
        x_validation=x_all[validation_mask],
        y_validation=y_all[validation_mask],
        x_test=x_all[test_mask],
        y_test=y_all[test_mask],
        train_dates=all_dates[train_mask],
        validation_dates=all_dates[validation_mask],
        test_dates=all_dates[test_mask],
        scaler=scaler,
    )
