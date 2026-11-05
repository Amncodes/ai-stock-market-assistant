"""Prepare chronological sequences for time-series models."""

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


@dataclass(frozen=True)
class FeatureSequences:
    """Multivariable sequence inputs and scaled price targets."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_dates: pd.Index
    validation_dates: pd.Index
    test_dates: pd.Index
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def _validate_split_settings(
    sequence_length: int,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Validate shared settings used by sequence preparation functions."""
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


def _get_split_positions(
    row_count: int,
    sequence_length: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int, np.ndarray]:
    """Return chronological split boundaries and target row positions."""
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))

    if train_end <= sequence_length:
        raise ValueError("Training data must be longer than sequence_length.")
    if validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Data is too short for the requested chronological splits.")

    target_positions = np.arange(sequence_length, row_count)
    return train_end, validation_end, target_positions


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

    _validate_split_settings(
        sequence_length,
        train_fraction,
        validation_fraction,
    )

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

    train_end, validation_end, target_positions = _get_split_positions(
        len(values),
        sequence_length,
        train_fraction,
        validation_fraction,
    )

    # Fit scaling using training prices only.
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end].reshape(-1, 1))
    scaled_values = scaler.transform(values.reshape(-1, 1)).astype(np.float32)

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


def prepare_feature_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 30,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> FeatureSequences:
    """Create multi-feature inputs for next-day price forecasts.

    Feature and target scalers are fitted on training-period rows only.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series.")

    _validate_split_settings(
        sequence_length,
        train_fraction,
        validation_fraction,
    )

    if features.empty or len(features.columns) == 0:
        raise ValueError("features must not be empty.")
    if not features.index.equals(target.index):
        raise ValueError("features and target must have matching indexes.")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("features must have unique dates in chronological order.")

    feature_values = features.apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    target_values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError("Features must be numeric and finite.")
    if not np.isfinite(target_values).all():
        raise ValueError("Target prices must be numeric and finite.")
    if (target_values <= 0).any():
        raise ValueError("Target prices must be greater than zero.")

    train_end, validation_end, target_positions = _get_split_positions(
        len(target_values),
        sequence_length,
        train_fraction,
        validation_fraction,
    )

    # Fit both scalers using training-period rows only.
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_values[:train_end])
    scaled_features = feature_scaler.transform(feature_values).astype(np.float32)

    target_scaler = MinMaxScaler()
    target_scaler.fit(target_values[:train_end].reshape(-1, 1))
    scaled_target = target_scaler.transform(
        target_values.reshape(-1, 1)
    ).astype(np.float32)

    x_all = np.array(
        [
            scaled_features[position - sequence_length : position]
            for position in target_positions
        ],
        dtype=np.float32,
    )
    y_all = scaled_target[target_positions]

    train_mask = target_positions < train_end
    validation_mask = (
        (target_positions >= train_end)
        & (target_positions < validation_end)
    )
    test_mask = target_positions >= validation_end
    all_dates = features.index.take(target_positions)

    return FeatureSequences(
        x_train=x_all[train_mask],
        y_train=y_all[train_mask],
        x_validation=x_all[validation_mask],
        y_validation=y_all[validation_mask],
        x_test=x_all[test_mask],
        y_test=y_all[test_mask],
        train_dates=all_dates[train_mask],
        validation_dates=all_dates[validation_mask],
        test_dates=all_dates[test_mask],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )