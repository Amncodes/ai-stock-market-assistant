"""Train a multivariable LSTM and compare it with forecast baselines."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.technical_indicators import add_technical_indicators
from src.models.baseline import evaluate_baselines
from src.models.lstm_forecaster import build_lstm_model
from src.models.sequences import prepare_feature_sequences


def _forecast_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate forecast errors in rupees and percent."""
    errors = actual - predicted

    return {
        "MAE_INR": float(np.mean(np.abs(errors))),
        "RMSE_INR": float(np.sqrt(np.mean(np.square(errors)))),
        "MAPE_percent": float(np.mean(np.abs(errors) / actual) * 100),
    }


def train_and_evaluate_multivariate_lstm(
    data: pd.DataFrame,
    sequence_length: int = 30,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    units: int = 32,
    max_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on several daily features and evaluate on held-out dates."""
    if max_epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("max_epochs, batch_size, and patience must be positive.")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("data must have unique dates in chronological order.")

    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    required_columns = {price_column, "Volume"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_seed)

    featured_data = add_technical_indicators(data, price_column=price_column)
    feature_columns = [
        price_column,
        "RSI_14",
        "MACD",
        "MACD_Hist",
        "Volume",
    ]

    # Indicator warm-up rows have missing values; remove them before making
    # sequences, while keeping all later dates in chronological order.
    model_data = featured_data.dropna(subset=feature_columns)
    if model_data.empty:
        raise ValueError("No rows remain after calculating technical indicators.")

    features = model_data[feature_columns]
    target = model_data[price_column]

    sequences = prepare_feature_sequences(
        features,
        target,
        sequence_length=sequence_length,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    model = build_lstm_model(
        sequence_length=sequence_length,
        units=units,
        feature_count=len(feature_columns),
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        sequences.x_train,
        sequences.y_train,
        validation_data=(sequences.x_validation, sequences.y_validation),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )

    scaled_predictions = model.predict(sequences.x_test, verbose=0)
    lstm_predictions = sequences.target_scaler.inverse_transform(
        scaled_predictions
    ).ravel()
    actual_prices = sequences.target_scaler.inverse_transform(
        sequences.y_test
    ).ravel()

    # Baselines are aligned to the exact dates used by the multivariable model.
    test_fraction = 1 - train_fraction - validation_fraction
    baseline_forecasts, _ = evaluate_baselines(
        data[price_column],
        test_fraction=test_fraction,
    )
    baseline_forecasts = baseline_forecasts.reindex(sequences.test_dates)

    if baseline_forecasts.isna().any().any():
        raise RuntimeError("Baseline forecasts are missing model test dates.")

    forecast_data = pd.DataFrame(
        {
            "Actual": actual_prices,
            "Previous_Close": baseline_forecasts[
                "Previous_Close_Forecast"
            ].to_numpy(),
            "Moving_Average": baseline_forecasts[
                "Moving_Average_Forecast"
            ].to_numpy(),
            "LSTM": lstm_predictions,
        },
        index=sequences.test_dates,
    )
    forecast_data.index.name = "Date"

    rows = []
    for model_name, forecast_column in (
        ("previous_close", "Previous_Close"),
        ("moving_average", "Moving_Average"),
        ("multivariate_lstm", "LSTM"),
    ):
        rows.append(
            {
                "Model": model_name,
                **_forecast_scores(
                    forecast_data["Actual"].to_numpy(),
                    forecast_data[forecast_column].to_numpy(),
                ),
            }
        )

    results = pd.DataFrame(rows).set_index("Model")
    training_info = {
        "epochs_trained": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
        "feature_columns": feature_columns,
        "forecast_data": forecast_data,
    }
    return results, training_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multivariable LSTM on historical stock data."
    )
    parser.add_argument("input_csv", help="Path to a raw stock-data CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    data = pd.read_csv(input_path, parse_dates=["Date"])
    data = data.sort_values("Date").set_index("Date")

    results, training_info = train_and_evaluate_multivariate_lstm(
        data,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    print("Input features:", ", ".join(training_info["feature_columns"]))
    print(f"Training epochs completed: {training_info['epochs_trained']}")
    print(
        "Best validation loss (scaled prices): "
        f"{training_info['best_validation_loss']:.6f}"
    )
    print("\nForecast errors on the same final test dates:")
    print(results.to_string())

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / f"{input_path.stem}_multivariate_forecasts.csv"

    if forecast_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {forecast_path}"
        )

    training_info["forecast_data"].to_csv(forecast_path)
    print(f"\nTest forecasts saved to: {forecast_path}")


if __name__ == "__main__":
    main()