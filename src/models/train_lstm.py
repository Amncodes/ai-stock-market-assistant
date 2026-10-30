"""Train a small LSTM and compare it with simple forecast baselines."""

import argparse
import math

import numpy as np
import pandas as pd

from src.models.baseline import evaluate_baselines
from src.models.lstm_forecaster import build_lstm_model
from src.models.sequences import prepare_price_sequences


def _forecast_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate forecast errors in rupees and percent."""
    errors = actual - predicted

    return {
        "MAE_INR": float(np.mean(np.abs(errors))),
        "RMSE_INR": float(np.sqrt(np.mean(np.square(errors)))),
        "MAPE_percent": float(np.mean(np.abs(errors) / actual) * 100),
    }


def train_and_evaluate_lstm(
    prices: pd.Series,
    sequence_length: int = 30,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    units: int = 32,
    max_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Train on earlier data and compare forecasts on the final test dates."""
    if max_epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("max_epochs, batch_size, and patience must be positive.")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_seed)

    sequences = prepare_price_sequences(
        prices,
        sequence_length=sequence_length,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    model = build_lstm_model(
        sequence_length=sequence_length,
        units=units,
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
    lstm_predictions = sequences.scaler.inverse_transform(
        scaled_predictions
    ).ravel()
    actual_prices = sequences.scaler.inverse_transform(
        sequences.y_test
    ).ravel()

    # Evaluate the simple baselines over exactly the same test dates.
    test_fraction = 1 - train_fraction - validation_fraction
    baseline_forecasts, _ = evaluate_baselines(
        prices,
        test_fraction=test_fraction,
    )

    if not baseline_forecasts.index.equals(sequences.test_dates):
        raise RuntimeError("Baseline and LSTM test dates do not match.")

    rows = []
    for baseline_name, forecast_column in (
        ("previous_close", "Previous_Close_Forecast"),
        ("moving_average", "Moving_Average_Forecast"),
    ):
        rows.append(
            {
                "Model": baseline_name,
                **_forecast_scores(
                    actual_prices,
                    baseline_forecasts[forecast_column].to_numpy(),
                ),
            }
        )

    rows.append(
        {
            "Model": "LSTM",
            **_forecast_scores(actual_prices, lstm_predictions),
        }
    )

    results = pd.DataFrame(rows).set_index("Model")
    training_info = {
        "epochs_trained": len(history.history["loss"]),
        "best_validation_loss": float(min(history.history["val_loss"])),
    }
    return results, training_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an LSTM on a stock's adjusted closing prices."
    )
    parser.add_argument("input_csv", help="Path to a raw stock-data CSV file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv, parse_dates=["Date"])
    data = data.sort_values("Date").set_index("Date")
    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"

    results, training_info = train_and_evaluate_lstm(
        data[price_column],
        max_epochs=args.epochs,
        patience=args.patience,
    )

    print(f"Price column: {price_column}")
    print(f"Training epochs completed: {training_info['epochs_trained']}")
    print(
        "Best validation loss (scaled prices): "
        f"{training_info['best_validation_loss']:.6f}"
    )
    print("\nForecast errors on the same final test dates:")
    print(results.to_string())


if __name__ == "__main__":
    main()