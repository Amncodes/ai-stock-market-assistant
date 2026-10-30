import unittest

import pandas as pd

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.models.lstm_forecaster import build_lstm_model
from src.models.train_lstm import train_and_evaluate_lstm


class LSTMForecasterTests(unittest.TestCase):
    @unittest.skipIf(tf is None, "TensorFlow is installed only in the LSTM environment")
    def test_model_returns_one_value_per_sequence(self):
        model = build_lstm_model(sequence_length=30, units=8)
        sample_input = tf.zeros((2, 30, 1))

        prediction = model(sample_input)

        self.assertEqual(tuple(prediction.shape), (2, 1))

    def test_rejects_invalid_model_settings(self):
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            build_lstm_model(sequence_length=0)

        with self.assertRaisesRegex(ValueError, "units"):
            build_lstm_model(units=0)

        with self.assertRaisesRegex(ValueError, "learning_rate"):
            build_lstm_model(learning_rate=0)

    @unittest.skipIf(tf is None, "TensorFlow is installed only in the LSTM environment")
    def test_trains_and_returns_comparable_forecast_metrics(self):
        prices = pd.Series(
            [100 + position * 0.5 for position in range(80)],
            index=pd.date_range("2025-01-01", periods=80, freq="D"),
            name="Adj Close",
        )

        results, training_info = train_and_evaluate_lstm(
            prices,
            sequence_length=10,
            train_fraction=0.70,
            validation_fraction=0.15,
            units=4,
            max_epochs=2,
            batch_size=8,
            patience=1,
        )

        self.assertEqual(
            set(results.index),
            {"previous_close", "moving_average", "LSTM"},
        )
        self.assertIn("MAE_INR", results.columns)
        self.assertIn("RMSE_INR", results.columns)
        self.assertIn("MAPE_percent", results.columns)
        self.assertLessEqual(training_info["epochs_trained"], 2)


if __name__ == "__main__":
    unittest.main()