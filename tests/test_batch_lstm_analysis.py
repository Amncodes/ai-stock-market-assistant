import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.batch_lstm_analysis import analyze_watchlist_lstm


class BatchLSTMAnalysisTests(unittest.TestCase):
    def create_input_csv(
        self,
        folder: str,
        name: str,
        first_price: int = 100,
    ) -> Path:
        close = list(range(first_price, first_price + 20))
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=20, freq="D"),
                "Open": close,
                "High": [price + 1 for price in close],
                "Low": [price - 1 for price in close],
                "Close": close,
                "Volume": [1000] * len(close),
            }
        )
        path = Path(folder) / f"{name}.csv"
        data.to_csv(path, index=False)
        return path

    def mock_model_results(self):
        results = pd.DataFrame(
            {
                "MAE_INR": [1.0, 2.0, 3.0],
                "RMSE_INR": [1.5, 2.5, 3.5],
                "MAPE_percent": [0.1, 0.2, 0.3],
            },
            index=["previous_close", "moving_average", "LSTM"],
        )
        training_info = {
            "epochs_trained": 10,
            "best_validation_loss": 0.002,
        }
        return results, training_info

    def test_creates_summary_for_multiple_stocks(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = {
                "RELIANCE.NS": self.create_input_csv(folder, "reliance"),
                "TCS.NS": self.create_input_csv(folder, "tcs", first_price=200),
            }
            summary_path = Path(folder) / "processed" / "lstm_summary.csv"

            with patch(
                "src.batch_lstm_analysis.train_and_evaluate_lstm",
                return_value=self.mock_model_results(),
            ):
                summary, errors = analyze_watchlist_lstm(
                    inputs,
                    summary_path,
                )

            self.assertEqual(
                set(summary["Ticker"]),
                {"RELIANCE.NS", "TCS.NS"},
            )
            self.assertEqual(errors, {})
            self.assertTrue(summary_path.is_file())
            self.assertIn("lstm_MAE_INR", summary.columns)
            self.assertIn("previous_close_MAE_INR", summary.columns)
            self.assertIn("moving_average_MAPE_percent", summary.columns)
            self.assertTrue((summary["Training_Epochs"] == 10).all())

    def test_one_stock_failure_does_not_stop_the_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = {
                "RELIANCE.NS": self.create_input_csv(folder, "reliance"),
                "TCS.NS": self.create_input_csv(
                    folder,
                    "tcs",
                    first_price=200,
                ),
                "INFY.NS": self.create_input_csv(
                    folder,
                    "infy",
                    first_price=300,
                ),
            }
            summary_path = Path(folder) / "processed" / "lstm_summary.csv"

            def train_one_stock(prices, **kwargs):
                if prices.iloc[0] == 200:
                    raise ValueError("sample training error")
                return self.mock_model_results()

            with patch(
                "src.batch_lstm_analysis.train_and_evaluate_lstm",
                side_effect=train_one_stock,
            ):
                summary, errors = analyze_watchlist_lstm(
                    inputs,
                    summary_path,
                )

            self.assertEqual(
                set(summary["Ticker"]),
                {"RELIANCE.NS", "INFY.NS"},
            )
            self.assertIn("TCS.NS", errors)
            self.assertIn("sample training error", errors["TCS.NS"])
            self.assertTrue(summary_path.is_file())

    def test_refuses_to_overwrite_existing_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            summary_path = Path(folder) / "summary.csv"
            summary_path.write_text("keep this summary")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                analyze_watchlist_lstm(
                    {"RELIANCE.NS": "raw.csv"},
                    summary_path,
                )

            self.assertEqual(summary_path.read_text(), "keep this summary")

    def test_uses_adjusted_close_when_available(self):
        with tempfile.TemporaryDirectory() as folder:
            input_path = self.create_input_csv(folder, "sample")
            data = pd.read_csv(input_path)
            data["Close"] = [100.0] * len(data)
            data["Adj Close"] = range(100, 120)
            data.to_csv(input_path, index=False)

            summary_path = Path(folder) / "processed" / "lstm_summary.csv"

            with patch(
                "src.batch_lstm_analysis.train_and_evaluate_lstm",
                return_value=self.mock_model_results(),
            ) as train_mock:
                summary, errors = analyze_watchlist_lstm(
                    {"TEST.NS": input_path},
                    summary_path,
                )

            self.assertEqual(errors, {})
            passed_prices = train_mock.call_args.args[0]
            self.assertEqual(passed_prices.name, "Adj Close")
            self.assertEqual(passed_prices.iloc[0], 100)
            self.assertEqual(summary.loc[0, "Price_Column"], "Adj Close")


if __name__ == "__main__":
    unittest.main()