import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.batch_analysis import analyze_watchlist_files


class BatchAnalysisTests(unittest.TestCase):
    def create_input_csv(self, folder: str, name: str) -> Path:
        close = list(range(100, 120))
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

    def test_creates_summary_for_multiple_stocks(self):
        risk = {
            "annualized_volatility": 0.2,
            "maximum_drawdown": -0.1,
        }

        with tempfile.TemporaryDirectory() as folder:
            inputs = {
                "RELIANCE.NS": self.create_input_csv(folder, "reliance"),
                "TCS.NS": self.create_input_csv(folder, "tcs"),
            }
            summary_path = Path(folder) / "processed" / "summary.csv"

            with patch(
                "src.batch_analysis.analyze_stock_file",
                return_value=(Path("processed.csv"), risk),
            ):
                summary, errors = analyze_watchlist_files(
                    inputs,
                    summary_path,
                    Path(folder) / "processed",
                )

            self.assertEqual(set(summary["Ticker"]), {"RELIANCE.NS", "TCS.NS"})
            self.assertEqual(errors, {})
            self.assertTrue(summary_path.is_file())
            self.assertIn("previous_close_MAPE_percent", summary.columns)
            self.assertIn("moving_average_MAPE_percent", summary.columns)
            self.assertTrue((summary["previous_close_MAPE_percent"] > 0).all())

    def test_one_stock_failure_does_not_stop_the_batch(self):
        risk = {
            "annualized_volatility": 0.2,
            "maximum_drawdown": -0.1,
        }

        with tempfile.TemporaryDirectory() as folder:
            inputs = {
                "RELIANCE.NS": self.create_input_csv(folder, "reliance"),
                "TCS.NS": self.create_input_csv(folder, "tcs"),
                "INFY.NS": self.create_input_csv(folder, "infy"),
            }
            summary_path = Path(folder) / "processed" / "summary.csv"

            def analyze_one_file(path, output_dir):
                if Path(path).stem == "tcs":
                    raise ValueError("sample analysis error")
                return Path("processed.csv"), risk

            with patch(
                "src.batch_analysis.analyze_stock_file",
                side_effect=analyze_one_file,
            ):
                summary, errors = analyze_watchlist_files(
                    inputs,
                    summary_path,
                    Path(folder) / "processed",
                )

            self.assertEqual(set(summary["Ticker"]), {"RELIANCE.NS", "INFY.NS"})
            self.assertIn("TCS.NS", errors)
            self.assertIn("sample analysis error", errors["TCS.NS"])
            self.assertTrue(summary_path.is_file())

    def test_refuses_to_overwrite_existing_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            summary_path = Path(folder) / "summary.csv"
            summary_path.write_text("keep this summary")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                analyze_watchlist_files(
                    {"RELIANCE.NS": "raw.csv"},
                    summary_path,
                )

            self.assertEqual(summary_path.read_text(), "keep this summary")


if __name__ == "__main__":
    unittest.main()