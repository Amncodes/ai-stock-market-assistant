import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis import analyze_stock_file


class AnalysisPipelineTests(unittest.TestCase):
    def create_input_csv(self, folder: str) -> Path:
        dates = pd.date_range("2025-01-01", periods=40, freq="D")
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": range(100, 140),
                "High": range(101, 141),
                "Low": range(99, 139),
                "Close": range(100, 140),
                "Volume": [1000] * 40,
            }
        ).iloc[::-1]  # Deliberately write newest dates first.

        input_path = Path(folder) / "sample.csv"
        data.to_csv(input_path, index=False)
        return input_path

    def test_analyzes_sorts_and_saves_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            input_path = self.create_input_csv(folder)
            output_dir = Path(folder) / "processed"

            output_path, metrics = analyze_stock_file(input_path, output_dir)

            result = pd.read_csv(output_path, parse_dates=["Date"])
            self.assertEqual(len(result), 40)
            self.assertTrue(result["Date"].is_monotonic_increasing)
            self.assertIn("RSI_14", result.columns)
            self.assertIn("MACD_Hist", result.columns)
            self.assertIn("annualized_volatility", metrics)
            self.assertIn("maximum_drawdown", metrics)
            self.assertEqual(metrics["maximum_drawdown"], 0.0)

    def test_refuses_to_overwrite_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            input_path = self.create_input_csv(folder)
            output_dir = Path(folder) / "processed"

            analyze_stock_file(input_path, output_dir)

            with self.assertRaisesRegex(FileExistsError, "Output already exists"):
                analyze_stock_file(input_path, output_dir)

    def test_rejects_csv_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / "incomplete.csv"
            pd.DataFrame({"Date": ["2025-01-01"], "Close": [100]}).to_csv(
                input_path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "missing"):
                analyze_stock_file(input_path, Path(folder) / "processed")


if __name__ == "__main__":
    unittest.main()