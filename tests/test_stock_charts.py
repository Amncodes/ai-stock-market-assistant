import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visualization.stock_charts import create_stock_analysis_chart


class StockChartTests(unittest.TestCase):
    def sample_data(self):
        return pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=40, freq="D"),
                "Close": range(100, 140),
                "RSI_14": [50.0] * 40,
                "MACD": [1.0] * 40,
                "MACD_Signal": [0.5] * 40,
                "MACD_Hist": [0.5] * 40,
            }
        )

    def test_saves_chart_image(self):
        with tempfile.TemporaryDirectory() as folder:
            output_path = Path(folder) / "stock_chart.png"

            result = create_stock_analysis_chart(
                self.sample_data(),
                "TEST.NS",
                output_path,
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_refuses_to_overwrite_existing_chart(self):
        with tempfile.TemporaryDirectory() as folder:
            output_path = Path(folder) / "stock_chart.png"
            output_path.write_text("keep this file")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                create_stock_analysis_chart(
                    self.sample_data(),
                    "TEST.NS",
                    output_path,
                )

            self.assertEqual(output_path.read_text(), "keep this file")

    def test_rejects_missing_indicator_columns(self):
        data = pd.DataFrame({"Date": ["2025-01-01"], "Close": [100]})

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "missing columns"):
                create_stock_analysis_chart(
                    data,
                    "TEST.NS",
                    Path(folder) / "stock_chart.png",
                )

    def test_rejects_invalid_adjusted_close_when_available(self):
        data = self.sample_data()
        data["Adj Close"] = [100.0] * len(data)
        data.loc[0, "Adj Close"] = 0

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Adj Close"):
                create_stock_analysis_chart(
                    data,
                    "TEST.NS",
                    Path(folder) / "stock_chart.png",
                )


if __name__ == "__main__":
    unittest.main()