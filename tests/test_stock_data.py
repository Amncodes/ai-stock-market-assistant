import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.stock_data import download_stock_data, save_raw_data


class StockDataTests(unittest.TestCase):
    def sample_data(self):
        return pd.DataFrame(
            {
                "Open": [100.0, 102.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 101.0],
                "Close": [103.0, 104.0],
                "Volume": [1000, 1200],
            },
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        )

    @patch("src.data.stock_data.yf.download")
    def test_download_returns_valid_sorted_data(self, mock_download):
        mock_download.return_value = self.sample_data()

        result = download_stock_data("RELIANCE.NS", start_date="2025-01-01")

        self.assertEqual(len(result), 2)
        self.assertEqual(result.index.name, "Date")
        self.assertEqual(
            set(result.columns),
            {"Open", "High", "Low", "Close", "Volume"},
        )
        mock_download.assert_called_once()

    @patch("src.data.stock_data.yf.download")
    def test_download_rejects_empty_data(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "No historical data"):
            download_stock_data("RELIANCE.NS", start_date="2025-01-01")

    def test_save_refuses_to_overwrite_existing_file(self):
        data = self.sample_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = Path(temp_dir) / "RELIANCE_NS_2026-10-18.csv"
            expected_path.write_text("keep this file")

            with patch("src.data.stock_data.date") as mock_date:
                mock_date.today.return_value.isoformat.return_value = "2026-10-18"

                with self.assertRaises(FileExistsError):
                    save_raw_data(data, "RELIANCE.NS", temp_dir)

            self.assertEqual(expected_path.read_text(), "keep this file")

    @patch("src.data.stock_data.yf.download")
    def test_preserves_adjusted_close_when_available(self, mock_download):
        data = self.sample_data()
        data["Adj Close"] = data["Close"] * 0.99
        mock_download.return_value = data

        result = download_stock_data("RELIANCE.NS", start_date="2025-01-01")

        self.assertIn("Adj Close", result.columns)
        self.assertTrue((result["Adj Close"] > 0).all())


if __name__ == "__main__":
    unittest.main()
