import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.watchlist import download_watchlist


class WatchlistTests(unittest.TestCase):
    @patch("src.data.watchlist.save_raw_data")
    @patch("src.data.watchlist.download_stock_data")
    def test_downloads_each_ticker(self, mock_download, mock_save):
        mock_download.return_value = pd.DataFrame({"Close": [100.0]})
        mock_save.side_effect = [
            Path("RELIANCE.csv"),
            Path("TCS.csv"),
        ]

        saved_files, errors = download_watchlist(
            ["RELIANCE.NS", "TCS.NS"],
            start_date="2025-01-01",
        )

        self.assertEqual(set(saved_files), {"RELIANCE.NS", "TCS.NS"})
        self.assertEqual(errors, {})
        self.assertEqual(mock_download.call_count, 2)
        self.assertEqual(mock_save.call_count, 2)

    @patch("src.data.watchlist.save_raw_data")
    @patch("src.data.watchlist.download_stock_data")
    def test_one_failure_does_not_stop_other_tickers(
        self,
        mock_download,
        mock_save,
    ):
        mock_download.side_effect = [
            pd.DataFrame({"Close": [100.0]}),
            RuntimeError("sample provider error"),
            pd.DataFrame({"Close": [200.0]}),
        ]
        mock_save.side_effect = [
            Path("RELIANCE.csv"),
            Path("INFY.csv"),
        ]

        saved_files, errors = download_watchlist(
            ["RELIANCE.NS", "BAD.NS", "INFY.NS"],
            start_date="2025-01-01",
        )

        self.assertEqual(set(saved_files), {"RELIANCE.NS", "INFY.NS"})
        self.assertIn("BAD.NS", errors)
        self.assertIn("sample provider error", errors["BAD.NS"])
        self.assertEqual(mock_download.call_count, 3)
        self.assertEqual(mock_save.call_count, 2)

    def test_rejects_empty_or_duplicate_tickers(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            download_watchlist([], start_date="2025-01-01")

        with self.assertRaisesRegex(ValueError, "duplicates"):
            download_watchlist(
                ["TCS.NS", "TCS.NS"],
                start_date="2025-01-01",
            )

    def test_rejects_a_single_string_instead_of_a_ticker_list(self):
        with self.assertRaisesRegex(TypeError, "iterable"):
            download_watchlist("TCS.NS", start_date="2025-01-01")


if __name__ == "__main__":
    unittest.main()