import unittest

import pandas as pd

from src.features.technical_indicators import add_technical_indicators


class TechnicalIndicatorTests(unittest.TestCase):
    def test_adds_indicators_and_expected_warmup_rows(self):
        data = pd.DataFrame({"Close": range(100, 140)})
        original = data.copy()

        result = add_technical_indicators(data)

        self.assertEqual(len(result), 40)
        self.assertTrue(data.equals(original))
        self.assertEqual(result["RSI_14"].isna().sum(), 14)
        self.assertEqual(result["MACD"].isna().sum(), 25)
        self.assertEqual(result["MACD_Signal"].isna().sum(), 33)
        self.assertEqual(result["MACD_Hist"].isna().sum(), 33)
        self.assertGreater(result["RSI_14"].iloc[-1], 99)
        self.assertGreater(result["MACD"].iloc[-1], 0)

    def test_flat_prices_have_neutral_rsi_and_zero_macd(self):
        data = pd.DataFrame({"Close": [100.0] * 40})

        result = add_technical_indicators(data)

        self.assertTrue((result["RSI_14"].dropna() == 50).all())
        self.assertTrue((result["MACD"].dropna() == 0).all())
        self.assertTrue((result["MACD_Hist"].dropna() == 0).all())

    def test_rejects_missing_or_invalid_close_prices(self):
        with self.assertRaisesRegex(ValueError, "Close"):
            add_technical_indicators(pd.DataFrame({"Open": [10, 11]}))

        with self.assertRaisesRegex(ValueError, "numeric"):
            add_technical_indicators(
                pd.DataFrame({"Close": [10, "bad", 12]})
            )

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            add_technical_indicators(pd.DataFrame({"Close": [10, 0, 12]}))

    def test_rejects_invalid_periods(self):
        data = pd.DataFrame({"Close": range(100, 140)})

        with self.assertRaisesRegex(ValueError, "positive"):
            add_technical_indicators(data, rsi_period=0)

        with self.assertRaisesRegex(ValueError, "smaller"):
            add_technical_indicators(data, macd_fast=26, macd_slow=12)

    def test_prefers_adjusted_close_when_available(self):
        data = pd.DataFrame(
            {
                "Close": [100.0] * 40,
                "Adj Close": range(100, 140),
            }
        )

        result = add_technical_indicators(data)

        # Rising adjusted prices produce a high RSI.
        self.assertGreater(result["RSI_14"].iloc[-1], 99)


if __name__ == "__main__":
    unittest.main()