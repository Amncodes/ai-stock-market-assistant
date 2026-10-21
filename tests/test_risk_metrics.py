import unittest

import numpy as np
import pandas as pd

from src.risk.risk_metrics import (
    calculate_annualized_volatility,
    calculate_daily_returns,
    calculate_max_drawdown,
)


class RiskMetricTests(unittest.TestCase):
    def test_calculates_daily_returns(self):
        prices = pd.Series([100.0, 110.0, 99.0])

        result = calculate_daily_returns(prices)

        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertAlmostEqual(result.iloc[1], 0.10)
        self.assertAlmostEqual(result.iloc[2], -0.10)
        self.assertEqual(result.name, "Daily_Return")

    def test_calculates_annualized_volatility(self):
        prices = pd.Series([100.0, 110.0, 99.0])

        result = calculate_annualized_volatility(prices)
        expected = pd.Series([0.10, -0.10]).std(ddof=1) * np.sqrt(252)

        self.assertAlmostEqual(result, expected)

    def test_calculates_maximum_drawdown(self):
        prices = pd.Series([100.0, 120.0, 90.0, 110.0])

        result = calculate_max_drawdown(prices)

        self.assertAlmostEqual(result, -0.25)

    def test_rejects_invalid_prices(self):
        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            calculate_daily_returns(pd.Series([100.0, "bad", 110.0]))

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_max_drawdown(pd.Series([100.0, 0.0]))

        with self.assertRaisesRegex(ValueError, "At least two"):
            calculate_daily_returns(pd.Series([100.0]))

    def test_rejects_invalid_volatility_settings(self):
        prices = pd.Series([100.0, 110.0, 105.0])

        with self.assertRaisesRegex(ValueError, "positive integer"):
            calculate_annualized_volatility(prices, trading_days_per_year=0)

        with self.assertRaisesRegex(ValueError, "three closing prices"):
            calculate_annualized_volatility(pd.Series([100.0, 110.0]))


if __name__ == "__main__":
    unittest.main()