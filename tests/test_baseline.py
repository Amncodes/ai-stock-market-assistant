import unittest

import pandas as pd

from src.models.baseline import evaluate_baselines


class BaselineForecastTests(unittest.TestCase):
    def test_forecasts_use_prior_prices_and_keep_test_dates(self):
        prices = pd.Series(
            [100, 110, 105, 120, 130, 125, 140, 150, 145, 160],
            index=pd.date_range("2025-01-01", periods=10, freq="D"),
            name="Close",
        )

        forecasts, metrics = evaluate_baselines(
            prices,
            test_fraction=0.3,
            moving_average_window=3,
        )

        self.assertEqual(len(forecasts), 3)
        self.assertTrue(forecasts.index.equals(prices.index[7:]))
        self.assertEqual(
            forecasts["Previous_Close_Forecast"].tolist(),
            [140, 150, 145],
        )
        self.assertAlmostEqual(
            forecasts["Moving_Average_Forecast"].iloc[0],
            (130 + 125 + 140) / 3,
        )
        self.assertAlmostEqual(metrics.loc["previous_close", "MAE"], 10.0)

    def test_rejects_invalid_prices(self):
        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            evaluate_baselines(pd.Series([100, "bad", 110]))

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            evaluate_baselines(pd.Series([100, 0, 110]))

    def test_rejects_invalid_settings(self):
        prices = pd.Series(range(100, 120))

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate_baselines(prices, test_fraction=1)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate_baselines(prices, moving_average_window=0)

    def test_requires_enough_historical_rows(self):
        prices = pd.Series([100, 101, 102, 103, 104, 105])

        with self.assertRaisesRegex(ValueError, "historical portion"):
            evaluate_baselines(
                prices,
                test_fraction=0.5,
                moving_average_window=5,
            )


if __name__ == "__main__":
    unittest.main()