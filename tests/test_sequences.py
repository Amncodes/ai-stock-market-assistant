import unittest

import numpy as np
import pandas as pd

from src.models.sequences import prepare_price_sequences


class PriceSequenceTests(unittest.TestCase):
    def sample_prices(self):
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        return pd.Series(
            np.arange(1, 101, dtype=float),
            index=dates,
            name="Adj Close",
        )

    def test_creates_chronological_sequences_and_splits(self):
        result = prepare_price_sequences(
            self.sample_prices(),
            sequence_length=10,
            train_fraction=0.70,
            validation_fraction=0.15,
        )

        self.assertEqual(result.x_train.shape, (60, 10, 1))
        self.assertEqual(result.y_train.shape, (60, 1))
        self.assertEqual(result.x_validation.shape, (15, 10, 1))
        self.assertEqual(result.y_validation.shape, (15, 1))
        self.assertEqual(result.x_test.shape, (15, 10, 1))
        self.assertEqual(result.y_test.shape, (15, 1))
        self.assertLess(result.train_dates[-1], result.validation_dates[0])
        self.assertLess(result.validation_dates[-1], result.test_dates[0])

    def test_scaler_uses_training_prices_only(self):
        result = prepare_price_sequences(
            self.sample_prices(),
            sequence_length=10,
            train_fraction=0.70,
            validation_fraction=0.15,
        )

        # 70% of 100 rows is 70 rows: prices 1 through 70.
        self.assertEqual(result.scaler.data_min_[0], 1.0)
        self.assertEqual(result.scaler.data_max_[0], 70.0)

        # Later prices can scale above 1 because they were not used to fit it.
        self.assertGreater(result.y_test.max(), 1.0)

    def test_rejects_invalid_price_values(self):
        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            prepare_price_sequences(pd.Series([100.0, "bad", 102.0]))

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            prepare_price_sequences(pd.Series([100.0, 0.0, 102.0]))

    def test_rejects_bad_splits_and_unordered_dates(self):
        prices = self.sample_prices()

        with self.assertRaisesRegex(ValueError, "leave test data"):
            prepare_price_sequences(
                prices,
                train_fraction=0.8,
                validation_fraction=0.2,
            )

        with self.assertRaisesRegex(ValueError, "chronological order"):
            prepare_price_sequences(prices.iloc[::-1])


if __name__ == "__main__":
    unittest.main()