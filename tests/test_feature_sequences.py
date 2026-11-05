import unittest

import numpy as np
import pandas as pd

from src.models.sequences import prepare_feature_sequences


class FeatureSequenceTests(unittest.TestCase):
    def make_data(self):
        dates = pd.date_range("2025-01-01", periods=40, freq="D")
        features = pd.DataFrame(
            {
                "Adj Close": np.arange(100.0, 140.0),
                "RSI_14": np.arange(30.0, 70.0),
                "MACD": np.arange(-5.0, 35.0),
                "Volume": np.arange(1000.0, 1040.0),
            },
            index=dates,
        )
        target = pd.Series(
            np.arange(100.0, 140.0),
            index=dates,
            name="Adj Close",
        )
        return features, target

    def test_creates_multivariable_chronological_sequences(self):
        features, target = self.make_data()

        result = prepare_feature_sequences(
            features,
            target,
            sequence_length=5,
            train_fraction=0.60,
            validation_fraction=0.20,
        )

        self.assertEqual(result.x_train.shape[1:], (5, 4))
        self.assertEqual(result.y_train.shape[1:], (1,))
        self.assertEqual(len(result.x_train), 19)
        self.assertEqual(len(result.x_validation), 8)
        self.assertEqual(len(result.x_test), 8)
        self.assertLess(result.train_dates[-1], result.validation_dates[0])
        self.assertLess(result.validation_dates[-1], result.test_dates[0])

    def test_scalers_use_training_rows_only(self):
        features, target = self.make_data()
        features.iloc[24:, :] = 10000
        target.iloc[24:] = 20000

        result = prepare_feature_sequences(
            features,
            target,
            sequence_length=5,
            train_fraction=0.60,
            validation_fraction=0.20,
        )

        self.assertEqual(result.feature_scaler.data_max_[0], 123.0)
        self.assertEqual(result.target_scaler.data_max_[0], 123.0)

    def test_rejects_misaligned_feature_and_target_indexes(self):
        features, target = self.make_data()
        target.index = target.index + pd.Timedelta(days=1)

        with self.assertRaisesRegex(ValueError, "matching indexes"):
            prepare_feature_sequences(features, target)

    def test_rejects_non_finite_features(self):
        features, target = self.make_data()
        features.iloc[5, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "numeric and finite"):
            prepare_feature_sequences(features, target)


if __name__ == "__main__":
    unittest.main()