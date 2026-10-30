"""Build a small LSTM model for one-step-ahead price experiments."""

import math


def build_lstm_model(
    sequence_length: int = 30,
    units: int = 32,
    learning_rate: float = 0.001,
):
    """Build and compile an LSTM that predicts one scaled price value."""
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer.")

    if not isinstance(units, int) or isinstance(units, bool) or units <= 0:
        raise ValueError("units must be a positive integer.")

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is optional. Install the packages from requirements-ml.txt "
            "in the LSTM environment."
        ) from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, 1)),
            tf.keras.layers.LSTM(units=units),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


if __name__ == "__main__":
    import tensorflow as tf

    model = build_lstm_model()
    model.summary()

    sample_input = tf.zeros((1, 30, 1))
    sample_prediction = model(sample_input)

    print("Sample input shape:", tuple(sample_input.shape))
    print("Prediction shape:", tuple(sample_prediction.shape))