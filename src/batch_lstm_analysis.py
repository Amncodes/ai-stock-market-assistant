"""Train and compare LSTM forecasts for multiple stock files."""

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.models.train_lstm import train_and_evaluate_lstm


def analyze_watchlist_lstm(
    input_files: Mapping[str, str | Path],
    summary_path: str | Path,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Evaluate baselines and an LSTM for each stock; save successful results."""
    if not input_files:
        raise ValueError("Provide at least one ticker and input file.")

    summary_file = Path(summary_path)
    if summary_file.exists():
        raise FileExistsError(f"Summary already exists: {summary_file}")

    rows = []
    errors = {}

    for ticker, input_path in input_files.items():
        try:
            data = pd.read_csv(input_path, parse_dates=["Date"])
            data = data.sort_values("Date").set_index("Date")

            price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
            if price_column not in data.columns:
                raise ValueError("Input CSV must contain Close or Adj Close.")

            results, training_info = train_and_evaluate_lstm(
                data[price_column],
                max_epochs=max_epochs,
                patience=patience,
            )

            row = {
                "Ticker": ticker,
                "Price_Column": price_column,
                "Training_Epochs": training_info["epochs_trained"],
                "Best_Validation_Loss": training_info["best_validation_loss"],
            }

            for model_name, metrics in results.iterrows():
                model_prefix = str(model_name).lower()
                for metric_name in ("MAE_INR", "RMSE_INR", "MAPE_percent"):
                    row[f"{model_prefix}_{metric_name}"] = float(
                        metrics[metric_name]
                    )

            rows.append(row)

        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_file, index=False)

    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LSTM and baseline forecasts across several stocks."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="TICKER=PATH",
        help="Ticker and raw CSV path; repeat this option for each stock",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path for the combined forecast metrics CSV",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    input_files = {}
    for item in args.input:
        if "=" not in item:
            parser.error(f"Input must be written as TICKER=PATH: {item}")

        ticker, path = item.split("=", maxsplit=1)
        if not ticker.strip() or not path.strip():
            parser.error(f"Input must include both ticker and path: {item}")
        if ticker in input_files:
            parser.error(f"Ticker was provided more than once: {ticker}")

        input_files[ticker] = path

    summary, errors = analyze_watchlist_lstm(
        input_files,
        args.summary,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    if not summary.empty:
        print(f"Summary saved to: {args.summary}")
        print("\nLSTM watchlist results:")
        print(summary.to_string(index=False))

    for ticker, error in errors.items():
        print(f"{ticker}: analysis failed: {error}")

    print(f"\nAnalyzed {len(summary)} of {len(input_files)} ticker(s).")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()