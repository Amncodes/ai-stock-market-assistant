"""Analyze multiple stock files and create a comparison summary."""

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.analysis import analyze_stock_file
from src.models.baseline import evaluate_baselines


def analyze_watchlist_files(
    input_files: Mapping[str, str | Path],
    summary_path: str | Path,
    output_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Analyze each ticker and save a summary for successful files.

    A failure for one ticker is recorded and does not stop the others.
    """
    if not input_files:
        raise ValueError("Provide at least one ticker and input file.")

    summary_file = Path(summary_path)
    if summary_file.exists():
        raise FileExistsError(f"Summary already exists: {summary_file}")

    rows = []
    errors = {}

    for ticker, input_path in input_files.items():
        try:
            _, risk = analyze_stock_file(input_path, output_dir)

            raw_data = pd.read_csv(input_path, parse_dates=["Date"])
            raw_data = raw_data.sort_values("Date").reset_index(drop=True)

            price_column = (
                "Adj Close" if "Adj Close" in raw_data.columns else "Close"
            )
            forecasts, forecast_metrics = evaluate_baselines(
                raw_data[price_column]
            )

            row = {
                "Ticker": ticker,
                "Annualized_Volatility": risk["annualized_volatility"],
                "Maximum_Drawdown": risk["maximum_drawdown"],
            }

            forecast_columns = {
                "previous_close": "Previous_Close_Forecast",
                "moving_average": "Moving_Average_Forecast",
            }

            for baseline_name, forecast_column in forecast_columns.items():
                actual = forecasts["Actual_Close"]
                predicted = forecasts[forecast_column]

                row[f"{baseline_name}_MAE_INR"] = forecast_metrics.loc[
                    baseline_name, "MAE"
                ]
                row[f"{baseline_name}_RMSE_INR"] = forecast_metrics.loc[
                    baseline_name, "RMSE"
                ]
                row[f"{baseline_name}_MAPE_percent"] = (
                    (actual - predicted).abs() / actual
                ).mean() * 100

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
        description="Analyze several stocks and compare their results."
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
        help="Path for the combined summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Folder for per-stock processed CSV files",
    )
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

    summary, errors = analyze_watchlist_files(
        input_files,
        args.summary,
        args.output_dir,
    )

    if not summary.empty:
        print(f"Summary saved to: {args.summary}")
        print("\nWatchlist results:")
        print(summary.to_string(index=False))

    for ticker, error in errors.items():
        print(f"{ticker}: analysis failed: {error}")

    print(f"\nAnalyzed {len(summary)} of {len(input_files)} ticker(s).")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()