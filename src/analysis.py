"""Run indicators and historical risk calculations on a saved stock CSV."""

import argparse
from pathlib import Path

import pandas as pd

from src.features.technical_indicators import add_technical_indicators
from src.risk.risk_metrics import (
    calculate_annualized_volatility,
    calculate_max_drawdown,
)


REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}


def analyze_stock_file(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
) -> tuple[Path, dict[str, float]]:
    """Analyze a raw OHLCV CSV, save indicators, and return risk metrics."""
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    data = pd.read_csv(input_file)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Input CSV contains no rows.")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isna().any():
        raise ValueError("Date column contains invalid or missing dates.")
    if data["Date"].duplicated().any():
        raise ValueError("Date column contains duplicate dates.")

    data = data.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(data["Close"], errors="coerce")
    if close.isna().any():
        raise ValueError("Close prices must be numeric and contain no missing values.")

    analyzed = add_technical_indicators(data)
    metrics = {
        "annualized_volatility": calculate_annualized_volatility(close),
        "maximum_drawdown": calculate_max_drawdown(close),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_file = destination / f"{input_file.stem}_analysis.csv"
    if output_file.exists():
        raise FileExistsError(
            f"Output already exists: {output_file}. "
            "Move or rename it before creating another analysis."
        )

    analyzed.to_csv(output_file, index=False)
    return output_file, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add technical indicators and calculate historical risk metrics."
    )
    parser.add_argument("input_csv", help="Path to a raw OHLCV CSV file")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Folder for the analyzed CSV (default: data/processed)",
    )
    args = parser.parse_args()

    output_file, metrics = analyze_stock_file(args.input_csv, args.output_dir)
    print(f"Analyzed data saved to: {output_file}")
    print(f"Annualized volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Maximum drawdown: {metrics['maximum_drawdown']:.2%}")


if __name__ == "__main__":
    main()