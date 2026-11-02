"""Plot actual stock prices against forecast model predictions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "Actual",
    "Previous_Close",
    "Moving_Average",
    "LSTM",
}


def create_forecast_chart(
    input_csv: str | Path,
    output_path: str | Path,
) -> Path:
    """Read a forecast CSV and save a chart comparing actual and predicted prices."""
    input_path = Path(input_csv)
    chart_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Forecast CSV not found: {input_path}")

    data = pd.read_csv(input_path, parse_dates=["Date"])

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Forecast CSV is missing columns: {sorted(missing_columns)}"
        )
    if data.empty:
        raise ValueError("Forecast CSV contains no rows.")

    value_columns = ["Actual", "Previous_Close", "Moving_Average", "LSTM"]
    data[value_columns] = data[value_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if data[["Date", *value_columns]].isna().any().any():
        raise ValueError("Forecast CSV contains missing or invalid values.")

    data = data.sort_values("Date")

    if chart_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing chart: {chart_path}")

    chart_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(data["Date"], data["Actual"], label="Actual", linewidth=2.2)
    ax.plot(data["Date"], data["Previous_Close"], label="Previous close", alpha=0.85)
    ax.plot(data["Date"], data["Moving_Average"], label="5-day moving average", alpha=0.85)
    ax.plot(data["Date"], data["LSTM"], label="LSTM", alpha=0.9)

    ax.set_title("TCS: Actual Prices vs Forecasts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close price (INR)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot actual prices against forecast model predictions."
    )
    parser.add_argument("input_csv", help="Path to a forecast comparison CSV")
    parser.add_argument("--output", required=True, help="Path for the PNG chart")
    args = parser.parse_args()

    chart_path = create_forecast_chart(args.input_csv, args.output)
    print(f"Forecast chart saved to: {chart_path}")


if __name__ == "__main__":
    main()