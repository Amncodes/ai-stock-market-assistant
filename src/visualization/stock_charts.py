"""Create a price, RSI, and MACD chart from processed stock data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without opening a GUI window.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "Close",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
}


def create_stock_analysis_chart(
    data: pd.DataFrame,
    ticker: str,
    output_path: str | Path,
) -> Path:
    """Save a three-panel chart for closing price, RSI, and MACD."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Input data must not be empty.")

    chart_data = data.copy()
    chart_data["Date"] = pd.to_datetime(chart_data["Date"], errors="coerce")
    chart_data["Close"] = pd.to_numeric(chart_data["Close"], errors="coerce")

    if chart_data["Date"].isna().any():
        raise ValueError("Date column contains invalid or missing dates.")
    if chart_data["Close"].isna().any() or (chart_data["Close"] <= 0).any():
        raise ValueError("Close prices must be valid numbers greater than zero.")

    chart_data = chart_data.sort_values("Date")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise FileExistsError(f"Chart already exists: {destination}")

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    axes[0].plot(chart_data["Date"], chart_data["Close"], color="navy")
    axes[0].set_title(f"{ticker} — Historical Price and Technical Indicators")
    axes[0].set_ylabel("Close price (INR)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(chart_data["Date"], chart_data["RSI_14"], color="darkorange")
    axes[1].axhline(70, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=1)
    axes[1].set_ylabel("RSI (14)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        chart_data["Date"], chart_data["MACD"], label="MACD", color="blue"
    )
    axes[2].plot(
        chart_data["Date"],
        chart_data["MACD_Signal"],
        label="Signal",
        color="orange",
    )
    axes[2].bar(
        chart_data["Date"],
        chart_data["MACD_Hist"],
        label="Histogram",
        color="gray",
        alpha=0.5,
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("MACD")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    date_locator = mdates.AutoDateLocator()
    axes[2].xaxis.set_major_locator(date_locator)
    axes[2].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(date_locator)
    )
    axes[2].set_xlabel("Date")

    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return destination