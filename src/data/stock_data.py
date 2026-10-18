"""Download and save historical stock data for research purposes."""

from datetime import date
from pathlib import Path
import re

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily historical data and validate its required fields."""
    if not ticker.strip():
        raise ValueError("Ticker must not be empty.")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    # yfinance may return a two-level column index, even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if data.empty:
        raise ValueError(f"No historical data returned for {ticker}.")
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")

    data = data.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if data.empty:
        raise ValueError(f"No valid OHLCV rows remain for {ticker}.")

    # Reject impossible prices and negative trading volume.
    price_columns = ["Open", "High", "Low", "Close"]
    if (data[price_columns] <= 0).any().any() or (data["Volume"] < 0).any():
        raise ValueError(f"Downloaded data contains invalid prices or volume for {ticker}.")

    data.index.name = "Date"
    return data.sort_index()


def save_raw_data(
    data: pd.DataFrame,
    ticker: str,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Save data to a ticker-specific CSV, refusing to overwrite a file."""
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{safe_ticker}_{date.today().isoformat()}.csv"
    if file_path.exists():
        raise FileExistsError(
            f"{file_path} already exists; move or rename it before saving again."
        )

    data.to_csv(file_path)
    return file_path

if __name__ == "__main__":
    history = download_stock_data("RELIANCE.NS", start_date="2023-01-01")
    saved_path = save_raw_data(history, "RELIANCE.NS")
    print(f"Rows: {len(history)}")
    print(f"Date range: {history.index.min().date()} to {history.index.max().date()}")
    print(f"Saved to: {saved_path}")