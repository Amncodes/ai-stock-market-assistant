"""Download historical data for a list of stock tickers."""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.stock_data import download_stock_data, save_raw_data


def download_watchlist(
    tickers: Iterable[str],
    start_date: str,
    end_date: str | None = None,
    output_dir: str | Path = "data/raw",
) -> tuple[dict[str, Path], dict[str, str]]:
    """Download each ticker, continuing if an individual ticker fails.

    Returns:
        saved_files: ticker-to-file-path mapping for successful downloads.
        errors: ticker-to-error-message mapping for failed downloads.
    """
    if isinstance(tickers, str):
        raise TypeError("tickers must be a list or other iterable of ticker strings.")

    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("Provide at least one ticker.")

    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in ticker_list):
        raise ValueError("Every ticker must be a non-empty string.")

    ticker_list = [ticker.strip() for ticker in ticker_list]
    if len(set(ticker_list)) != len(ticker_list):
        raise ValueError("Ticker list must not contain duplicates.")

    saved_files: dict[str, Path] = {}
    errors: dict[str, str] = {}

    for ticker in ticker_list:
        try:
            data = download_stock_data(ticker, start_date, end_date)
            saved_files[ticker] = save_raw_data(data, ticker, output_dir)
        except Exception as exc:
            # One failure should not prevent the remaining tickers from running.
            errors[ticker] = f"{type(exc).__name__}: {exc}"

    return saved_files, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical data for multiple stock tickers."
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols, for example RELIANCE.NS TCS.NS INFY.NS",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="First date to download, in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        help="Last date boundary, in YYYY-MM-DD format (optional)",
    )
    args = parser.parse_args()

    saved_files, errors = download_watchlist(
        args.tickers,
        args.start_date,
        args.end_date,
    )

    for ticker, path in saved_files.items():
        print(f"{ticker}: saved to {path}")

    for ticker, error in errors.items():
        print(f"{ticker}: download failed: {error}")

    print(f"\nDownloaded {len(saved_files)} of {len(args.tickers)} ticker(s).")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()