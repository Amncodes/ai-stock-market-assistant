"""Streamlit dashboard for local Indian stock market analysis."""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.features.technical_indicators import add_technical_indicators
from src.models.baseline import evaluate_baselines
from src.risk.risk_metrics import (
    calculate_annualized_volatility,
    calculate_max_drawdown,
)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def find_latest_stock_files() -> dict[str, Path]:
    """Find the newest dated CSV file for each ticker."""
    latest_files = {}

    for path in sorted(RAW_DIR.glob("*.csv")):
        match = re.fullmatch(r"(.+)_\d{4}-\d{2}-\d{2}", path.stem)
        if not match:
            continue

        ticker = match.group(1).replace("_", ".")
        latest_files[ticker] = path

    return latest_files


@st.cache_data
def load_stock_data(csv_path: str) -> pd.DataFrame:
    """Load and sort one local raw stock-data CSV."""
    data = pd.read_csv(csv_path, parse_dates=["Date"])
    data = data.sort_values("Date").set_index("Date")
    return data


def main() -> None:
    st.set_page_config(
        page_title="Indian Stock Market Assistant",
        page_icon="📈",
        layout="wide",
    )

    st.title("AI-Driven Indian Stock Market Assistant")
    st.caption(
        "Historical research dashboard for Indian stocks. "
        "Forecasts are experimental and are not investment advice."
    )

    stock_files = find_latest_stock_files()
    if not stock_files:
        st.error("No dated stock CSV files found in data/raw/.")
        st.stop()

    ticker = st.sidebar.selectbox(
        "Choose a stock",
        options=sorted(stock_files),
    )
    data = load_stock_data(str(stock_files[ticker]))

    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"

    # Calculate indicators from the full history before applying the date filter.
    featured_data = add_technical_indicators(
        data,
        price_column=price_column,
    )

    earliest_date = featured_data.index.min().date()
    latest_date = featured_data.index.max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(earliest_date, latest_date),
        min_value=earliest_date,
        max_value=latest_date,
    )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Select both a start date and an end date.")
        st.stop()

    start_date, end_date = date_range
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    featured_data = featured_data.loc[
        (featured_data.index >= pd.Timestamp(start_date))
        & (featured_data.index < end_exclusive)
    ]

    if len(featured_data) < 8:
        st.warning("Select a range containing at least 8 trading days.")
        st.stop()

    prices = featured_data[price_column]
    _, baseline_metrics = evaluate_baselines(prices)

    latest = featured_data.iloc[-1]
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Latest adjusted close",
        f"₹{latest[price_column]:,.2f}",
    )
    metric_columns[1].metric(
        "RSI (14)",
        f"{latest['RSI_14']:.2f}",
    )
    metric_columns[2].metric(
        "Annualized volatility",
        f"{calculate_annualized_volatility(prices):.2%}",
    )
    metric_columns[3].metric(
        "Maximum drawdown",
        f"{calculate_max_drawdown(prices):.2%}",
    )

    st.caption(
        f"Data file: {stock_files[ticker].name} · "
        f"Showing {featured_data.index.min().date()} to "
        f"{featured_data.index.max().date()}"
    )

    st.subheader(f"{ticker} adjusted closing price")
    st.line_chart(featured_data[[price_column]])

    left, right = st.columns(2)
    with left:
        st.subheader("RSI (14)")
        st.line_chart(featured_data[["RSI_14"]])
    with right:
        st.subheader("MACD")
        st.line_chart(featured_data[["MACD", "MACD_Signal"]])

    st.subheader("Baseline forecast errors")
    display_metrics = baseline_metrics.rename_axis("Model").reset_index()
    display_metrics = display_metrics.rename(
        columns={"MAE": "MAE (INR)", "RMSE": "RMSE (INR)"}
    )
    st.dataframe(display_metrics, hide_index=True, width="stretch")

    best_model = baseline_metrics["MAE"].idxmin()
    st.info(
        f"On this selected historical period, **{best_model}** had the "
        f"lowest baseline MAE: "
        f"₹{baseline_metrics.loc[best_model, 'MAE']:.2f}."
    )
    st.subheader("Watchlist comparison")
    compare_tickers = st.multiselect(
        "Choose stocks to compare",
        options=sorted(stock_files),
        default=sorted(stock_files),
    )

    normalized_prices = {}
    comparison_rows = []

    for compare_ticker in compare_tickers:
        compare_data = load_stock_data(str(stock_files[compare_ticker]))
        compare_price_column = (
            "Adj Close" if "Adj Close" in compare_data.columns else "Close"
        )
        compare_prices = pd.to_numeric(
            compare_data[compare_price_column],
            errors="coerce",
        )

        compare_prices = compare_prices.loc[
            (compare_prices.index >= pd.Timestamp(start_date))
            & (compare_prices.index < end_exclusive)
        ].dropna()

        if len(compare_prices) < 2:
            continue

        normalized_prices[compare_ticker] = (
            compare_prices / compare_prices.iloc[0] * 100
        )

        comparison_rows.append(
            {
                "Ticker": compare_ticker,
                "Period return (%)": (
                    compare_prices.iloc[-1] / compare_prices.iloc[0] - 1
                )
                * 100,
                "Annualized volatility": calculate_annualized_volatility(
                    compare_prices
                ),
                "Maximum drawdown": calculate_max_drawdown(compare_prices),
            }
        )

    if normalized_prices:
        st.caption(
            "Each line starts at 100 on its first available date in the "
            "selected range."
        )
        st.line_chart(pd.DataFrame(normalized_prices))

        comparison_table = pd.DataFrame(comparison_rows).sort_values(
            "Period return (%)",
            ascending=False,
        )
        st.dataframe(
            comparison_table,
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Choose at least one stock with data in the selected date range.")

    ticker_file_prefix = ticker.replace(".", "_")
    saved_forecasts = sorted(
        PROCESSED_DIR.glob(f"{ticker_file_prefix}_*_forecasts.csv")
    )

    st.subheader("Saved LSTM forecast comparison")
    if not saved_forecasts:
        st.write(
            "No saved LSTM forecast files were found for this stock yet."
        )
    else:
        forecast_options = [""] + [str(path) for path in saved_forecasts]
        selected_forecast = st.selectbox(
            "Choose a saved forecast run",
            options=forecast_options,
            format_func=lambda value: (
                "Select a forecast file" if not value else Path(value).name
            ),
        )

        if selected_forecast:
            forecast_data = pd.read_csv(
                selected_forecast,
                parse_dates=["Date"],
            ).sort_values("Date")

            forecast_data = forecast_data.loc[
                (forecast_data["Date"] >= pd.Timestamp(start_date))
                & (forecast_data["Date"] < end_exclusive)
            ]

            forecast_columns = [
                column
                for column in (
                    "Actual",
                    "Previous_Close",
                    "Moving_Average",
                    "LSTM",
                )
                if column in forecast_data.columns
            ]

            if "Actual" not in forecast_columns:
                st.warning(
                    "This forecast file does not contain an Actual column."
                )
            elif forecast_data.empty:
                st.info(
                    "This forecast file has no results within the selected "
                    "date range."
                )
            else:
                st.line_chart(
                    forecast_data.set_index("Date")[forecast_columns]
                )

                actual = forecast_data["Actual"]
                forecast_rows = []
                for column in forecast_columns:
                    if column == "Actual":
                        continue

                    errors = actual - forecast_data[column]
                    forecast_rows.append(
                        {
                            "Model": column,
                            "MAE (INR)": errors.abs().mean(),
                            "RMSE (INR)": (errors.pow(2).mean()) ** 0.5,
                        }
                    )

                if forecast_rows:
                    st.dataframe(
                        pd.DataFrame(forecast_rows),
                        hide_index=True,
                        width="stretch",
                    )


main()