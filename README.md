# AI-Driven Indian Stock Market Assistant

A research dashboard for exploring Indian stock-market history, technical indicators, risk measures, and experimental price forecasts. It supports multiple tickers and includes baseline and LSTM forecasting experiments.

Forecasts are experimental. They are not guaranteed returns or personalized investment advice.

## Dashboard

The Streamlit dashboard reads saved CSV files and displays:

- Adjusted closing-price history
- RSI and MACD indicators
- Annualized volatility and maximum drawdown
- Previous-close and moving-average forecast errors
- Saved LSTM forecast comparisons, when available

The dashboard does not automatically download fresh prices or train an LSTM when opened. Download stock data first.

## Setup

From the project folder, create and activate the main environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Download stock data

For example, to download three stocks starting from January 2023:

```bash
python -m src.data.watchlist RELIANCE.NS TCS.NS INFY.NS --start-date 2023-01-01
```

The downloaded CSV files are saved in `data/raw/`.

## Run the dashboard

With `.venv` activated, run:

```bash
streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

## Project folders

- `src/` — data download, analysis, indicators, forecasting, and dashboard code
- `tests/` — automated checks
- `data/raw/` — downloaded stock data
- `data/processed/` — analysis results and forecast files

## Disclaimer

This project is for educational research and decision support. Forecasts are uncertain and should not be treated as guaranteed returns or personalized investment advice.
