# download_data.py
# Fetch historical stock data from Yahoo Finance and save to a local CSV file

import argparse
import os
import yfinance as yf
import pandas as pd

def fetch_and_save(ticker: str, interval: str, period: str, output_file: str):
    """
    Fetches historical data for the given ticker and saves it to CSV.

    Args:
        ticker: Stock symbol, e.g. 'AAPL'.
        interval: Data interval, e.g. '1h', '1d'.
        period: Period to fetch, e.g. '3mo', '1y'.
        output_file: Path to save CSV.
    """
    print(f"Fetching data for {ticker}, interval={interval}, period={period}...")
    df = yf.download(ticker, interval=interval, period=period)
    if df.empty:
        print("No data fetched. Please check ticker/interval/period.")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file)
    print(f"Fetched {len(df)} rows. Data saved to {output_file}.")


def main():
    parser = argparse.ArgumentParser(description="Download stock data and save locally")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--interval", default="1h", help="Data interval, e.g. '1h', '1d'")
    parser.add_argument("--period", default="3mo", help="Data period, e.g. '3mo', '1y'")
    parser.add_argument("--output", default="data/AAPL_1h_3mo.csv", help="Output CSV file path")
    args = parser.parse_args()

    fetch_and_save(args.ticker, args.interval, args.period, args.output)

if __name__ == "__main__":
    main()
