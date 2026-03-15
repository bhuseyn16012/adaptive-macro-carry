# data/pipeline.py
# Fetches, cleans, and saves all raw data
# Run this once to populate data/raw/ folder
# Every other module loads from there

import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from config import FRED_API_KEY, FRED_SERIES, YAHOO_TICKERS, START_DATE, END_DATE, DATA_DIR


def setup_storage():
    """Create the data/raw directory if it does not exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Storage directory ready: {DATA_DIR}")


def fetch_fred_data():
    """
    Fetch all FRED series defined in config.py
    FRED data is mostly monthly — we will handle
    alignment to daily later in the signal module
    """
    fred = Fred(api_key=FRED_API_KEY)
    
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id}) from FRED...")
        
        try:
            series = fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=END_DATE
            )
            
            # Convert to DataFrame with clean column name
            df = series.to_frame(name=name)
            df.index.name = "date"
            
            # Save to CSV
            path = os.path.join(DATA_DIR, f"{name}.csv")
            df.to_csv(path)
            print(f"  Saved to {path} — {len(df)} observations")
            
        except Exception as e:
            print(f"  ERROR fetching {name}: {e}")


def fetch_yahoo_data():
    """
    Fetch all Yahoo Finance series defined in config.py
    Yahoo data is daily
    """
    for name, ticker in YAHOO_TICKERS.items():
        print(f"Fetching {name} ({ticker}) from Yahoo Finance...")
        
        try:
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False
            )
            
            # We only need the closing price
            df = raw[["Close"]].copy()
            df.columns = [name]
            df.index.name = "date"
            
            # Save to CSV
            path = os.path.join(DATA_DIR, f"{name}.csv")
            df.to_csv(path)
            print(f"  Saved to {path} — {len(df)} observations")
            
        except Exception as e:
            print(f"  ERROR fetching {name}: {e}")


def fetch_boj_rate():
    """
    Bank of Japan policy rate
    This is not on FRED so we hardcode the historical
    rate change dates based on BOJ official announcements
    We will later interpolate this to monthly frequency
    """
    # Historical BOJ policy rate changes (date, new_rate)
    # Source: Bank of Japan official announcements
    boj_changes = [
        ("2000-01-01",  0.00),
        ("2000-08-11",  0.25),
        ("2001-03-19",  0.10),
        ("2001-09-19",  0.00),
        ("2006-07-14",  0.25),
        ("2007-02-21",  0.50),
        ("2008-10-31",  0.30),
        ("2008-12-19",  0.10),
        ("2010-10-05",  0.00),
        ("2016-01-29", -0.10),
        ("2024-03-19",  0.10),
        ("2024-07-31",  0.25),
        ("2025-01-24",  0.50),
    ]
    
    # Build a monthly date range
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    df = pd.DataFrame(index=date_range)
    df.index.name = "date"
    df["boj_rate"] = None
    
    # Forward fill the rate from each change date
    for date_str, rate in boj_changes:
        date = pd.Timestamp(date_str)
        df.loc[df.index >= date, "boj_rate"] = rate
    
    df["boj_rate"] = pd.to_numeric(df["boj_rate"])
    
    path = os.path.join(DATA_DIR, "boj_rate.csv")
    df.to_csv(path)
    print(f"BOJ rate saved to {path} — {len(df)} observations")


def run():
    """Run the full data pipeline"""
    print("=" * 50)
    print("ADAPTIVE MACRO CARRY — DATA PIPELINE")
    print("=" * 50)
    
    setup_storage()
    fetch_fred_data()
    fetch_yahoo_data()
    fetch_boj_rate()
    
    print("=" * 50)
    print("Data pipeline complete. All files saved to data/raw/")
    print("=" * 50)


if __name__ == "__main__":
    run()