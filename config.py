# config.py
# Central configuration file for the entire project
# All parameters live here so nothing is hardcoded elsewhere

import os
from dotenv import load_dotenv

# Explicitly load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Date range
START_DATE = "2000-01-01"
END_DATE = "2026-02-28"

# Train/test split — minimum 7 years in test set
TRAIN_END = "2016-12-31"
TEST_START = "2017-01-01"

# FRED Series IDs
FRED_SERIES = {
    "fed_funds_rate":    "FEDFUNDS",        # US Federal Funds Rate
    "us_cpi":            "CPIAUCSL",        # US CPI
    "japan_cpi":         "JPNCPIALLAINMEI", # Japan CPI All Items
    "us_current_acct":   "NETFI",           # US Current Account Balance
    "yield_curve":       "T10Y2Y",          # US 10Y minus 2Y spread
    "us_1yr_treasury":   "DGS1",            # Risk free rate for metrics
}

# Yahoo Finance tickers
YAHOO_TICKERS = {
    "usdjpy":  "USDJPY=X",
    "eurusd":  "EURUSD=X",
    "vix":     "^VIX",
    "sp500":   "^GSPC",
}

# Data storage path
DATA_DIR = "data/raw/"