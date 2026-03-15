# signals/factors.py
# Builds the three individual factors that drive the composite signal
# Now uses Bloomberg data for Japan CPI, current account, and 2yr yields

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_END


def load_raw(filename):
    """Load a CSV from data/raw/ and return a clean Series"""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df.squeeze()


def load_bloomberg_monthly():
    """
    Load monthly Bloomberg data.
    Contains: japan_cpi, japan_current_acct
    Dates are end-of-month so we resample to month-start for alignment.
    """
    path = os.path.join(DATA_DIR, "monthly_bloomberg.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    # Resample to month-start frequency for alignment with FRED data
    df = df.resample("MS").first()
    
    print(f"Bloomberg monthly loaded: {len(df)} rows, "
          f"{df.index.min().date()} to {df.index.max().date()}")
    return df


def load_bloomberg_daily():
    """
    Load daily Bloomberg data.
    Contains: japan_2yr_yield, us_2yr_yield
    Fixes basis points issue and resamples to monthly.
    """
    path = os.path.join(DATA_DIR, "daily_bloomberg.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    # Fix basis points issue — if value > 10 it is in basis points not percent
    for col in ["japan_2yr_yield", "us_2yr_yield"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x/100 if x > 10 else x)
    
    # Resample to monthly by taking last value of each month
    df_monthly = df.resample("MS").last()
    
    print(f"Bloomberg daily loaded and resampled: {len(df_monthly)} monthly rows")
    return df_monthly


def build_real_rate_differential():
    """
    Factor 1: Real Rate Differential
    
    Formula: (US 2yr Yield - US CPI YoY) - (Japan 2yr Yield - Japan CPI YoY)
    
    Using 2-year yields instead of policy rates because they are
    forward-looking — they reflect where markets expect rates to go,
    not just where they are today. This makes them a stronger signal.
    
    Using real rates (nominal minus inflation) because they capture
    true purchasing power return of holding a currency.
    """
    # Load Bloomberg data
    bloomberg_monthly = load_bloomberg_monthly()
    bloomberg_daily = load_bloomberg_daily()
    
    # Load FRED data
    us_cpi = load_raw("us_cpi.csv")
    fed_funds = load_raw("fed_funds_rate.csv")
    
    # Get Japan CPI and yields from Bloomberg
    japan_cpi = bloomberg_monthly["japan_cpi"]
    japan_2yr = bloomberg_daily["japan_2yr_yield"]
    us_2yr = bloomberg_daily["us_2yr_yield"]
    
    # Compute US CPI YoY from FRED (already in levels so compute pct change)
    us_inflation = us_cpi.pct_change(12) * 100
    
    # Japan CPI from Bloomberg is already YoY percentage so use directly
    jp_inflation = japan_cpi
    
    # Compute real rates
    us_real = us_2yr - us_inflation
    jp_real = japan_2yr - jp_inflation
    
    # Align all series
    combined = pd.concat([us_real, jp_real], axis=1)
    combined.columns = ["us_real", "jp_real"]
    combined = combined.ffill().dropna()
    
    combined["real_rate_diff"] = combined["us_real"] - combined["jp_real"]
    
    print(f"Real rate differential built: {len(combined)} monthly observations")
    print(f"Range: {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"Current value: {combined['real_rate_diff'].iloc[-1]:.2f}%")
    
    return combined["real_rate_diff"]


def build_current_account_differential():
    """
    Factor 2: Current Account Differential
    
    US current account from FRED minus Japan current account from Bloomberg.
    Both normalized by computing YoY percentage change to make them comparable
    regardless of currency or scale differences.
    """
    bloomberg_monthly = load_bloomberg_monthly()
    us_ca = load_raw("us_current_acct.csv")
    japan_ca = bloomberg_monthly["japan_current_acct"]
    
    # Compute YoY change for both
    us_ca_yoy = us_ca.resample("MS").ffill().pct_change(12) * 100
    jp_ca_yoy = japan_ca.pct_change(12) * 100
    
    # Differential — positive means US current account improving relative to Japan
    combined = pd.concat([us_ca_yoy, jp_ca_yoy], axis=1)
    combined.columns = ["us_ca_yoy", "jp_ca_yoy"]
    combined = combined.ffill().dropna()
    
    combined["ca_diff"] = combined["us_ca_yoy"] - combined["jp_ca_yoy"]
    
    print(f"Current account differential built: {len(combined)} monthly observations")
    print(f"Range: {combined.index.min().date()} to {combined.index.max().date()}")
    
    return combined["ca_diff"]


def build_fundamental_momentum(real_rate_diff, lookback=3):
    """
    Factor 3: Fundamental Momentum
    
    Rate of change of the real rate differential over a lookback window.
    Captures whether the fundamental driver is accelerating or decelerating.
    """
    momentum = real_rate_diff.diff(lookback)
    momentum.name = "fundamental_momentum"
    momentum = momentum.dropna()
    
    print(f"Fundamental momentum built: {len(momentum)} monthly observations")
    print(f"Lookback: {lookback} months")
    
    return momentum


def plot_factors(real_rate_diff, current_acct, momentum, usdjpy):
    """Visualize all three factors alongside USD/JPY price"""
    usdjpy_monthly = usdjpy.resample("MS").last()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Adaptive Macro Carry — Factor Analysis (Bloomberg Data)",
                 fontsize=14, fontweight="bold")
    
    axes[0].plot(usdjpy_monthly.index, usdjpy_monthly.values,
                 color="black", linewidth=1.5)
    axes[0].set_title("USD/JPY Exchange Rate")
    axes[0].set_ylabel("USD/JPY")
    axes[0].axvline(pd.Timestamp(TRAIN_END), color="red",
                    linestyle="--", alpha=0.7, label="Train/Test Split")
    axes[0].legend()
    
    axes[1].plot(real_rate_diff.index, real_rate_diff.values,
                 color="blue", linewidth=1.2)
    axes[1].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[1].axvline(pd.Timestamp(TRAIN_END), color="red",
                    linestyle="--", alpha=0.7)
    axes[1].set_title("Factor 1: Real Rate Differential (US minus Japan, 2yr yields)")
    axes[1].set_ylabel("Percentage Points")
    
    axes[2].plot(current_acct.index, current_acct.values,
                 color="green", linewidth=1.2)
    axes[2].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[2].axvline(pd.Timestamp(TRAIN_END), color="red",
                    linestyle="--", alpha=0.7)
    axes[2].set_title("Factor 2: Current Account Differential (US minus Japan, YoY)")
    axes[2].set_ylabel("Percent Change")
    
    axes[3].bar(momentum.index, momentum.values,
                color=["blue" if x > 0 else "red" for x in momentum.values],
                alpha=0.6, width=20)
    axes[3].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[3].axvline(pd.Timestamp(TRAIN_END), color="red",
                    linestyle="--", alpha=0.7)
    axes[3].set_title("Factor 3: Fundamental Momentum (3-month change in real rate diff)")
    axes[3].set_ylabel("Percentage Points")
    
    plt.tight_layout()
    plt.savefig("data/raw/factor_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Factor plot saved to data/raw/factor_analysis.png")


def run():
    """Build all three factors and visualize them"""
    print("=" * 50)
    print("FACTOR CONSTRUCTION")
    print("=" * 50)
    
    real_rate_diff = build_real_rate_differential()
    current_acct   = build_current_account_differential()
    momentum       = build_fundamental_momentum(real_rate_diff, lookback=3)
    usdjpy         = load_raw("usdjpy.csv")
    
    plot_factors(real_rate_diff, current_acct, momentum, usdjpy)
    
    real_rate_diff.to_csv("data/raw/factor_real_rate_diff.csv")
    current_acct.to_csv("data/raw/factor_current_acct.csv")
    momentum.to_csv("data/raw/factor_momentum.csv")
    
    print("=" * 50)
    print("All factors saved to data/raw/")
    print("=" * 50)
    
    return real_rate_diff, current_acct, momentum


if __name__ == "__main__":
    run()