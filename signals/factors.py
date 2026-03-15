# signals/factors.py
# Builds the three individual factors that drive the composite signal
# Each factor is built independently, verified, then exported
# All factors are monthly frequency — we align to daily later

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# This makes sure Python can find config.py from any location
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_END


def load_raw(filename):
    """
    Load a CSV from data/raw/ and return a clean Series
    Parses dates automatically and sets date as index
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df.squeeze()  # converts single-column DataFrame to Series


def build_real_rate_differential():
    """
    Factor 1: Real Rate Differential
    
    Formula: (Fed Funds Rate - US CPI YoY) - (BOJ Rate - Japan CPI YoY)
    
    Why YoY CPI: We want the inflation rate (how fast prices are rising)
    not the price level itself. Year-over-year percentage change gives us that.
    
    Why real rates: Nominal rates alone are misleading. If the US offers 5%
    but inflation is 6%, the real return is negative. Real rates capture
    the true purchasing power return of holding a currency.
    """
    # Load raw series
    fed_rate = load_raw("fed_funds_rate.csv")
    boj_rate = load_raw("boj_rate.csv")
    us_cpi   = load_raw("us_cpi.csv")
    jp_cpi   = load_raw("japan_cpi.csv")

    # Compute year-over-year inflation rates (percentage change vs 12 months ago)
    us_inflation = us_cpi.pct_change(12) * 100
    jp_inflation = jp_cpi.pct_change(12) * 100

    # Align all series to the same monthly index
    # We use inner join so we only keep dates where ALL series have data
    combined = pd.concat([fed_rate, boj_rate, us_inflation, jp_inflation], axis=1)
    combined.columns = ["fed_rate", "boj_rate", "us_inflation", "jp_inflation"]
    combined = combined.dropna()

    # Compute real rates for each country
    combined["us_real_rate"] = combined["fed_rate"] - combined["us_inflation"]
    combined["jp_real_rate"] = combined["boj_rate"] - combined["jp_inflation"]

    # Compute the differential
    combined["real_rate_diff"] = combined["us_real_rate"] - combined["jp_real_rate"]

    print(f"Real rate differential built: {len(combined)} monthly observations")
    print(f"Range: {combined.index.min().date()} to {combined.index.max().date()}")
    print(f"Current value: {combined['real_rate_diff'].iloc[-1]:.2f}%")

    return combined["real_rate_diff"]


def build_current_account_differential():
    """
    Factor 2: Current Account Differential
    
    The US current account balance from FRED is already the net position.
    Japan current account data is quarterly from FRED.
    
    A negative and deteriorating US current account means more dollars
    flowing out of the US, creating structural selling pressure on USD.
    We take US minus Japan so a positive value favors USD strength.
    
    Note: Current account data is quarterly so we forward fill to monthly.
    This is legitimate because current account positions change slowly.
    """
    us_ca = load_raw("us_current_acct.csv")

    # Resample to monthly frequency using forward fill
    # This carries the last quarterly value forward until the next reading
    us_ca_monthly = us_ca.resample("MS").ffill()

    # Normalize by computing the 12-month change to capture the trend
    # rather than the absolute level (which varies by economic size)
    us_ca_change = us_ca_monthly.pct_change(12) * 100

    us_ca_change.name = "current_acct_diff"
    us_ca_change = us_ca_change.dropna()

    print(f"Current account differential built: {len(us_ca_change)} monthly observations")
    print(f"Range: {us_ca_change.index.min().date()} to {us_ca_change.index.max().date()}")

    return us_ca_change


def build_fundamental_momentum(real_rate_diff, lookback=3):
    """
    Factor 3: Fundamental Momentum
    
    Measures how fast the real rate differential is changing.
    
    Formula: RealDiff(t) - RealDiff(t - lookback)
    
    A large positive value means the differential is widening rapidly,
    which signals accelerating capital flow pressure toward USD.
    A negative value means the differential is narrowing, a warning sign.
    
    The lookback period is a hyperparameter we will optimize later.
    Default of 3 months captures medium-term momentum without too much noise.
    """
    momentum = real_rate_diff.diff(lookback)
    momentum.name = "fundamental_momentum"
    momentum = momentum.dropna()

    print(f"Fundamental momentum built: {len(momentum)} monthly observations")
    print(f"Lookback: {lookback} months")

    return momentum


def plot_factors(real_rate_diff, current_acct, momentum, usdjpy):
    """
    Visualize all three factors alongside USD/JPY price
    This is your sanity check — factors should visually relate to price moves
    """
    # Resample USD/JPY to monthly for comparison
    usdjpy_monthly = usdjpy.resample("MS").last()

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Adaptive Macro Carry — Factor Analysis", fontsize=14, fontweight="bold")

    # Plot USD/JPY price
    axes[0].plot(usdjpy_monthly.index, usdjpy_monthly.values, color="black", linewidth=1.5)
    axes[0].set_title("USD/JPY Exchange Rate")
    axes[0].set_ylabel("USD/JPY")
    axes[0].axvline(pd.Timestamp(TRAIN_END), color="red", linestyle="--", alpha=0.7, label="Train/Test Split")
    axes[0].legend()

    # Plot real rate differential
    axes[1].plot(real_rate_diff.index, real_rate_diff.values, color="blue", linewidth=1.2)
    axes[1].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[1].axvline(pd.Timestamp(TRAIN_END), color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Factor 1: Real Rate Differential (US minus Japan)")
    axes[1].set_ylabel("Percentage Points")

    # Plot current account
    axes[2].plot(current_acct.index, current_acct.values, color="green", linewidth=1.2)
    axes[2].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[2].axvline(pd.Timestamp(TRAIN_END), color="red", linestyle="--", alpha=0.7)
    axes[2].set_title("Factor 2: US Current Account YoY Change")
    axes[2].set_ylabel("Percent Change")

    # Plot momentum
    axes[3].bar(momentum.index, momentum.values, color=["blue" if x > 0 else "red" for x in momentum.values], alpha=0.6, width=20)
    axes[3].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[3].axvline(pd.Timestamp(TRAIN_END), color="red", linestyle="--", alpha=0.7)
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

    # Build each factor
    real_rate_diff = build_real_rate_differential()
    current_acct   = build_current_account_differential()
    momentum       = build_fundamental_momentum(real_rate_diff, lookback=3)

    # Load USD/JPY for visualization
    usdjpy = load_raw("usdjpy.csv")

    # Plot all factors against price
    plot_factors(real_rate_diff, current_acct, momentum, usdjpy)

    # Save factors to CSV for use in composite signal module
    real_rate_diff.to_csv("data/raw/factor_real_rate_diff.csv")
    current_acct.to_csv("data/raw/factor_current_acct.csv")
    momentum.to_csv("data/raw/factor_momentum.csv")

    print("=" * 50)
    print("All factors saved to data/raw/")
    print("=" * 50)

    return real_rate_diff, current_acct, momentum


if __name__ == "__main__":
    run()