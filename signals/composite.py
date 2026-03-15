import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_END


def load_factor(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.squeeze()


def standardize(series, window=36):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def build_composite_signal(w1=0.5, w2=0.25, w3=0.25, window=36):
    real_rate_diff = load_factor("factor_real_rate_diff.csv")
    current_acct = load_factor("factor_current_acct.csv")
    momentum = load_factor("factor_momentum.csv")

    z1 = standardize(real_rate_diff, window)
    z2 = standardize(current_acct, window)
    z3 = standardize(momentum, window)

    combined = pd.concat([z1, z2, z3], axis=1, join="inner")
    combined.columns = ["z_real_rate", "z_current_acct", "z_momentum"]
    combined = combined.dropna()

    combined["composite"] = (
        w1 * combined["z_real_rate"] +
        w2 * combined["z_current_acct"] +
        w3 * combined["z_momentum"]
    )


    print("Composite signal built: " + str(len(combined)) + " monthly observations")
    print("Range: " + str(combined.index.min().date()) + " to " + str(combined.index.max().date()))
    print("Current score: " + str(round(combined["composite"].iloc[-1], 3)))

    return combined


def plot_composite(combined, usdjpy):
    usdjpy_monthly = usdjpy.resample("MS").last()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Adaptive Macro Carry - Composite Signal vs USD/JPY",
                 fontsize=14, fontweight="bold")

    ax1.plot(usdjpy_monthly.index, usdjpy_monthly.values,
             color="black", linewidth=1.5)
    ax1.set_title("USD/JPY Exchange Rate")
    ax1.set_ylabel("USD/JPY")
    ax1.axvline(pd.Timestamp(TRAIN_END), color="red",
                linestyle="--", alpha=0.7, label="Train/Test Split")
    ax1.legend()

    composite = combined["composite"]
    colors = []
    for x in composite.values:
        if x > 1.0:
            colors.append("green")
        elif x < -1.0:
            colors.append("red")
        else:
            colors.append("gray")

    ax2.bar(composite.index, composite.values,
            color=colors, alpha=0.7, width=20)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.axhline(1.0, color="green", linestyle="--", alpha=0.6,
                label="Long signal (above 1.0)")
    ax2.axhline(-1.0, color="red", linestyle="--", alpha=0.6,
                label="Short signal (below -1.0)")
    ax2.axvline(pd.Timestamp(TRAIN_END), color="red",
                linestyle="--", alpha=0.7, label="Train/Test Split")
    ax2.set_title("Composite Signal (green = long USD/JPY, red = short, gray = neutral)")
    ax2.set_ylabel("Z-Score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("data/raw/composite_signal.png", dpi=150, bbox_inches="tight")
    plt.show()

def run():
    print("=" * 50)
    print("COMPOSITE SIGNAL CONSTRUCTION")
    print("=" * 50)

    combined = build_composite_signal(w1=0.5, w2=0.25, w3=0.25)

    usdjpy = pd.read_csv(
        os.path.join(DATA_DIR, "usdjpy.csv"),
        index_col="date", parse_dates=True
    ).squeeze()

    plot_composite(combined, usdjpy)

    combined.to_csv(os.path.join(DATA_DIR, "composite_signal.csv"))
    print("Composite signal saved to data/raw/composite_signal.csv")
    print("=" * 50)

    return combined


if __name__ == "__main__":
    run()