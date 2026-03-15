import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
fed_rate = pd.read_csv("data/raw/fed_funds_rate.csv", index_col="date", parse_dates=True).squeeze()
boj_rate = pd.read_csv("data/raw/boj_rate.csv", index_col="date", parse_dates=True).squeeze()
usdjpy = pd.read_csv("data/raw/usdjpy.csv", index_col="date", parse_dates=True).squeeze()

# Align to monthly and training period only
rate_diff = (fed_rate - boj_rate).ffill()
rate_diff = rate_diff["2000-01-01":"2016-12-31"]

# Resample USD/JPY to monthly returns
usdjpy_monthly = usdjpy.resample("MS").last()
usdjpy_returns = usdjpy_monthly.pct_change()
usdjpy_returns = usdjpy_returns["2000-01-01":"2016-12-31"]

# Hyperparameter grid
lookbacks = [1, 2, 3, 4, 6, 9, 12]
thresholds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

# Build heatmap matrix
results = pd.DataFrame(index=lookbacks, columns=thresholds, dtype=float)

for lb in lookbacks:
    for th in thresholds:
        # Compute momentum of rate differential
        momentum = rate_diff.diff(lb)
        
        # Generate signal: +1 long, -1 short, 0 flat
        signal = pd.Series(0, index=momentum.index)
        signal[momentum > th] = 1
        signal[momentum < -th] = -1
        
        # Align signal with returns (shift by 1 to avoid lookahead)
        signal_aligned = signal.reindex(usdjpy_returns.index).ffill().shift(1)
        
        # Calculate strategy return
        strategy_returns = signal_aligned * usdjpy_returns
        annualized_return = strategy_returns.mean() * 12 * 100
        
        results.loc[lb, th] = round(annualized_return, 2)

# Plot heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(
    results.astype(float),
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Annualized Return (%)"}
)
plt.title("Hyperparameter Heatmap — Training Set (2000-2016)\nLookback Window vs Entry Threshold", fontsize=13)
plt.xlabel("Entry Threshold (percentage points)")
plt.ylabel("Lookback Window (months)")
plt.tight_layout()
plt.savefig("heatmap_training.png", dpi=150, bbox_inches="tight")
plt.show()
print("Heatmap saved as heatmap_training.png")