import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from hmmlearn import hmm
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_END


def load_raw(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df.squeeze()


def build_regime_features():
    vix = load_raw("vix.csv")
    usdjpy = load_raw("usdjpy.csv")

    vix_monthly = vix.resample("MS").mean()
    vix_monthly.name = "vix"

    usdjpy_returns = np.log(usdjpy / usdjpy.shift(1))
    realized_vol = usdjpy_returns.resample("MS").std() * np.sqrt(21) * 100
    realized_vol.name = "realized_vol"

    features = pd.concat([vix_monthly, realized_vol], axis=1).dropna()

    print("Regime features built: " + str(len(features)) + " monthly observations")
    print("Range: " + str(features.index.min().date()) + " to " + str(features.index.max().date()))

    return features


def train_hmm(features, n_states=2, n_iter=1000):
    X = features[["vix", "realized_vol"]].values
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42
    )
    model.fit(X)
    return model


def label_regimes(model, features):
    X = features[["vix", "realized_vol"]].values
    raw_states = model.predict(X)

    temp = pd.DataFrame({"state": raw_states, "vix": features["vix"].values})
    avg_vix = temp.groupby("state")["vix"].mean().sort_values()
    state_map = {old: new for new, old in enumerate(avg_vix.index)}

    labeled = pd.Series(
        [state_map[s] for s in raw_states],
        index=features.index,
        name="regime"
    )

    smoothed = labeled.rolling(3, center=True).apply(
        lambda x: pd.Series(x).mode()[0]
    ).fillna(labeled).astype(int)
    smoothed.name = "regime"

    return smoothed


def get_regime_multiplier(regime_series):
    multiplier = regime_series.map({0: 1.0, 1: 0.0})
    multiplier.name = "regime_multiplier"
    return multiplier


def plot_regimes(features, regimes, usdjpy):
    usdjpy_monthly = usdjpy.resample("MS").last()

    regime_colors = {0: "lightgreen", 1: "lightcoral"}
    regime_labels = {0: "Risk-On", 1: "Crisis"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Adaptive Macro Carry - Regime Detection",
                 fontsize=14, fontweight="bold", y=0.99)

    for ax in axes:
        ax.axvline(pd.Timestamp(TRAIN_END), color="red", linestyle="--", alpha=0.7)
        prev_date = regimes.index[0]
        prev_regime = regimes.iloc[0]
        for date, regime in regimes.items():
            if regime != prev_regime:
                ax.axvspan(prev_date, date, alpha=0.3,
                          color=regime_colors[prev_regime])
                prev_date = date
                prev_regime = regime
        ax.axvspan(prev_date, regimes.index[-1], alpha=0.3,
                  color=regime_colors[prev_regime])

    # Panel 1: USD/JPY
    axes[0].plot(usdjpy_monthly.index, usdjpy_monthly.values,
                color="black", linewidth=1.5)
    axes[0].set_ylabel("USD/JPY")
    axes[0].set_title("USD/JPY Exchange Rate", pad=4)
    legend_elements = [Patch(facecolor=regime_colors[i], alpha=0.5,
                            label=regime_labels[i]) for i in range(2)]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=8)

    # Panel 2: VIX and Realized Vol on dual axis
    ax2b = axes[1].twinx()
    axes[1].plot(features.index, features["vix"].values,
                color="purple", linewidth=1.2, label="VIX (left)")
    axes[1].axhline(20, color="orange", linestyle="--", alpha=0.4, label="VIX 20")
    axes[1].axhline(30, color="red", linestyle="--", alpha=0.4, label="VIX 30")
    axes[1].set_ylabel("VIX", color="purple")
    axes[1].set_title("VIX and USD/JPY Realized Volatility", pad=4)
    ax2b.plot(features.index, features["realized_vol"].values,
             color="darkorange", linewidth=1.2, alpha=0.7, label="Realized Vol (right)")
    ax2b.set_ylabel("Realized Vol %", color="darkorange")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Panel 3: Regime
    axes[2].plot(regimes.index, regimes.values,
                color="black", linewidth=1.0, drawstyle="steps-post")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Risk-On", "Crisis"])
    axes[2].set_title("Detected Regime", pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("data/raw/regime_detection.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Regime plot saved to data/raw/regime_detection.png")


def run():
    print("=" * 50)
    print("REGIME DETECTION")
    print("=" * 50)

    features = build_regime_features()
    model = train_hmm(features)
    regimes = label_regimes(model, features)
    multipliers = get_regime_multiplier(regimes)

    counts = regimes.value_counts().sort_index()
    labels = {0: "Risk-On", 1: "Crisis"}
    for state, count in counts.items():
        pct = round(count / len(regimes) * 100, 1)
        print(labels[state] + ": " + str(count) + " months (" + str(pct) + "%)")

    usdjpy = load_raw("usdjpy.csv")
    plot_regimes(features, regimes, usdjpy)

    regimes.to_csv(os.path.join(DATA_DIR, "regimes.csv"))
    multipliers.to_csv(os.path.join(DATA_DIR, "regime_multipliers.csv"))
    print("Regimes saved to data/raw/regimes.csv")
    print("=" * 50)

    return regimes, multipliers, model


if __name__ == "__main__":
    run()