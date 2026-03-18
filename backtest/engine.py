import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TRAIN_END, TEST_START
from signals.factors import (
    load_raw, build_carry_signal, build_momentum_signal, build_composite,
)

# Hyperparameter grid
LOOKBACKS = [6, 9, 12, 18]
KALMAN_COVS = [0.01, 0.05, 0.10, 0.25, 0.50]
THRESHOLD = 0.0

# Fixed structural parameters
W_CARRY = 0.30
W_MOMENTUM = 0.70
Z_WINDOW = 36


def strategy_returns(usdjpy, lookback, kalman_cov,
                     threshold=THRESHOLD, start=None, end=None):
    """
    Full signal pipeline for given hyperparameters.
    Builds carry with specified Kalman smoothing, momentum with
    specified lookback, composites them, generates lagged positions,
    returns monthly strategy returns.

    Positions are lagged by 1 month to prevent look-ahead bias:
    signal generated at end of month t, position taken at start of
    month t+1, return earned during month t+1.
    """
    carry, _ = build_carry_signal(transition_cov=kalman_cov)
    momentum = build_momentum_signal(usdjpy, lookback=lookback,
                                     verbose=False)
    comp = build_composite(carry, momentum, window=Z_WINDOW,
                           w_carry=W_CARRY, w_momentum=W_MOMENTUM,
                           threshold=threshold)

    usdjpy_monthly = usdjpy.resample("MS").last()
    rets = usdjpy_monthly.pct_change()

    # Lag position by 1 month to prevent look-ahead
    position = comp["position"].shift(1)
    strat = (position * rets).dropna()
    strat.name = "strategy_return"

    if start:
        strat = strat[strat.index >= start]
    if end:
        strat = strat[strat.index <= end]

    return strat, comp


def evaluate(returns):
    """Sharpe ratio for hyperparameter optimization."""
    if len(returns) < 12 or returns.std() == 0:
        return -999.0
    return (returns.mean() * 12) / (returns.std() * np.sqrt(12))


# -------------------------------------------------------------------
# Heatmap: hyperparameter search on training data
# -------------------------------------------------------------------

def build_heatmap(usdjpy, train_end=TRAIN_END):
    """
    2D heatmap: momentum_lookback vs Kalman transition_covariance.
    Evaluation metric: annualized return on training period.

    The Kalman transition_cov is analogous to the EWMA span
    parameter in Tornell & Kunz (2026):
        - Low transition_cov (0.01) = very smooth, slow to react
          (like EWMA with large span)
        - High transition_cov (0.50) = responsive, tracks noise
          (like EWMA with small span)
    """
    results = pd.DataFrame(index=LOOKBACKS, columns=KALMAN_COVS,
                           dtype=float)
    results.index.name = "lookback"

    for lb in LOOKBACKS:
        for kc in KALMAN_COVS:
            ret, _ = strategy_returns(usdjpy, lb, kc, end=train_end)
            ann_ret = ret.mean() * 12 * 100
            results.loc[lb, kc] = round(ann_ret, 2)

    print("\nHeatmap (annualized return %):")
    print(results.to_string())
    return results


def plot_heatmap(heatmap_df,
                 title="Hyperparameter Optimization -- Training Set "
                       "(2000-2016)\nAnnualized Return (%)",
                 save_path=None):
    """
    Professional heatmap matching academic paper style.
    Clean grid, annotated cells, diverging colormap centered at zero.
    """
    data = heatmap_df.astype(float)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Color normalization centered at zero
    vals = data.values
    vmax = max(abs(vals.min()), abs(vals.max()))
    if vmax == 0:
        vmax = 1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Draw heatmap
    im = ax.imshow(vals, cmap="RdYlGn", norm=norm, aspect="auto")

    # Axis labels
    row_labels = [str(r) for r in data.index]
    col_labels = [str(c) for c in data.columns]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    ax.set_xlabel("Kalman Transition Covariance "
                  "(low = smooth, high = responsive)",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Momentum Lookback (months)", fontsize=11,
                  labelpad=8)

    # Cell annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = vals[i, j]
            brightness = (val - vals.min()) / (vals.max() - vals.min()) \
                if vals.max() != vals.min() else 0.5
            color = "white" if brightness > 0.75 or brightness < 0.25 \
                else "black"
            fontweight = "bold" if val == vals.max() else "normal"
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight=fontweight)

    # Mark best cell with a border
    best_idx = np.unravel_index(vals.argmax(), vals.shape)
    rect = plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5),
                          1, 1, linewidth=2.5, edgecolor="black",
                          facecolor="none")
    ax.add_patch(rect)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Annualized Return (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "heatmap_cmv.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("Heatmap saved to " + save_path)


# -------------------------------------------------------------------
# Walk-forward backtest with annual re-optimization
# -------------------------------------------------------------------

def walk_forward_backtest(usdjpy):
    """
    Walk-forward with annual re-optimization over both
    momentum_lookback and kalman_transition_covariance.

    For each year Y in the test period:
        1. Train on all data from start through end of Y-1
        2. Optimize lookback and kalman_cov by maximizing Sharpe
        3. Generate OOS signals for year Y using optimal params
        4. Record returns and optimal parameters
    """
    test_years = range(int(TEST_START[:4]), 2027)
    all_returns = []
    all_positions = []
    param_history = []

    print("\n" + "=" * 60)
    print("WALK-FORWARD BACKTEST (Annual Re-optimization)")
    print("=" * 60)

    for year in test_years:
        train_end = str(year - 1) + "-12-31"
        test_start = str(year) + "-01-01"
        test_end = str(year) + "-12-31"

        best_sharpe = -999
        best_lb, best_kc = 12, 0.10

        for lb in LOOKBACKS:
            for kc in KALMAN_COVS:
                ret, _ = strategy_returns(usdjpy, lb, kc,
                                          end=train_end)
                sr = evaluate(ret)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_lb = lb
                    best_kc = kc

        # OOS with optimal params
        ret_oos, comp_oos = strategy_returns(
            usdjpy, best_lb, best_kc,
            start=test_start, end=test_end
        )

        pos_oos = comp_oos["position"].shift(1)
        pos_oos = pos_oos[(pos_oos.index >= test_start)
                          & (pos_oos.index <= test_end)]

        all_returns.append(ret_oos)
        all_positions.append(pos_oos)
        param_history.append({
            "year": year,
            "lookback": best_lb,
            "kalman_cov": best_kc,
            "train_sharpe": round(best_sharpe, 3),
        })

        print(f"  {year}: lookback={best_lb:2d}, "
              f"kalman_cov={best_kc:.2f}, "
              f"train_sharpe={best_sharpe:+.3f}, "
              f"OOS_months={len(ret_oos)}")

    oos_returns = pd.concat(all_returns)
    oos_returns = oos_returns[
        ~oos_returns.index.duplicated(keep="first")]
    positions = pd.concat(all_positions)
    positions = positions[
        ~positions.index.duplicated(keep="first")]

    return oos_returns, param_history, positions


# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------

def plot_param_history(param_history, save_path=None):
    """Show how hyperparameters evolve over walk-forward period."""
    df = pd.DataFrame(param_history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    sharex=True)
    fig.suptitle("Walk-Forward Hyperparameter Evolution",
                 fontsize=13, fontweight="bold")

    ax1.step(df["year"], df["lookback"], where="mid",
             color="blue", linewidth=2, marker="o", markersize=6)
    ax1.set_ylabel("Momentum Lookback (months)")
    ax1.set_ylim(0, max(LOOKBACKS) + 3)
    ax1.grid(True, alpha=0.3)

    ax2.step(df["year"], df["kalman_cov"], where="mid",
             color="red", linewidth=2, marker="s", markersize=6)
    ax2.set_ylabel("Kalman Transition Covariance")
    ax2.set_xlabel("Year")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "param_evolution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Parameter chart saved to " + save_path)


def plot_equity_curve(oos_returns, usdjpy, positions=None,
                      save_path=None):
    """
    Cumulative PnL: strategy vs buy-and-hold, with drawdown
    and position panels.
    """
    usdjpy_monthly = usdjpy.resample("MS").last()
    bh = usdjpy_monthly.pct_change().reindex(
        oos_returns.index).fillna(0)

    cum_strat = (1 + oos_returns).cumprod()
    cum_bh = (1 + bh).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             sharex=True)
    fig.suptitle("Adaptive Macro Carry -- OOS Equity Curve "
                 "(Walk-Forward)",
                 fontsize=14, fontweight="bold")

    axes[0].plot(cum_strat.index, cum_strat.values,
                 color="blue", linewidth=1.5, label="CMV Strategy")
    axes[0].plot(cum_bh.index, cum_bh.values,
                 color="gray", linewidth=1.0, alpha=0.7,
                 label="Buy-and-Hold USD/JPY")
    axes[0].axhline(1.0, color="black", linestyle="-", alpha=0.3)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    peak = cum_strat.cummax()
    dd = (cum_strat - peak) / peak
    axes[1].fill_between(dd.index, dd.values, 0,
                         color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)

    if positions is not None:
        pos_a = positions.reindex(oos_returns.index).fillna(0)
        colors = ["green" if p > 0 else "red" if p < 0 else "gray"
                  for p in pos_a.values]
        axes[2].bar(pos_a.index, pos_a.values,
                    color=colors, alpha=0.6, width=20)
        axes[2].set_ylabel("Position")
        axes[2].set_yticks([-1, 0, 1])
        axes[2].set_yticklabels(["Short", "Flat", "Long"])

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "equity_curve_oos.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Equity curve saved to " + save_path)


def plot_signals_on_price(usdjpy, composite_df, positions,
                          save_path=None):
    """
    Plot forecast signals overlaid on USD/JPY price.
    Green triangles = long entry, red triangles = short entry.
    """
    usdjpy_monthly = usdjpy.resample("MS").last()
    pos = positions.reindex(usdjpy_monthly.index).fillna(0)

    # Detect position changes (entries)
    entries = pos[(pos != 0) & (pos.shift(1) != pos)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    sharex=True)
    fig.suptitle("Forecast Signals on USD/JPY -- Test Period",
                 fontsize=13, fontweight="bold")

    test_price = usdjpy_monthly[
        usdjpy_monthly.index >= TEST_START]
    ax1.plot(test_price.index, test_price.values,
             color="black", linewidth=1.2)

    for dt in entries.index:
        if dt in usdjpy_monthly.index \
                and dt >= pd.Timestamp(TEST_START):
            c = "green" if entries.loc[dt] > 0 else "red"
            marker = "^" if entries.loc[dt] > 0 else "v"
            ax1.scatter(dt, usdjpy_monthly.loc[dt], color=c,
                        marker=marker, s=80, zorder=5)

    ax1.set_ylabel("USD/JPY")
    ax1.set_title("USD/JPY with Trade Signals "
                  "(green=long, red=short)")
    ax1.grid(True, alpha=0.3)

    comp_test = composite_df["composite"][
        composite_df.index >= TEST_START]
    comp_colors = ["green" if x > 0 else "red"
                   for x in comp_test.values]
    ax2.bar(comp_test.index, comp_test.values,
            color=comp_colors, alpha=0.6, width=20)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Composite Z-Score")
    ax2.set_title("CMV Composite Signal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "signals_on_price.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Signal plot saved to " + save_path)


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

def run():
    """Execute full backtest pipeline."""
    usdjpy = load_raw("usdjpy.csv")

    # 1. Heatmap on training data
    print("\n--- HEATMAP ---")
    hm = build_heatmap(usdjpy)
    plot_heatmap(hm)

    # 2. Walk-forward backtest
    oos_returns, param_history, positions = walk_forward_backtest(
        usdjpy)

    # 3. Plots
    plot_param_history(param_history)

    # Build composite with final year's params for signal plot
    final_p = param_history[-1]
    carry, _ = build_carry_signal(
        transition_cov=final_p["kalman_cov"])
    momentum = build_momentum_signal(
        usdjpy, lookback=final_p["lookback"])
    comp_final = build_composite(carry, momentum)

    plot_equity_curve(oos_returns, usdjpy, positions)
    plot_signals_on_price(usdjpy, comp_final, positions)

    # Save results
    oos_returns.to_csv(os.path.join(DATA_DIR, "oos_returns.csv"))
    positions.to_csv(os.path.join(DATA_DIR, "oos_positions.csv"))
    pd.DataFrame(param_history).to_csv(
        os.path.join(DATA_DIR, "param_history.csv"), index=False)

    print("\nAll backtest results saved to data/raw/")
    return oos_returns, param_history, positions


if __name__ == "__main__":
    run()