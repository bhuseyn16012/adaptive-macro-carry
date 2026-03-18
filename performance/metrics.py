import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


def load_benchmark_and_rf():
    sp500 = pd.read_csv(os.path.join(DATA_DIR, "sp500.csv"),
                        index_col="date", parse_dates=True).squeeze()
    sp500_monthly = sp500.resample("MS").last().pct_change()
    sp500_monthly.name = "sp500"

    rfr = pd.read_csv(os.path.join(DATA_DIR, "us_1yr_treasury.csv"),
                      index_col="date", parse_dates=True).squeeze()
    rfr_monthly = rfr.resample("MS").last() / 100 / 12
    rfr_monthly.name = "rf"

    return sp500_monthly, rfr_monthly


def annualized_return(returns):
    total = (1 + returns).prod()
    n_years = len(returns) / 12
    if n_years <= 0 or total <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def ccror(returns):
    return (1 + returns).prod() - 1


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def max_drawdown_duration(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    underwater = cum < peak
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    underwater_groups = groups[underwater]
    if len(underwater_groups) == 0:
        return 0
    return int(underwater_groups.value_counts().max())


def return_on_account(returns):
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return ccror(returns) / mdd


def gini_coefficient(returns):
    arr = np.sort(np.abs(returns.values))
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / \
           (n * np.sum(arr))


def compute_alpha_beta(strategy_returns, benchmark_returns,
                       rf_returns):
    aligned = pd.concat([strategy_returns, benchmark_returns,
                         rf_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["strat", "bench", "rf"]
    y = aligned["strat"] - aligned["rf"]
    x = aligned["bench"] - aligned["rf"]
    if len(y) < 3 or x.std() == 0:
        return 0.0, 0.0
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return intercept * 12, slope


def compute_up_down_alpha_beta(strategy_returns,
                                benchmark_returns, rf_returns):
    aligned = pd.concat([strategy_returns, benchmark_returns,
                         rf_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["strat", "bench", "rf"]
    y = aligned["strat"] - aligned["rf"]
    x = aligned["bench"] - aligned["rf"]
    results = {}
    for label, mask in [("up", aligned["bench"] > 0),
                        ("down", aligned["bench"] <= 0)]:
        yy, xx = y[mask], x[mask]
        if len(yy) < 3 or xx.std() == 0:
            results[label + "_alpha"] = 0.0
            results[label + "_beta"] = 0.0
        else:
            slope, intercept, _, _, _ = stats.linregress(xx, yy)
            results[label + "_alpha"] = intercept * 12
            results[label + "_beta"] = slope
    return results


def sharpe_ratio(returns, rf_returns):
    aligned = pd.concat([returns, rf_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["ret", "rf"]
    excess = aligned["ret"] - aligned["rf"]
    if excess.std() == 0:
        return 0.0
    return (excess.mean() * 12) / (excess.std() * np.sqrt(12))


def sortino_ratio(returns, rf_returns):
    aligned = pd.concat([returns, rf_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["ret", "rf"]
    excess = aligned["ret"] - aligned["rf"]
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return (excess.mean() * 12) / (downside.std() * np.sqrt(12))


def weighted_directional_test(strategy_returns, asset_returns):
    aligned = pd.concat([strategy_returns, asset_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["strat", "asset"]
    correct = np.sign(aligned["strat"]) == np.sign(aligned["asset"])
    weights = aligned["asset"].abs()
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0, 1.0, "Insufficient data"
    wda = (correct * weights).sum() / total_weight
    n = len(aligned)
    se = np.sqrt(0.25 / n)
    z = (wda - 0.5) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    if p_value < 0.05:
        interp = (f"Reject H0 at 5% (WDA={wda:.3f}, z={z:.2f}, "
                  f"p={p_value:.4f}). Strategy has significant "
                  f"directional ability.")
    else:
        interp = (f"Fail to reject H0 (WDA={wda:.3f}, z={z:.2f}, "
                  f"p={p_value:.4f}). No significant directional "
                  f"ability detected.")
    return wda, p_value, interp


def binomial_forecast_accuracy(strategy_returns, asset_returns):
    aligned = pd.concat([strategy_returns, asset_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["strat", "asset"]
    active = aligned[aligned["strat"] != 0]
    if len(active) < 5:
        return 0.0, 1.0, "Insufficient active months"
    correct = np.sign(active["strat"]) == np.sign(active["asset"])
    n_correct = correct.sum()
    n_total = len(active)
    accuracy = n_correct / n_total
    result = stats.binomtest(n_correct, n_total, 0.5,
                             alternative="two-sided")
    p_value = result.pvalue
    if p_value < 0.05:
        interp = (f"Reject H0 at 5% (accuracy={accuracy:.1%}, "
                  f"{n_correct}/{n_total}, p={p_value:.4f}). "
                  f"Forecast accuracy significantly different "
                  f"from random.")
    else:
        interp = (f"Fail to reject H0 (accuracy={accuracy:.1%}, "
                  f"{n_correct}/{n_total}, p={p_value:.4f}). "
                  f"Forecast accuracy not significantly different "
                  f"from random.")
    return accuracy, p_value, interp


def calmar_ratio(returns):
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return annualized_return(returns) / mdd


def omega_ratio(returns, threshold=0.0):
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess <= 0].sum())
    if losses == 0:
        return float("inf")
    return gains / losses


def tail_ratio(returns):
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    if p5 == 0:
        return float("inf")
    return abs(p95) / abs(p5)


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return gains / losses


def win_rate(returns):
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).mean()


def avg_win_loss_ratio(returns):
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return float("inf")
    return abs(wins.mean() / losses.mean())


def information_ratio(strategy_returns, benchmark_returns):
    aligned = pd.concat([strategy_returns, benchmark_returns],
                        axis=1, join="inner").dropna()
    aligned.columns = ["strat", "bench"]
    active = aligned["strat"] - aligned["bench"]
    te = active.std() * np.sqrt(12)
    if te == 0:
        return 0.0
    return (active.mean() * 12) / te


def kurtosis_excess(returns):
    return returns.kurtosis()


def var_95(returns):
    return np.percentile(returns, 5)


def cvar_95(returns):
    var = var_95(returns)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return tail.mean()


def monthly_hit_rate_by_year(returns):
    yearly = returns.groupby(returns.index.year)
    result = {}
    for year, group in yearly:
        active = group[group != 0]
        if len(active) > 0:
            result[year] = round((active > 0).mean() * 100, 1)
    return result


def compute_all_metrics(strategy_returns, asset_returns=None,
                        benchmark_returns=None, rf_returns=None,
                        label="Strategy", include_tests=True):
    if benchmark_returns is None or rf_returns is None:
        benchmark_returns, rf_returns = load_benchmark_and_rf()
    if asset_returns is None:
        usdjpy = pd.read_csv(
            os.path.join(DATA_DIR, "usdjpy.csv"),
            index_col="date", parse_dates=True).squeeze()
        asset_returns = usdjpy.resample("MS").last().pct_change()

    sr = strategy_returns.dropna()
    ar = asset_returns
    alpha, beta = compute_alpha_beta(sr, benchmark_returns,
                                     rf_returns)
    ud = compute_up_down_alpha_beta(sr, benchmark_returns,
                                    rf_returns)

    metrics = {}
    metrics["Annualized Return"] = f"{annualized_return(sr)*100:.2f}%"
    metrics["CCROR"] = f"{ccror(sr)*100:.2f}%"
    metrics["Max Drawdown"] = f"{max_drawdown(sr)*100:.2f}%"
    metrics["Skewness"] = f"{sr.skew():.3f}"
    metrics["Highest Month RoR"] = f"{sr.max()*100:.2f}%"
    metrics["Lowest Month RoR"] = f"{sr.min()*100:.2f}%"
    metrics["Gini Coefficient"] = f"{gini_coefficient(sr):.3f}"
    metrics["Annualized Std Dev"] = f"{sr.std()*np.sqrt(12)*100:.2f}%"
    metrics["Alpha"] = f"{alpha*100:.2f}%"
    metrics["Beta"] = f"{beta:.3f}"
    metrics["Up Alpha"] = f"{ud['up_alpha']*100:.2f}%"
    metrics["Up Beta"] = f"{ud['up_beta']:.3f}"
    metrics["Down Alpha"] = f"{ud['down_alpha']*100:.2f}%"
    metrics["Down Beta"] = f"{ud['down_beta']:.3f}"
    metrics["Sharpe Ratio"] = f"{sharpe_ratio(sr, rf_returns):.3f}"
    metrics["Sortino Ratio"] = f"{sortino_ratio(sr, rf_returns):.3f}"
    metrics["Return on Account"] = f"{return_on_account(sr):.3f}"
    metrics["Calmar Ratio"] = f"{calmar_ratio(sr):.3f}"
    metrics["Omega Ratio"] = f"{omega_ratio(sr):.3f}"
    metrics["Profit Factor"] = f"{profit_factor(sr):.3f}"
    metrics["Win Rate"] = f"{win_rate(sr)*100:.1f}%"
    metrics["Avg Win/Loss Ratio"] = f"{avg_win_loss_ratio(sr):.3f}"
    metrics["Tail Ratio"] = f"{tail_ratio(sr):.3f}"
    metrics["Information Ratio"] = \
        f"{information_ratio(sr, benchmark_returns):.3f}"
    metrics["Excess Kurtosis"] = f"{kurtosis_excess(sr):.3f}"
    metrics["VaR (95%)"] = f"{var_95(sr)*100:.2f}%"
    metrics["CVaR (95%)"] = f"{cvar_95(sr)*100:.2f}%"
    metrics["Max DD Duration (months)"] = \
        f"{max_drawdown_duration(sr)}"

    if include_tests:
        _, _, wda_interp = weighted_directional_test(sr, ar)
        _, _, bfa_interp = binomial_forecast_accuracy(sr, ar)
        metrics["Weighted Directional Test"] = wda_interp
        metrics["Binomial Forecast Accuracy"] = bfa_interp

    return metrics


def print_metrics_table(strat_metrics, bh_metrics):
    print("\n" + "=" * 85)
    print(f"{'PERFORMANCE METRICS':^85s}")
    print("=" * 85)
    print(f"  {'Metric':<30s} {'Strategy':>24s} {'Buy-and-Hold':>24s}")
    print("-" * 85)
    skip = ["Weighted Directional Test", "Binomial Forecast Accuracy"]
    for key in strat_metrics:
        if key in skip:
            continue
        sv = str(strat_metrics[key])[:24]
        bv = str(bh_metrics.get(key, "N/A"))[:24]
        print(f"  {key:<30s} {sv:>24s} {bv:>24s}")
    print("=" * 85)
    print("\nStatistical Tests (Strategy only):")
    print("-" * 85)
    for key in skip:
        if key in strat_metrics:
            print(f"\n  {key}:")
            print(f"  {strat_metrics[key]}")
    print()


def plot_metrics_table(strat_metrics, bh_metrics, save_path=None):
    skip = ["Weighted Directional Test", "Binomial Forecast Accuracy"]
    rows = []
    strat_vals = []
    bh_vals = []
    for key in strat_metrics:
        if key in skip:
            continue
        rows.append(key)
        strat_vals.append(str(strat_metrics[key]))
        bh_vals.append(str(bh_metrics.get(key, "N/A")))

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.38 + 1.5))
    ax.axis("off")
    table_data = list(zip(rows, strat_vals, bh_vals))
    col_labels = ["Metric", "CMV Strategy", "Buy-and-Hold USD/JPY"]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#2c3e50", "#2c3e50", "#2c3e50"],
    )
    for j in range(3):
        cell = table[0, j]
        cell.set_text_props(color="white", fontweight="bold",
                            fontsize=10)
        cell.set_height(0.045)
    for i in range(1, len(table_data) + 1):
        bg = "#f7f7f7" if i % 2 == 0 else "white"
        cell_name = table[i, 0]
        cell_name.set_text_props(fontweight="bold", fontsize=9,
                                 ha="left")
        cell_name.set_facecolor(bg)
        cell_name.set_height(0.038)
        cell_name.PAD = 0.05
        for j in [1, 2]:
            cell = table[i, j]
            cell.set_text_props(fontsize=9)
            cell.set_facecolor(bg)
            cell.set_height(0.038)
    table.auto_set_column_width([0, 1, 2])
    for j, w in enumerate([0.35, 0.325, 0.325]):
        for i in range(len(table_data) + 1):
            table[i, j].set_width(w)
    required_count = 17
    if len(table_data) > required_count:
        for j in range(3):
            cell = table[required_count + 1, j]
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)

    fig.suptitle("Performance Metrics -- OOS Period (2017-2026)",
                 fontsize=14, fontweight="bold", y=0.98)
    wda_text = strat_metrics.get("Weighted Directional Test", "")
    bfa_text = strat_metrics.get("Binomial Forecast Accuracy", "")
    footnote = ("Statistical Tests (Strategy):\n"
                + "  WDT: " + wda_text[:90] + "\n"
                + "  BFA: " + bfa_text[:90])
    fig.text(0.05, 0.01, footnote, fontsize=7, family="monospace",
             va="bottom")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "metrics_table.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("Metrics table saved to " + save_path)


def bootstrap_significance_test(strategy_returns, n_sims=10000,
                                 save_path=None):
    """
    Bootstrap Significance Test.

    Randomly assigns +1/-1 positions to actual USD/JPY returns
    10,000 times to create a distribution of "luck-based" Sharpe
    ratios. If the real Sharpe is in the top 5%, the signal has
    statistically significant predictive power beyond chance.

    Makes no assumptions about return distribution (nonparametric).
    """
    real_sharpe = (strategy_returns.mean() * 12) / \
                  (strategy_returns.std() * np.sqrt(12))

    usdjpy = pd.read_csv(os.path.join(DATA_DIR, "usdjpy.csv"),
                         index_col="date",
                         parse_dates=True).squeeze()
    asset_rets = usdjpy.resample("MS").last().pct_change()
    asset_rets = asset_rets.reindex(
        strategy_returns.index).dropna()

    np.random.seed(42)
    random_sharpes = []
    n_months = len(asset_rets)

    for _ in range(n_sims):
        random_pos = np.random.choice([-1, 1], size=n_months)
        random_strat = random_pos * asset_rets.values
        mu = random_strat.mean() * 12
        sigma = random_strat.std() * np.sqrt(12)
        s = mu / sigma if sigma > 0 else 0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    percentile = (random_sharpes < real_sharpe).mean() * 100
    p_value = 1 - percentile / 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(random_sharpes, bins=80, color="#cccccc",
            edgecolor="white", linewidth=0.3, density=True,
            label="Random signals (n=10,000)")
    ax.axvline(real_sharpe, color="blue", linewidth=2.5,
               linestyle="-",
               label=f"Real strategy: {real_sharpe:.3f}")
    cutoff_95 = np.percentile(random_sharpes, 95)
    ax.axvline(cutoff_95, color="red", linewidth=1.5,
               linestyle="--", alpha=0.7,
               label=f"95th percentile: {cutoff_95:.3f}")
    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Bootstrap Significance Test\n",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    result_text = (f"Real Sharpe: {real_sharpe:.3f}\n"
                   f"Percentile: {percentile:.1f}%\n"
                   f"p-value: {p_value:.4f}\n"
                   f"{'Significant at 5%' if p_value < 0.05 else 'Not significant'}")
    ax.text(0.97, 0.95, result_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray",
                      alpha=0.9))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "bootstrap_test.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("Bootstrap test saved to " + save_path)
    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Percentile: {percentile:.1f}%")
    print(f"  p-value: {p_value:.4f}")
    return real_sharpe, percentile, p_value


def random_walk_benchmark(strategy_returns, n_sims=10000,
                          save_path=None):
    """
    Random Walk Benchmark Test.

    Generates 10,000 random strategies (coin flip: +1 or -1 each
    month) and compares cumulative returns against the real strategy.
    Addresses the Meese-Rogoff (1983) puzzle: random walks are
    famously hard to beat in FX forecasting.

    If the real strategy beats most random strategies, the signal
    adds genuine value beyond noise.
    """
    usdjpy = pd.read_csv(os.path.join(DATA_DIR, "usdjpy.csv"),
                         index_col="date",
                         parse_dates=True).squeeze()
    usdjpy_monthly = usdjpy.resample("MS").last()
    asset_returns = usdjpy_monthly.pct_change()
    aligned_returns = asset_returns.reindex(
        strategy_returns.index).dropna()

    real_cum = (1 + strategy_returns).prod() - 1

    np.random.seed(42)
    random_cums = []
    n_months = len(aligned_returns)

    for _ in range(n_sims):
        random_pos = np.random.choice([-1, 1], size=n_months)
        random_strat = random_pos * aligned_returns.values
        random_cums.append((1 + random_strat).prod() - 1)

    random_cums = np.array(random_cums)
    percentile = (random_cums < real_cum).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(random_cums * 100, bins=80, color="#cccccc",
            edgecolor="white", linewidth=0.3, density=True,
            label="Random coin-flip strategies (n=10,000)")
    ax.axvline(real_cum * 100, color="blue", linewidth=2.5,
               linestyle="-",
               label=f"Real strategy: {real_cum*100:.1f}%")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-",
               alpha=0.3)
    ax.set_xlabel("Cumulative Return (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Random Walk Benchmark\n",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    result_text = (f"Real CCROR: {real_cum*100:.1f}%\n"
                   f"Median random: "
                   f"{np.median(random_cums)*100:.1f}%\n"
                   f"Percentile: {percentile:.1f}%\n"
                   f"Beat {percentile:.0f}% of random strategies")
    ax.text(0.97, 0.95, result_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray",
                      alpha=0.9))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "random_walk_test.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print("Random walk test saved to " + save_path)
    print(f"  Real CCROR: {real_cum*100:.1f}%")
    print(f"  Beat {percentile:.0f}% of random strategies")
    return real_cum, percentile


def run(oos_returns=None):
    if oos_returns is None:
        oos_returns = pd.read_csv(
            os.path.join(DATA_DIR, "oos_returns.csv"),
            index_col=0, parse_dates=True
        ).squeeze()

    benchmark, rf = load_benchmark_and_rf()
    usdjpy = pd.read_csv(os.path.join(DATA_DIR, "usdjpy.csv"),
                         index_col="date",
                         parse_dates=True).squeeze()
    asset = usdjpy.resample("MS").last().pct_change()
    bh = asset.reindex(oos_returns.index).dropna()

    strat_metrics = compute_all_metrics(
        oos_returns, asset, benchmark, rf,
        label="CMV Strategy", include_tests=True)
    bh_metrics = compute_all_metrics(
        bh, asset, benchmark, rf,
        label="Buy-and-Hold", include_tests=False)

    print_metrics_table(strat_metrics, bh_metrics)
    plot_metrics_table(strat_metrics, bh_metrics)

    print("\n--- ROBUSTNESS TESTS ---")
    bootstrap_significance_test(oos_returns)
    random_walk_benchmark(oos_returns)

    hits = monthly_hit_rate_by_year(oos_returns)
    print("\nMonthly Hit Rate by Year:")
    for year, pct in hits.items():
        print(f"  {year}: {pct}%")

    return strat_metrics, bh_metrics


if __name__ == "__main__":
    run()