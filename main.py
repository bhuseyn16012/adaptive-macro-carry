import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("ADAPTIVE MACRO CARRY -- FULL PIPELINE")
    print("=" * 60)

    # Step 1: Build CMV factors
    print("\n[1/4] BUILDING FACTORS...")
    from signals.factors import run as run_factors
    carry, nominal_diff, momentum, composite_df = run_factors()

    # Step 2: Regime detection
    print("\n[2/4] REGIME DETECTION...")
    from regime.hmm import run as run_regime
    regimes, multipliers, hmm_model = run_regime()

    # Step 3: Walk-forward backtest
    print("\n[3/4] WALK-FORWARD BACKTEST...")
    from backtest.engine import run as run_backtest
    oos_returns, param_history, positions = run_backtest()

    # Step 4: Performance metrics
    print("\n[4/4] PERFORMANCE METRICS...")
    from performance.metrics import run as run_metrics
    strat_metrics, bh_metrics = run_metrics(oos_returns)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print("\nGenerated files in data/raw/:")
    for f in sorted(os.listdir("data/raw/")):
        if f.endswith((".png", ".csv")):
            size = os.path.getsize(os.path.join("data/raw/", f))
            if size > 0:
                print(f"  {f:40s} {size:>8,d} bytes")

    print("\nKey results:")
    print(f"  Strategy Sharpe:      {strat_metrics['Sharpe Ratio']}")
    print(f"  Strategy Return:      {strat_metrics['Annualized Return']}")
    print(f"  Strategy CCROR:       {strat_metrics['CCROR']}")
    print(f"  Max Drawdown:         {strat_metrics['Max Drawdown']}")
    print(f"  Omega Ratio:          {strat_metrics['Omega Ratio']}")
    print(f"  Profit Factor:        {strat_metrics['Profit Factor']}")
    print(f"  Win Rate:             {strat_metrics['Win Rate']}")
    print(f"  Forecast Accuracy:    {strat_metrics['Binomial Forecast Accuracy'][:50]}")
    print(f"  OOS Months:           {len(oos_returns)}")


if __name__ == "__main__":
    main()