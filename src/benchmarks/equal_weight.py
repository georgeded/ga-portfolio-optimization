"""
Benchmark: Equal-Weight Portfolio (1/N)

Constructs a naive equal-weight portfolio that assigns weight 1/N
to every stock in the eligible universe at each rebalancing date t.

Key properties:
- No estimation required
- Rebalanced monthly back to equal weights
- Full eligible universe (~867 stocks) — consistent with DeMiguel et al. (2009)
- Transaction costs applied via turnover calculation
- Weights drift between rebalancing dates

Reference: DeMiguel et al. (2009) "Optimal Versus Naive Diversification"
"""

import numpy as np
import pandas as pd
import os

from src.evaluation.metrics import (
    portfolio_turnover,
    herfindahl_index,
    TRANSACTION_COST
)
from src.utils.data import load_data
from src.utils.portfolio import (
    get_monthly_returns,
    get_rf_for_month,
    compute_drift_weights,
    align_drifted_weights,
    print_results,
)


def run_equal_weight(universe: pd.DataFrame,
                     returns:  pd.DataFrame,
                     gamma:    float = TRANSACTION_COST) -> pd.DataFrame:
    """
    Run the equal-weight strategy over all rebalancing dates.

    At each rebalancing date t:
    1. Get eligible stocks from universe
    2. Assign equal weights (1/N)
    3. Apply weights to month t returns
    4. Compute turnover vs drifted weights from last period
    5. Record results

    Args:
        universe: DataFrame with columns [date, permno]
        returns:  DataFrame with columns [date, permno, ret, rf, excess_ret]
        gamma:    proportional transaction cost rate

    Returns:
        DataFrame with monthly results
    """
    rebalance_dates  = sorted(universe["date"].unique())
    all_return_dates = sorted(returns["date"].unique())

    results      = []
    prev_weights = None
    prev_permnos = None

    for t in rebalance_dates:

        # Find end-of-month date for this rebalancing month
        apply_dates = [
            d for d in all_return_dates
            if d.year == t.year and d.month == t.month
        ]
        if not apply_dates:
            continue
        apply_date = apply_dates[0]

        # Step 1: Get eligible stocks
        eligible = universe[universe["date"] == t]["permno"].tolist()
        N = len(eligible)
        if N == 0:
            continue

        # Step 2: Equal weights
        weights = np.ones(N) / N

        # Step 3: Get returns for month t
        month_ret     = get_monthly_returns(returns, eligible, apply_date)
        stock_returns = month_ret.values

        portfolio_gross  = float(np.dot(weights, stock_returns))
        rf               = get_rf_for_month(returns, apply_date)
        portfolio_excess = portfolio_gross - rf

        # Step 4: Compute turnover
        if prev_weights is not None and prev_permnos is not None:
            prev_ret      = get_monthly_returns(returns, prev_permnos, apply_date)
            drifted       = compute_drift_weights(prev_weights, prev_ret.values)
            drifted_array = align_drifted_weights(drifted, prev_permnos, eligible)
            turnover      = portfolio_turnover(weights, drifted_array)
        else:
            turnover = 1.0

        # Step 5: Transaction cost and net return
        cost       = gamma * turnover
        net_excess = portfolio_excess - cost

        results.append({
            "date"          : apply_date,
            "portfolio_ret" : portfolio_gross,
            "excess_ret"    : portfolio_excess,
            "rf"            : rf,
            "net_excess_ret": net_excess,
            "turnover"      : turnover,
            "cost"          : cost,
            "hhi"           : herfindahl_index(weights),
            "n_stocks"      : N,
        })

        # Drift weights for next period's turnover calculation
        prev_weights = compute_drift_weights(weights, stock_returns)
        prev_permnos = eligible

    return pd.DataFrame(results)


if __name__ == "__main__":
    universe, returns = load_data()

    print("\nRunning 1/N — Full Universe...")
    results = run_equal_weight(universe, returns)
    print_results(results, "1/N FULL UNIVERSE", show_theoretical_hhi=True)

    os.makedirs("results/benchmarks", exist_ok=True)
    results.to_parquet(
        "results/benchmarks/equal_weight_full.parquet", index=False
    )
    print("\nSaved: results/benchmarks/equal_weight_full.parquet")