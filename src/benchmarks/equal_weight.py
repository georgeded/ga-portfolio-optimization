"""Naive equal-weight (1/N) benchmark. Rebalanced monthly weights drift between periods."""

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


def run_equal_weight(universe: pd.DataFrame, returns: pd.DataFrame,
                     gamma: float = TRANSACTION_COST) -> pd.DataFrame:
    rebalance_dates = sorted(universe["date"].unique())
    all_return_dates = sorted(returns["date"].unique())

    results = []
    prev_weights = None
    prev_permnos = None

    for t in rebalance_dates:

        apply_dates = [
            d for d in all_return_dates
            if d.year == t.year and d.month == t.month
        ]
        if not apply_dates:
            continue
        apply_date = apply_dates[0]

        eligible = universe[universe["date"] == t]["permno"].tolist()
        N = len(eligible)
        if N == 0:
            continue

        weights = np.ones(N) / N

        month_ret = get_monthly_returns(returns, eligible, apply_date)
        stock_returns = month_ret.values

        portfolio_gross = float(np.dot(weights, stock_returns))
        rf = get_rf_for_month(returns, apply_date)
        portfolio_excess = portfolio_gross - rf

        if prev_weights is not None and prev_permnos is not None:
            prev_ret = get_monthly_returns(returns, prev_permnos, apply_date)
            drifted = compute_drift_weights(prev_weights, prev_ret.values)
            drifted_array = align_drifted_weights(drifted, prev_permnos, eligible)
            turnover = portfolio_turnover(weights, drifted_array)
        else:
            turnover = 1.0

        cost = gamma * turnover
        net_excess = portfolio_excess - cost

        results.append({
            "date": apply_date,
            "portfolio_ret": portfolio_gross,
            "excess_ret": portfolio_excess,
            "rf": rf,
            "net_excess_ret": net_excess,
            "turnover": turnover,
            "cost": cost,
            "hhi": herfindahl_index(weights),
            "n_stocks": N,
        })

        # drift weights for next period's turnover calculation
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
