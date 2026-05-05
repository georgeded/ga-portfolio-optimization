"""
Mean-variance optimization - two variants:
- Unconstrained MVO: maximize Sharpe, long-only [0, 1], no cardinality constraint.
- Constrained MVO: maximize Sharpe, long-only, max weight 0.15, no cardinality constraint.

Both use the 60-month estimation window, full universe (approx. 867 stocks), gamma = 0.3%.
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from tqdm import tqdm

from src.evaluation.metrics import (
    portfolio_turnover,
    transaction_cost,
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
    get_estimation_window
)

W_MIN_UNCONSTRAINED = 0.0
W_MAX_UNCONSTRAINED = 1.0
W_MIN_CONSTRAINED = 0.0
W_MAX_CONSTRAINED = 0.15


def negative_sharpe(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Objective minimized by scipy SLSQP."""
    port_return = float(weights @ mu)
    port_var = float(weights @ sigma @ weights)
    if port_var <= 0:
        return 0.0
    return -port_return / np.sqrt(port_var)


def optimize_mvo(mu: np.ndarray, sigma: np.ndarray, w_min: float, w_max: float,
                 n_restarts: int = 1, rng: np.random.Generator = None) -> np.ndarray:
    """Maximize Sharpe (SLSQP, n_restarts random starts). Falls back to equal weights on failure."""
    if rng is None:
        rng = np.random.default_rng(seed=42)

    N = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(w_min, w_max)] * N

    best_weights = None
    best_sharpe = -np.inf

    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(N))
        w0 = np.clip(w0, w_min, w_max)
        w0 = w0 / w0.sum()

        try:
            result = minimize(
                fun=negative_sharpe,
                x0=w0,
                args=(mu, sigma),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-6}
            )
            if result.success and (-result.fun) > best_sharpe:
                best_sharpe = -result.fun
                best_weights = result.x
        except Exception:
            continue

    if best_weights is None:
        best_weights = np.ones(N) / N

    best_weights = np.clip(best_weights, w_min, w_max)
    best_weights = best_weights / best_weights.sum()
    return best_weights


def _process_single_period(t, apply_date, universe, returns, w_min, w_max, gamma,
                           prev_weights, prev_permnos, rng) -> tuple:
    """Process one rebalancing period. Returns (result_dict, new_weights, new_permnos)."""
    eligible = universe[universe["date"] == t]["permno"].tolist()
    if len(eligible) == 0:
        return None, prev_weights, prev_permnos

    mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
    if mu is None or len(valid_permnos) < 2:
        return None, prev_weights, prev_permnos

    weights = optimize_mvo(mu, sigma, w_min, w_max, rng=rng)

    month_ret = get_monthly_returns(returns, valid_permnos, apply_date)
    stock_returns = month_ret.values
    rf = get_rf_for_month(returns, apply_date)

    portfolio_gross = float(np.dot(weights, stock_returns))
    portfolio_excess = portfolio_gross - rf

    if prev_weights is not None and prev_permnos is not None:
        prev_ret = get_monthly_returns(returns, prev_permnos, apply_date)
        drifted = compute_drift_weights(prev_weights, prev_ret.values)
        drifted_array = align_drifted_weights(drifted, prev_permnos, valid_permnos)
        to = portfolio_turnover(weights, drifted_array)
    else:
        to = 1.0

    cost = transaction_cost(to, gamma)
    net_excess = portfolio_excess - cost

    result = {
        "date": apply_date,
        "n_stocks": len(valid_permnos),
        "portfolio_ret": portfolio_gross,
        "excess_ret": portfolio_excess,
        "rf": rf,
        "net_excess_ret": net_excess,
        "turnover": to,
        "cost": cost,
        "hhi": herfindahl_index(weights),
    }

    return result, compute_drift_weights(weights, stock_returns), valid_permnos


def run_mvo(universe: pd.DataFrame, returns: pd.DataFrame, constrained: bool = False,
            gamma: float = TRANSACTION_COST, seed: int = 42) -> pd.DataFrame:
    """Run MVO over all rebalancing dates.
    constrained=True uses bounds [0, 0.15]; False uses [0, 1.0].
    """
    rng = np.random.default_rng(seed)
    w_min = W_MIN_CONSTRAINED if constrained else W_MIN_UNCONSTRAINED
    w_max = W_MAX_CONSTRAINED if constrained else W_MAX_UNCONSTRAINED
    label = "Constrained MVO" if constrained else "Unconstrained MVO"

    rebalance_dates = sorted(universe["date"].unique())
    all_return_dates = sorted(returns["date"].unique())

    results = []
    prev_weights = None
    prev_permnos = None

    for t in tqdm(rebalance_dates, desc=f"Running {label}"):
        apply_dates = [
            d for d in all_return_dates
            if d.year == t.year and d.month == t.month
        ]
        if not apply_dates:
            continue

        result, prev_weights, prev_permnos = _process_single_period(
            t, apply_dates[0], universe, returns,
            w_min, w_max, gamma,
            prev_weights, prev_permnos, rng
        )
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    universe, returns = load_data()

    print(f"Universe: {universe['date'].nunique()} rebalancing dates, "
          f"avg {universe.groupby('date').size().mean():.0f} stocks/month")

    os.makedirs("results/benchmarks", exist_ok=True)

    print("\nRunning Unconstrained MVO...")
    results_unconstrained = run_mvo(universe, returns, constrained=False)
    print_results(results_unconstrained, "UNCONSTRAINED MVO")
    results_unconstrained.to_parquet(
        "results/benchmarks/mvo_unconstrained.parquet", index=False
    )
    print("Saved: results/benchmarks/mvo_unconstrained.parquet")

    print("\nRunning Constrained MVO...")
    results_constrained = run_mvo(universe, returns, constrained=True)
    print_results(results_constrained, "CONSTRAINED MVO (QP)")
    results_constrained.to_parquet(
        "results/benchmarks/mvo_constrained.parquet", index=False
    )
    print("Saved: results/benchmarks/mvo_constrained.parquet")
