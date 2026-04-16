"""
src/benchmarks/equal_weight.py
Benchmark 2: Equal-Weight Portfolio (1/N)

Constructs a naive equal-weight portfolio that assigns weight 1/N
to every stock in the eligible universe at each rebalancing date t.

Key properties:
- No estimation required (no expected returns or covariance matrix)
- Rebalanced monthly back to equal weights
- Full eligible universe used (consistent with DeMiguel et al. 2009)
- Transaction costs applied via turnover calculation
- Weights drift between rebalancing dates

Reference: DeMiguel et al. (2009) "Optimal Versus Naive Diversification"
"""

import numpy as np
import pandas as pd
import os
from src.evaluation.metrics import (
    compute_all_metrics,
    portfolio_turnover,
    format_metrics,
    herfindahl_index
)


def load_data(
    universe_path: str = "data/processed/universe.parquet",
    returns_path:  str = "data/processed/returns.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load universe and returns data."""
    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])

    returns = pd.read_parquet(returns_path)
    returns["date"] = pd.to_datetime(returns["date"])

    print(f"Universe: {universe['date'].nunique()} rebalancing dates")
    print(f"Returns : {len(returns):,} rows")
    return universe, returns


def get_monthly_returns(returns:        pd.DataFrame,
                        permnos:        list,
                        month:          pd.Timestamp) -> pd.Series:
    """
    Get actual returns for a set of stocks in a given month.

    Args:
        returns: full returns DataFrame
        permnos: list of eligible PERMNOs
        month:   the month we want returns for (end-of-month date)

    Returns:
        Series indexed by permno with 'ret' values.
        Missing stocks get return of 0 (conservative assumption).
    """
    mask = (
        (returns["date"] == month) &
        (returns["permno"].isin(permnos))
    )
    month_ret = returns.loc[mask].set_index("permno")["ret"]

    # Fill missing with 0 — conservative, avoids dropping stocks
    month_ret = month_ret.reindex(permnos, fill_value=0.0).fillna(0.0)
    return month_ret


def get_rf_for_month(returns:  pd.DataFrame,
                     month:    pd.Timestamp) -> float:
    """Get the risk-free rate for a given month."""
    mask = returns["date"] == month
    rf_vals = returns.loc[mask, "rf"].dropna()
    if len(rf_vals) == 0:
        return 0.0
    return float(rf_vals.iloc[0])


def compute_drift_weights(weights:      np.ndarray,
                          stock_returns: np.ndarray) -> np.ndarray:
    """
    Compute how weights drift after one month of returns.

    After holding for one month, each position grows by (1 + r_i).
    The new (drifted) weights are renormalized to sum to 1.

    This is the pre-rebalance weight vector used to compute turnover.

    Args:
        weights:       weight vector at start of month (sums to 1)
        stock_returns: realized returns for each stock that month

    Returns:
        Drifted weight vector (sums to 1)
    """
    stock_returns = np.nan_to_num(stock_returns, nan=0.0)
    drifted = weights * (1 + stock_returns)
    total = drifted.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)
    return drifted / total


def run_equal_weight(universe: pd.DataFrame,
                     returns:  pd.DataFrame,
                     gamma:    float = 0.003) -> pd.DataFrame:
    """
    Run the equal-weight strategy over all rebalancing dates.

    At each rebalancing date t:
    1. Get eligible stocks from universe
    2. Assign equal weights (1/N)
    3. Apply weights to next month's returns
    4. Compute turnover vs drifted weights from last period
    5. Record results

    Args:
        universe: DataFrame with columns [date, permno]
        returns:  DataFrame with columns [date, permno, ret, rf, excess_ret]
        gamma:    proportional transaction cost rate

    Returns:
        DataFrame with monthly results:
        [date, portfolio_ret, excess_ret, rf, turnover, cost, net_ret, hhi, n_stocks]
    """
    rebalance_dates = sorted(universe["date"].unique())

    # Get the next month's end-of-month date for each rebalancing date
    # Rebalancing date t is month-start (e.g. 2005-01-01)
    # Returns are recorded at end-of-month (e.g. 2005-01-31)
    # So we need to find the end-of-month date for month t
    all_return_dates = sorted(returns["date"].unique())

    results = []
    prev_weights = None
    prev_permnos = None

    for t in rebalance_dates:

        # Find the end-of-month date for this rebalancing month
        t_year  = t.year
        t_month = t.month
        apply_dates = [
            d for d in all_return_dates
            if d.year == t_year and d.month == t_month
        ]
        if len(apply_dates) == 0:
            continue
        apply_date = apply_dates[0]

        # Step 1: Get eligible stocks at t
        eligible = universe[universe["date"] == t]["permno"].tolist()
        N = len(eligible)
        if N == 0:
            continue

        # Step 2: Equal weights
        weights = np.ones(N) / N

        # Step 3: Get returns for month t
        month_ret = get_monthly_returns(returns, eligible, apply_date)
        stock_returns = month_ret.values

        # Portfolio gross return
        portfolio_gross = float(np.dot(weights, stock_returns))

        # Risk-free rate for this month
        rf = get_rf_for_month(returns, apply_date)

        # Excess return
        portfolio_excess = portfolio_gross - rf

        # Step 4: Compute turnover
        if prev_weights is not None and prev_permnos is not None:
            # Drift the previous weights using this month's returns
            # Only for stocks that were in both periods
            prev_returns = get_monthly_returns(
                returns, prev_permnos, apply_date
            )
            drifted = compute_drift_weights(
                prev_weights, prev_returns.values
            )
            # Map drifted weights to current universe
            # Stocks that left get weight 0, new stocks start at 0
            drifted_series = pd.Series(
                drifted, index=prev_permnos
            ).reindex(eligible, fill_value=0.0)
            drifted_array = drifted_series.values

            # Normalize (in case some stocks left universe)
            if drifted_array.sum() > 0:
                drifted_array = drifted_array / drifted_array.sum()

            turnover = portfolio_turnover(weights, drifted_array)
        else:
            # First period: buying from cash, turnover = 1.0
            turnover = 1.0

        # Step 5: Transaction cost and net return
        cost = gamma * turnover
        net_excess = portfolio_excess - cost

        # HHI concentration
        hhi = herfindahl_index(weights)

        results.append({
            "date"          : apply_date,
            "portfolio_ret" : portfolio_gross,
            "excess_ret"    : portfolio_excess,
            "rf"            : rf,
            "net_excess_ret": net_excess,
            "turnover"      : turnover,
            "cost"          : cost,
            "hhi"           : hhi,
            "n_stocks"      : N,
        })

        # Update previous weights for next iteration
        # Drift current weights to end of month for next period's turnover calc
        prev_weights = compute_drift_weights(weights, stock_returns)
        prev_permnos = eligible

    return pd.DataFrame(results)


def print_results(results: pd.DataFrame, gamma: float = 0.003) -> None:
    """Print performance summary."""
    clean     = results.dropna(subset=["excess_ret", "rf", "turnover"])
    excess    = clean["excess_ret"].values
    rf        = clean["rf"].values
    turnovers = clean["turnover"].values

    metrics = compute_all_metrics(excess, rf, turnovers, gamma)

    print("\n" + "="*50)
    print("EQUAL-WEIGHT (1/N) PORTFOLIO RESULTS")
    print("="*50)
    print(f"Period : {results['date'].min().date()} "
          f"to {results['date'].max().date()}")
    print(f"Months : {len(results)}")
    print(f"Avg N  : {results['n_stocks'].mean():.0f} stocks")
    print()
    print(f"Sharpe Ratio (net)     : {metrics['sharpe_net']:.4f}")
    print(f"Sharpe Ratio (gross)   : {metrics['sharpe_gross']:.4f}")
    print(f"Annualized Return (net): {metrics['ann_return_net']*100:.2f}%")
    print(f"Annualized Volatility  : {metrics['ann_vol']*100:.2f}%")
    print(f"Max Drawdown (net)     : {metrics['max_drawdown_net']*100:.2f}%")
    print(f"Avg Monthly Turnover   : {metrics['avg_turnover']*100:.2f}%")
    print(f"Avg Transaction Cost   : {metrics['avg_transaction_cost']*100:.4f}%")
    print(f"Avg HHI                : {results['hhi'].mean():.6f}")
    print(f"Theoretical min HHI    : {(1/results['n_stocks'].mean()):.6f}")
    print("="*50)


if __name__ == "__main__":
    universe, returns = load_data()

    print("\nRunning equal-weight strategy...")
    results = run_equal_weight(universe, returns)

    print_results(results)

    # Save results
    os.makedirs("results/benchmarks", exist_ok=True)
    results.to_parquet(
        "results/benchmarks/equal_weight.parquet", index=False
    )
    print("\nSaved to results/benchmarks/equal_weight.parquet")