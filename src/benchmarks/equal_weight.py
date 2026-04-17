"""
Benchmark: Equal-Weight Portfolio (1/N)

Two versions as per updated Step 12 methodology:

(a) 1/N Full Universe:
    - Equal weight across all eligible stocks (~867/month)
    - Primary naive benchmark, consistent with DeMiguel et al. (2009)
    - Hardest comparator — maximum diversification

(b) 1/N Top-200:
    - Equal weight across top 200 stocks by market cap
    - Same universe as GA and MVO
    - Secondary benchmark for direct comparison
    - Isolates effect of weight optimization vs naive allocation

Performance decomposition enabled by running both:
    1/N Full → 1/N Top-200:   effect of universe restriction
    1/N Top-200 → GA:         effect of weight optimization
    GA → Constrained MVO:     effect of cardinality constraint

Reference: DeMiguel et al. (2009) "Optimal Versus Naive Diversification"
"""

import numpy as np
import pandas as pd
import os
from src.evaluation.metrics import (
    compute_all_metrics,
    portfolio_turnover,
    herfindahl_index,
    TRANSACTION_COST
)


# ── Data Loading ──────────────────────────────────────────────────────────────

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


# ── Universe Capping ──────────────────────────────────────────────────────────

def cap_universe(universe: pd.DataFrame,
                 returns:  pd.DataFrame,
                 top_n:    int = 200) -> pd.DataFrame:
    """
    At each rebalancing date, keep only the top N stocks by market cap.

    Market cap = |prc| × shrout × 1000 (in millions).
    Reduces universe from ~867 to top_n stocks for direct comparison
    with MVO and GA benchmarks.

    Args:
        universe: full eligible universe DataFrame [date, permno]
        returns:  returns DataFrame with prc and shrout columns
        top_n:    number of stocks to keep per rebalancing date

    Returns:
        Capped universe DataFrame [date, permno]
    """
    returns = returns.copy()
    returns["mktcap"] = returns["prc"].abs() * returns["shrout"] * 1000

    result_rows = []
    for date, group in universe.groupby("date"):
        permnos = group["permno"].tolist()

        date_ret = returns[
            (returns["date"].dt.year  == date.year) &
            (returns["date"].dt.month == date.month) &
            (returns["permno"].isin(permnos))
        ][["permno", "mktcap"]].drop_duplicates("permno")

        top = date_ret.nlargest(top_n, "mktcap")["permno"].tolist()
        for p in top:
            result_rows.append({"date": date, "permno": p})

    return pd.DataFrame(result_rows)


# ── Portfolio Helpers ─────────────────────────────────────────────────────────

def get_monthly_returns(returns: pd.DataFrame,
                        permnos: list,
                        month:   pd.Timestamp) -> pd.Series:
    """
    Get actual returns for a set of stocks in a given month.

    Missing stocks get return of 0 (conservative assumption).
    Existing NaN returns also filled with 0.
    """
    mask = (
        (returns["date"] == month) &
        (returns["permno"].isin(permnos))
    )
    month_ret = returns.loc[mask].set_index("permno")["ret"]
    return month_ret.reindex(permnos, fill_value=0.0).fillna(0.0)


def get_rf_for_month(returns: pd.DataFrame,
                     month:   pd.Timestamp) -> float:
    """Get the risk-free rate for a given month."""
    rf_vals = returns.loc[returns["date"] == month, "rf"].dropna()
    return float(rf_vals.iloc[0]) if len(rf_vals) > 0 else 0.0


def compute_drift_weights(weights:       np.ndarray,
                          stock_returns: np.ndarray) -> np.ndarray:
    """
    Compute how weights drift after one month of returns.

    After holding for one month, each position grows by (1 + r_i).
    The drifted weights are renormalized to sum to 1.
    Used to compute turnover at the next rebalancing date.
    """
    stock_returns = np.nan_to_num(stock_returns, nan=0.0)
    drifted = weights * (1 + stock_returns)
    total   = drifted.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)
    return drifted / total


# ── Main Runner ───────────────────────────────────────────────────────────────

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
            drifted_series = (pd.Series(drifted, index=prev_permnos)
                              .reindex(eligible, fill_value=0.0))
            drifted_array  = drifted_series.values
            if drifted_array.sum() > 0:
                drifted_array = drifted_array / drifted_array.sum()
            turnover = portfolio_turnover(weights, drifted_array)
        else:
            turnover = 1.0  # First period: full investment from cash

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


# ── Results Printing ──────────────────────────────────────────────────────────

def print_results(results: pd.DataFrame,
                  label:   str,
                  gamma:   float = TRANSACTION_COST) -> None:
    """Print performance summary."""
    clean     = results.dropna(subset=["excess_ret", "rf", "turnover"])
    excess    = clean["excess_ret"].values
    rf        = clean["rf"].values
    turnovers = clean["turnover"].values

    metrics = compute_all_metrics(excess, rf, turnovers, gamma)

    print("\n" + "="*50)
    print(f"{label} RESULTS")
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


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    universe_full, returns = load_data()

    # ── Run 1: Full universe (~867 stocks) ────────────────────────────────────
    print("\nRunning 1/N — Full Universe...")
    results_full = run_equal_weight(universe_full, returns)
    print_results(results_full, "1/N FULL UNIVERSE (~867 stocks)")

    # ── Run 2: Top-200 universe ───────────────────────────────────────────────
    print("\nCapping universe to top 200 stocks by market cap...")
    universe_200 = cap_universe(universe_full, returns, top_n=200)
    print(f"Capped: {universe_200.groupby('date').size().mean():.0f} "
          f"stocks/month on average")

    print("\nRunning 1/N — Top-200 Universe...")
    results_200 = run_equal_weight(universe_200, returns)
    print_results(results_200, "1/N TOP-200 UNIVERSE")

    # ── Save both ─────────────────────────────────────────────────────────────
    os.makedirs("results/benchmarks", exist_ok=True)

    results_full.to_parquet(
        "results/benchmarks/equal_weight_full.parquet", index=False
    )
    results_200.to_parquet(
        "results/benchmarks/equal_weight_top200.parquet", index=False
    )
    print("\nSaved:")
    print("  results/benchmarks/equal_weight_full.parquet")
    print("  results/benchmarks/equal_weight_top200.parquet")