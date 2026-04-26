import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics, TRANSACTION_COST
ESTIMATION_WINDOW   = 60


def get_monthly_returns(returns: pd.DataFrame,
                        permnos: list,
                        month:   pd.Timestamp) -> pd.Series:
    """Get actual returns for a set of stocks in a given month."""
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
    """Compute drifted weights after one month of returns."""
    stock_returns = np.nan_to_num(stock_returns, nan=0.0)
    drifted       = weights * (1 + stock_returns)
    total         = drifted.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)
    return drifted / total


# NOTE: cap_universe() is kept for reference but not used in the
# main experiment. All models now use the full eligible universe
# per supervisor feedback (April 2026).
def cap_universe(universe: pd.DataFrame,
                 returns:  pd.DataFrame,
                 top_n:    int = 200) -> pd.DataFrame:
    """At each rebalancing date, keep only the top N stocks by market cap."""
    returns           = returns.copy()
    returns["mktcap"] = returns["prc"].abs() * returns["shrout"] * 1000

    result_rows = []
    for date, group in universe.groupby("date"):
        permnos  = group["permno"].tolist()
        date_ret = returns[
            (returns["date"].dt.year  == date.year) &
            (returns["date"].dt.month == date.month) &
            (returns["permno"].isin(permnos))
        ][["permno", "mktcap"]].drop_duplicates("permno")

        top = date_ret.nlargest(top_n, "mktcap")["permno"].tolist()
        for p in top:
            result_rows.append({"date": date, "permno": p})

    return pd.DataFrame(result_rows)


def align_drifted_weights(prev_weights: np.ndarray,
                          prev_permnos: list,
                          curr_permnos: list) -> np.ndarray:
    """Align drifted weights from previous period to current universe.

    Stocks leaving the portfolio get weight 0.
    Stocks entering the portfolio get weight 0.
    Result is normalized to sum to 1.
    If all previous stocks left the universe, falls back to equal weight
    (treated as full portfolio replacement).
    """
    aligned = (pd.Series(prev_weights, index=prev_permnos)
               .reindex(curr_permnos, fill_value=0.0)
               .values)
    total = aligned.sum()
    if total > 0:
        aligned = aligned / total
    else:
        # All previous stocks left universe — treat as full replacement
        aligned = np.ones(len(curr_permnos)) / len(curr_permnos)
    return aligned

# ── Estimation ────────────────────────────────────────────────────────────────

def get_estimation_window(returns:        pd.DataFrame,
                          permnos:        list,
                          rebalance_date: pd.Timestamp) -> tuple:
    """
    Get expected returns and covariance matrix for the estimation window.

    Uses excess returns from [t-60, t-1] months (60-month rolling window).
    Drops stocks with any missing return in the window.

    Returns:
        mu:            expected excess returns vector (N,)
        sigma:         covariance matrix (N, N)
        valid_permnos: stocks with complete 60-month history
    """
    window_end   = rebalance_date - pd.DateOffset(days=1)
    window_start = rebalance_date - pd.DateOffset(months=ESTIMATION_WINDOW)

    mask = (
        (returns["date"] >= window_start) &
        (returns["date"] <= window_end) &
        (returns["permno"].isin(permnos))
    )
    window_data = returns[mask][["date", "permno", "excess_ret"]]

    ret_matrix    = window_data.pivot(
        index="date", columns="permno", values="excess_ret"
    ).dropna(axis=1)
    valid_permnos = ret_matrix.columns.tolist()

    if len(valid_permnos) < 2:
        return None, None, []

    ret_array = ret_matrix.values
    mu        = ret_array.mean(axis=0)
    sigma     = np.cov(ret_array, rowvar=False)
    sigma = sigma + 1e-4 * np.eye(len(valid_permnos))

    return mu, sigma, valid_permnos


def print_results(results: pd.DataFrame,
                  label:   str,
                  gamma:   float = TRANSACTION_COST,
                  show_theoretical_hhi: bool = False) -> None:
    """Print performance summary.

    Args:
        results:               DataFrame of monthly portfolio results
        label:                 display name for the model
        gamma:                 transaction cost rate
        show_theoretical_hhi:  if True, print 1/N theoretical HHI minimum
                               (only meaningful for equal-weight portfolios)
    """
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
    if show_theoretical_hhi:
        print(f"Theoretical min HHI    : {(1/results['n_stocks'].mean()):.6f}")
    print("="*50)