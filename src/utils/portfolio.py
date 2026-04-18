import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics, TRANSACTION_COST


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


def cap_universe(universe: pd.DataFrame,
                 returns:  pd.DataFrame,
                 top_n:    int = 200) -> pd.DataFrame:
    """At each rebalancing date, keep only the top N stocks by market cap."""
    returns       = returns.copy()
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
    """Align drifted weights from previous period to current universe."""
    aligned = (pd.Series(prev_weights, index=prev_permnos)
               .reindex(curr_permnos, fill_value=0.0)
               .values)
    total = aligned.sum()
    if total > 0:
        aligned = aligned / total
    return aligned


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
