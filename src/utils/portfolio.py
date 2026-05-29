import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.evaluation.metrics import compute_all_metrics, TRANSACTION_COST
ESTIMATION_WINDOW = 60


def get_monthly_returns(returns: pd.DataFrame, permnos: list, month: pd.Timestamp) -> pd.Series:
    # Missing application-month returns are filled with 0.0, a backtest assumption.
    mask = (
        (returns["date"] == month) &
        (returns["permno"].isin(permnos))
    )
    month_ret = returns.loc[mask].set_index("permno")["ret"]
    return month_ret.reindex(permnos, fill_value=0.0).fillna(0.0)


def get_rf_for_month(returns: pd.DataFrame, month: pd.Timestamp) -> float:
    rf_vals = returns.loc[returns["date"] == month, "rf"].dropna()
    return float(rf_vals.iloc[0]) if len(rf_vals) > 0 else 0.0


def compute_drift_weights(weights: np.ndarray, stock_returns: np.ndarray) -> np.ndarray:
    stock_returns = np.nan_to_num(stock_returns, nan=0.0)
    drifted = weights * (1 + stock_returns)
    total = drifted.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)
    return drifted / total


def align_drifted_weights(prev_weights: np.ndarray, prev_permnos: list, curr_permnos: list) -> np.ndarray:
    # New or departed stocks get zero weight.
    aligned = (pd.Series(prev_weights, index=prev_permnos)
               .reindex(curr_permnos, fill_value=0.0)
               .values)
    total = aligned.sum()
    if total > 0:
        aligned = aligned / total
    else:
        # All previous stocks left the universe.
        aligned = np.ones(len(curr_permnos)) / len(curr_permnos)
    return aligned


def get_estimation_window(returns: pd.DataFrame, permnos: list, rebalance_date: pd.Timestamp) -> tuple:
    # Uses [t-60, t-1] and drops stocks with incomplete history.
    window_end = rebalance_date - pd.DateOffset(days=1)
    window_start = rebalance_date - pd.DateOffset(months=ESTIMATION_WINDOW)

    mask = (
        (returns["date"] >= window_start) &
        (returns["date"] <= window_end) &
        (returns["permno"].isin(permnos))
    )
    window_data = returns[mask][["date", "permno", "excess_ret"]]

    ret_matrix = window_data.pivot(
        index="date", columns="permno", values="excess_ret"
    ).dropna(axis=1)
    valid_permnos = ret_matrix.columns.tolist()

    if len(valid_permnos) < 2:
        return None, None, []

    ret_array = ret_matrix.values
    mu = ret_array.mean(axis=0)
    lw = LedoitWolf()
    lw.fit(ret_array)
    sigma = lw.covariance_

    return mu, sigma, valid_permnos


def print_results(results: pd.DataFrame, label: str, gamma: float = TRANSACTION_COST,
                  show_theoretical_hhi: bool = False) -> None:
    # show_theoretical_hhi prints the 1/K lower bound for equal-weight.
    clean = results.dropna(subset=["excess_ret", "rf", "turnover"])
    excess = clean["excess_ret"].values
    rf = clean["rf"].values
    turnovers = clean["turnover"].values

    metrics = compute_all_metrics(excess, rf, turnovers, gamma)

    print(f"\n{label}")
    print(f"Period: {results['date'].min().date()} "
          f"to {results['date'].max().date()}")
    print(f"Months: {len(results)}")
    print(f"Avg N: {results['n_stocks'].mean():.0f} stocks")
    print()
    print(f"Sharpe (net): {metrics['sharpe_net']:.4f}")
    print(f"Sharpe (gross): {metrics['sharpe_gross']:.4f}")
    print(f"Annualized Return (net): {metrics['ann_return_net']*100:.2f}%")
    print(f"Annualized Volatility: {metrics['ann_vol']*100:.2f}%")
    print(f"Max Drawdown (net): {metrics['max_drawdown_net']*100:.2f}%")
    print(f"Avg Monthly Turnover: {metrics['avg_turnover']*100:.2f}%")
    print(f"Avg Transaction Cost: {metrics['avg_transaction_cost']*100:.4f}%")
    print(f"Avg HHI: {results['hhi'].mean():.6f}")
    if show_theoretical_hhi:
        print(f"Theoretical min HHI: {(1/results['n_stocks'].mean()):.6f}")
