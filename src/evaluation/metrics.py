"""Evaluation Metrics

All performance metrics used to evaluate portfolio performance.
All returns are assumed to be monthly. Annualization uses:
- Returns:    × 12
- Volatility: × √12
- Sharpe:     annualized excess return / annualized volatility

References:
- Sharpe (1994): Sharpe ratio
- Sortino & Price (1994): Sortino ratio  
- DeMiguel et al. (2009): turnover and transaction cost definitions
"""
import pandas as pd 
import numpy as np
from typing import Optional

ANNUALIZATION_FACTOR = 12 # monthly returns to annual returns
SQRT_12 = np.sqrt(ANNUALIZATION_FACTOR)
TRANSACTION_COST = 0.003 # Step 7

def annualized_return(excess_returns: np.ndarray) -> float:
    """
    Annualized mean excess return.

    Args:
        excess_returns: array of monthly excess returns (ret - rf)

    Returns:
        Annualized mean excess return (scalar)
    """
    return float(np.mean(excess_returns) * ANNUALIZATION_FACTOR)


def annualized_volatility(excess_returns: np.ndarray) -> float:
    """
    Annualized standard deviation of excess returns.

    Args:
        excess_returns: array of monthly excess returns

    Returns:
        Annualized volatility (scalar)
    """
    return float(np.std(excess_returns, ddof=1) * SQRT_12)


def sharpe_ratio(excess_returns: np.ndarray) -> float:
    """
    Annualized Sharpe ratio.

    Sharpe = annualized mean excess return / annualized volatility

    Note: computed on excess returns (already rf-subtracted),
    so no additional rf subtraction needed here.

    Args:
        excess_returns: array of monthly excess returns (ret - rf)

    Returns:
        Annualized Sharpe ratio (scalar). Returns 0.0 if vol is zero.
    """
    vol = annualized_volatility(excess_returns)
    if vol == 0:
        return 0.0
    return annualized_return(excess_returns) / vol


def sortino_ratio(excess_returns: np.ndarray) -> float:
    """
    Annualized Sortino ratio.

    Sortino = annualized mean excess return / annualized downside deviation
    Downside deviation uses only returns below zero (below risk-free rate).

    Args:
        excess_returns: array of monthly excess returns (ret - rf)

    Returns:
        Annualized Sortino ratio (scalar). Returns 0.0 if downside vol is zero.
    """
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_vol = float(np.sqrt(np.mean(downside ** 2)) * SQRT_12)
    if downside_vol == 0:
        return 0.0
    return annualized_return(excess_returns) / downside_vol


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Maximum drawdown — largest peak-to-trough loss.

    Args:
        cumulative_returns: array of cumulative portfolio values
                           (e.g. starting at 1.0, growing over time)

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.45 = -45%)
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(np.min(drawdown))


def compute_cumulative_returns(excess_returns: np.ndarray,
                                rf: np.ndarray) -> np.ndarray:
    """
    Compute cumulative portfolio value from monthly excess returns.

    Total return = excess return + rf
    Cumulative value starts at 1.0 and compounds monthly.

    Args:
        excess_returns: array of monthly excess returns
        rf:             array of monthly risk-free rates (same length)

    Returns:
        Array of cumulative portfolio values starting at 1.0
    """
    total_returns = excess_returns + rf
    return np.cumprod(1 + total_returns)


def portfolio_turnover(weights_current: np.ndarray,
                       weights_previous: np.ndarray) -> float:
    """
    Portfolio turnover between two rebalancing periods.

    Turnover = sum(|w_t - w_{t-1}|) / 2

    where w_{t-1} is the pre-rebalance weight AFTER drift
    (not the previous target weight).

    Args:
        weights_current:  target weights at time t (sums to 1)
        weights_previous: pre-rebalance weights at t (after drift, sums to 1)

    Returns:
        Turnover as a fraction between 0 and 1
    """
    return float(np.sum(np.abs(weights_current - weights_previous)) / 2)


def transaction_cost(turnover: float,
                     gamma: float = TRANSACTION_COST) -> float:
    """
    Proportional transaction cost.

    Cost = γ × turnover

    Args:
        turnover: portfolio turnover (fraction)
        gamma:    proportional cost per unit traded (default 0.3%)

    Returns:
        Transaction cost as a fraction of portfolio value
    """
    return float(gamma * turnover)


def net_return(gross_return: float,
               turnover:     float,
               gamma:        float = TRANSACTION_COST) -> float:
    """
    Net portfolio return after transaction costs.

    net_return = gross_return - γ × turnover

    Args:
        gross_return: gross portfolio return for the period
        turnover:     portfolio turnover for the period
        gamma:        proportional transaction cost

    Returns:
        Net return after costs
    """
    return gross_return - transaction_cost(turnover, gamma)


def herfindahl_index(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index (HHI) — portfolio concentration.

    HHI = sum(w_i^2)

    Interpretation:
    - HHI = 1/N for equal-weight portfolio (minimum concentration)
    - HHI = 1.0  for single-asset portfolio (maximum concentration)

    Args:
        weights: array of portfolio weights (sums to 1, all >= 0)

    Returns:
        HHI value between 1/N and 1.0
    """
    return float(np.sum(weights ** 2))


# ── Summary Function ──────────────────────────────────────────────────────────

def compute_all_metrics(excess_returns:  np.ndarray,
                        rf:              np.ndarray,
                        turnovers:       np.ndarray,
                        gamma:           float = TRANSACTION_COST
                        ) -> dict:
    """
    Compute all evaluation metrics for a portfolio return series.

    This is the main function called by the results tables.
    All inputs should cover the full out-of-sample period.

    Args:
        excess_returns: monthly excess returns (ret - rf), length T
        rf:             monthly risk-free rates, length T
        turnovers:      monthly portfolio turnovers, length T
        gamma:          proportional transaction cost rate

    Returns:
        Dictionary of annualized performance metrics
    """
    # Net returns after transaction costs
    costs = np.array([transaction_cost(to, gamma) for to in turnovers])
    net_excess = excess_returns - costs

    # Cumulative returns for drawdown
    cumulative = compute_cumulative_returns(excess_returns, rf)
    cumulative_net = compute_cumulative_returns(net_excess, rf)

    return {
        # Gross metrics (before costs)
        "sharpe_gross"        : sharpe_ratio(excess_returns),
        "sortino_gross"       : sortino_ratio(excess_returns),
        "ann_return_gross"    : annualized_return(excess_returns),
        "ann_vol"             : annualized_volatility(excess_returns),
        "max_drawdown_gross"  : max_drawdown(cumulative),

        # Net metrics (after costs) — PRIMARY REPORTED METRICS
        "sharpe_net"          : sharpe_ratio(net_excess),
        "sortino_net"         : sortino_ratio(net_excess),
        "ann_return_net"      : annualized_return(net_excess),
        "max_drawdown_net"    : max_drawdown(cumulative_net),

        # Trading activity
        "avg_turnover"        : float(np.mean(turnovers)),
        "avg_transaction_cost": float(np.mean(costs)),
        "total_cost"          : float(np.sum(costs)),
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def format_metrics(metrics: dict) -> pd.DataFrame:
    """
    Format metrics dictionary into a readable DataFrame for thesis Table 1.

    Args:
        metrics: output of compute_all_metrics()

    Returns:
        Single-row DataFrame with formatted metric names
    """
    display_names = {
        "ann_return_net"      : "Annualized Return (net)",
        "ann_vol"             : "Annualized Volatility",
        "sharpe_net"          : "Sharpe Ratio (net)",
        "sortino_net"         : "Sortino Ratio (net)",
        "max_drawdown_net"    : "Max Drawdown (net)",
        "avg_turnover"        : "Avg Monthly Turnover",
        "avg_transaction_cost": "Avg Transaction Cost",
        "sharpe_gross"        : "Sharpe Ratio (gross)",
        "ann_return_gross"    : "Annualized Return (gross)",
        "max_drawdown_gross"  : "Max Drawdown (gross)",
    }
    return pd.DataFrame(
        {display_names.get(k, k): [round(v, 4)]
         for k, v in metrics.items()
         if k in display_names}
    )