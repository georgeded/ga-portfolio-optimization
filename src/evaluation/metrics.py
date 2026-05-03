"""
Monthly returns throughout. Annualization: returns × 12, vol × √12, Sharpe = ann. excess / ann. vol.
"""

import pandas as pd
import numpy as np

ANNUALIZATION_FACTOR = 12
SQRT_12              = np.sqrt(ANNUALIZATION_FACTOR)
TRANSACTION_COST     = 0.003  # γ = 0.3%


def annualized_return(excess_returns: np.ndarray) -> float:
    """Annualized mean excess return (μ̄ × 12)."""
    return float(np.mean(excess_returns) * ANNUALIZATION_FACTOR)


def annualized_volatility(excess_returns: np.ndarray) -> float:
    """
    Annualized std of excess returns (ddof=1, × √12).
    Clipped to 0.0 to guard against floating-point noise.
    """
    vol = float(np.std(excess_returns, ddof=1) * SQRT_12)
    return max(vol, 0.0)


def sharpe_ratio(excess_returns: np.ndarray) -> float:
    """
    Input is excess returns — no additional rf subtraction.
    Returns 0.0 when vol < 1e-10.
    """
    vol = annualized_volatility(excess_returns)
    if vol < 1e-10:
        return 0.0
    return annualized_return(excess_returns) / vol


def sortino_ratio(excess_returns: np.ndarray) -> float:
    """
    Downside deviation uses returns below zero (below risk-free rate).
    Returns 0.0 if no downside returns or downside vol < 1e-10.
    """
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_vol = float(np.sqrt(np.mean(downside ** 2)) * SQRT_12)
    if downside_vol < 1e-10:
        return 0.0
    return annualized_return(excess_returns) / downside_vol


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Largest peak-to-trough loss. Returns negative fraction (e.g. −0.45 = −45%)."""
    peak     = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(np.min(drawdown))


def compute_cumulative_returns(excess_returns: np.ndarray,
                               rf:             np.ndarray) -> np.ndarray:
    """Compound monthly (excess_ret + rf) into a cumulative value series starting at 1.0."""
    total_returns = excess_returns + rf
    return np.cumprod(1 + total_returns)


def portfolio_turnover(weights_current:  np.ndarray,
                       weights_previous: np.ndarray) -> float:
    """
    Σ|w_t − w_{t-1}| / 2 where w_{t-1} is the post-drift weight before rebalancing.
    """
    return float(np.sum(np.abs(weights_current - weights_previous)) / 2)


def transaction_cost(turnover: float,
                     gamma:    float = TRANSACTION_COST) -> float:
    """γ × turnover."""
    return float(gamma * turnover)


def net_return(gross_return: float,
               turnover:     float,
               gamma:        float = TRANSACTION_COST) -> float:
    """gross_return − γ × turnover."""
    return gross_return - transaction_cost(turnover, gamma)


def herfindahl_index(weights: np.ndarray) -> float:
    """HHI = Σw_i². Range: [1/K, 1.0] for equal-weight to single-asset."""
    return float(np.sum(weights ** 2))


def compute_all_metrics(excess_returns: np.ndarray,
                        rf:             np.ndarray,
                        turnovers:      np.ndarray,
                        gamma:          float = TRANSACTION_COST
                        ) -> dict:
    """Returns annualized gross and net performance metrics as a dict."""
    costs      = np.array([transaction_cost(to, gamma) for to in turnovers])
    net_excess = excess_returns - costs

    cumulative     = compute_cumulative_returns(excess_returns, rf)
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

        "avg_turnover"        : float(np.mean(turnovers)),
        "avg_transaction_cost": float(np.mean(costs)),
        "total_cost"          : float(np.sum(costs)),
    }


def format_metrics(metrics: dict) -> pd.DataFrame:
    """Format compute_all_metrics() output into a single-row DataFrame for Table 1."""
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