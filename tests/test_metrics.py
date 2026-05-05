"""
Unit tests for src/evaluation/metrics.py.
All tests use known inputs with hand-verifiable answers.
Run with: python3 -m pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    compute_cumulative_returns,
    portfolio_turnover,
    transaction_cost,
    net_return,
    herfindahl_index,
    compute_all_metrics,
    SQRT_12,
)


# --- fixtures ---

@pytest.fixture
def flat_returns():
    """60 months of constant 1% excess return — all math is trivial."""
    return np.full(60, 0.01)


@pytest.fixture
def mixed_returns():
    """Alternating +3% and -1% — known mean and std."""
    return np.tile([0.03, -0.01], 30)  # 60 months


@pytest.fixture
def declining_returns():
    """Steadily declining — known drawdown."""
    return np.array([-0.1, -0.1, -0.1, 0.05, 0.05])


@pytest.fixture
def equal_weights_5():
    """Equal weights for 5 stocks."""
    return np.full(5, 0.2)


# --- annualized_return ---

def test_annualized_return_flat(flat_returns):
    """Constant 1% monthly → 12% annualized."""
    result = annualized_return(flat_returns)
    assert abs(result - 0.12) < 1e-10


def test_annualized_return_zero():
    """Zero returns → zero annualized return."""
    result = annualized_return(np.zeros(60))
    assert abs(result) < 1e-10


def test_annualized_return_negative():
    """Negative mean → negative annualized return."""
    result = annualized_return(np.full(60, -0.005))
    assert abs(result - (-0.06)) < 1e-10


def test_annualized_return_mixed(mixed_returns):
    """Mean of [0.03, -0.01] = 0.01 → 12% annualized."""
    result = annualized_return(mixed_returns)
    assert abs(result - 0.12) < 1e-10


# --- annualized_volatility ---

def test_annualized_volatility_flat(flat_returns):
    """Constant returns → zero volatility."""
    result = annualized_volatility(flat_returns)
    assert abs(result) < 1e-10


def test_annualized_volatility_known():
    """
    Two-value series [0.02, -0.02] repeated.
    Verify annualization: result = std(ddof=1) × sqrt(12).
    """
    returns = np.tile([0.02, -0.02], 30)
    expected = np.std(returns, ddof=1) * SQRT_12
    result = annualized_volatility(returns)
    assert abs(result - expected) < 1e-10


def test_annualized_volatility_positive():
    """Volatility is always non-negative."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.01, 0.05, 60)
    assert annualized_volatility(returns) >= 0


# --- sharpe_ratio ---

def test_sharpe_ratio_flat(flat_returns):
    """Zero volatility → Sharpe returns 0.0 (not infinity)."""
    result = sharpe_ratio(flat_returns)
    assert abs(result) < 1e-10


def test_sharpe_ratio_known():
    """
    Known returns: mean=0.01/month, std=0.05/month.
    Ann return = 0.12, Ann vol = 0.05 × sqrt(12).
    Sharpe = 0.12 / (0.05 × sqrt(12)).
    """
    returns = np.array([0.01 + 0.05 * x for x in
                        (np.arange(60) - 29.5) / 29.5])
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    expected = ann_ret / ann_vol if ann_vol > 0 else 0.0
    result = sharpe_ratio(returns)
    assert abs(result - expected) < 1e-10


def test_sharpe_ratio_positive_for_positive_mean():
    """Positive mean excess return → positive Sharpe."""
    returns = np.tile([0.02, 0.01], 30)
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_negative_for_negative_mean():
    """Negative mean excess return → negative Sharpe."""
    returns = np.tile([-0.02, -0.01], 30)
    assert sharpe_ratio(returns) < 0


# --- sortino_ratio ---

def test_sortino_ratio_no_downside():
    """All positive returns → no downside → returns 0.0."""
    returns = np.full(60, 0.01)
    result = sortino_ratio(returns)
    assert abs(result) < 1e-10


def test_sortino_ratio_greater_than_sharpe():
    """
    Sortino >= Sharpe when returns are not symmetric.
    Downside deviation uses only negative returns,
    which is <= total std when positives exist.
    """
    returns = np.tile([0.03, -0.01], 30)
    sr = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    assert so >= sr


def test_sortino_ratio_known():
    """
    Returns: [0.02, -0.02] repeated.
    Downside returns: [-0.02] only.
    Downside vol = sqrt(mean([-0.02]^2)) × sqrt(12) = 0.02 × sqrt(12).
    Ann return = 0.0 (mean is 0) → Sortino = 0.0.
    """
    returns = np.tile([0.02, -0.02], 30)
    result = sortino_ratio(returns)
    assert abs(result) < 1e-10


# --- max_drawdown ---

def test_max_drawdown_no_loss():
    """Monotonically increasing → zero drawdown."""
    cumulative = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    result = max_drawdown(cumulative)
    assert abs(result) < 1e-10


def test_max_drawdown_known():
    """
    Cumulative: [1.0, 1.5, 1.0, 1.2].
    Peak at 1.5, trough at 1.0 → drawdown = (1.0 - 1.5) / 1.5 = -0.3333...
    """
    cumulative = np.array([1.0, 1.5, 1.0, 1.2])
    result = max_drawdown(cumulative)
    assert abs(result - (-1/3)) < 1e-10


def test_max_drawdown_full_loss():
    """Complete loss → drawdown = -1.0."""
    cumulative = np.array([1.0, 0.5, 0.0])
    result = max_drawdown(cumulative)
    assert abs(result - (-1.0)) < 1e-10


def test_max_drawdown_negative():
    """Max drawdown is always <= 0."""
    rng = np.random.default_rng(42)
    cumulative = np.cumprod(1 + rng.normal(0.01, 0.05, 60))
    assert max_drawdown(cumulative) <= 0.0


# --- compute_cumulative_returns ---

def test_cumulative_returns_known():
    """
    excess_ret = [0.05], rf = [0.01] → total = 0.06 → cumulative = [1.06].
    """
    excess = np.array([0.05])
    rf     = np.array([0.01])
    result = compute_cumulative_returns(excess, rf)
    assert abs(result[0] - 1.06) < 1e-10


def test_cumulative_returns_compounding():
    """
    Two months: total returns [0.1, 0.1] → cumulative = [1.1, 1.21].
    """
    excess = np.array([0.07, 0.07])
    rf     = np.array([0.03, 0.03])
    result = compute_cumulative_returns(excess, rf)
    assert abs(result[0] - 1.10) < 1e-10
    assert abs(result[1] - 1.21) < 1e-10


def test_cumulative_returns_always_positive():
    """Cumulative value should always be positive for reasonable returns."""
    excess = np.tile([0.01, -0.005], 30)
    rf     = np.full(60, 0.003)
    result = compute_cumulative_returns(excess, rf)
    assert (result > 0).all()


# --- portfolio_turnover ---

def test_turnover_no_change():
    """Identical weights → zero turnover."""
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(portfolio_turnover(w, w)) < 1e-10


def test_turnover_complete_change():
    """
    Move all weight from stock A to stock B.
    |0-1| + |1-0| = 2, divided by 2 = 1.0.
    """
    w_new = np.array([0.0, 1.0])
    w_old = np.array([1.0, 0.0])
    result = portfolio_turnover(w_new, w_old)
    assert abs(result - 1.0) < 1e-10


def test_turnover_known():
    """
    w_new = [0.5, 0.5], w_old = [0.3, 0.7].
    |0.5-0.3| + |0.5-0.7| = 0.4 → turnover = 0.2.
    """
    w_new = np.array([0.5, 0.5])
    w_old = np.array([0.3, 0.7])
    result = portfolio_turnover(w_new, w_old)
    assert abs(result - 0.2) < 1e-10


def test_turnover_between_zero_and_one():
    """Turnover is always between 0 and 1."""
    rng   = np.random.default_rng(42)
    w_new = rng.dirichlet(np.ones(10))
    w_old = rng.dirichlet(np.ones(10))
    result = portfolio_turnover(w_new, w_old)
    assert 0.0 <= result <= 1.0


# --- transaction_cost ---

def test_transaction_cost_default_gamma():
    """10% turnover × 0.3% = 0.03%."""
    result = transaction_cost(0.1)
    assert abs(result - 0.0003) < 1e-10


def test_transaction_cost_zero_turnover():
    """Zero turnover → zero cost."""
    assert abs(transaction_cost(0.0)) < 1e-10


def test_transaction_cost_custom_gamma():
    """Custom gamma: 20% turnover × 0.5% = 0.1%."""
    result = transaction_cost(0.2, gamma=0.005)
    assert abs(result - 0.001) < 1e-10


# --- net_return ---

def test_net_return_known():
    """
    gross = 0.05, turnover = 0.1, gamma = 0.003.
    cost = 0.003 × 0.1 = 0.0003 → net = 0.0497.
    """
    result = net_return(0.05, 0.1)
    assert abs(result - 0.0497) < 1e-10


def test_net_return_less_than_gross():
    """Net return is always <= gross return."""
    assert net_return(0.05, 0.1) <= 0.05


# --- herfindahl_index ---

def test_hhi_equal_weights(equal_weights_5):
    """Equal weights of 1/N → HHI = 1/N."""
    result = herfindahl_index(equal_weights_5)
    assert abs(result - 0.2) < 1e-10


def test_hhi_single_stock():
    """All weight in one stock → HHI = 1.0."""
    w = np.array([1.0, 0.0, 0.0])
    assert abs(herfindahl_index(w) - 1.0) < 1e-10


def test_hhi_minimum_for_n_stocks():
    """HHI is minimized at 1/N for equal weights."""
    n = 20
    w_equal = np.full(n, 1/n)
    rng     = np.random.default_rng(42)
    w_random = rng.dirichlet(np.ones(n))
    assert herfindahl_index(w_equal) <= herfindahl_index(w_random)


def test_hhi_between_zero_and_one():
    """HHI is always between 0 and 1."""
    rng = np.random.default_rng(42)
    w   = rng.dirichlet(np.ones(10))
    result = herfindahl_index(w)
    assert 0.0 <= result <= 1.0


# --- compute_all_metrics ---

def test_compute_all_metrics_keys():
    """Output contains all expected metric keys."""
    rng       = np.random.default_rng(42)
    excess    = rng.normal(0.008, 0.05, 60)
    rf        = np.full(60, 0.003)
    turnovers = np.full(60, 0.1)

    metrics = compute_all_metrics(excess, rf, turnovers)

    expected_keys = [
        "sharpe_gross", "sortino_gross", "ann_return_gross",
        "ann_vol", "max_drawdown_gross",
        "sharpe_net", "sortino_net", "ann_return_net",
        "max_drawdown_net", "avg_turnover",
        "avg_transaction_cost", "total_cost",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"


def test_compute_all_metrics_net_less_than_gross():
    """Net Sharpe should be <= gross Sharpe when turnover > 0."""
    rng       = np.random.default_rng(42)
    excess    = rng.normal(0.008, 0.05, 60)
    rf        = np.full(60, 0.003)
    turnovers = np.full(60, 0.1)

    metrics = compute_all_metrics(excess, rf, turnovers)
    assert metrics["sharpe_net"] <= metrics["sharpe_gross"]


def test_compute_all_metrics_zero_turnover_equals_gross():
    """Zero turnover → net metrics equal gross metrics."""
    rng       = np.random.default_rng(42)
    excess    = rng.normal(0.008, 0.05, 60)
    rf        = np.full(60, 0.003)
    turnovers = np.zeros(60)

    metrics = compute_all_metrics(excess, rf, turnovers)
    assert abs(metrics["sharpe_net"] - metrics["sharpe_gross"]) < 1e-10


def test_compute_all_metrics_no_nan():
    """No NaN values in output for valid inputs."""
    rng       = np.random.default_rng(42)
    excess    = rng.normal(0.008, 0.05, 60)
    rf        = np.full(60, 0.003)
    turnovers = np.full(60, 0.05)

    metrics = compute_all_metrics(excess, rf, turnovers)
    for key, val in metrics.items():
        assert not np.isnan(val), f"NaN found in metric: {key}"
