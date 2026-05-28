import numpy as np
import pandas as pd

import src.benchmarks.mvo as mvo
from src.data.universe import build_monthly_universe
from src.evaluation.metrics import portfolio_turnover
from src.utils.portfolio import (
    align_drifted_weights,
    compute_drift_weights,
    get_estimation_window,
    get_monthly_returns,
)


# ---------------------------------------------------------------------------
# compute_drift_weights
# ---------------------------------------------------------------------------

def test_compute_drift_weights_total_loss_falls_back_to_equal_weight():
    """When the whole portfolio goes to zero, drift weights should fall back to equal weight."""
    weights = np.array([0.4, 0.6])
    stock_returns = np.array([-1.0, -1.0])

    result = compute_drift_weights(weights, stock_returns)

    np.testing.assert_allclose(result, np.array([0.5, 0.5]), atol=1e-10)


def test_compute_drift_weights_sums_to_one():
    """Drifted weights should always sum to 1 for any non-catastrophic return vector."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        weights = rng.dirichlet(np.ones(10))
        returns = rng.uniform(-0.2, 0.3, 10)
        result = compute_drift_weights(weights, returns)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12,
                                   err_msg="drifted weights do not sum to 1")


# ---------------------------------------------------------------------------
# align_drifted_weights
# ---------------------------------------------------------------------------

def test_align_drifted_weights_all_previous_holdings_leave_universe():
    """When all previous holdings leave the universe, the result should be equal weight over the new universe."""
    prev_weights = np.array([0.25, 0.75])
    prev_permnos = [101, 102]
    curr_permnos = [201, 202, 203, 204]

    result = align_drifted_weights(prev_weights, prev_permnos, curr_permnos)

    np.testing.assert_allclose(result, np.full(4, 0.25), atol=1e-10)


# ---------------------------------------------------------------------------
# portfolio_turnover
# ---------------------------------------------------------------------------

def test_portfolio_turnover_exiting_positions():
    """An asset held at 0.5 that exits completely contributes 0.5 to the
    half-sum turnover, matching the weight that enters to replace it.
    """
    # Stock A exits (0.5 -> 0), Stock B enters (0 -> 0.5), Stock C unchanged (0.5 -> 0.5)
    w_old = np.array([0.5, 0.0, 0.5])
    w_new = np.array([0.0, 0.5, 0.5])
    # |0.5| + |0.5| + |0| = 1.0 -> /2 = 0.5
    result = portfolio_turnover(w_new, w_old)
    np.testing.assert_allclose(result, 0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# build_monthly_universe
# ---------------------------------------------------------------------------

def test_build_monthly_universe_requires_market_cap_and_full_history():
    """Eligible stocks must have market cap >= 2B and a complete 60-month return history."""
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    rows = []
    for permno, mktcap in [(1, 3.0), (2, 3.0), (3, 1.5)]:
        for i, date in enumerate(dates):
            ret = np.nan if permno == 2 and i == 0 else 0.01
            rows.append({
                "date": date,
                "permno": permno,
                "ret": ret,
                "mktcap": mktcap,
            })
    df = pd.DataFrame(rows)

    result = build_monthly_universe(df, pd.Timestamp("2005-01-01"))

    assert result.tolist() == [1]


# ---------------------------------------------------------------------------
# get_estimation_window
# ---------------------------------------------------------------------------

def test_get_estimation_window_returns_mu_sigma_and_valid_permnos():
    """Verifies mu and sigma are computed from the complete 60-month return matrix.
    Sigma uses Ledoit-Wolf shrinkage so only structural properties are checked.
    """
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    stock_1 = np.full(60, 0.01)
    stock_2 = np.linspace(-0.02, 0.04, 60)
    rows = []
    for date, r1, r2 in zip(dates, stock_1, stock_2):
        rows.append({"date": date, "permno": 1, "excess_ret": r1})
        rows.append({"date": date, "permno": 2, "excess_ret": r2})
        rows.append({"date": date, "permno": 3, "excess_ret": np.nan})
    returns = pd.DataFrame(rows)

    mu, sigma, valid_permnos = get_estimation_window(
        returns, [1, 2, 3], pd.Timestamp("2005-01-01")
    )
    expected_matrix = np.column_stack([stock_1, stock_2])
    expected_mu = expected_matrix.mean(axis=0)

    assert valid_permnos == [1, 2]
    np.testing.assert_allclose(mu, expected_mu, atol=1e-10)

    assert sigma.shape == (2, 2)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(sigma)
    assert np.all(eigenvalues > 0), f"sigma not positive definite: {eigenvalues}"


def test_get_estimation_window_excludes_application_month():
    """The window ends strictly before the rebalance date, so the application month
    return is never used to estimate mu. Use an extreme return (50%) in the
    application month so leakage would be obvious in the returned mu.
    """
    window_dates = pd.date_range("2000-01-31", periods=60, freq="ME")  # Jan 2000 - Dec 2004
    rows = []
    for date in window_dates:
        rows.append({"date": date, "permno": 1, "excess_ret": 0.0})
        rows.append({"date": date, "permno": 2, "excess_ret": 0.0})

    # application month: give stock 1 an extreme return that would shift mu noticeably
    rows.append({"date": pd.Timestamp("2005-01-31"), "permno": 1, "excess_ret": 0.5})
    rows.append({"date": pd.Timestamp("2005-01-31"), "permno": 2, "excess_ret": -0.5})
    returns = pd.DataFrame(rows)

    mu, sigma, valid_permnos = get_estimation_window(
        returns, [1, 2], pd.Timestamp("2005-01-01")
    )

    assert valid_permnos == [1, 2]
    # Window is all zeros; if application month leaked, mu would be non-zero
    np.testing.assert_allclose(mu, [0.0, 0.0], atol=1e-10,
                               err_msg="application-month return leaked into estimation window")


def test_get_estimation_window_excludes_data_older_than_60_months():
    """The window starts 60 months before the rebalance date, so t-61 data is excluded.
    Put an extreme return at t-61 and zeros in t-60 to t-1; if the old data leaked
    in, mu would be noticeably non-zero.
    """
    # t-61: one month before the window opens
    old_date = pd.Timestamp("1999-12-31")
    window_dates = pd.date_range("2000-01-31", periods=60, freq="ME")  # Jan 2000 - Dec 2004

    rows = []
    # extreme return at t-61
    rows.append({"date": old_date, "permno": 1, "excess_ret": 0.5})
    rows.append({"date": old_date, "permno": 2, "excess_ret": -0.5})
    # zeros in the actual window
    for date in window_dates:
        rows.append({"date": date, "permno": 1, "excess_ret": 0.0})
        rows.append({"date": date, "permno": 2, "excess_ret": 0.0})
    returns = pd.DataFrame(rows)

    mu, sigma, valid_permnos = get_estimation_window(
        returns, [1, 2], pd.Timestamp("2005-01-01")
    )

    assert valid_permnos == [1, 2]
    # Window contains only zeros; t-61 must not have been included
    np.testing.assert_allclose(mu, [0.0, 0.0], atol=1e-10,
                               err_msg="data older than 60 months leaked into estimation window")


# ---------------------------------------------------------------------------
# optimize_mvo fallback
# ---------------------------------------------------------------------------

def test_optimize_mvo_solver_failure_falls_back_to_equal_weight(monkeypatch):
    """Verifies optimizer failure returns equal weights, computed as 1/N after the fallback path."""
    def fail_minimize(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(mvo, "minimize", fail_minimize)
    mvo._solver_failures = 0
    mu = np.array([0.01, 0.02, 0.03])
    sigma = np.eye(3)

    result = mvo.optimize_mvo(
        mu, sigma, 0.0, 1.0, rng=np.random.default_rng(42)
    )

    np.testing.assert_allclose(result, np.full(3, 1 / 3), atol=1e-10)
    assert mvo._solver_failures == 1
