import numpy as np
import pandas as pd

import src.benchmarks.mvo as mvo
from src.data.universe import build_monthly_universe
from src.utils.portfolio import (
    align_drifted_weights,
    compute_drift_weights,
    get_estimation_window,
)


def test_compute_drift_weights_total_loss_falls_back_to_equal_weight():
    """Verifies full portfolio loss returns equal weights, computed as 1/N for each asset."""
    weights = np.array([0.4, 0.6])
    stock_returns = np.array([-1.0, -1.0])

    result = compute_drift_weights(weights, stock_returns)

    np.testing.assert_allclose(result, np.array([0.5, 0.5]), atol=1e-10)


def test_align_drifted_weights_all_previous_holdings_leave_universe():
    """Verifies all departed holdings return equal weights over the current universe, computed as 1/N."""
    prev_weights = np.array([0.25, 0.75])
    prev_permnos = [101, 102]
    curr_permnos = [201, 202, 203, 204]

    result = align_drifted_weights(prev_weights, prev_permnos, curr_permnos)

    np.testing.assert_allclose(result, np.full(4, 0.25), atol=1e-10)


def test_build_monthly_universe_requires_market_cap_and_full_history():
    """Verifies eligibility is the intersection of mktcap>=2B and exactly 60 non-missing returns."""
    dates = pd.date_range("2000-01-31", periods=60, freq="ME")
    rows = []
    for permno, mktcap_lagged in [(1, 3.0), (2, 3.0), (3, 1.5)]:
        for i, date in enumerate(dates):
            ret = np.nan if permno == 2 and i == 0 else 0.01
            rows.append({
                "date": date,
                "permno": permno,
                "ret": ret,
                "mktcap_lagged": mktcap_lagged,
            })
    df = pd.DataFrame(rows)

    result = build_monthly_universe(df, pd.Timestamp("2005-01-01"))

    assert result.tolist() == [1]


def test_get_estimation_window_returns_mu_sigma_and_valid_permnos():
    """Verifies mu and sigma are computed from the complete 60-month return matrix plus 1e-4 ridge."""
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
    expected_sigma = np.cov(expected_matrix, rowvar=False) + 1e-4 * np.eye(2)

    assert valid_permnos == [1, 2]
    np.testing.assert_allclose(mu, expected_mu, atol=1e-10)
    np.testing.assert_allclose(sigma, expected_sigma, atol=1e-10)


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
