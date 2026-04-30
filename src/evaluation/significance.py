"""
Statistical Significance Testing
Compares GA performance against MVO and 1/N benchmarks using:

1. Paired t-test on monthly net excess returns
   - H0: mean(GA returns) - mean(benchmark returns) = 0

2. Jobson-Korkie test with Memmel (2003) correction
   - H0: Sharpe(GA) - Sharpe(benchmark) = 0
   - Status quo method in portfolio comparison literature
   - Used by DeMiguel et al. (2009)

References:
- Jobson & Korkie (1981): Performance hypothesis testing with the
  Sharpe and Treynor measures. Journal of Finance, 36(4):889-908.
- Memmel (2003): Performance hypothesis testing with the Sharpe ratio.
  Finance Letters, 1:21-23.
- DeMiguel et al. (2009): Optimal versus naive diversification.
  Review of Financial Studies, 22(5):1915-1953.
"""

import numpy as np
import pandas as pd
from scipy import stats
import os


# ── Jobson-Korkie Test (Memmel 2003 correction) ───────────────────────────────

def jobson_korkie_test(r1: np.ndarray, r2: np.ndarray) -> dict:
    """
    Jobson-Korkie (1981) test for equality of two Sharpe ratios,
    with Memmel (2003) correction.

    Tests H0: SR(r1) = SR(r2) against H1: SR(r1) != SR(r2).

    Variance formula (Memmel 2003 corrected):
        var = (1/T) * [2*(1-rho) + 0.5*SR1^2 + 0.5*SR2^2 - SR1*SR2*(1+rho)]

    Note: var_diff can be negative when strategies are highly correlated
    (e.g. identical series). The guard (var_diff <= 0 → z=0, p=1) is
    mathematically necessary, not just defensive.

    Args:
        r1: monthly net excess returns for strategy 1 (GA)
        r2: monthly net excess returns for strategy 2 (benchmark)

    Returns:
        dict with z-statistic, p-value, SR difference, annualized SRs
    """
    T = len(r1)
    assert len(r2) == T, "Return series must have equal length"

    # Sample moments
    mu1, mu2 = r1.mean(), r2.mean()
    s1,  s2  = r1.std(ddof=1), r2.std(ddof=1)
    s12      = np.cov(r1, r2, ddof=1)[0, 1]

    # Guard against degenerate inputs
    if s1 <= 0 or s2 <= 0:
        return {
            "sr1_ann": 0.0, "sr2_ann": 0.0, "sr_diff_ann": 0.0,
            "z_stat": 0.0, "p_value": 1.0, "significant": False,
        }

    # Monthly Sharpe ratios (annualisation applied to output only)
    sr1 = mu1 / s1
    sr2 = mu2 / s2

    # Correlation between the two return series
    rho = s12 / (s1 * s2)
    rho = np.clip(rho, -1.0, 1.0)  # numerical safety

    # Memmel (2003) corrected variance of (SR1 - SR2)
    # When rho=1 and sr1=sr2, this is negative → handled by guard below
    var_diff = (1 / T) * (
        2 * (1 - rho)
        + 0.5 * sr1 ** 2
        + 0.5 * sr2 ** 2
        - sr1 * sr2 * (1 + rho)
    )

    # Guard: var_diff <= 0 implies strategies are indistinguishable
    if var_diff <= 0:
        z_stat  = 0.0
        p_value = 1.0
    else:
        z_stat  = (sr1 - sr2) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    sqrt12  = np.sqrt(12)

    return {
        "sr1_ann"    : sr1 * sqrt12,
        "sr2_ann"    : sr2 * sqrt12,
        "sr_diff_ann": (sr1 - sr2) * sqrt12,
        "z_stat"     : z_stat,
        "p_value"    : p_value,
        "significant": p_value < 0.05,
    }


# ── Paired t-test ─────────────────────────────────────────────────────────────

def paired_ttest(r1: np.ndarray, r2: np.ndarray) -> dict:
    """
    Paired t-test on monthly net excess return differences.

    Tests H0: mean(r1 - r2) = 0 against H1: mean(r1 - r2) != 0.

    Args:
        r1: monthly net excess returns for strategy 1 (GA)
        r2: monthly net excess returns for strategy 2 (benchmark)

    Returns:
        dict with t-statistic, p-value, mean difference (annualized)
    """
    diff            = r1 - r2
    t_stat, p_value = stats.ttest_1samp(diff, popmean=0)

    return {
        "mean_diff_ann": diff.mean() * 12,
        "t_stat"       : t_stat,
        "p_value"      : p_value,
        "significant"  : p_value < 0.05,
    }


# ── Main Comparison ───────────────────────────────────────────────────────────

def run_significance_tests(
    ga_path      : str = "results/ga/ga_results.parquet",
    ew_path      : str = "results/benchmarks/equal_weight_full.parquet",
    mvo_unc_path : str = "results/benchmarks/mvo_unconstrained.parquet",
    mvo_con_path : str = "results/benchmarks/mvo_constrained.parquet",
) -> None:
    """
    Run all significance tests and print results table.
    Saves CSV to results/tables/significance_tests.csv.
    """
    # Load results
    ga      = pd.read_parquet(ga_path)
    ew      = pd.read_parquet(ew_path)
    mvo_unc = pd.read_parquet(mvo_unc_path)
    mvo_con = pd.read_parquet(mvo_con_path)

    # Strict date alignment — all series must cover identical periods
    dates = ga["date"].values
    assert (ew["date"].values      == dates).all(), "Date mismatch: GA vs 1/N"
    assert (mvo_unc["date"].values == dates).all(), "Date mismatch: GA vs Unc MVO"
    assert (mvo_con["date"].values == dates).all(), "Date mismatch: GA vs Con MVO"

    r_ga      = ga["net_excess_ret"].values
    r_ew      = ew["net_excess_ret"].values
    r_mvo_unc = mvo_unc["net_excess_ret"].values
    r_mvo_con = mvo_con["net_excess_ret"].values

    benchmarks = {
        "GA vs 1/N"              : r_ew,
        "GA vs Unconstrained MVO": r_mvo_unc,
        "GA vs Constrained MVO"  : r_mvo_con,
    }

    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS — GA vs Benchmarks")
    print("="*70)
    print(f"Sample : {len(r_ga)} monthly observations "
          f"({ga['date'].min().date()} to {ga['date'].max().date()})")
    print("Alpha  : 0.05 (two-tailed)\n")

    # ── Test 1: Paired t-test ─────────────────────────────────────────────────
    print("-"*70)
    print("TEST 1: Paired t-test on monthly net excess returns")
    print("H0: mean(GA return) - mean(benchmark return) = 0")
    print("-"*70)
    print(f"{'Comparison':<30} {'Mean Diff (ann)':>16} {'t-stat':>10} "
          f"{'p-value':>10} {'Sig':>6}")
    print("-"*70)

    ttest_results = {}
    for label, r_bench in benchmarks.items():
        res = paired_ttest(r_ga, r_bench)
        ttest_results[label] = res
        sig = "✓" if res["significant"] else "✗"
        print(f"{label:<30} {res['mean_diff_ann']*100:>15.2f}% "
              f"{res['t_stat']:>10.3f} {res['p_value']:>10.4f} {sig:>6}")

    # ── Test 2: Jobson-Korkie ─────────────────────────────────────────────────
    print("\n" + "-"*70)
    print("TEST 2: Jobson-Korkie test (Memmel 2003 correction)")
    print("H0: Sharpe(GA) - Sharpe(benchmark) = 0")
    print("-"*70)
    print(f"{'Comparison':<30} {'SR(GA)':>8} {'SR(bench)':>10} "
          f"{'SR Diff':>8} {'z-stat':>8} {'p-value':>10} {'Sig':>6}")
    print("-"*70)

    jk_results = {}
    for label, r_bench in benchmarks.items():
        res = jobson_korkie_test(r_ga, r_bench)
        jk_results[label] = res
        sig = "✓" if res["significant"] else "✗"
        print(f"{label:<30} {res['sr1_ann']:>8.4f} {res['sr2_ann']:>10.4f} "
              f"{res['sr_diff_ann']:>8.4f} {res['z_stat']:>8.3f} "
              f"{res['p_value']:>10.4f} {sig:>6}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for label in benchmarks:
        tt = ttest_results[label]
        jk = jk_results[label]
        print(f"\n{label}:")
        print(f"  Paired t-test : p={tt['p_value']:.4f} "
              f"({'significant' if tt['significant'] else 'not significant'} at α=0.05)")
        print(f"  Jobson-Korkie : p={jk['p_value']:.4f} "
              f"({'significant' if jk['significant'] else 'not significant'} at α=0.05)")

    print("\n" + "="*70)
    print("NOTE: Jobson-Korkie assumes normally distributed, i.i.d. returns.")
    print("Financial returns typically exhibit heavier tails and volatility")
    print("clustering — results should be interpreted with this caveat.")
    print("="*70)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results/tables", exist_ok=True)

    rows = []
    for label, r_bench in benchmarks.items():
        tt = ttest_results[label]
        jk = jk_results[label]
        rows.append({
            "comparison"    : label,
            "mean_diff_ann" : tt["mean_diff_ann"],
            "t_stat"        : tt["t_stat"],
            "t_pvalue"      : tt["p_value"],
            "t_significant" : tt["significant"],
            "sr_ga_ann"     : jk["sr1_ann"],
            "sr_bench_ann"  : jk["sr2_ann"],
            "sr_diff_ann"   : jk["sr_diff_ann"],
            "jk_z_stat"     : jk["z_stat"],
            "jk_pvalue"     : jk["p_value"],
            "jk_significant": jk["significant"],
        })

    pd.DataFrame(rows).to_csv(
        "results/tables/significance_tests.csv", index=False
    )
    print("\nSaved: results/tables/significance_tests.csv")


if __name__ == "__main__":
    run_significance_tests()