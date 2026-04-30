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
import matplotlib.pyplot as plt
from scipy import stats
import os

OUT_DIR = "results/tables"

# ── Column name constants ─────────────────────────────────────────────────────
COL_COMPARISON  = "Comparison"
COL_MEAN_DIFF   = "Mean Diff (ann.)"
COL_T_STAT      = "t-stat"
COL_T_PVAL      = "t p-value"
COL_T_SIG       = "t Sig."
COL_SR_GA       = "SR(GA)"
COL_SR_BENCH    = "SR(Bench)"
COL_SR_DIFF     = "SR Diff"
COL_Z_STAT      = "z-stat"
COL_JK_PVAL     = "JK p-value"
COL_JK_SIG      = "JK Sig."


# ── Jobson-Korkie Test (Memmel 2003 correction) ───────────────────────────────

def jobson_korkie_test(r1: np.ndarray, r2: np.ndarray) -> dict:
    """
    Jobson-Korkie (1981) test for equality of two Sharpe ratios,
    with Memmel (2003) correction.

    Variance formula (Memmel 2003 corrected):
        var = (1/T) * [2*(1-rho) + 0.5*SR1^2 + 0.5*SR2^2 - SR1*SR2*(1+rho)]

    Note: var_diff can be negative when strategies are highly correlated.
    The guard (var_diff <= 0 → z=0, p=1) is mathematically necessary.
    """
    T = len(r1)
    assert len(r2) == T, "Return series must have equal length"

    mu1, mu2 = r1.mean(), r2.mean()
    s1,  s2  = r1.std(ddof=1), r2.std(ddof=1)
    s12      = np.cov(r1, r2, ddof=1)[0, 1]

    if s1 <= 0 or s2 <= 0:
        return {
            "sr1_ann": 0.0, "sr2_ann": 0.0, "sr_diff_ann": 0.0,
            "z_stat": 0.0, "p_value": 1.0, "significant": False,
        }

    sr1 = mu1 / s1
    sr2 = mu2 / s2
    rho = np.clip(s12 / (s1 * s2), -1.0, 1.0)

    var_diff = (1 / T) * (
        2 * (1 - rho)
        + 0.5 * sr1 ** 2
        + 0.5 * sr2 ** 2
        - sr1 * sr2 * (1 + rho)
    )

    if var_diff <= 0:
        z_stat  = 0.0
        p_value = 1.0
    else:
        z_stat  = (sr1 - sr2) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    sqrt12 = np.sqrt(12)
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
    """Paired t-test on monthly net excess return differences."""
    diff            = r1 - r2
    t_stat, p_value = stats.ttest_1samp(diff, popmean=0)
    return {
        "mean_diff_ann": diff.mean() * 12,
        "t_stat"       : t_stat,
        "p_value"      : p_value,
        "significant"  : p_value < 0.05,
    }


# ── Build Table 2 ─────────────────────────────────────────────────────────────

def build_table2(
    ga_path      : str = "results/ga/ga_results.parquet",
    ew_path      : str = "results/benchmarks/equal_weight_full.parquet",
    mvo_unc_path : str = "results/benchmarks/mvo_unconstrained.parquet",
    mvo_con_path : str = "results/benchmarks/mvo_constrained.parquet",
) -> tuple:
    """
    Run all significance tests and return raw + formatted DataFrames.
    Returns: (raw_df, fmt_df)
    """
    ga      = pd.read_parquet(ga_path)
    ew      = pd.read_parquet(ew_path)
    mvo_unc = pd.read_parquet(mvo_unc_path)
    mvo_con = pd.read_parquet(mvo_con_path)

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

    rows = []
    for label, r_bench in benchmarks.items():
        tt = paired_ttest(r_ga, r_bench)
        jk = jobson_korkie_test(r_ga, r_bench)
        rows.append({
            COL_COMPARISON : label,
            COL_MEAN_DIFF  : tt["mean_diff_ann"],
            COL_T_STAT     : tt["t_stat"],
            COL_T_PVAL     : tt["p_value"],
            COL_T_SIG      : tt["significant"],
            COL_SR_GA      : jk["sr1_ann"],
            COL_SR_BENCH   : jk["sr2_ann"],
            COL_SR_DIFF    : jk["sr_diff_ann"],
            COL_Z_STAT     : jk["z_stat"],
            COL_JK_PVAL    : jk["p_value"],
            COL_JK_SIG     : jk["significant"],
        })

    raw = pd.DataFrame(rows).set_index(COL_COMPARISON)

    # Formatted version
    fmt = raw.copy()
    fmt[COL_MEAN_DIFF] = (raw[COL_MEAN_DIFF] * 100).map("{:.2f}%".format)
    fmt[COL_T_STAT]    = raw[COL_T_STAT].map("{:.3f}".format)
    fmt[COL_T_PVAL]    = raw[COL_T_PVAL].map("{:.4f}".format)
    fmt[COL_T_SIG]     = raw[COL_T_SIG].map(lambda x: "✓" if x else "✗")
    fmt[COL_SR_GA]     = raw[COL_SR_GA].map("{:.4f}".format)
    fmt[COL_SR_BENCH]  = raw[COL_SR_BENCH].map("{:.4f}".format)
    fmt[COL_SR_DIFF]   = raw[COL_SR_DIFF].map("{:.4f}".format)
    fmt[COL_Z_STAT]    = raw[COL_Z_STAT].map("{:.3f}".format)
    fmt[COL_JK_PVAL]   = raw[COL_JK_PVAL].map("{:.4f}".format)
    fmt[COL_JK_SIG]    = raw[COL_JK_SIG].map(lambda x: "✓" if x else "✗")

    return raw, fmt


# ── Print ─────────────────────────────────────────────────────────────────────

def print_table2(fmt: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("TABLE 2 — Statistical Significance of Performance Differences")
    print("="*80)
    print("Sample: 252 monthly observations | α = 0.05 (two-tailed)\n")

    print("TEST 1: Paired t-test (H0: mean return difference = 0)")
    print(fmt[[COL_MEAN_DIFF, COL_T_STAT, COL_T_PVAL, COL_T_SIG]].to_string())

    print("\nTEST 2: Jobson-Korkie (Memmel 2003) (H0: Sharpe difference = 0)")
    print(fmt[[COL_SR_GA, COL_SR_BENCH, COL_SR_DIFF,
               COL_Z_STAT, COL_JK_PVAL, COL_JK_SIG]].to_string())

    print("\nNOTE: JK assumes i.i.d. normal returns — interpret with caution.")
    print("="*80)


# ── PNG ───────────────────────────────────────────────────────────────────────

def to_png(fmt: pd.DataFrame, path: str) -> None:
    """Save Table 2 as a PNG image."""
    display = fmt[[COL_MEAN_DIFF, COL_T_STAT, COL_T_PVAL, COL_T_SIG,
                   COL_SR_GA, COL_SR_BENCH, COL_SR_DIFF,
                   COL_Z_STAT, COL_JK_PVAL, COL_JK_SIG]].copy()

    n_rows, n_cols = display.shape
    fig_w = 2 + n_cols * 1.3
    fig_h = 0.5 + n_rows * 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText  = display.values,
        rowLabels = display.index.tolist(),
        colLabels = display.columns.tolist(),
        cellLoc   = "center",
        rowLoc    = "left",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(-1, n_cols)))

    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#222222")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "white")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def to_latex(fmt: pd.DataFrame) -> str:
    """Generate LaTeX table for thesis inclusion."""
    display = fmt[[COL_MEAN_DIFF, COL_T_STAT, COL_T_PVAL, COL_T_SIG,
                   COL_SR_DIFF, COL_Z_STAT, COL_JK_PVAL, COL_JK_SIG]]
    return display.to_latex(
        caption=(
            "Statistical significance of GA performance differences "
            "vs benchmarks (January 2005--December 2025, $T=252$). "
            "Paired $t$-test: $H_0$: mean return difference $= 0$. "
            "Jobson--Korkie (Memmel 2003): $H_0$: Sharpe difference $= 0$. "
            "$\\alpha = 0.05$ (two-tailed). "
            "JK assumes i.i.d.\\ normal returns."
        ),
        label="tab:significance",
        column_format="l" + "r" * display.shape[1],
        escape=False,
        bold_rows=False,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Running significance tests...")
    raw, fmt = build_table2()

    print_table2(fmt)

    raw.to_csv(f"{OUT_DIR}/significance_tests.csv")
    print(f"\nSaved: {OUT_DIR}/significance_tests.csv")

    fmt.to_csv(f"{OUT_DIR}/significance_tests_formatted.csv")
    print(f"Saved: {OUT_DIR}/significance_tests_formatted.csv")

    latex = to_latex(fmt)
    with open(f"{OUT_DIR}/table2_significance.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {OUT_DIR}/table2_significance.tex")

    to_png(fmt, f"{OUT_DIR}/table2_significance.png")