"""
Thesis Tables
Generates Table 1 (Main Performance Comparison) and saves as CSV and LaTeX.

Table 1 columns (all net of transaction costs):
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Avg Monthly Turnover
- Avg Transaction Cost

Reference: DeMiguel et al. (2009) — evaluation protocol
"""

import numpy as np
import pandas as pd
import os

SQRT_12 = np.sqrt(12)
OUT_DIR = "results/tables"

# ── Column name constants ─────────────────────────────────────────────────────
COL_RETURN   = "Annualized Return (net)"
COL_VOL      = "Annualized Volatility"
COL_SHARPE   = "Sharpe Ratio (net)"
COL_SORTINO  = "Sortino Ratio (net)"
COL_MDD      = "Max Drawdown (net)"
COL_TURNOVER = "Avg Monthly Turnover"
COL_COST     = "Avg Transaction Cost"

COLUMNS = [COL_RETURN, COL_VOL, COL_SHARPE, COL_SORTINO,
           COL_MDD, COL_TURNOVER, COL_COST]

# LaTeX column header mapping
COL_LATEX = {
    COL_RETURN  : "Ann. Return",
    COL_VOL     : "Ann. Vol.",
    COL_SHARPE  : "Sharpe",
    COL_SORTINO : "Sortino",
    COL_MDD     : "Max DD",
    COL_TURNOVER: "Turnover",
    COL_COST    : "Avg Cost",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_table1_metrics(df: pd.DataFrame) -> dict:
    """Compute all Table 1 metrics for a single strategy."""
    net = df["net_excess_ret"].values
    rf  = df["rf"].values
    to  = df["turnover"].values

    # Annualized return and volatility
    ann_ret = float(np.mean(net) * 12)
    ann_vol = float(np.std(net, ddof=1) * SQRT_12)

    # Sharpe ratio (net)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino ratio (net) — downside deviation using returns below zero
    downside = net[net < 0]
    if len(downside) > 0:
        dd_vol  = float(np.sqrt(np.mean(downside ** 2)) * SQRT_12)
        sortino = ann_ret / dd_vol if dd_vol > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown (net) — computed on total return for correct compounding
    cum  = np.cumprod(1 + net + rf)
    peak = np.maximum.accumulate(cum)
    mdd  = float(np.min((cum - peak) / peak))

    # Turnover and cost — use stored simulation values for consistency
    avg_to   = float(np.mean(to))
    avg_cost = float(df["cost"].mean())

    return {
        COL_RETURN  : ann_ret,
        COL_VOL     : ann_vol,
        COL_SHARPE  : sharpe,
        COL_SORTINO : sortino,
        COL_MDD     : mdd,
        COL_TURNOVER: avg_to,
        COL_COST    : avg_cost,
    }


# ── Table 1 ───────────────────────────────────────────────────────────────────

def build_table1(
    ga_path      : str = "results/ga/ga_results.parquet",
    ew_path      : str = "results/benchmarks/equal_weight_full.parquet",
    mvo_unc_path : str = "results/benchmarks/mvo_unconstrained.parquet",
    mvo_con_path : str = "results/benchmarks/mvo_constrained.parquet",
) -> pd.DataFrame:

    strategies = {
        "GA (adaptive K)"   : ga_path,
        "Constrained MVO"   : mvo_con_path,
        "Unconstrained MVO" : mvo_unc_path,
        "1/N (~867 stocks)" : ew_path,
    }

    rows = {}
    for name, path in strategies.items():
        df         = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        rows[name] = compute_table1_metrics(df)

    return pd.DataFrame(rows).T


def format_table1(table: pd.DataFrame) -> pd.DataFrame:
    """Format raw values for display."""
    fmt = table.copy()
    fmt[COL_RETURN]   = (table[COL_RETURN]   * 100).map("{:.2f}%".format)
    fmt[COL_VOL]      = (table[COL_VOL]       * 100).map("{:.2f}%".format)
    fmt[COL_SHARPE]   =  table[COL_SHARPE].map("{:.4f}".format)
    fmt[COL_SORTINO]  =  table[COL_SORTINO].map("{:.4f}".format)
    fmt[COL_MDD]      = (table[COL_MDD]       * 100).map("{:.2f}%".format)
    fmt[COL_TURNOVER] = (table[COL_TURNOVER]  * 100).map("{:.2f}%".format)
    fmt[COL_COST]     = (table[COL_COST]      * 100).map("{:.4f}%".format)
    return fmt


def print_table1(fmt: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("TABLE 1 — Main Performance Comparison (net of transaction costs)")
    print("="*80)
    print("Period: 2005-01-31 to 2025-12-31 | 252 monthly observations")
    print("Transaction cost: γ = 0.3% per unit traded\n")
    print(fmt.to_string())
    print("="*80)


def to_latex(fmt: pd.DataFrame) -> str:
    """
    Generate LaTeX table string for thesis inclusion.
    Formatted for ACM style — compatible with booktabs.
    """
    fmt_latex = fmt.rename(columns=COL_LATEX)

    return fmt_latex.to_latex(
        caption=(
            "Main performance comparison (net of transaction costs, "
            "January 2005--December 2025). "
            "All metrics computed on monthly net excess returns. "
            "Transaction cost $\\gamma = 0.3\\%$ per unit traded."
        ),
        label="tab:performance",
        column_format="l" + "r" * len(COL_LATEX),
        escape=False,
        bold_rows=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building Table 1...")
    table = build_table1()
    fmt   = format_table1(table)

    print_table1(fmt)

    # Raw values CSV
    table.to_csv(f"{OUT_DIR}/table1_performance.csv")
    print(f"\nSaved: {OUT_DIR}/table1_performance.csv")

    # Formatted CSV
    fmt.to_csv(f"{OUT_DIR}/table1_performance_formatted.csv")
    print(f"Saved: {OUT_DIR}/table1_performance_formatted.csv")

    # LaTeX
    latex = to_latex(fmt)
    with open(f"{OUT_DIR}/table1_performance.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {OUT_DIR}/table1_performance.tex")