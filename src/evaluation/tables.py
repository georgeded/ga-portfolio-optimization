"""
Thesis Tables
Generates Table 1 (Main Performance Comparison) and saves as CSV, LaTeX and PNG.

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
import matplotlib.pyplot as plt
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

    ann_ret = float(np.mean(net) * 12)
    ann_vol = float(np.std(net, ddof=1) * SQRT_12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = net[net < 0]
    if len(downside) > 0:
        dd_vol  = float(np.sqrt(np.mean(downside ** 2)) * SQRT_12)
        sortino = ann_ret / dd_vol if dd_vol > 0 else 0.0
    else:
        sortino = 0.0

    cum  = np.cumprod(1 + net + rf)
    peak = np.maximum.accumulate(cum)
    mdd  = float(np.min((cum - peak) / peak))

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
    """Generate LaTeX table string for thesis inclusion."""
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


def to_png(fmt: pd.DataFrame, path: str) -> None:
    """
    Save formatted table as a PNG image.
    Clean white background, no axes — ready for thesis or slides.
    """
    n_rows, n_cols = fmt.shape
    fig_w = 2 + n_cols * 1.6
    fig_h = 0.5 + n_rows * 0.45

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText  = fmt.values,
        rowLabels = fmt.index.tolist(),
        colLabels = fmt.columns.tolist(),
        cellLoc   = "center",
        rowLoc    = "left",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building Table 1...")
    table = build_table1()
    fmt   = format_table1(table)

    print_table1(fmt)

    table.to_csv(f"{OUT_DIR}/table1_performance.csv")
    print(f"\nSaved: {OUT_DIR}/table1_performance.csv")

    fmt.to_csv(f"{OUT_DIR}/table1_performance_formatted.csv")
    print(f"Saved: {OUT_DIR}/table1_performance_formatted.csv")

    latex = to_latex(fmt)
    with open(f"{OUT_DIR}/table1_performance.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {OUT_DIR}/table1_performance.tex")

    to_png(fmt, f"{OUT_DIR}/table1_performance.png")