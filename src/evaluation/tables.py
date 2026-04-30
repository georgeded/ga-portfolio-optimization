"""
Thesis Tables
Generates Table 1 (Main Performance Comparison) and
Table 3 (Portfolio Characteristics) as CSV, LaTeX and PNG.

Table 1 columns (all net of transaction costs):
- Annualized Return, Annualized Volatility, Sharpe Ratio,
  Sortino Ratio, Max Drawdown, Avg Monthly Turnover, Avg Transaction Cost

Table 3 columns (portfolio characteristics — answers RQ3):
- Avg Portfolio Size (K), Avg HHI, Avg Monthly Turnover, Avg Transaction Cost

Reference: DeMiguel et al. (2009) — evaluation protocol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SQRT_12 = np.sqrt(12)
OUT_DIR = "results/tables"

# ── Table 1 column constants ──────────────────────────────────────────────────
COL_RETURN   = "Annualized Return (net)"
COL_VOL      = "Annualized Volatility"
COL_SHARPE   = "Sharpe Ratio (net)"
COL_SORTINO  = "Sortino Ratio (net)"
COL_MDD      = "Max Drawdown (net)"
COL_TURNOVER = "Avg Monthly Turnover"
COL_COST     = "Avg Transaction Cost"

COLUMNS_T1 = [COL_RETURN, COL_VOL, COL_SHARPE, COL_SORTINO,
              COL_MDD, COL_TURNOVER, COL_COST]

COL_LATEX_T1 = {
    COL_RETURN  : "Ann. Return",
    COL_VOL     : "Ann. Vol.",
    COL_SHARPE  : "Sharpe",
    COL_SORTINO : "Sortino",
    COL_MDD     : "Max DD",
    COL_TURNOVER: "Turnover",
    COL_COST    : "Avg Cost",
}

# ── Table 3 column constants ──────────────────────────────────────────────────
COL_K        = "Avg Portfolio Size (K)"
COL_HHI      = "Avg HHI"
COL_TO3      = "Avg Monthly Turnover"
COL_COST3    = "Avg Transaction Cost"

COLUMNS_T3 = [COL_K, COL_HHI, COL_TO3, COL_COST3]

COL_LATEX_T3 = {
    COL_K    : "Avg K",
    COL_HHI  : "Avg HHI",
    COL_TO3  : "Turnover",
    COL_COST3: "Avg Cost",
}


# ── Shared PNG helper ─────────────────────────────────────────────────────────

def to_png(fmt: pd.DataFrame, path: str, fontsize: int = 10) -> None:
    """Save a formatted DataFrame as a PNG table image."""
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
    tbl.set_fontsize(fontsize)
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


# ── Table 1 ───────────────────────────────────────────────────────────────────

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

    return {
        COL_RETURN  : ann_ret,
        COL_VOL     : ann_vol,
        COL_SHARPE  : sharpe,
        COL_SORTINO : sortino,
        COL_MDD     : mdd,
        COL_TURNOVER: float(np.mean(to)),
        COL_COST    : float(df["cost"].mean()),
    }


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


def to_latex_t1(fmt: pd.DataFrame) -> str:
    fmt_latex = fmt.rename(columns=COL_LATEX_T1)
    return fmt_latex.to_latex(
        caption=(
            "Main performance comparison (net of transaction costs, "
            "January 2005--December 2025). "
            "All metrics computed on monthly net excess returns. "
            "Transaction cost $\\gamma = 0.3\\%$ per unit traded."
        ),
        label="tab:performance",
        column_format="l" + "r" * len(COL_LATEX_T1),
        escape=False,
        bold_rows=False,
    )


# ── Table 3 ───────────────────────────────────────────────────────────────────

def compute_table3_metrics(df: pd.DataFrame) -> dict:
    """
    Compute portfolio characteristics for Table 3 (RQ3).
    Avg K: for MVO and 1/N this equals the eligible universe size (~867)
    and is not a design parameter. For GA, K reflects adaptive cardinality
    selection subject to the constraint K ∈ [10, 30].
    """
    return {
        COL_K    : float(df["n_stocks"].mean()),
        COL_HHI  : float(df["hhi"].mean()),
        COL_TO3  : float(df["turnover"].mean()),
        COL_COST3: float(df["cost"].mean()),
    }


def build_table3(
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
        rows[name] = compute_table3_metrics(df)

    return pd.DataFrame(rows).T


def format_table3(table: pd.DataFrame) -> pd.DataFrame:
    fmt = table.copy()
    fmt[COL_K]    =  table[COL_K].map("{:.1f}".format)
    fmt[COL_HHI]  =  table[COL_HHI].map("{:.6f}".format)
    fmt[COL_TO3]  = (table[COL_TO3]   * 100).map("{:.2f}%".format)
    fmt[COL_COST3]= (table[COL_COST3] * 100).map("{:.4f}%".format)
    return fmt


def print_table3(fmt: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("TABLE 3 — Portfolio Characteristics (RQ3)")
    print("="*80)
    print("Period: 2005-01-31 to 2025-12-31 | 252 monthly observations\n")
    print(fmt.to_string())
    print("="*80)


def to_latex_t3(fmt: pd.DataFrame) -> str:
    fmt_latex = fmt.rename(columns=COL_LATEX_T3)
    return fmt_latex.to_latex(
        caption=(
            "Portfolio characteristics across strategies "
            "(January 2005--December 2025). "
            "\\textit{Avg K}: average number of stocks held per period. "
            "For the GA, $K \\in [10, 30]$ is enforced by the cardinality constraint "
            "(adaptive selection); for MVO and 1/N, K reflects the full eligible "
            "universe ($\\approx 867$) and is not a design parameter. "
            "\\textit{HHI}: Herfindahl-Hirschman Index; theoretical minimum is $1/K$, "
            "so values are not directly comparable across strategies with different $K$ "
            "(GA lower bound $\\approx 0.071$; MVO/1/N lower bound $\\approx 0.001$). "
            "Transaction cost $\\gamma = 0.3\\%$ per unit traded."
        ),
        label="tab:characteristics",
        column_format="l" + "r" * len(COL_LATEX_T3),
        escape=False,
        bold_rows=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Table 1 ───────────────────────────────────────────────────────────────
    print("Building Table 1...")
    table1 = build_table1()
    fmt1   = format_table1(table1)

    print_table1(fmt1)

    table1.to_csv(f"{OUT_DIR}/table1_performance.csv")
    print(f"\nSaved: {OUT_DIR}/table1_performance.csv")

    fmt1.to_csv(f"{OUT_DIR}/table1_performance_formatted.csv")
    print(f"Saved: {OUT_DIR}/table1_performance_formatted.csv")

    with open(f"{OUT_DIR}/table1_performance.tex", "w") as f:
        f.write(to_latex_t1(fmt1))
    print(f"Saved: {OUT_DIR}/table1_performance.tex")

    to_png(fmt1, f"{OUT_DIR}/table1_performance.png")

    # ── Table 3 ───────────────────────────────────────────────────────────────
    print("\nBuilding Table 3...")
    table3 = build_table3()
    fmt3   = format_table3(table3)

    print_table3(fmt3)

    table3.to_csv(f"{OUT_DIR}/table3_characteristics.csv")
    print(f"\nSaved: {OUT_DIR}/table3_characteristics.csv")

    fmt3.to_csv(f"{OUT_DIR}/table3_characteristics_formatted.csv")
    print(f"Saved: {OUT_DIR}/table3_characteristics_formatted.csv")

    with open(f"{OUT_DIR}/table3_characteristics.tex", "w") as f:
        f.write(to_latex_t3(fmt3))
    print(f"Saved: {OUT_DIR}/table3_characteristics.tex")

    to_png(fmt3, f"{OUT_DIR}/table3_characteristics.png")