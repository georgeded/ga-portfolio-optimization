"""
K-sensitivity comparison table (RQ3): performance and portfolio
characteristics across fixed K values [10, 15, 20, 25, 30].

Columns: K | Sharpe (net) | Ann Return | Ann Vol | Max Drawdown |
         Avg Turnover | Avg Cost | Avg HHI | Avg K (actual)

Avg K (actual) = mean n_stocks per period. Should equal k since K is
fixed, but deviations flag constraint violations.

Outputs:
  results/k_sensitivity/table_k_sensitivity.csv
  results/k_sensitivity/table_k_sensitivity.tex
  results/k_sensitivity/table_k_sensitivity.png

python3 -m src.evaluation.k_sensitivity_tables
"""

import os

import numpy as np
import pandas as pd

from src.evaluation.tables import to_png
from src.optimization.k_sensitivity import K_VALUES, OUTPUT_DIR

TABLE_DIR = "results/tables"

SQRT_12 = np.sqrt(12)

COL_K = "K (target)"
COL_SHARPE = "Sharpe (net)"
COL_RETURN = "Ann Return"
COL_VOL = "Ann Vol"
COL_MDD = "Max Drawdown"
COL_TURNOVER = "Avg Turnover"
COL_COST = "Avg Cost"
COL_HHI = "Avg HHI"
COL_K_ACTUAL = "Avg K (actual)"

COLUMNS = [
    COL_K, COL_SHARPE, COL_RETURN, COL_VOL, COL_MDD,
    COL_TURNOVER, COL_COST, COL_HHI, COL_K_ACTUAL,
]

COL_LATEX = {
    COL_K:        "$K$",
    COL_SHARPE:   "Sharpe",
    COL_RETURN:   "Ann.~Ret.",
    COL_VOL:      "Ann.~Vol.",
    COL_MDD:      "Max DD",
    COL_TURNOVER: "Turnover",
    COL_COST:     "Avg Cost",
    COL_HHI:      "HHI",
    COL_K_ACTUAL: "$\\bar{K}_{\\text{actual}}$",
}


def load_k_results(k_values: list = K_VALUES) -> dict:
    """Load parquets for all K values. Raises FileNotFoundError if missing."""
    data = {}
    for k in k_values:
        path = os.path.join(OUTPUT_DIR, f"k{k:02d}_results.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Results not found: {path}\n"
                f"Run: python3 -m src.optimization.k_sensitivity --k {k}"
            )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        data[k] = df.sort_values("date").reset_index(drop=True)
    return data


def compute_metrics(df: pd.DataFrame, k: int) -> dict:
    """Compute all table metrics for a single K run."""
    net = df["net_excess_ret"].values
    rf = df["rf"].values
    to = df["turnover"].values

    ann_ret = float(net.mean() * 12)
    ann_vol = float(net.std(ddof=1) * SQRT_12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = np.cumprod(1 + net + rf)
    peak = np.maximum.accumulate(cum)
    mdd = float(np.min((cum - peak) / peak))

    return {
        COL_K:        k,
        COL_SHARPE:   sharpe,
        COL_RETURN:   ann_ret,
        COL_VOL:      ann_vol,
        COL_MDD:      mdd,
        COL_TURNOVER: float(to.mean()),
        COL_COST:     float(df["cost"].mean()),
        COL_HHI:      float(df["hhi"].mean()),
        COL_K_ACTUAL: float(df["n_stocks"].mean()),
    }


def build_table(data: dict) -> pd.DataFrame:
    """Build the raw (numeric) table across all K values."""
    rows = [compute_metrics(data[k], k) for k in sorted(data.keys())]
    return pd.DataFrame(rows).set_index(COL_K)


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    fmt = table.copy().astype(object)
    fmt[COL_SHARPE] = table[COL_SHARPE].map("{:.4f}".format)
    fmt[COL_RETURN] = (table[COL_RETURN] * 100).map("{:.2f}%".format)
    fmt[COL_VOL] = (table[COL_VOL] * 100).map("{:.2f}%".format)
    fmt[COL_MDD] = (table[COL_MDD] * 100).map("{:.2f}%".format)
    fmt[COL_TURNOVER] = (table[COL_TURNOVER] * 100).map("{:.2f}%".format)
    fmt[COL_COST] = (table[COL_COST] * 100).map("{:.4f}%".format)
    fmt[COL_HHI] = table[COL_HHI].map("{:.6f}".format)
    fmt[COL_K_ACTUAL] = table[COL_K_ACTUAL].map("{:.1f}".format)
    return fmt


def to_latex(fmt: pd.DataFrame) -> str:
    fmt_latex = fmt.copy()
    fmt_latex.index.name = COL_LATEX[COL_K]
    fmt_latex = fmt_latex.rename(columns=COL_LATEX)
    return fmt_latex.to_latex(
        caption=(
            "K-sensitivity analysis: GA performance and portfolio characteristics "
            "with fixed cardinality $K \\in \\{10, 15, 20, 25, 30\\}$ "
            "(January 2005--December 2025). "
            "Fixed hyperparameters (pc=0.6054, pm=0.1370, $\\sigma_m$=0.1469, "
            "$\\lambda$=1.8437) are used throughout to isolate the effect of $K$. "
            "$\\bar{K}_{\\text{actual}}$ is the mean number of stocks held per "
            "period and should equal $K$; deviations flag constraint violations."
        ),
        label="tab:k_sensitivity",
        column_format="r" + "r" * fmt_latex.shape[1],
        escape=False,
        bold_rows=False,
    )


def print_table(fmt: pd.DataFrame) -> None:
    print("\nK-Sensitivity Analysis")
    print("Period: 2005-01-31 to 2025-12-31 | Fixed hyperparameters")
    print("Transaction cost: gamma = 0.3% per unit traded\n")
    print(fmt.to_string())


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    print("Loading K-sensitivity results...")
    data = load_k_results()

    print("Building table...")
    table = build_table(data)
    fmt = format_table(table)

    print_table(fmt)

    csv_path = os.path.join(TABLE_DIR, "T4_k_sensitivity.csv")
    table.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    fmt_csv_path = os.path.join(TABLE_DIR, "T4_k_sensitivity_formatted.csv")
    fmt.to_csv(fmt_csv_path)
    print(f"Saved: {fmt_csv_path}")

    tex_path = os.path.join(TABLE_DIR, "T4_k_sensitivity.tex")
    with open(tex_path, "w") as f:
        f.write(to_latex(fmt))
    print(f"Saved: {tex_path}")

    png_path = os.path.join(TABLE_DIR, "T4_k_sensitivity.png")
    to_png(fmt, png_path)
