#!/usr/bin/env python3
"""
Lambda=0 ablation.

Runs the full 252-period GA with lambda=0 (no turnover penalty),
then compares against the main tuned run (lambda=1.8437) and
produces a summary table.

Runtime: approx. 35 minutes for the GA run.

Usage: python3 -m src.ablation.ablation_lambda
"""

import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.evaluation.metrics import compute_all_metrics, TRANSACTION_COST
from src.evaluation.tables import to_png
from src.optimization.runner import run

MAIN_PATH = "results/ga/ga_results.parquet"
OUTPUT_DIR = "results/ablation"
TABLE_DIR = "results/tables"


def metrics_row(df: pd.DataFrame, label: str) -> dict:
    m = compute_all_metrics(
        df["excess_ret"].values,
        df["rf"].values,
        df["turnover"].values,
        TRANSACTION_COST,
    )
    return {
        "Lambda":       label,
        "Sharpe (net)": round(m["sharpe_net"], 4),
        "Ann Return":   f"{m['ann_return_net'] * 100:.2f}%",
        "Ann Vol":      f"{m['ann_vol'] * 100:.2f}%",
        "Max Drawdown": f"{m['max_drawdown_net'] * 100:.2f}%",
        "Avg Turnover": f"{m['avg_turnover'] * 100:.2f}%",
        "Avg K":        round(float(df["n_stocks"].mean()), 1),
    }


def _is_complete(path: str, expected_rows: int = 252) -> bool:
    """Return True if the parquet exists, has the expected number of rows,
    and covers at least Jan 2005 to Dec 2025."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
        if len(df) < expected_rows:
            return False
        dates = pd.to_datetime(df["date"])
        if dates.min().year > 2005 or dates.max().year < 2025:
            return False
        return True
    except Exception:
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    lam0_path = f"{OUTPUT_DIR}/lambda0_results.parquet"
    if _is_complete(lam0_path):
        print(f"Found complete lambda=0 results at {lam0_path}, skipping GA run.")
        df_lam0 = pd.read_parquet(lam0_path)
    else:
        if os.path.exists(lam0_path):
            existing = pd.read_parquet(lam0_path)
            print(f"lambda=0 parquet exists but is incomplete "
                  f"({len(existing)} rows). Re-running GA.")
        else:
            print("Running GA with lambda=0 ...")
        df_lam0 = run(
            lambda_val=0.0,
            output_path=lam0_path,
            clear_checkpoint=False,
        )

    print("Loading main GA results ...")
    df_main = pd.read_parquet(MAIN_PATH)

    row_lam0 = metrics_row(df_lam0, "0 (no turnover penalty)")
    row_main = metrics_row(df_main, "1.8437 (tuned)")

    summary = pd.DataFrame([row_lam0, row_main])

    csv_path = f"{OUTPUT_DIR}/lambda_ablation_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\n" + summary.to_string(index=False))

    png_path = f"{TABLE_DIR}/T5_lambda_ablation.png"
    to_png(summary.set_index("Lambda"), png_path)


if __name__ == "__main__":
    main()
