"""Post-run checks for GA behaviour, robustness, and transaction costs."""

import numpy as np
import pandas as pd
import os

from src.evaluation.metrics import sharpe_ratio, TRANSACTION_COST
from src.optimization.genetic_algorithm import K_MIN, K_MAX

# Monthly windows used for the robustness table.
SUBPERIODS = [
    ("Pre-GFC",  "2005-01", "2007-12"),
    ("GFC",      "2008-01", "2009-12"),
    ("Recovery", "2010-01", "2019-12"),
    ("COVID+",   "2020-01", "2025-12"),
]

GAMMAS = [0.001, 0.002, 0.003, 0.005, 0.010]

OUTPUT_DIR = "results/post_processing"


def _to_period_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M")
    out = out.set_index("date").sort_index()
    return out


def k_behavior_summary(ga_results: pd.DataFrame) -> dict:
    k = ga_results["n_stocks"]
    stats = {
        "mean":   float(k.mean()),
        "std":    float(k.std(ddof=1)),
        "min":    float(k.min()),
        "max":    float(k.max()),
        "median": float(k.median()),
        "pct_at_lower_bound": float((k == K_MIN).mean() * 100),
        "pct_at_upper_bound": float((k == K_MAX).mean() * 100),
    }

    lines = [
        "=" * 50,
        "GA Cardinality K — Behaviour Summary",
        "=" * 50,
        f"  Mean      : {stats['mean']:.2f}",
        f"  Std       : {stats['std']:.2f}",
        f"  Min       : {stats['min']:.0f}",
        f"  Max       : {stats['max']:.0f}",
        f"  Median    : {stats['median']:.1f}",
        f"  % at K=10 : {stats['pct_at_lower_bound']:.1f}%  (lower bound binding)",
        f"  % at K=30 : {stats['pct_at_upper_bound']:.1f}%  (upper bound binding)",
    ]
    output = "\n".join(lines)
    print(output)

    with open(f"{OUTPUT_DIR}/k_behavior.txt", "w") as f:
        f.write(output + "\n")

    pd.DataFrame([stats]).to_csv(f"{OUTPUT_DIR}/k_behavior.csv", index=False)
    print(f"  Saved -> {OUTPUT_DIR}/k_behavior.txt + .csv\n")

    return stats


def subperiod_robustness(results: dict) -> pd.DataFrame:
    indexed = {name: _to_period_index(df) for name, df in results.items()}
    model_names = list(results.keys())

    rows = {}
    for label, start, end in SUBPERIODS:
        row = {}
        for name in model_names:
            df = indexed[name]
            mask = (df.index >= start) & (df.index <= end)
            sub = df.loc[mask, "net_excess_ret"].values
            row[name] = round(sharpe_ratio(sub), 4) if len(sub) > 1 else np.nan
        rows[label] = row

    table = pd.DataFrame(rows).T
    table.index.name = "Subperiod"

    header = "\n".join([
        "=" * 70,
        "Subperiod Robustness — Net Sharpe Ratio",
        "=" * 70,
    ])
    print(header)
    print(table.to_string())
    print()

    table.to_csv(f"{OUTPUT_DIR}/subperiod_robustness.csv")
    with open(f"{OUTPUT_DIR}/subperiod_robustness.txt", "w") as f:
        f.write(header + "\n")
        f.write(table.to_string() + "\n")
    print(f"  Saved -> {OUTPUT_DIR}/subperiod_robustness.txt + .csv\n")

    return table


def transaction_cost_sensitivity(results: dict) -> pd.DataFrame:
    model_names = list(results.keys())

    rows = {}
    for gamma in GAMMAS:
        row = {}
        for name in model_names:
            df = results[name]
            net = (df["excess_ret"] - gamma * df["turnover"]).values
            row[name] = round(sharpe_ratio(net), 4)
        rows[gamma] = row

    table = pd.DataFrame(rows).T
    table.index.name = "Gamma"

    header = "\n".join([
        "=" * 70,
        "Transaction-Cost Sensitivity — Net Sharpe Ratio",
        f"  (baseline gamma = {TRANSACTION_COST})",
        "=" * 70,
    ])
    print(header)
    print(table.to_string())
    print()

    table.to_csv(f"{OUTPUT_DIR}/tc_sensitivity.csv")
    with open(f"{OUTPUT_DIR}/tc_sensitivity.txt", "w") as f:
        f.write(header + "\n")
        f.write(table.to_string() + "\n")
    print(f"  Saved -> {OUTPUT_DIR}/tc_sensitivity.txt + .csv\n")

    return table


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "1/N":               pd.read_parquet("results/benchmarks/equal_weight_full.parquet"),
        "Unconstrained MVO": pd.read_parquet("results/benchmarks/mvo_unconstrained.parquet"),
        "Constrained MVO":   pd.read_parquet("results/benchmarks/mvo_constrained.parquet"),
        "GA":                pd.read_parquet("results/ga/ga_results.parquet"),
    }

    k_behavior_summary(results["GA"])
    subperiod_robustness(results)
    transaction_cost_sensitivity(results)

    print(f"All outputs saved to {OUTPUT_DIR}/")