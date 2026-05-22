"""
Post-processing analysis: cardinality behaviour, subperiod robustness, and
transaction-cost sensitivity.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import sharpe_ratio, TRANSACTION_COST
from src.optimization.genetic_algorithm import K_MIN, K_MAX

# ---------------------------------------------------------------------------
# Subperiod definitions: (label, start_inclusive, end_inclusive)
# ---------------------------------------------------------------------------
SUBPERIODS = [
    ("Pre-GFC",  "2005-01", "2007-12"),
    ("GFC",      "2008-01", "2009-12"),
    ("Recovery", "2010-01", "2019-12"),
    ("COVID+",   "2020-01", "2025-12"),
]

GAMMAS = [0.001, 0.002, 0.003, 0.005, 0.010]


def _to_period_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with date as a Period[M] index for easy slicing."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.to_period("M")
    out = out.set_index("date").sort_index()
    return out


# ---------------------------------------------------------------------------
# 1. Cardinality K behaviour
# ---------------------------------------------------------------------------

def k_behavior_summary(ga_results: pd.DataFrame) -> dict:
    """Compute and print descriptive stats on the GA's cardinality K (n_stocks)."""
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

    print("=" * 50)
    print("GA Cardinality K — Behaviour Summary")
    print("=" * 50)
    print(f"  Mean      : {stats['mean']:.2f}")
    print(f"  Std       : {stats['std']:.2f}")
    print(f"  Min       : {stats['min']:.0f}")
    print(f"  Max       : {stats['max']:.0f}")
    print(f"  Median    : {stats['median']:.1f}")
    print(f"  % at K=10 : {stats['pct_at_lower_bound']:.1f}%  (lower bound binding)")
    print(f"  % at K=30 : {stats['pct_at_upper_bound']:.1f}%  (upper bound binding)")
    print()

    return stats


# ---------------------------------------------------------------------------
# 2. Subperiod robustness
# ---------------------------------------------------------------------------

def subperiod_robustness(results: dict) -> pd.DataFrame:
    """Net Sharpe by subperiod for each model.

    results : {model_name: DataFrame with columns date, net_excess_ret, ...}
    Returns a DataFrame (rows=subperiods, cols=models).
    """
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

    table = pd.DataFrame(rows).T          # rows=subperiods, cols=models
    table.index.name = "Subperiod"

    print("=" * 70)
    print("Subperiod Robustness — Net Sharpe Ratio")
    print("=" * 70)
    print(table.to_string())
    print()

    return table


# ---------------------------------------------------------------------------
# 3. Transaction-cost sensitivity
# ---------------------------------------------------------------------------

def transaction_cost_sensitivity(results: dict) -> pd.DataFrame:
    """Sharpe ratio sensitivity to gamma for each model.

    For each gamma, net_excess = excess_ret − gamma × turnover.
    Returns a DataFrame (rows=gamma values, cols=models).
    """
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

    print("=" * 70)
    print("Transaction-Cost Sensitivity — Net Sharpe Ratio")
    print(f"  (baseline gamma = {TRANSACTION_COST})")
    print("=" * 70)
    print(table.to_string())
    print()

    return table


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = {
        "1/N":               pd.read_parquet("results/benchmarks/equal_weight_full.parquet"),
        "Unconstrained MVO": pd.read_parquet("results/benchmarks/mvo_unconstrained.parquet"),
        "Constrained MVO":   pd.read_parquet("results/benchmarks/mvo_constrained.parquet"),
        "GA":                pd.read_parquet("results/ga/ga_results.parquet"),
    }

    k_behavior_summary(results["GA"])
    subperiod_robustness(results)
    transaction_cost_sensitivity(results)
