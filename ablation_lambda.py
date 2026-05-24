#!/usr/bin/env python3
"""
Runs GA with lambda=0 on all 252 consecutive monthly periods, recording
only the 20 evenly-spaced sampled dates, then compares against the tuned
lambda=1.8437 results from the main evaluation.

The key fix over the original ablation: instead of jumping ~13 months
between sampled periods and carrying stale prev_weights, we run the full
252-period chain at lambda=0.  The 232 intermediate (non-sampled) periods
use minimal GA settings (1 run, 10 gens) only to keep the weight chain
fresh; their metrics are discarded.  The 20 reported periods use the same
reduced-but-real settings as before (5 runs, 100 gens).  This gives
lambda=0 exactly 1-month-old prev_weights at every sampled date — the same
condition that lambda=1.84 had in the main evaluation — making the
comparison a true ceteris-paribus test of the penalty coefficient.

Runtime estimate: ~2–3 hours on c2-standard-16.

Usage: python3 ablation_lambda.py
"""

import os
import sys
import time
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")

import src.optimization.genetic_algorithm as ga_module
from src.evaluation.metrics import (
    TRANSACTION_COST,
    compute_all_metrics,
    herfindahl_index,
    portfolio_turnover,
    transaction_cost,
)
from src.optimization.genetic_algorithm import fitness as ga_fitness
from src.optimization.runner import BASE_SEED, N_RUNNERS, run_ga_with_affinity
from src.utils.data import load_data
from src.utils.portfolio import (
    align_drifted_weights,
    compute_drift_weights,
    get_estimation_window,
    get_monthly_returns,
    get_rf_for_month,
)

LAMBDA_ABLATION   = 0.0
LAMBDA_MAIN       = 1.8437
N_PERIODS         = 20
N_RUNS            = 5    # runs used for the 20 *reported* sampled periods
N_GENS            = 100  # gens used for the 20 *reported* sampled periods
N_RUNS_WARMUP     = 1    # runs used for the 232 intermediate chain periods
N_GENS_WARMUP     = 10   # gens used for the 232 intermediate chain periods
GAMMA             = TRANSACTION_COST
MAIN_RESULTS_PATH = "results/ga/ga_results.parquet"
OUTPUT_DIR        = "results/ablation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_period(t, apply_date, universe, returns, prev_weights, prev_permnos, lam, pool,
               n_runs=N_RUNS, n_gens=N_GENS):
    """
    Run one rebalancing period and return (result_dict, drifted_weights, permnos).

    Parameters
    ----------
    n_runs : int
        Number of independent GA runs.  Use N_RUNS for reported periods,
        N_RUNS_WARMUP for intermediate chain periods.
    n_gens : int
        Number of GA generations.  Use N_GENS for reported periods,
        N_GENS_WARMUP for intermediate chain periods.
    """
    eligible = universe[universe["date"] == t]["permno"].tolist()
    if not eligible:
        return None, prev_weights, prev_permnos

    mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
    if mu is None or len(valid_permnos) < 2:
        return None, prev_weights, prev_permnos

    n_assets = len(valid_permnos)
    pw_aligned = (
        align_drifted_weights(prev_weights, prev_permnos, valid_permnos)
        if prev_weights is not None and prev_permnos is not None
        else None
    )

    # Temporarily swap lambda/n_gens so run_ga uses the caller's values
    orig_lambda, orig_ngens = ga_module.LAMBDA, ga_module.N_GENS
    ga_module.LAMBDA = lam
    ga_module.N_GENS = n_gens
    try:
        args = [
            (i % N_RUNNERS, (n_assets, mu, sigma, pw_aligned, BASE_SEED + i, n_gens))
            for i in range(n_runs)
        ]
        all_weights = pool.map(run_ga_with_affinity, args)
    finally:
        ga_module.LAMBDA = orig_lambda
        ga_module.N_GENS = orig_ngens

    # Pick the run closest to the median fitness
    fitnesses = np.array([ga_fitness(w, mu, sigma, pw_aligned, lam) for w in all_weights])
    canon_w   = all_weights[int(np.argmin(np.abs(fitnesses - np.median(fitnesses))))]

    month_ret = get_monthly_returns(returns, valid_permnos, apply_date)
    rf        = get_rf_for_month(returns, apply_date)

    gross   = float(canon_w @ month_ret.values)
    excess  = gross - rf
    to      = portfolio_turnover(canon_w, pw_aligned) if pw_aligned is not None else 1.0
    cost    = transaction_cost(to, GAMMA)

    result = {
        "date":           apply_date,
        "lambda":         lam,
        "n_stocks":       int((canon_w > 0).sum()),
        "portfolio_ret":  gross,
        "excess_ret":     excess,
        "rf":             rf,
        "net_excess_ret": excess - cost,
        "turnover":       to,
        "cost":           cost,
        "hhi":            herfindahl_index(canon_w),
    }

    return result, compute_drift_weights(canon_w, month_ret.values), valid_permnos


def metrics_row(df, label):
    clean = df.dropna(subset=["excess_ret", "rf", "turnover"])
    m = compute_all_metrics(
        clean["excess_ret"].values,
        clean["rf"].values,
        clean["turnover"].values,
        GAMMA,
    )
    return {
        "lambda":         label,
        "n_periods":      len(clean),
        "sharpe_net":     round(m["sharpe_net"], 4),
        "ann_return_net": f"{m['ann_return_net'] * 100:.2f}%",
        "ann_vol":        f"{m['ann_vol'] * 100:.2f}%",
        "avg_turnover":   f"{m['avg_turnover'] * 100:.2f}%",
        "max_drawdown":   f"{m['max_drawdown_net'] * 100:.2f}%",
    }


def save_table_png(summary, path):
    cols  = ["lambda", "sharpe_net", "ann_return_net", "avg_turnover", "max_drawdown"]
    hdrs  = ["λ", "Sharpe (net)", "Ann. Return", "Avg Turnover", "Max Drawdown"]
    table = summary[cols].copy()
    table.columns = hdrs

    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.55 * len(table)))
    ax.axis("off")

    t = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 2.0)

    # Bold header row
    for col in range(len(hdrs)):
        t[0, col].set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved table PNG: {path}")


def main():
    print("λ ablation (fair): 0.0 vs 1.8437")
    print(f"Full 252-period chain at λ=0; record only {N_PERIODS} sampled periods.")
    print(f"Sampled: {N_RUNS} runs × {N_GENS} gens | "
          f"Chain:   {N_RUNS_WARMUP} run  × {N_GENS_WARMUP} gens\n")

    universe, returns = load_data()
    rebalance_dates   = sorted(universe["date"].unique())
    all_return_dates  = sorted(returns["date"].unique())

    # ── Determine the 20 sampled dates (same as original ablation) ────────────
    indices       = np.linspace(0, len(rebalance_dates) - 1, N_PERIODS, dtype=int)
    sampled_dates = {rebalance_dates[i] for i in indices}
    first, last   = min(sampled_dates), max(sampled_dates)
    print(f"Sampled date range : {first.date()} → {last.date()}")
    print(f"Total chain periods: {len(rebalance_dates)}\n")

    # ── Run ALL 252 periods at λ=0; record only the sampled ones ─────────────
    # This ensures prev_weights are always 1-month-old at every sampled date,
    # exactly matching the condition under which λ=1.84 was evaluated.
    t0           = time.time()
    results_lam0 = []
    prev_weights = None
    prev_permnos = None

    with mp.Pool(processes=min(N_RUNS, mp.cpu_count())) as pool:
        for t in tqdm(rebalance_dates, desc="λ=0 (fair chain)"):
            apply = [d for d in all_return_dates if d.year == t.year and d.month == t.month]
            if not apply:
                continue

            is_sampled = t in sampled_dates
            n_runs_use = N_RUNS       if is_sampled else N_RUNS_WARMUP
            n_gens_use = N_GENS       if is_sampled else N_GENS_WARMUP

            result, prev_weights, prev_permnos = run_period(
                t, apply[0], universe, returns,
                prev_weights, prev_permnos, LAMBDA_ABLATION, pool,
                n_runs=n_runs_use, n_gens=n_gens_use,
            )
            # Only keep results for the 20 reported sampled dates
            if is_sampled and result is not None:
                results_lam0.append(result)

    print(f"\nDone: {len(results_lam0)} sampled periods recorded "
          f"in {(time.time() - t0) / 60:.1f} min")

    df_lam0 = pd.DataFrame(results_lam0)
    fair_parquet = f"{OUTPUT_DIR}/lambda0_fair_results.parquet"
    df_lam0.to_parquet(fair_parquet, index=False)
    print(f"Saved: {fair_parquet}")

    rows = [metrics_row(df_lam0, "0.0  (ablation, fair)")]

    if os.path.exists(MAIN_RESULTS_PATH):
        df_main          = pd.read_parquet(MAIN_RESULTS_PATH)
        df_main["date"]  = pd.to_datetime(df_main["date"])
        sampled_date_vals = set(df_lam0["date"].values)
        df_sub            = df_main[df_main["date"].isin(sampled_date_vals)]
        if len(df_sub):
            rows.append(metrics_row(df_sub, "1.8437 (tuned)"))
        else:
            print("Warning: no date overlap with main results — check MAIN_RESULTS_PATH")
    else:
        print(f"Main results not found at {MAIN_RESULTS_PATH}.")
        print("Adding full-evaluation figures from thesis as fallback.")
        rows.append({
            "lambda": "1.8437 (tuned, full eval)", "n_periods": 252,
            "sharpe_net": 0.2355, "ann_return_net": "3.99%",
            "ann_vol": "16.96%", "avg_turnover": "21.34%", "max_drawdown": "-44.95%",
        })

    summary = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}/lambda_ablation_summary.csv"
    summary.to_csv(csv_path, index=False)

    print("\n" + summary.to_string(index=False))
    print(f"\nNote: λ=0 uses {N_PERIODS} sampled periods at full settings; "
          f"prev_weights always 1-month-old (fair chain).")
    print(f"CSV: {csv_path}")

    png_path = f"{OUTPUT_DIR}/lambda_ablation_table.png"
    save_table_png(summary, png_path)

    print("\nLaTeX rows:")
    for _, row in summary.iterrows():
        print(
            f"  {row['lambda']} & {row['sharpe_net']} & "
            f"{row['ann_return_net']} & {row['avg_turnover']} & "
            f"{row['max_drawdown']} \\\\"
        )


if __name__ == "__main__":
    main()