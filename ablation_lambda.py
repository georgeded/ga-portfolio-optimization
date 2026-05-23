#!/usr/bin/env python3
"""
Runs GA with lambda=0 on 20 evenly-spaced periods at reduced settings
(5 runs, 100 gens) and compares against the tuned lambda=1.8437 results.
Runtime: ~1.5 hours on c2-standard-16.

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
N_RUNS            = 5
N_GENS            = 100
GAMMA             = TRANSACTION_COST
MAIN_RESULTS_PATH = "results/ga/ga_results.parquet"
OUTPUT_DIR        = "results/ablation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_period(t, apply_date, universe, returns, prev_weights, prev_permnos, lam, pool):
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

    # Temporarily swap lambda so run_ga uses the ablation value
    orig_lambda, orig_ngens = ga_module.LAMBDA, ga_module.N_GENS
    ga_module.LAMBDA = lam
    ga_module.N_GENS = N_GENS
    try:
        args = [
            (i % N_RUNNERS, (n_assets, mu, sigma, pw_aligned, BASE_SEED + i, N_GENS))
            for i in range(N_RUNS)
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
    print("λ ablation: 0.0 vs 1.8437")
    print(f"{N_PERIODS} periods  |  {N_RUNS} runs  |  {N_GENS} gens\n")

    universe, returns = load_data()
    rebalance_dates   = sorted(universe["date"].unique())
    all_return_dates  = sorted(returns["date"].unique())

    indices       = np.linspace(0, len(rebalance_dates) - 1, N_PERIODS, dtype=int)
    sampled_dates = [rebalance_dates[i] for i in indices]
    print(f"Period range: {sampled_dates[0].date()} → {sampled_dates[-1].date()}\n")

    t0           = time.time()
    results_lam0 = []
    prev_weights = None
    prev_permnos = None

    with mp.Pool(processes=min(N_RUNS, mp.cpu_count())) as pool:
        for t in tqdm(sampled_dates, desc="λ=0"):
            apply = [d for d in all_return_dates if d.year == t.year and d.month == t.month]
            if not apply:
                continue
            result, prev_weights, prev_permnos = run_period(
                t, apply[0], universe, returns,
                prev_weights, prev_permnos, LAMBDA_ABLATION, pool,
            )
            if result is not None:
                results_lam0.append(result)

    print(f"\nDone: {len(results_lam0)} periods in {(time.time() - t0) / 60:.1f} min")

    df_lam0 = pd.DataFrame(results_lam0)
    df_lam0.to_parquet(f"{OUTPUT_DIR}/lambda0_results.parquet", index=False)

    rows = [metrics_row(df_lam0, "0.0  (ablation)")]

    if os.path.exists(MAIN_RESULTS_PATH):
        df_main     = pd.read_parquet(MAIN_RESULTS_PATH)
        df_main["date"] = pd.to_datetime(df_main["date"])
        sampled_set = set(df_lam0["date"].values)
        df_sub      = df_main[df_main["date"].isin(sampled_set)]
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
    print(f"\nNote: λ=0 is on {N_PERIODS} sampled periods at reduced settings.")
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