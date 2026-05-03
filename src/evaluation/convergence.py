"""
Figure A1 — GA Convergence (Tuned vs Default Parameters)

Three dates: Jan 2007 (pre-crisis), Jan 2010 (post-crisis), Jan 2020 (recent).
N_RUNS_CONV=8 independent runs; fitness = pure in-sample Sharpe (prev_weights=None, λ inactive).
λ excluded from this diagnostic — comparison isolates pc/pm/sigma_m only.

Outputs: A1_convergence_tuned.png, A1_convergence_default.png, A1_convergence_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.utils.data import load_data
from src.utils.portfolio import get_estimation_window
from src.optimization import genetic_algorithm as ga_module
from src.optimization.genetic_algorithm import run_ga, N_GENS

REPR_DATES  = ["2007-01-01", "2010-01-01", "2020-01-01"]
DATE_LABELS = ["Pre-crisis (Jan 2007)",
               "Post-crisis (Jan 2010)",
               "Recent (Jan 2020)"]
N_RUNS_CONV = 8
BASE_SEED   = 42
OUT_DIR     = "results/figures"
TUNED_LABEL = "Tuned (Optuna)"

# Tuned parameters (Optuna — best_params.json)
TUNED_PARAMS = {
    "pc"      : 0.6054,
    "pm"      : 0.1370,
    "sigma_m" : 0.1469,
    "lambda_" : 1.8437,  # inactive in this diagnostic (prev_weights=None)
    "label"   : TUNED_LABEL,
}

# Default parameters (standard EA literature defaults)
DEFAULT_PARAMS = {
    "pc"      : 0.8,
    "pm"      : 0.1,
    "sigma_m" : 0.05,
    "lambda_" : 0.0,   # also inactive (prev_weights=None)
    "label"   : "Default",
}

CONFIG_STYLES = {
    TUNED_LABEL      : {"lw": 2.0, "ls": "-",  "alpha_band": 0.15},
    "Default"       : {"lw": 1.6, "ls": "--", "alpha_band": 0.10},
}

DATE_COLORS = ["#000000", "#555555", "#999999"]

# Y-axis label — explicit that λ is inactive and fitness = pure Sharpe
YAXIS_LABEL = "In-sample Sharpe  (prev_weights=None, λ inactive)"

plt.rcParams.update({
    "figure.dpi"       : 300,
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 10,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})


def run_convergence_for_config(mu:       np.ndarray,
                               sigma:    np.ndarray,
                               n_assets: int,
                               params:   dict) -> tuple:
    """
    Temporarily overrides ga_module constants; restores after run regardless of exceptions.
    prev_weights=None: λ inactive, fitness = pure in-sample Sharpe.
    Returns (median_traj, q25, q75, generations, median_stop).
    """
    orig_pc      = ga_module.PC
    orig_pm      = ga_module.PM
    orig_sigma_m = ga_module.SIGMA_M
    orig_lambda  = ga_module.LAMBDA

    ga_module.PC      = params["pc"]
    ga_module.PM      = params["pm"]
    ga_module.SIGMA_M = params["sigma_m"]
    ga_module.LAMBDA  = params["lambda_"]

    try:
        all_histories = []
        for run_i in range(N_RUNS_CONV):
            rng = np.random.default_rng(BASE_SEED + run_i)
            _, history = run_ga(
                n_assets       = n_assets,
                mu             = mu,
                sigma          = sigma,
                prev_weights   = None,  # λ inactive — see module docstring
                rng            = rng,
                return_history = True,
            )
            all_histories.append(history)

    finally:
        ga_module.PC      = orig_pc
        ga_module.PM      = orig_pm
        ga_module.SIGMA_M = orig_sigma_m
        ga_module.LAMBDA  = orig_lambda

    # Pad shorter runs with final value (early stopping → converged)
    max_len = max(len(h) for h in all_histories)
    padded  = np.array([
        h + [h[-1]] * (max_len - len(h))
        for h in all_histories
    ])

    median_traj = np.median(padded, axis=0)
    q25         = np.percentile(padded, 25, axis=0)
    q75         = np.percentile(padded, 75, axis=0)
    generations = np.arange(1, max_len + 1)
    median_stop = int(np.median([len(h) for h in all_histories]))

    return median_traj, q25, q75, generations, median_stop


def plot_single_config(results:    dict,
                       config_lbl: str,
                       filename:   str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (date_str, date_label) in enumerate(zip(REPR_DATES, DATE_LABELS)):
        median_traj, q25, q75, generations, median_stop = \
            results[config_lbl][date_str]
        color = DATE_COLORS[i]
        style = CONFIG_STYLES[config_lbl]

        ax.plot(generations, median_traj,
                color=color, lw=style["lw"], ls=style["ls"],
                label=f"{date_label} (stop gen: {median_stop})")
        ax.fill_between(generations, q25, q75,
                        alpha=style["alpha_band"], color=color)

    ax.set_title(
        f"Figure A1: GA Convergence — {config_lbl} Parameters\n"
        f"(median ± IQR across {N_RUNS_CONV} independent runs)"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel(YAXIS_LABEL)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out_path = f"{OUT_DIR}/{filename}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_comparison(results: dict) -> None:
    """
    Primary figure: tuned vs default overlaid, one subplot per date.
    This is the key figure for RQ3 — isolates pc/pm/sigma_m effect.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, (date_str, date_label) in enumerate(zip(REPR_DATES, DATE_LABELS)):
        ax = axes[i]

        for config_lbl in [TUNED_LABEL, "Default"]:
            median_traj, q25, q75, generations, median_stop = \
                results[config_lbl][date_str]
            style = CONFIG_STYLES[config_lbl]
            color = "#000000" if config_lbl == TUNED_LABEL else "#888888"

            ax.plot(generations, median_traj,
                    color=color, lw=style["lw"], ls=style["ls"],
                    label=f"{config_lbl} (stop: {median_stop})")
            ax.fill_between(generations, q25, q75,
                            alpha=style["alpha_band"], color=color)

        ax.set_title(date_label)
        ax.set_xlabel("Generation")
        if i == 0:
            ax.set_ylabel(YAXIS_LABEL)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig.suptitle(
        "Figure A1: GA Convergence — Tuned vs Default Parameters\n"
        f"(median ± IQR, {N_RUNS_CONV} runs; "
        "fitness = in-sample Sharpe, λ inactive)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = f"{OUT_DIR}/A1_convergence_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_convergence() -> None:
    print("Loading data...")
    universe, returns = load_data()

    results = {
        TUNED_PARAMS["label"]  : {},
        DEFAULT_PARAMS["label"]: {},
    }

    for date_str, date_label in zip(REPR_DATES, DATE_LABELS):
        t = pd.Timestamp(date_str)
        print(f"\n{'='*60}")
        print(f"Date: {date_label}")
        print(f"{'='*60}")

        eligible = universe[universe["date"] == t]["permno"].tolist()
        if not eligible:
            print(f"WARNING: No eligible stocks at {date_str}, skipping.")
            continue

        mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
        if mu is None:
            print(f"WARNING: Estimation window failed at {date_str}, skipping.")
            continue

        n_assets = len(valid_permnos)
        print(f"Universe: {n_assets} stocks")

        for params in [TUNED_PARAMS, DEFAULT_PARAMS]:
            lbl = params["label"]
            print(f"\n  Running {lbl} "
                  f"(pc={params['pc']}, pm={params['pm']}, "
                  f"sigma_m={params['sigma_m']}) "
                  f"[λ inactive — prev_weights=None]...")

            median_traj, q25, q75, generations, median_stop = \
                run_convergence_for_config(mu, sigma, n_assets, params)

            results[lbl][date_str] = (
                median_traj, q25, q75, generations, median_stop
            )

            print(f"    Median final fitness : {median_traj[-1]:.4f}")
            print(f"    Median stopping gen  : {median_stop} / {N_GENS}")
            print(f"    IQR at final gen     : "
                  f"[{q25[-1]:.4f}, {q75[-1]:.4f}]")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print("Saving figures...")

    plot_single_config(results, TUNED_PARAMS["label"],
                       "A1_convergence_tuned.png")
    plot_single_config(results, DEFAULT_PARAMS["label"],
                       "A1_convergence_default.png")
    plot_comparison(results)

    print("\nAll convergence figures saved.")


if __name__ == "__main__":
    run_convergence()