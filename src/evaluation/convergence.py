"""
Appendix Figure A1 — GA Convergence Plot

Runs the GA with return_history=True at 3 representative rebalancing dates:
- Pre-crisis : 2007-01-01
- Post-crisis: 2010-01-01
- Recent     : 2020-01-01

For each date, runs N_RUNS_CONV independent GA instances and plots the
median best-fitness trajectory across runs, with IQR band showing dispersion.

Purpose: confirms 200 generations is sufficient for convergence and that
early stopping does not terminate prematurely.

Note: prev_weights=None is used for all dates, so fitness reflects
in-sample Sharpe without turnover penalty. This is appropriate for a
convergence diagnostic but differs slightly from the main experiment
where lambda * turnover is active.

Output: results/figures/A1_convergence.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.utils.data import load_data
from src.utils.portfolio import get_estimation_window
from src.optimization.genetic_algorithm import run_ga, N_GENS

# ── Config ────────────────────────────────────────────────────────────────────
REPR_DATES  = ["2007-01-01", "2010-01-01", "2020-01-01"]
DATE_LABELS = ["Pre-crisis (Jan 2007)",
               "Post-crisis (Jan 2010)",
               "Recent (Jan 2020)"]
N_RUNS_CONV = 8    # runs per date — matches main experiment
BASE_SEED   = 42   # matches runner.py
OUT_DIR     = "results/figures"

# Greyscale line styles — consistent with thesis figures
LINE_STYLES = [
    {"color": "#000000", "lw": 2.0, "ls": "-"},
    {"color": "#555555", "lw": 1.8, "ls": "--"},
    {"color": "#999999", "lw": 1.8, "ls": "-."},
]

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


# ── Main ──────────────────────────────────────────────────────────────────────

def run_convergence() -> None:
    print("Loading data...")
    universe, returns = load_data()

    fig, ax = plt.subplots(figsize=(10, 5))

    for date_str, label, style in zip(REPR_DATES, DATE_LABELS, LINE_STYLES):
        t = pd.Timestamp(date_str)

        # Get eligible universe at this date
        eligible = universe[universe["date"] == t]["permno"].tolist()
        if len(eligible) == 0:
            print(f"WARNING: No eligible stocks at {date_str}, skipping.")
            continue

        # Build estimation window — identical to main runner
        mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
        if mu is None or len(valid_permnos) < 2:
            print(f"WARNING: Estimation window failed at {date_str}, skipping.")
            continue

        n_assets = len(valid_permnos)
        print(f"\n{label}: N={n_assets} stocks, "
              f"running {N_RUNS_CONV} GA instances...")

        # Run N_RUNS_CONV independent GA instances with history
        all_histories = []
        for run_i in range(N_RUNS_CONV):
            rng = np.random.default_rng(BASE_SEED + run_i)
            _, history = run_ga(
                n_assets       = n_assets,
                mu             = mu,
                sigma          = sigma,
                prev_weights   = None,  # no prior portfolio — see module note
                rng            = rng,
                return_history = True,
            )
            all_histories.append(history)
            print(f"  Run {run_i+1}/{N_RUNS_CONV}: "
                  f"{len(history)} generations, "
                  f"best fitness = {history[-1]:.4f}")

        # Pad histories to same length with final value.
        # Runs that early-stopped have genuinely converged — padding
        # reflects that best fitness does not change after stopping.
        max_len = max(len(h) for h in all_histories)
        padded  = np.array([
            h + [h[-1]] * (max_len - len(h))
            for h in all_histories
        ])

        # Median trajectory and IQR band across runs
        median_traj = np.median(padded, axis=0)
        q25         = np.percentile(padded, 25, axis=0)
        q75         = np.percentile(padded, 75, axis=0)
        generations = np.arange(1, max_len + 1)

        # Median stopping generation (correct convergence diagnostic)
        median_stop = int(np.median([len(h) for h in all_histories]))

        ax.plot(generations, median_traj, label=label, **style)
        ax.fill_between(generations, q25, q75,
                        alpha=0.15, color=style["color"])

        print(f"  Median final fitness : {median_traj[-1]:.4f}")
        print(f"  Median stopping gen  : {median_stop} / {N_GENS}")
        print(f"  IQR at final gen     : [{q25[-1]:.4f}, {q75[-1]:.4f}]")

    ax.set_title(
        "Figure A1: GA Convergence — Best Fitness per Generation\n"
        "(median ± IQR across 8 independent runs, 3 representative dates)"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness  (Sharpe − λ·Turnover, in-sample)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/A1_convergence.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_convergence()