"""
Figure 6 — Mean-Variance Efficient Frontier (selected periods)

For 3 representative rebalancing dates:
- Pre-crisis : 2007-01-01
- Post-crisis: 2010-01-01
- Recent     : 2020-01-01

At each date:
1. Recompute mu and sigma from the 60-month estimation window
2. Trace the efficient frontier via parametric sweep
3. Overlay actual portfolio positions of GA, constrained MVO,
   unconstrained MVO, and 1/N as scatter points

GA weights are obtained by running run_ga once at each date with a
reduced generation budget (N_GENS_FRONTIER=50) for speed. This is
illustrative only — the main experiment used 200 generations.

Note: All positions use in-sample estimates (mu, sigma from 60-month
window). This figure is illustrative — out-of-sample performance is
evaluated separately.

Output: results/figures/F6_frontier.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import minimize
import os

from src.utils.portfolio import get_estimation_window
from src.utils.data import load_data
from src.benchmarks.mvo import optimize_mvo
from src.optimization import genetic_algorithm as ga_module
from src.optimization.genetic_algorithm import run_ga

# ── Config ────────────────────────────────────────────────────────────────────
REPR_DATES  = ["2007-01-01", "2010-01-01", "2020-01-01"]
DATE_LABELS = ["Pre-crisis (Jan 2007)",
               "Post-crisis (Jan 2010)",
               "Recent (Jan 2020)"]

N_FRONTIER_POINTS = 30    # 30 points sufficient for smooth curve
N_GENS_FRONTIER   = 50    # reduced from 200 — illustrative only
BASE_SEED         = 42
OUT_DIR           = "results/figures"

PORTFOLIO_STYLES = {
    "GA"               : {"marker": "*", "ms": 14, "color": "#000000",
                          "zorder": 6,  "offset_vol": 0.0, "offset_ret": 0.0},
    "Constrained MVO"  : {"marker": "s", "ms":  8, "color": "#444444",
                          "zorder": 5,  "offset_vol": 0.0, "offset_ret": 0.0},
    "Unconstrained MVO": {"marker": "D", "ms":  9, "color": "#333333",
                          "zorder": 10, "offset_vol": 0.4, "offset_ret": 0.8},
    "1/N"              : {"marker": "o", "ms":  8, "color": "#bbbbbb",
                          "zorder": 3,  "offset_vol": 0.0, "offset_ret": 0.0},
}

plt.rcParams.update({
    "figure.dpi"       : 300,
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "legend.fontsize"  : 9,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})


# ── Frontier Computation ──────────────────────────────────────────────────────

def min_variance_portfolio(sigma:         np.ndarray,
                           target_return: float,
                           mu:            np.ndarray,
                           w_max:         float = 1.0) -> np.ndarray | None:
    """Minimum variance portfolio for a given target return."""
    N = len(mu)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
    ]
    bounds = [(0.0, w_max)] * N
    w0     = np.ones(N) / N

    try:
        result = minimize(
            fun         = lambda w: float(w @ sigma @ w),
            x0          = w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 200, "ftol": 1e-8},
        )
        if result.success:
            return result.x
    except Exception:
        pass
    return None


def compute_frontier(mu:       np.ndarray,
                     sigma:    np.ndarray,
                     n_points: int   = N_FRONTIER_POINTS,
                     w_max:    float = 1.0) -> tuple:
    """Trace the efficient frontier. Returns annualized (vols, returns)."""
    r_min = mu.min()
    r_max = mu.max() * 0.95

    frontier_vols = []
    frontier_rets = []

    for r_target in np.linspace(r_min, r_max, n_points):
        w = min_variance_portfolio(sigma, r_target, mu, w_max)
        if w is not None:
            frontier_vols.append(float(np.sqrt(w @ sigma @ w)) * np.sqrt(12))
            frontier_rets.append(float(w @ mu) * 12)

    return np.array(frontier_vols), np.array(frontier_rets)


def portfolio_position(weights: np.ndarray,
                       mu:      np.ndarray,
                       sigma:   np.ndarray) -> tuple:
    """Annualized (vol, return) for a weight vector."""
    vol = float(np.sqrt(weights @ sigma @ weights)) * np.sqrt(12)
    ret = float(weights @ mu) * 12
    return vol, ret


# ── Portfolio Weights ─────────────────────────────────────────────────────────

def get_portfolio_weights(mu:       np.ndarray,
                          sigma:    np.ndarray,
                          n_assets: int) -> dict:
    """
    Compute portfolio weights for each strategy at a given date.

    GA: single run_ga call with reduced N_GENS_FRONTIER=50 generations
        (illustrative only — main experiment used 200 generations).
        Module-level N_GENS is temporarily overridden and restored.
    MVO: optimize_mvo with same bounds as main experiment.
    1/N: equal weight.
    """
    rng = np.random.default_rng(BASE_SEED)

    # GA — exact weights, reduced generations for speed
    print("    Running GA (50 gens — illustrative)...")
    orig_n_gens    = ga_module.N_GENS
    ga_module.N_GENS = N_GENS_FRONTIER
    try:
        ga_w = run_ga(
            n_assets     = n_assets,
            mu           = mu,
            sigma        = sigma,
            prev_weights = None,
            rng          = rng,
        )
    finally:
        ga_module.N_GENS = orig_n_gens  # always restore

    # MVO unconstrained
    print("    Running Unconstrained MVO...")
    rng = np.random.default_rng(BASE_SEED)
    mvo_unc_w = optimize_mvo(
        mu    = mu,
        sigma = sigma,
        w_min = 0.0,
        w_max = 1.0,
        rng   = rng,
    )

    # MVO constrained
    print("    Running Constrained MVO...")
    rng = np.random.default_rng(BASE_SEED)
    mvo_con_w = optimize_mvo(
        mu    = mu,
        sigma = sigma,
        w_min = 0.0,
        w_max = 0.15,
        rng   = rng,
    )

    # 1/N
    ew_w = np.ones(n_assets) / n_assets

    return {
        "GA"               : ga_w,
        "Unconstrained MVO": mvo_unc_w,
        "Constrained MVO"  : mvo_con_w,
        "1/N"              : ew_w,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_frontier() -> None:
    print("Loading data...")
    universe, returns = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, date_str, label in zip(axes, REPR_DATES, DATE_LABELS):
        t = pd.Timestamp(date_str)
        print(f"\nProcessing {label}...")

        eligible = universe[universe["date"] == t]["permno"].tolist()
        if not eligible:
            ax.set_title(f"{label}\n(no data)")
            continue

        mu, sigma, valid_permnos = get_estimation_window(returns, eligible, t)
        if mu is None:
            ax.set_title(f"{label}\n(estimation failed)")
            continue

        n_assets = len(valid_permnos)
        print(f"  Universe: {n_assets} stocks")

        # Compute frontiers
        print("  Computing unconstrained frontier...")
        vols_unc, rets_unc = compute_frontier(mu, sigma, w_max=1.0)

        print("  Computing constrained frontier (w<=0.15)...")
        vols_con, rets_con = compute_frontier(mu, sigma, w_max=0.15)

        if len(vols_unc) > 1:
            ax.plot(vols_unc * 100, rets_unc * 100,
                    color="#000000", lw=1.5, ls="-",
                    label="Frontier (unconstrained)", zorder=2)
        if len(vols_con) > 1:
            ax.plot(vols_con * 100, rets_con * 100,
                    color="#666666", lw=1.5, ls="--",
                    label="Frontier (w ≤ 0.15)", zorder=2)

        # Portfolio positions
        print("  Computing portfolio positions...")
        weights = get_portfolio_weights(mu, sigma, n_assets)

        for name, w in weights.items():
            vol, ret = portfolio_position(w, mu, sigma)
            s = PORTFOLIO_STYLES[name]
            ax.scatter(
                vol * 100 + s["offset_vol"],
                ret * 100 + s["offset_ret"],
                marker    = s["marker"],
                s         = s["ms"] ** 2,
                color     = s["color"],
                zorder    = s["zorder"],
                label     = name,
                edgecolor = "black",
                linewidth = 0.6,
            )

        ax.set_title(label)
        ax.set_xlabel("Annualized Volatility (%)")
        if ax is axes[0]:
            ax.set_ylabel("Annualized Expected Return (%)")

    # Shared legend
    handles = [
        mlines.Line2D([], [], color="#000000", ls="-",
                      label="Frontier (unconstrained)"),
        mlines.Line2D([], [], color="#666666", ls="--",
                      label="Frontier (w ≤ 0.15)"),
    ]
    for name, s in PORTFOLIO_STYLES.items():
        handles.append(plt.scatter([], [],
                                   marker    = s["marker"],
                                   s         = s["ms"] ** 2,
                                   color     = s["color"],
                                   edgecolors= "black",
                                   linewidths= 0.6,
                                   label     = name))

    fig.legend(handles=handles, loc="lower center",
               ncol=6, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(
        "Figure 6: Mean-Variance Efficient Frontier at Representative Dates\n"
        "(in-sample estimates from 60-month window; positions are in-sample only)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/F6_frontier.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_frontier()