"""
K-sensitivity figures (RQ3): effect of fixed cardinality K on
Sharpe ratio, turnover, and diversification.

Produces two figures:
  F_K1: three-panel Sharpe / Turnover / HHI vs K with error bars
  F_K2: cumulative net returns for all K values on one plot

python3 -m src.evaluation.k_sensitivity_figures
"""

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.k_sensitivity import K_VALUES, OUTPUT_DIR

plt.rcParams.update({
    "figure.dpi":        300,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# Grayscale palette: darkest for K=10, lightest for K=30
_GRAYS = ["#000000", "#333333", "#666666", "#999999", "#bbbbbb"]
# Line styles for additional distinction in B&W print
_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

K_STYLES = {
    k: {"color": _GRAYS[i], "ls": _LINESTYLES[i], "lw": 1.8}
    for i, k in enumerate(K_VALUES)
}

SQRT_12 = np.sqrt(12)


def add_caption(fig, text: str) -> None:
    fig.text(
        0.5, -0.10, text,
        ha="center", va="top",
        fontsize=9, style="italic",
        color="#444444",
        transform=fig.transFigure,
    )


def load_k_results(k_values: list = K_VALUES) -> dict:
    """Load parquets for all K values. Raises if a file is missing."""
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
        df = df.sort_values("date").reset_index(drop=True)
        data[k] = df
    return data


def cumulative_returns(df: pd.DataFrame) -> np.ndarray:
    total = df["net_excess_ret"].values + df["rf"].values
    return np.cumprod(1 + total)


def _rolling_sharpe_series(net: np.ndarray, window: int = 12) -> np.ndarray:
    """Annualized rolling Sharpe over a trailing window. Returns non-NaN values."""
    s = pd.Series(net)
    mu = s.rolling(window).mean() * 12
    vol = s.rolling(window).std(ddof=1) * SQRT_12
    sr = (mu / vol.replace(0.0, np.nan)).dropna()
    return sr.values


def _rolling_mean_series(values: np.ndarray, window: int = 12) -> np.ndarray:
    """Rolling mean over a trailing window. Returns non-NaN values."""
    return pd.Series(values).rolling(window).mean().dropna().values


def compute_sharpe_net(df: pd.DataFrame) -> float:
    net = df["net_excess_ret"].values
    ann_ret = float(net.mean() * 12)
    ann_vol = float(net.std(ddof=1) * SQRT_12)
    return ann_ret / ann_vol if ann_vol > 0 else 0.0


def plot_fk1(data: dict) -> None:
    """Three-panel figure: net Sharpe, avg turnover, avg HHI vs K.

    Central value: full-period metric.
    Error bar half-width: std of rolling 12-month estimates across periods.
    """
    k_list = sorted(data.keys())
    sharpes = []
    sharpe_e = []
    turnovers = []
    turnover_e = []
    hhis = []
    hhi_e = []

    for k in k_list:
        df = data[k]
        net = df["net_excess_ret"].values
        to = df["turnover"].values
        hhi = df["hhi"].values

        sharpes.append(compute_sharpe_net(df))
        sharpe_e.append(float(np.std(_rolling_sharpe_series(net), ddof=1)))

        turnovers.append(float(to.mean()))
        turnover_e.append(float(np.std(_rolling_mean_series(to), ddof=1)))

        hhis.append(float(hhi.mean()))
        hhi_e.append(float(np.std(_rolling_mean_series(hhi), ddof=1)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    panel_data = [
        (axes[0], sharpes,    sharpe_e,   "Net Sharpe Ratio (annualized)",
         "Figure K1a: Sharpe vs K"),
        (axes[1], [t * 100 for t in turnovers],
                  [e * 100 for e in turnover_e],
         "Avg Monthly Turnover (%)",
         "Figure K1b: Turnover vs K"),
        (axes[2], hhis, hhi_e,
         "Avg HHI",
         "Figure K1c: Concentration vs K"),
    ]

    for ax, vals, errs, ylabel, title in panel_data:
        ax.errorbar(
            k_list, vals, yerr=errs,
            fmt="o-",
            color="#000000",
            lw=1.8,
            capsize=4,
            capthick=1.2,
            elinewidth=1.0,
            markersize=6,
        )
        ax.set_xticks(k_list)
        ax.set_xlabel("Fixed Cardinality K")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(
        "Figure K1: Effect of Fixed Cardinality K on Performance Metrics",
        fontsize=13,
        y=1.02,
    )

    caption_text = (
        "Effect of fixed cardinality K on net Sharpe ratio, average monthly "
        "turnover, and portfolio concentration (HHI). All models use fixed "
        "hyperparameters (pc=0.6054, pm=0.1370, sigma_m=0.1469, lambda=1.8437) "
        "to isolate the effect of K. Error bars show ±1 std of the "
        "rolling 12-month estimates across periods."
    )
    add_caption(fig, caption_text)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUTPUT_DIR, "FK1_sharpe_turnover_hhi_vs_k.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fk2(data: dict) -> None:
    """Cumulative net portfolio value for all K values on one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for k in sorted(data.keys()):
        df = data[k]
        s = K_STYLES[k]
        ax.plot(
            df["date"],
            cumulative_returns(df),
            color=s["color"],
            lw=s["lw"],
            ls=s["ls"],
            label=f"K={k}",
        )

    ax.set_title("Figure K2: Cumulative Net Portfolio Value by Fixed K (2005-2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (net of costs, initial = 1.0)")

    handles = [
        mlines.Line2D(
            [], [],
            color=K_STYLES[k]["color"],
            lw=K_STYLES[k]["lw"],
            ls=K_STYLES[k]["ls"],
            label=f"K={k}",
        )
        for k in sorted(data.keys())
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(K_VALUES),
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    add_caption(fig,
        "Cumulative net portfolio value for each fixed K. All returns net of "
        "transaction costs (gamma = 0.3%). Initial portfolio value normalised to 1.0. "
        "Fixed hyperparameters: pc=0.6054, pm=0.1370, sigma_m=0.1469, lambda=1.8437."
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = os.path.join(OUTPUT_DIR, "FK2_cumulative_returns_by_k.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading K-sensitivity results...")
    data = load_k_results()

    print(f"\nGenerating figures in {OUTPUT_DIR}/\n")
    plot_fk1(data)
    plot_fk2(data)

    print("\nAll K-sensitivity figures saved.")
