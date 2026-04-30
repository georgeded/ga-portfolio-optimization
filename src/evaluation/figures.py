"""
Thesis Figures
Generates all publication-quality figures for the thesis.

Figures produced:
- F1: Cumulative net returns (all 4 models)
- F2: Rolling 12-month Sharpe ratio
- F3: Monthly portfolio turnover
- F4: Herfindahl-Hirschman Index (concentration)
- F5: GA cardinality (K) over time

All figures saved to results/figures/ as 300 DPI PNGs.
Figures are greyscale-compatible (readable in black-and-white print).

Reference: SRC-RPT — "Figures must be readable in a black-and-white print."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ── Style ─────────────────────────────────────────────────────────────────────
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

# Greyscale-compatible styles (readable in B&W print)
STYLES = {
    "GA"               : {"color": "#000000", "lw": 2.0, "ls": "-",  "zorder": 5},
    "Unconstrained MVO": {"color": "#444444", "lw": 1.5, "ls": "--", "zorder": 4},
    "Constrained MVO"  : {"color": "#888888", "lw": 1.5, "ls": "-.", "zorder": 3},
    "1/N"              : {"color": "#bbbbbb", "lw": 1.5, "ls": ":",  "zorder": 2},
}

OUT_DIR    = "results/figures"
LEGEND_LOC = "upper right"

# Crisis periods — applied consistently across all time-series figures
CRISES = [
    {"start": "2008-09-01", "end": "2009-06-01",
     "color": "black", "label": "GFC (2008–09)"},
    {"start": "2020-02-01", "end": "2020-04-01",
     "color": "grey",  "label": "COVID (2020)"},
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all(
    ga_path      : str = "results/ga/ga_results.parquet",
    ew_path      : str = "results/benchmarks/equal_weight_full.parquet",
    mvo_unc_path : str = "results/benchmarks/mvo_unconstrained.parquet",
    mvo_con_path : str = "results/benchmarks/mvo_constrained.parquet",
) -> dict:
    data = {
        "GA"               : pd.read_parquet(ga_path),
        "Unconstrained MVO": pd.read_parquet(mvo_unc_path),
        "Constrained MVO"  : pd.read_parquet(mvo_con_path),
        "1/N"              : pd.read_parquet(ew_path),
    }
    for name, df in data.items():
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        data[name] = df

    # Strict date alignment — all series must cover identical periods
    dates = data["GA"]["date"].values
    for name, df in data.items():
        assert (df["date"].values == dates).all(), \
            f"Date mismatch: GA vs {name}"

    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def cumulative_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Cumulative net portfolio value starting at 1.0.
    Total return = net_excess_ret + rf (net of transaction costs).
    """
    total = df["net_excess_ret"].values + df["rf"].values
    return np.cumprod(1 + total)


def rolling_sharpe(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """
    Rolling annualized Sharpe on net excess returns.
    Returns Series with NaNs dropped (first window-1 periods).
    Clipped to [-5, 5] to guard against near-zero volatility periods.
    """
    net = pd.Series(df["net_excess_ret"].values, index=df["date"])
    mu  = net.rolling(window).mean() * 12
    vol = net.rolling(window).std(ddof=1) * np.sqrt(12)
    rs  = (mu / vol.replace(0, np.nan)).clip(-5, 5)
    return rs.dropna()


def smooth_series(series: pd.Series, window: int = 6) -> pd.Series:
    """Rolling mean with leading NaNs dropped."""
    return series.rolling(window).mean().dropna()


def add_crisis_shading(ax, include_legend: bool = True) -> None:
    """
    Add shaded crisis periods to a time-series axes object.
    Applied consistently across all time-series figures.
    """
    for crisis in CRISES:
        ax.axvspan(
            pd.Timestamp(crisis["start"]),
            pd.Timestamp(crisis["end"]),
            alpha=0.08,
            color=crisis["color"],
            label=crisis["label"] if include_legend else None,
            zorder=1,
        )


# ── F1: Cumulative Returns ────────────────────────────────────────────────────

def plot_cumulative_returns(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        cum = cumulative_returns(df)
        ax.plot(df["date"], cum, label=name, **STYLES[name])

    add_crisis_shading(ax, include_legend=True)

    ax.set_title("Figure 1: Cumulative Net Portfolio Value (2005–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (net of costs, initial = 1.0)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/F1_cumulative_returns.png")
    plt.close(fig)
    print("Saved: F1_cumulative_returns.png")


# ── F2: Rolling 12-Month Sharpe ───────────────────────────────────────────────

def plot_rolling_sharpe(data: dict, window: int = 12) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        rs = rolling_sharpe(df, window)  # NaNs already dropped
        ax.plot(rs.index, rs.values, label=name, **STYLES[name])

    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.5)
    add_crisis_shading(ax, include_legend=True)

    ax.set_title(f"Figure 2: Rolling {window}-Month Sharpe Ratio (net, annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/F2_rolling_sharpe.png")
    plt.close(fig)
    print("Saved: F2_rolling_sharpe.png")


# ── F3: Monthly Turnover ──────────────────────────────────────────────────────

def plot_turnover(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        # turnover is stored as fraction → multiply by 100 for %
        to = pd.Series(df["turnover"].values * 100, index=df["date"])
        to_smooth = smooth_series(to, window=6)  # NaNs dropped
        ax.plot(to_smooth.index, to_smooth.values, label=name, **STYLES[name])

    add_crisis_shading(ax, include_legend=True)

    ax.set_title("Figure 3: Monthly Portfolio Turnover (6-month rolling avg)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc=LEGEND_LOC, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/F3_turnover.png")
    plt.close(fig)
    print("Saved: F3_turnover.png")


# ── F4: HHI Concentration ─────────────────────────────────────────────────────

def plot_hhi(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        hhi = pd.Series(df["hhi"].values, index=df["date"])
        hhi_smooth = smooth_series(hhi, window=6)  # NaNs dropped
        ax.plot(hhi_smooth.index, hhi_smooth.values, label=name, **STYLES[name])

    add_crisis_shading(ax, include_legend=True)

    ax.set_title("Figure 4: Portfolio Concentration — HHI (6-month rolling avg)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Herfindahl-Hirschman Index")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc=LEGEND_LOC, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/F4_hhi.png")
    plt.close(fig)
    print("Saved: F4_hhi.png")


# ── F5: GA Cardinality (K) Over Time ─────────────────────────────────────────

def plot_cardinality(data: dict) -> None:
    df  = data["GA"]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["date"], df["n_stocks"], color="#000000", lw=1.2,
            alpha=0.6, label="K (monthly)")

    k_smooth = smooth_series(
        pd.Series(df["n_stocks"].values, index=df["date"]), window=12
    )
    ax.plot(k_smooth.index, k_smooth.values, color="#000000", lw=2.0,
            label="K (12-month avg)")

    ax.axhline(df["n_stocks"].mean(), color="#666666", lw=1.2, ls="--",
               label=f"Mean K = {df['n_stocks'].mean():.1f}")

    add_crisis_shading(ax, include_legend=True)

    ax.set_title("Figure 5: GA Selected Portfolio Size (K) Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks Selected (K)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.legend(loc=LEGEND_LOC, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/F5_cardinality.png")
    plt.close(fig)
    print("Saved: F5_cardinality.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    data = load_all()

    print(f"\nGenerating figures → {OUT_DIR}/\n")
    plot_cumulative_returns(data)
    plot_rolling_sharpe(data)
    plot_turnover(data)
    plot_hhi(data)
    plot_cardinality(data)

    print("\nAll figures saved successfully.")