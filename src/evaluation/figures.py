"""
Generates F1–F5: cumulative returns, rolling Sharpe, turnover, HHI, and GA cardinality.
Greyscale-compatible (readable in B&W print). Legends below figure to avoid obscuring data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os

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
    "GA"               : {"color": "#000000", "lw": 2.0, "ls": "-"},
    "Unconstrained MVO": {"color": "#444444", "lw": 1.5, "ls": "--"},
    "Constrained MVO"  : {"color": "#888888", "lw": 1.5, "ls": "-."},
    "1/N"              : {"color": "#bbbbbb", "lw": 1.5, "ls": ":"},
}

OUT_DIR = "results/figures"

# Crisis periods — applied consistently across all time-series figures
CRISES = [
    {"start": "2008-09-01", "end": "2009-06-01",
     "color": "black", "label": "GFC (2008–09)"},
    {"start": "2020-02-01", "end": "2020-04-01",
     "color": "grey",  "label": "COVID (2020)"},
]

# Shared legend handles for the 4 strategies — built once, reused
def make_strategy_handles() -> list:
    handles = []
    for name, s in STYLES.items():
        handles.append(mlines.Line2D(
            [], [], color=s["color"], lw=s["lw"], ls=s["ls"], label=name
        ))
    return handles

def make_crisis_handles() -> list:
    return [
        mpatches.Patch(color=c["color"], alpha=0.25, label=c["label"])
        for c in CRISES
    ]


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

    dates = data["GA"]["date"].values
    for name, df in data.items():
        assert (df["date"].values == dates).all(), \
            f"Date mismatch: GA vs {name}"

    return data


def cumulative_returns(df: pd.DataFrame) -> np.ndarray:
    total = df["net_excess_ret"].values + df["rf"].values
    return np.cumprod(1 + total)


def rolling_sharpe(df: pd.DataFrame, window: int = 12) -> pd.Series:
    net = pd.Series(df["net_excess_ret"].values, index=df["date"])
    mu  = net.rolling(window).mean() * 12
    vol = net.rolling(window).std(ddof=1) * np.sqrt(12)
    rs  = (mu / vol.replace(0, np.nan)).clip(-5, 5)
    return rs.dropna()


def smooth_series(series: pd.Series, window: int = 6) -> pd.Series:
    return series.rolling(window).mean().dropna()


def add_crisis_shading(ax) -> None:
    """Add shaded crisis periods. No legend label — handled by fig.legend."""
    for crisis in CRISES:
        ax.axvspan(
            pd.Timestamp(crisis["start"]),
            pd.Timestamp(crisis["end"]),
            alpha=0.08, color=crisis["color"], zorder=1,
        )


def add_bottom_legend(fig, extra_handles: list = None,
                      ncol: int = 6) -> None:
    """
    Place a shared legend below the figure.
    Includes strategy lines + crisis patches.
    extra_handles: additional handles specific to a figure (e.g. mean K line).
    """
    handles = make_strategy_handles() + make_crisis_handles()
    if extra_handles:
        handles += extra_handles

    fig.legend(
        handles        = handles,
        loc            = "lower center",
        ncol           = ncol,
        framealpha     = 0.9,
        bbox_to_anchor = (0.5, -0.02),
        fontsize       = 10,
    )


def add_caption(fig, text: str) -> None:
    fig.text(
        0.5, -0.10, text,
        ha="center", va="top",
        fontsize=9, style="italic",
        color="#444444",
        transform=fig.transFigure,
    )


def fmt_axes(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))


def plot_cumulative_returns(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        s = STYLES[name]
        ax.plot(df["date"], cumulative_returns(df),
                color=s["color"], lw=s["lw"], ls=s["ls"])

    add_crisis_shading(ax)
    fmt_axes(ax)

    ax.set_title("Figure 1: Cumulative Net Portfolio Value (2005–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (net of costs, initial = 1.0)")

    add_bottom_legend(fig, ncol=6)
    add_caption(fig,
        "All returns net of transaction costs (γ = 0.3%). Initial portfolio value "
        "normalised to 1.0. Shaded regions indicate the Global Financial Crisis "
        "(Sep 2008 – Jun 2009) and COVID-19 shock (Feb – Apr 2020)."
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{OUT_DIR}/F1_cumulative_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: F1_cumulative_returns.png")


def plot_rolling_sharpe(data: dict, window: int = 12) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        s  = STYLES[name]
        rs = rolling_sharpe(df, window)
        ax.plot(rs.index, rs.values,
                color=s["color"], lw=s["lw"], ls=s["ls"])

    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.5)
    add_crisis_shading(ax)
    fmt_axes(ax)

    ax.set_title(f"Figure 2: Rolling {window}-Month Sharpe Ratio (net, annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")

    add_bottom_legend(fig, ncol=6)
    add_caption(fig,
        "Computed on monthly net excess returns over a trailing 12-month window. "
        "Clipped to [−5, 5] to suppress near-zero volatility artefacts. "
        "Shaded regions indicate crisis periods."
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{OUT_DIR}/F2_rolling_sharpe.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: F2_rolling_sharpe.png")


def plot_turnover(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        s  = STYLES[name]
        to = pd.Series(df["turnover"].values * 100, index=df["date"])
        to_smooth = smooth_series(to, window=6)
        ax.plot(to_smooth.index, to_smooth.values,
                color=s["color"], lw=s["lw"], ls=s["ls"])

    add_crisis_shading(ax)
    fmt_axes(ax)

    ax.set_title("Figure 3: Monthly Portfolio Turnover (6-month rolling avg)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover (%)")

    add_bottom_legend(fig, ncol=6)
    add_caption(fig,
        "Turnover = ½ · Σ|w_t − w_{t−1}| where w_{t−1} reflects post-drift weights "
        "before rebalancing. Smoothed with a 6-month rolling average for readability. "
        "First-period turnover (100%) excluded after smoothing."
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{OUT_DIR}/F3_turnover.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: F3_turnover.png")


def plot_hhi(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, df in data.items():
        s   = STYLES[name]
        hhi = pd.Series(df["hhi"].values, index=df["date"])
        hhi_smooth = smooth_series(hhi, window=6)
        ax.plot(hhi_smooth.index, hhi_smooth.values,
                color=s["color"], lw=s["lw"], ls=s["ls"])

    add_crisis_shading(ax)
    fmt_axes(ax)

    ax.set_title("Figure 4: Portfolio Concentration — HHI (6-month rolling avg)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Herfindahl-Hirschman Index")

    add_bottom_legend(fig, ncol=6)
    add_caption(fig,
        "HHI = Σw_i². Theoretical minimum is 1/K: GA lower bound ≈ 0.071 (avg K=14), "
        "MVO/1/N lower bound ≈ 0.001 (K≈867). Values are not directly comparable "
        "across strategies with different K."
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{OUT_DIR}/F4_hhi.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: F4_hhi.png")


def plot_cardinality(data: dict) -> None:
    df  = data["GA"]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["date"], df["n_stocks"],
            color="#000000", lw=1.2, alpha=0.6)

    k_smooth = smooth_series(
        pd.Series(df["n_stocks"].values, index=df["date"]), window=12
    )
    ax.plot(k_smooth.index, k_smooth.values,
            color="#000000", lw=2.0)

    ax.axhline(
        df["n_stocks"].mean(), color="#666666", lw=1.2, ls="--"
    )

    add_crisis_shading(ax)
    fmt_axes(ax)

    ax.set_title("Figure 5: GA Selected Portfolio Size (K) Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks Selected (K)")

    # F5 has its own legend entries — no strategy comparison needed
    extra = [
        mlines.Line2D([], [], color="#000000", lw=1.2, alpha=0.6,
                      label="K (monthly)"),
        mlines.Line2D([], [], color="#000000", lw=2.0,
                      label="K (12-month avg)"),
        mlines.Line2D([], [], color="#666666", lw=1.2, ls="--",
                      label=f"Mean K = {df['n_stocks'].mean():.1f}"),
    ]
    fig.legend(
        handles        = extra + make_crisis_handles(),
        loc            = "lower center",
        ncol           = 5,
        framealpha     = 0.9,
        bbox_to_anchor = (0.5, -0.02),
        fontsize       = 10,
    )

    add_caption(fig,
        "K is selected adaptively by the GA within the constraint K ∈ [10, 30]. "
        "Monthly values (light) and 12-month rolling average (bold) are shown. "
        "Post-GFC reduction to K≈10 reflects the GA preferring concentrated "
        "portfolios during high estimation-error periods."
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(f"{OUT_DIR}/F5_cardinality.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: F5_cardinality.png")


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