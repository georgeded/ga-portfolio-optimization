import pandas as pd
import numpy as np
import os

MIN_MARKET_CAP_B = 2.0  # billion USD
ESTIMATION_WINDOW = 60  # months
SHROUT_SCALE = 1_000  # CRSP ShrOut is in thousands of shares


def load_raw_crsp(path: str = "data/raw/crsp_returns.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Raw CRSP loaded: {len(df):,} rows, "
          f"{df['permno'].nunique():,} unique stocks")
    return df


def compute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mktcap"] = df["prc"].abs() * df["shrout"] * SHROUT_SCALE / 1e9
    df = df.sort_values(["permno", "date"])
    df["mktcap_lagged"] = df.groupby("permno")["mktcap"].shift(1)
    return df


def filter_basic(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["exchcd"].isin(['N', 'Q'])]
    print(f"After exchange filter (NYSE/NASDAQ): {len(df):,} rows "
          f"(removed {before - len(df):,})")

    before = len(df)
    df = df[
        (df["sharetype"] == "NS") &
        (df["securitytype"] == "EQTY") &
        (df["securitysubtype"] == "COM") &
        (df["usincflg"] == "Y") &
        (df["issuertype"].isin(["CORP", "ACOR"]))
    ]
    print(f"After share class filter (common stocks): {len(df):,} rows "
          f"(removed {before - len(df):,})")

    return df


def build_monthly_universe(df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.Index:
    window_end = rebalance_date - pd.DateOffset(days=1)
    window_start = rebalance_date - pd.DateOffset(months=ESTIMATION_WINDOW)

    window_data = df[
        (df["date"] >= window_start) &
        (df["date"] <= window_end)
    ]

    # Match t-1 by month to handle month-end CRSP dates.
    t_minus_1 = rebalance_date - pd.DateOffset(months=1)
    last_obs = df[
        (df["date"].dt.year == t_minus_1.year) &
        (df["date"].dt.month == t_minus_1.month)
    ]
    eligible_mktcap = last_obs[
        last_obs["mktcap"] >= MIN_MARKET_CAP_B
    ]["permno"]

    return_counts = (
        window_data[window_data["ret"].notna()]
        .groupby("permno")["ret"]
        .count()
    )
    eligible_history = return_counts[
        return_counts == ESTIMATION_WINDOW
    ].index

    eligible = eligible_mktcap[eligible_mktcap.isin(eligible_history)]

    return pd.Index(eligible)


def build_full_universe(df: pd.DataFrame, start_date: str = "2005-01-01",
                        end_date: str = "2025-12-01") -> dict:
    df = compute_market_cap(df)
    df = filter_basic(df)

    rebalance_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="MS"  # Month Start
    )

    universe = {}
    for t in rebalance_dates:
        eligible = build_monthly_universe(df, t)
        universe[t] = eligible.tolist()

    sizes = [len(v) for v in universe.values()]
    print(f"\nUniverse built across {len(universe)} rebalancing dates")
    print(f"Avg universe size : {np.mean(sizes):.0f} stocks")
    print(f"Min universe size : {min(sizes)} stocks")
    print(f"Max universe size : {max(sizes)} stocks")

    return universe


def validate_universe(universe: dict) -> None:
    sizes = pd.Series(
        {date: len(stocks) for date, stocks in universe.items()}
    )

    empty = sizes[sizes == 0]
    assert len(empty) == 0, f"Empty universe at: {empty.index.tolist()}"
    print("No empty universes")

    # Sudden jump (>50% MoM) signals a data error, not methodology.
    pct_change = sizes.pct_change().abs()
    large_jumps = pct_change[pct_change > 0.5]
    if len(large_jumps) > 0:
        print(f"Large universe size jumps at: "
              f"{large_jumps.index.tolist()}")
    else:
        print("Universe size stable over time")

    print("\nUniverse size summary (descriptive):")
    print(f"Mean: {sizes.mean():.0f} stocks")
    print(f"Median: {sizes.median():.0f} stocks")
    print(f"Min: {sizes.min()} stocks")
    print(f"Max: {sizes.max()} stocks")
    print("Note: typical literature estimates are 100-300.")
    print("Actual size reflects the full $2B market cap filter on real data.")


if __name__ == "__main__":
    df = load_raw_crsp()
    uni = build_full_universe(df)
    validate_universe(uni)

    os.makedirs("data/processed", exist_ok=True)
    rows = [
        {"date": date, "permno": permno}
        for date, permnos in uni.items()
        for permno in permnos
    ]
    universe_df = pd.DataFrame(rows)
    universe_df.to_parquet("data/processed/universe.parquet", index=False)
    print(f"\nSaved {len(universe_df):,} rows to "
          f"data/processed/universe.parquet")
