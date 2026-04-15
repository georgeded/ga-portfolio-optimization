"""
Step 15: Universe Construction & Validation
Filters raw CRSP data to construct the eligible investment universe
at each rebalancing date t, following Step 5 of the methodology.

Filters applied:
- NYSE and NASDAQ only (primaryexch IN ('N', 'Q'))
- Common stocks only (ShareType/SecuritySubType/IssuerType combination)
- Market cap >= $2 billion (|PRC| x SHROUT, lagged by 1 month)
- Minimum 60 months of history prior to t
- No missing returns within the 60-month estimation window
- Assets included up to delisting date (MthRet includes delisting return)
"""

import pandas as pd
import numpy as np
import os

# ── Constants (from Step 5 & Step 6) ────────────────────────────────────────
MIN_MARKET_CAP_B  = 2.0    # billion USD
ESTIMATION_WINDOW = 60     # months
SHROUT_SCALE      = 1_000  # CRSP ShrOut is in thousands of shares


def load_raw_crsp(path: str = "data/raw/crsp_returns.parquet") -> pd.DataFrame:
    """Load raw CRSP data saved by loader.py."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Raw CRSP loaded: {len(df):,} rows, "
          f"{df['permno'].nunique():,} unique stocks")
    return df


def compute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market cap as |prc| x shrout (in billions).
    shrout is in thousands of shares, prc is price per share.
    Lags market cap by 1 month — at time t, uses t-1 value as per Step 5.
    """
    df = df.copy()
    df["mktcap"] = df["prc"].abs() * df["shrout"] * SHROUT_SCALE / 1e9
    df = df.sort_values(["permno", "date"])
    df["mktcap_lagged"] = df.groupby("permno")["mktcap"].shift(1)
    return df


def filter_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply static filters to remove ineligible securities.

    CIZ format equivalents of legacy SIZ filters:
    - exchcd IN ('N','Q')           = NYSE and NASDAQ only
    - sharetype = 'NS'              )
    - securitytype = 'EQTY'         ) = common stocks only
    - securitysubtype = 'COM'       ) (equivalent to legacy shrcd IN (10,11))
    - usincflg = 'Y'                )
    - issuertype IN ('CORP','ACOR') )
    """
    before = len(df)
    df = df[df["exchcd"].isin(['N', 'Q'])]
    print(f"After exchange filter (NYSE/NASDAQ): {len(df):,} rows "
          f"(removed {before - len(df):,})")

    before = len(df)
    df = df[
        (df["sharetype"]       == "NS")   &
        (df["securitytype"]    == "EQTY") &
        (df["securitysubtype"] == "COM")  &
        (df["usincflg"]        == "Y")    &
        (df["issuertype"].isin(["CORP", "ACOR"]))
    ]
    print(f"After share class filter (common stocks): {len(df):,} rows "
          f"(removed {before - len(df):,})")

    return df


def build_monthly_universe(df: pd.DataFrame,
                           rebalance_date: pd.Timestamp) -> pd.Index:
    """
    Build the eligible universe at a single rebalancing date t.

    Rules applied:
    1. Market cap >= $2B at t-1 (uses last available obs before t,
       robust to stocks with missing observation exactly at window_end)
    2. Exactly 60 non-missing returns in window [t-60, t-1]
    3. Delisted assets auto-excluded (no return data after delisting)

    Returns: pd.Index of eligible PERMNOs
    """
    # CRSP dates are end-of-month; subtract 1 day so end-of-month dates
    # fall inside the window (e.g. Dec 31 is included for Jan 1 rebalance)
    window_end   = rebalance_date - pd.DateOffset(days=1)
    window_start = rebalance_date - pd.DateOffset(months=ESTIMATION_WINDOW)

    # Subset to estimation window
    window_data = df[
        (df["date"] >= window_start) &
        (df["date"] <= window_end)
    ]

    # Rule 1: Market cap filter using last available observation before t
    last_obs = (
        df[df["date"] <= window_end]
        .sort_values("date")
        .groupby("permno")
        .tail(1)
    )
    eligible_mktcap = last_obs[
        last_obs["mktcap_lagged"] >= MIN_MARKET_CAP_B
    ]["permno"]

    # Rules 2 & 3: Must have exactly 60 non-missing returns in window
    return_counts = (
        window_data[window_data["ret"].notna()]
        .groupby("permno")["ret"]
        .count()
    )
    eligible_history = return_counts[
        return_counts == ESTIMATION_WINDOW
    ].index

    # Must pass both filters
    eligible = eligible_mktcap[eligible_mktcap.isin(eligible_history)]

    return pd.Index(eligible)


def build_full_universe(df: pd.DataFrame,
                        start_date: str = "2005-01-01",
                        end_date:   str = "2025-12-01") -> dict:
    """
    Build the eligible universe at every monthly rebalancing date.

    Returns dict: {pd.Timestamp -> list of eligible PERMNOs}

    First valid rebalancing date is 2005-01-01:
    - Data starts 2000-01-01
    - 60-month burn-in window = first usable date is 2005-01-01
    """
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
    """
    Validation checks from Step 16 (Baseline Verification Design).

    Universe size is descriptive only — 100-300 was the original estimate
    from Step 5 based on literature. Actual size emerges from filters
    applied to real data and is not constrained.
    """
    sizes = pd.Series(
        {date: len(stocks) for date, stocks in universe.items()}
    )

    # Check 1: No empty universe at any rebalancing date
    empty = sizes[sizes == 0]
    assert len(empty) == 0, f"Empty universe at: {empty.index.tolist()}"
    print("✓ No empty universes")

    # Check 2: No sudden jumps (>50% change month-over-month)
    # A sudden jump signals a data error, not a methodology issue
    pct_change = sizes.pct_change().abs()
    large_jumps = pct_change[pct_change > 0.5]
    if len(large_jumps) > 0:
        print(f"⚠ Large universe size jumps at: "
              f"{large_jumps.index.tolist()}")
    else:
        print("✓ Universe size stable over time")

    # Descriptive summary for thesis reporting
    print("\nUniverse size summary (descriptive):")
    print(f"  Mean   : {sizes.mean():.0f} stocks")
    print(f"  Median : {sizes.median():.0f} stocks")
    print(f"  Min    : {sizes.min()} stocks")
    print(f"  Max    : {sizes.max()} stocks")
    print("  Note   : Step 5 estimate was 100-300 based on literature.")
    print("           Actual size reflects full $2B filter on real data.")


if __name__ == "__main__":
    df = load_raw_crsp()
    uni = build_full_universe(df)
    validate_universe(uni)

    # Save as flat DataFrame for use by returns.py and all models
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