"""
Merges CRSP returns with rf to compute monthly excess returns (MthRet − rf).
Output used by GA, MVO, 1/N, and evaluation modules.
"""

import pandas as pd
import numpy as np
import os


def load_universe(path: str = "data/processed/universe.parquet") -> pd.DataFrame:
    """Load the eligible universe (permno + date pairs)."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Universe loaded: {len(df):,} rows, "
          f"{df['permno'].nunique():,} unique stocks, "
          f"{df['date'].nunique()} rebalancing dates")
    return df


def load_crsp(path: str = "data/raw/crsp_returns.parquet") -> pd.DataFrame:
    """Load processed CRSP returns."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_rf(path: str = "data/processed/risk_free_rate.parquet") -> pd.DataFrame:
    """Load monthly risk-free rate."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_excess_returns(universe:  pd.DataFrame,
                           crsp:      pd.DataFrame,
                           rf:        pd.DataFrame) -> pd.DataFrame:
    """
    CRSP dates are end-of-month (e.g. 2005-01-31); FRED dates are start-of-month.
    Matched on year-month period — same calendar month, different day convention.
    """
    crsp = crsp.copy()
    crsp["year_month"] = crsp["date"].dt.to_period("M")
    rf = rf.copy()
    rf["year_month"] = rf["date"].dt.to_period("M")

    crsp = crsp.merge(
        rf[["year_month", "rf"]],
        on="year_month",
        how="left"
    )

    crsp["excess_ret"] = crsp["ret"] - crsp["rf"]

    eligible_permnos = universe["permno"].unique()
    crsp_eligible = crsp[crsp["permno"].isin(eligible_permnos)].copy()

    print(f"Stocks in universe: {len(eligible_permnos):,}")
    print(f"CRSP rows for eligible stocks: {len(crsp_eligible):,}")

    return crsp_eligible


def validate_excess_returns(df: pd.DataFrame) -> None:
    """Sanity checks on excess returns."""
    print("\n--- Validation ---")

    missing_rf = df["rf"].isna().sum()
    assert missing_rf == 0, f"Missing rf for {missing_rf:,} rows"
    print("✓ No missing risk-free rate values")

    print("✓ Excess return stats:")
    print(f"  Mean  : {df['excess_ret'].mean():.4f}")
    print(f"  Std   : {df['excess_ret'].std():.4f}")
    print(f"  Min   : {df['excess_ret'].min():.4f}")
    print(f"  Max   : {df['excess_ret'].max():.4f}")

    print(f"✓ Date range: {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    print(f"✓ rf range: {df['rf'].min():.6f} to {df['rf'].max():.6f}")

    print("--- Validation passed ---\n")


if __name__ == "__main__":
    universe = load_universe()
    crsp     = load_crsp()
    rf       = load_rf()

    print("\nComputing excess returns...")
    df = compute_excess_returns(universe, crsp, rf)
    validate_excess_returns(df)

    os.makedirs("data/processed", exist_ok=True)
    cols = ["date", "permno", "ticker", "ret", "rf", "excess_ret",
            "prc", "shrout", "exchcd"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_parquet("data/processed/returns.parquet", index=False)
    print(f"Saved {len(df):,} rows to data/processed/returns.parquet")