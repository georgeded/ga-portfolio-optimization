"""
Step 15: Data Loading & Validation
Loads the manually downloaded CRSP monthly stock data (CIZ format)
and saves it as parquet for use by the rest of the pipeline.

Data downloaded manually from WRDS:
- Database: CRSP Annual Update > Stock Version 2 (CIZ) > Monthly Stock File
- Date range: 2000-01-01 to 2025-12-31
- Variables: permno, primaryexch, securitytype, sharetype, securitysubtype,
             usincflg, issuertype, secinfostartdt, secinfoenddt,
             mthcaldt, mthret, mthretx, mthprc, shrout, ticker
"""

import pandas as pd
import numpy as np
import os


def load_crsp_csv(path: str = "data/raw/crsp_returns.csv") -> pd.DataFrame:
    """
    Load raw CRSP CSV downloaded from WRDS web query.
    Standardises column names to lowercase.
    """
    print(f"Loading CRSP data from {path}...")

    # parse_dates uses original column name (before lowercase conversion)
    df = pd.read_csv(path, parse_dates=["MthCalDt"], low_memory=False)

    # Standardise all column names to lowercase
    df.columns = df.columns.str.lower()

    # Deduplicate before rename (WRDS warning: rare duplicates from
    # multiple distribution events in same month)
    before = len(df)
    df = df.drop_duplicates(subset=['permno', 'mthcaldt'], keep='first')
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed:,} duplicate rows (multiple distributions)")

    # Rename to consistent internal names
    df = df.rename(columns={
        "mthcaldt"       : "date",
        "mthret"         : "ret",
        "mthretx"        : "retx",
        "mthprc"         : "prc",
        "primaryexch"    : "exchcd",
        "secinfostartdt" : "start_dt",
        "secinfoenddt"   : "end_dt",
    })

    print(f"Loaded {len(df):,} rows, "
          f"{df['permno'].nunique():,} unique stocks")
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Sanity checks on raw downloaded data before any filtering.
    """
    print("\n--- Validation ---")

    # Date range
    print(f"Date range: {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    # Row count
    assert len(df) > 100_000, f"Too few rows: {len(df)}"
    print(f"✓ Row count: {len(df):,}")

    # Stock count
    n_stocks = df['permno'].nunique()
    assert n_stocks > 500, f"Too few unique stocks: {n_stocks}"
    print(f"✓ Unique stocks: {n_stocks:,}")

    # Both exchanges present
    exchanges = df['exchcd'].dropna().unique()
    print(f"✓ Exchanges present: {sorted(exchanges)}")

    # Return coverage
    ret_pct = df['ret'].notna().mean() * 100
    print(f"✓ Return coverage: {ret_pct:.1f}% non-missing")

    # Share type distribution
    print("\nShare type distribution:")
    print(df['sharetype'].value_counts().head(10))

    print("\n--- Validation passed ---\n")


if __name__ == "__main__":
    df = load_crsp_csv()
    validate_raw_data(df)

    # Save as parquet for faster loading by rest of pipeline
    os.makedirs("data/raw", exist_ok=True)
    df.to_parquet("data/raw/crsp_returns.parquet", index=False)
    print("Saved to data/raw/crsp_returns.parquet")