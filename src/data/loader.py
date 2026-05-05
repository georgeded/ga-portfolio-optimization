"""
Load raw CRSP monthly data (CIZ format) from WRDS and save as parquet.
WRDS source: CRSP Annual Update → Stock Version 2 (CIZ) → Monthly Stock File.
Date range: 2000-01-01 to 2025-12-31.
"""

import pandas as pd
import os


def load_crsp_csv(path: str = "data/raw/crsp_returns.csv") -> pd.DataFrame:
    """Load raw CRSP CSV from WRDS; standardise column names and deduplicate."""
    print(f"Loading CRSP data from {path}...")

    # parse_dates uses the original column name before the lowercase conversion below
    df = pd.read_csv(path, parse_dates=["MthCalDt"], low_memory=False)
    df.columns = df.columns.str.lower()

    # Deduplicate before rename (WRDS warning: rare duplicates from
    # multiple distribution events in same month)
    before = len(df)
    df = df.drop_duplicates(subset=['permno', 'mthcaldt'], keep='first')
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed:,} duplicate rows (multiple distributions)")

    df = df.rename(columns={
        "mthcaldt": "date",
        "mthret": "ret",
        "mthretx": "retx",
        "mthprc": "prc",
        "primaryexch": "exchcd",
        "secinfostartdt": "start_dt",
        "secinfoenddt": "end_dt",
    })

    print(f"Loaded {len(df):,} rows, "
          f"{df['permno'].nunique():,} unique stocks")
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Sanity checks on raw data before any filtering."""
    print("\n--- Validation ---")

    print(f"Date range: {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    assert len(df) > 100_000, f"Too few rows: {len(df)}"
    print(f"✓ Row count: {len(df):,}")

    n_stocks = df['permno'].nunique()
    assert n_stocks > 500, f"Too few unique stocks: {n_stocks}"
    print(f"✓ Unique stocks: {n_stocks:,}")

    exchanges = df['exchcd'].dropna().unique()
    print(f"✓ Exchanges present: {sorted(exchanges)}")

    ret_pct = df['ret'].notna().mean() * 100
    print(f"✓ Return coverage: {ret_pct:.1f}% non-missing")

    print("\nShare type distribution:")
    print(df['sharetype'].value_counts().head(10))

    print("\n--- Validation passed ---\n")


if __name__ == "__main__":
    df = load_crsp_csv()
    validate_raw_data(df)

    os.makedirs("data/raw", exist_ok=True)
    df.to_parquet("data/raw/crsp_returns.parquet", index=False)
    print("Saved to data/raw/crsp_returns.parquet")
