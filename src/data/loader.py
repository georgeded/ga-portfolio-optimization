# Input: data/raw/crsp_returns.csv

import pandas as pd
import os


def load_crsp_csv(path: str = "data/raw/crsp_returns.csv") -> pd.DataFrame:
    print(f"Loading CRSP data from {path}...")

    # parse_dates uses the original column name before lowercase conversion.
    df = pd.read_csv(path, parse_dates=["MthCalDt"], low_memory=False)
    df.columns = df.columns.str.lower()

    # Rare WRDS duplicates can come from multiple distributions in one month.
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
    print("\nValidation")

    print(f"Date range: {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    assert len(df) > 100_000, f"Too few rows: {len(df)}"
    print(f"Row count: {len(df):,}")

    n_stocks = df['permno'].nunique()
    assert n_stocks > 500, f"Too few unique stocks: {n_stocks}"
    print(f"Unique stocks: {n_stocks:,}")

    exchanges = df['exchcd'].dropna().unique()
    print(f"Exchanges present: {sorted(exchanges)}")

    ret_pct = df['ret'].notna().mean() * 100
    print(f"Return coverage: {ret_pct:.1f}% non-missing")

    print("\nShare type distribution:")
    print(df['sharetype'].value_counts().head(10))

    print("\nValidation passed\n")


if __name__ == "__main__":
    df = load_crsp_csv()
    validate_raw_data(df)

    os.makedirs("data/raw", exist_ok=True)
    df.to_parquet("data/raw/crsp_returns.parquet", index=False)
    print("Saved to data/raw/crsp_returns.parquet")
