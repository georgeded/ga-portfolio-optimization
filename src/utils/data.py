import pandas as pd


def load_data(
    universe_path: str = "data/processed/universe.parquet",
    returns_path:  str = "data/processed/returns.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load universe and returns data."""
    universe = pd.read_parquet(universe_path)
    universe["date"] = pd.to_datetime(universe["date"])

    returns = pd.read_parquet(returns_path)
    returns["date"] = pd.to_datetime(returns["date"])

    print(f"Universe: {universe['date'].nunique()} rebalancing dates")
    print(f"Returns : {len(returns):,} rows")
    return universe, returns
