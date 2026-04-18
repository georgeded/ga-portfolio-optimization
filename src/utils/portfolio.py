import numpy as np
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

def get_monthly_returns(returns: pd.DataFrame,
                        permnos: list,
                        month:   pd.Timestamp) -> pd.Series:
    """
    Get actual returns for a set of stocks in a given month.

    Missing stocks get return of 0 (conservative assumption).
    Existing NaN returns also filled with 0.
    """
    mask = (
        (returns["date"] == month) &
        (returns["permno"].isin(permnos))
    )
    month_ret = returns.loc[mask].set_index("permno")["ret"]
    return month_ret.reindex(permnos, fill_value=0.0).fillna(0.0)

def get_rf_for_month(returns: pd.DataFrame,
                     month:   pd.Timestamp) -> float:
    """Get the risk-free rate for a given month."""
    rf_vals = returns.loc[returns["date"] == month, "rf"].dropna()
    return float(rf_vals.iloc[0]) if len(rf_vals) > 0 else 0.0

def compute_drift_weights(weights:       np.ndarray,
                          stock_returns: np.ndarray) -> np.ndarray:
    """Compute drifted weights after one month of returns."""
    stock_returns = np.nan_to_num(stock_returns, nan=0.0)
    drifted       = weights * (1 + stock_returns)
    total         = drifted.sum()
    if total <= 0:
        return np.ones(len(weights)) / len(weights)
    return drifted / total

def cap_universe(universe: pd.DataFrame,
                 returns:  pd.DataFrame,
                 top_n:    int = 200) -> pd.DataFrame:
    """
    At each rebalancing date, keep only the top N stocks by market cap.

    Market cap = |prc| × shrout × 1000 (in millions).
    Reduces universe from ~867 to top_n stocks for computational
    tractability. Standard practice in empirical portfolio optimisation.

    Args:
        universe: full eligible universe DataFrame [date, permno]
        returns:  returns DataFrame with prc and shrout columns
        top_n:    number of stocks to keep per rebalancing date

    Returns:
        Capped universe DataFrame [date, permno]
    """
    returns       = returns.copy()
    returns["mktcap"] = returns["prc"].abs() * returns["shrout"] * 1000

    result_rows = []
    for date, group in universe.groupby("date"):
        permnos  = group["permno"].tolist()
        date_ret = returns[
            (returns["date"].dt.year  == date.year) &
            (returns["date"].dt.month == date.month) &
            (returns["permno"].isin(permnos))
        ][["permno", "mktcap"]].drop_duplicates("permno")

        top = date_ret.nlargest(top_n, "mktcap")["permno"].tolist()
        for p in top:
            result_rows.append({"date": date, "permno": p})

    return pd.DataFrame(result_rows)


def align_drifted_weights(prev_weights: np.ndarray,
                          prev_permnos: list,
                          curr_permnos: list) -> np.ndarray:
    """
    Align drifted weights from previous period to current universe.

    Stocks leaving the portfolio get weight 0.
    Stocks entering the portfolio get weight 0 (not previously held).
    Result is normalized to sum to 1.

    Args:
        prev_weights: drifted weight vector from previous period
        prev_permnos: stock list from previous period
        curr_permnos: stock list for current period

    Returns:
        Weight vector aligned to curr_permnos, normalized to sum to 1
    """
    aligned = (pd.Series(prev_weights, index=prev_permnos)
               .reindex(curr_permnos, fill_value=0.0)
               .values)
    total = aligned.sum()
    if total > 0:
        aligned = aligned / total
    return aligned