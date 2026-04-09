# Risk-free rate processing. Source: FRED DTB3 - 3-Month Treasury Bill, Monthly Average
import pandas as pd 

def load_risk_free_rate(path: str = "data/raw/risk_free_rate.csv") -> pd.DataFrame:
    # Converts from annual percentage -> monthly decimal monthly_rf = (annual_rate / 100) /12
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns = {"observation_date": "date", "DTB3": "rf_annual_pct"})

    # Conversion from annual percent to monthly decimal
    df["rf"] = (df["rf_annual_pct"] / 100) / 12

    # Keep only whats needed
    df = df[["date", "rf"]].copy()

    # Validation step (important)
    assert df["rf"].isna().sum() == 0, "Missing values in risk-free rate"
    assert (df["rf"] >= 0).all(), "Negative risk-free rates found"
    
    print(f"Risk-free rate loaded: {len(df)} months")
    print(f"Range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Monthly rf range: {df['rf'].min():.6f} to {df['rf'].max():.6f}")
    
    return df

if __name__ == "__main__":
    df = load_risk_free_rate()
    df.to_parquet("data/processed/risk_free_rate.parquet", index=False)
    print("Saved to data/processed/risk_free_rate.parquet")