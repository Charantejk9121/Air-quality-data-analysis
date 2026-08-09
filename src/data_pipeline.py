"""
data_pipeline.py
-----------------
Loads the raw CPCB city-day air quality dataset, maps cities to states,
audits data completeness, and produces a cleaned dataframe.

Cleaning philosophy (kept from the original analysis):
  - Drop rows with no AQI value (unusable for ranking/modeling).
  - Impute missing pollutant readings using CITY-LEVEL median only.
    No global-median fallback -- if a city never measured a pollutant,
    it stays NaN rather than being fabricated from other cities' data.
"""

import pandas as pd

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

CITY_STATE = {
    "Ahmedabad": "Gujarat", "Aizawl": "Mizoram", "Amaravati": "Andhra Pradesh",
    "Amritsar": "Punjab", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Brajrajnagar": "Odisha", "Chandigarh": "Chandigarh", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Delhi": "Delhi", "Ernakulam": "Kerala",
    "Gurugram": "Haryana", "Guwahati": "Assam", "Hyderabad": "Telangana",
    "Jaipur": "Rajasthan", "Jorapokhar": "Jharkhand", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Patna": "Bihar", "Shillong": "Meghalaya", "Talcher": "Odisha",
    "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
}


def load_raw(path: str = "data/city_day.csv") -> pd.DataFrame:
    """Load the raw CSV and attach a State column."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["State"] = df["City"].map(CITY_STATE)
    return df


def completeness_audit(df: pd.DataFrame) -> dict:
    """Return a summary of missing-data patterns per city x pollutant."""
    coverage = df.groupby("City")[POLLUTANT_COLS].apply(lambda g: g.notna().mean() * 100)

    zero_coverage = {
        col: coverage.index[coverage[col] == 0].tolist()
        for col in POLLUTANT_COLS
        if (coverage[col] == 0).any()
    }
    low_coverage = coverage[(coverage > 0) & (coverage < 50)]

    return {
        "coverage_table": coverage,
        "zero_coverage": zero_coverage,
        "low_coverage": low_coverage,
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no AQI, impute pollutants using city-level median only."""
    df_clean = df.dropna(subset=["AQI"]).copy()

    for col in POLLUTANT_COLS:
        df_clean[col] = df_clean.groupby("City")[col].transform(lambda x: x.fillna(x.median()))

    df_clean["Month"] = df_clean["Date"].dt.month
    df_clean["Year"] = df_clean["Date"].dt.year
    return df_clean


def physical_consistency_report(df_clean: pd.DataFrame) -> dict:
    """Check for impossible values: negatives, duplicates, PM2.5 > PM10."""
    neg_found = {
        c: int((df_clean[c] < 0).sum())
        for c in POLLUTANT_COLS + ["AQI"]
        if (df_clean[c] < 0).sum() > 0
    }
    dupes = int(df_clean.duplicated(subset=["City", "Date"]).sum())

    both_present = df_clean.dropna(subset=["PM2.5", "PM10"])
    pm_violation = both_present[both_present["PM2.5"] > both_present["PM10"]]
    violation_rate = len(pm_violation) / len(both_present) * 100 if len(both_present) else 0.0

    return {
        "negative_values": neg_found,
        "duplicate_city_date_rows": dupes,
        "pm25_gt_pm10_rows": len(pm_violation),
        "pm25_gt_pm10_rate_pct": round(violation_rate, 2),
    }


def load_and_clean(path: str = "data/city_day.csv") -> pd.DataFrame:
    """Convenience wrapper: load raw data and return the cleaned dataframe."""
    raw = load_raw(path)
    return clean(raw)


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw records: {len(raw):,} | Cities: {raw['City'].nunique()} | States: {raw['State'].nunique()}")

    audit = completeness_audit(raw)
    print(f"Pollutants with zero coverage for some city: {list(audit['zero_coverage'].keys())}")

    cleaned = clean(raw)
    print(f"Cleaned records: {len(cleaned):,}")
    cleaned.to_csv("outputs/city_day_clean.csv", index=False)

    consistency = physical_consistency_report(cleaned)
    print(consistency)
