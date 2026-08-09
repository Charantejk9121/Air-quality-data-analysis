"""
main.py
-------
Runs the full pipeline end to end:
  1. Load + clean the raw dataset (data/city_day.csv)
  2. Print data-quality audit + physical consistency checks
  3. Generate and save all 10 EDA visualizations to visuals/
  4. Train the AQI-bucket classifier and save it + a metrics report to outputs/

Usage:
    pip install -r requirements.txt
    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import load_raw, completeness_audit, clean, physical_consistency_report
from eda import run_all
from model import train


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("visuals", exist_ok=True)

    print("=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)
    raw = load_raw("data/city_day.csv")
    print(f"Raw records: {len(raw):,} | Cities: {raw['City'].nunique()} | States: {raw['State'].nunique()}")
    print(f"Date range: {raw['Date'].min().date()} to {raw['Date'].max().date()}")

    print("\n" + "=" * 60)
    print("2. DATA QUALITY AUDIT")
    print("=" * 60)
    audit = completeness_audit(raw)
    print(f"Pollutants with zero real coverage for at least one city: "
          f"{list(audit['zero_coverage'].keys())}")

    df_clean = clean(raw)
    df_clean.to_csv("outputs/city_day_clean.csv", index=False)
    print(f"Cleaned records: {len(df_clean):,} (saved to outputs/city_day_clean.csv)")

    consistency = physical_consistency_report(df_clean)
    print(f"Physical consistency report: {consistency}")

    print("\n" + "=" * 60)
    print("3. GENERATING VISUALIZATIONS")
    print("=" * 60)
    run_all(df_clean)

    print("\n" + "=" * 60)
    print("4. TRAINING AQI-BUCKET CLASSIFIER")
    print("=" * 60)
    train(df_clean)

    print("\nDone. See visuals/ for charts and outputs/ for model + reports.")


if __name__ == "__main__":
    main()
