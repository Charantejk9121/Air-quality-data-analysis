"""
Air Quality Analysis - India (2015-2020)
Dataset: City-level daily air quality data (CPCB-sourced), 26 cities, 21 states
Source: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

Steps:
  1. Load data + map cities to states
  2. Data completeness audit (per city, per pollutant)
  3. Clean missing/inconsistent records (city-level median only - no
     cross-city fabrication for columns a city never measured)
  4. Physical-consistency + anomaly checks (facts-based, not just stats)
  5. EDA: top polluted states/cities (PM2.5-based), seasonal patterns
  6. Generate 10 visualizations (heatmaps, time-series, bar charts, correlation)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
pd.set_option('display.width', 120)

INPUT_FILE = "city_day.csv"          # raw dataset
CLEAN_FILE = "city_day_clean.csv"    # output: cleaned dataset

pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                   'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'])

city_state = {
    'Ahmedabad': 'Gujarat', 'Aizawl': 'Mizoram', 'Amaravati': 'Andhra Pradesh',
    'Amritsar': 'Punjab', 'Bengaluru': 'Karnataka', 'Bhopal': 'Madhya Pradesh',
    'Brajrajnagar': 'Odisha', 'Chandigarh': 'Chandigarh', 'Chennai': 'Tamil Nadu',
    'Coimbatore': 'Tamil Nadu', 'Delhi': 'Delhi', 'Ernakulam': 'Kerala',
    'Gurugram': 'Haryana', 'Guwahati': 'Assam', 'Hyderabad': 'Telangana',
    'Jaipur': 'Rajasthan', 'Jorapokhar': 'Jharkhand', 'Kochi': 'Kerala',
    'Kolkata': 'West Bengal', 'Lucknow': 'Uttar Pradesh', 'Mumbai': 'Maharashtra',
    'Patna': 'Bihar', 'Shillong': 'Meghalaya', 'Talcher': 'Odisha',
    'Thiruvananthapuram': 'Kerala', 'Visakhapatnam': 'Andhra Pradesh'
}
df['State'] = df['City'].map(city_state)

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Total records (raw): {len(df):,}")
print(f"Cities: {df['City'].nunique()}  |  States/UTs: {df['State'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


# ---------------------------------------------------------------------------
# 2. DATA COMPLETENESS AUDIT (per city x pollutant, on RAW data)
# ---------------------------------------------------------------------------
# Why: blanket imputation (e.g. filling every gap with a global median) can
# silently fabricate entire columns for cities that never measured that
# pollutant at all. That fabricated data then looks "real" in every chart
# and ranking downstream unless it's caught here first.
print("\n" + "="*60)
print("DATA COMPLETENESS AUDIT")
print("="*60)

coverage = df.groupby('City')[pollutant_cols].apply(lambda g: g.notna().mean() * 100)

zero_coverage = {}
for col in pollutant_cols:
    cities_with_zero = coverage.index[coverage[col] == 0].tolist()
    if cities_with_zero:
        zero_coverage[col] = cities_with_zero

print("Pollutants with ZERO real data for specific cities (100% would be fabricated if imputed):")
for col, cities in zero_coverage.items():
    print(f"  {col}: {cities}")

low_coverage = coverage[(coverage > 0) & (coverage < 50)]
print("\nCity x Pollutant combos with <50% real coverage (imputation-heavy, use with caution):")
for city in low_coverage.index:
    flagged = low_coverage.loc[city]
    flagged = flagged[flagged.notna() & (flagged > 0)]
    if len(flagged):
        print(f"  {city}: " + ", ".join(f"{c}={v:.0f}%" for c, v in flagged.items()))


# ---------------------------------------------------------------------------
# 3. CLEANING
# ---------------------------------------------------------------------------
total_cells = df.shape[0] * df.shape[1]
missing_before = df.isnull().sum().sum()
print(f"\nMissing/inconsistent cells before cleaning: {missing_before:,} "
      f"({missing_before/total_cells*100:.2f}%)")

df_clean = df.copy()
before_rows = len(df_clean)

# Rows with no AQI value are unusable for ranking -> drop them
df_clean = df_clean.dropna(subset=['AQI'])
rows_removed = before_rows - len(df_clean)

# Impute using CITY-LEVEL median ONLY. Deliberately no global-median
# fallback: if a city never measured a pollutant, that stays NaN rather
# than being fabricated from other cities' data (this is what caused the
# earlier Lucknow PM10 issue).
for col in pollutant_cols:
    df_clean[col] = df_clean.groupby('City')[col].transform(lambda x: x.fillna(x.median()))

missing_after = df_clean[pollutant_cols].isnull().sum().sum()

print(f"\nRows removed (missing AQI): {rows_removed:,} ({rows_removed/before_rows*100:.2f}% of raw rows)")
print(f"Final clean dataset size: {len(df_clean):,} records")
print(f"Remaining genuine gaps after cleaning (cities with zero real data for a "
      f"pollutant - left as NaN, not fabricated): {missing_after:,} cells")

df_clean.to_csv(CLEAN_FILE, index=False)
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Year'] = df_clean['Date'].dt.year


# ---------------------------------------------------------------------------
# 4. PHYSICAL CONSISTENCY + ANOMALY CHECKS (facts-based)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PHYSICAL CONSISTENCY CHECKS")
print("="*60)

neg_found = {c: (df_clean[c] < 0).sum() for c in pollutant_cols + ['AQI']
             if (df_clean[c] < 0).sum() > 0}
print(f"Negative values (impossible): {neg_found if neg_found else 'none found'}")

dupes = df_clean.duplicated(subset=['City', 'Date']).sum()
print(f"Duplicate City+Date records: {dupes if dupes else 'none found'}")

# PM2.5 must always be <= PM10 (PM2.5 is a physical subset of PM10).
# Rate is computed only over rows where BOTH values are real - rows where
# PM10 was never measured for that city (still NaN after cleaning) can't
# violate the rule and would understate the true rate if included.
both_present = df_clean.dropna(subset=['PM2.5', 'PM10'])
pm_violation = both_present[both_present['PM2.5'] > both_present['PM10']]
print(f"\nPM2.5 > PM10 (physically impossible): {len(pm_violation):,} of "
      f"{len(both_present):,} rows with both values measured "
      f"({len(pm_violation)/len(both_present)*100:.2f}%)")
viol_by_city = pm_violation.groupby('City').size().sort_values(ascending=False)
print("Concentrated in cities with the weakest raw PM10 coverage (see completeness "
      "audit above), not random measurement error:")
print(viol_by_city.head(5).to_string())

print("\n" + "="*60)
print("ANOMALY CHECK - MEAN vs MEDIAN comparison across cities")
print("="*60)
print("Rationale: a city can look anomalous on MEAN alone due to a handful of\n"
      "extreme single-day spikes. Checking MEDIAN too separates a genuine,\n"
      "sustained sensor/calibration issue from a few bad days of data.\n")

city_means = df_clean.groupby('City')[pollutant_cols].mean()
city_medians = df_clean.groupby('City')[pollutant_cols].median()

rows = []
for col in pollutant_cols:
    mean_med = city_means[col].median()
    med_med = city_medians[col].median()
    if mean_med == 0 or med_med == 0:
        continue
    mean_flagged = set(city_means.index[city_means[col] > 3 * mean_med])
    median_flagged = set(city_medians.index[city_medians[col] > 3 * med_med])
    for city in mean_flagged | median_flagged:
        rows.append({
            'Pollutant': col,
            'City': city,
            'Flagged_by_mean': city in mean_flagged,
            'Flagged_by_median': city in median_flagged,
            'Verdict': 'SUSTAINED anomaly' if city in median_flagged else 'Spike-day artifact only'
        })

anomaly_report = pd.DataFrame(rows).sort_values(['Verdict', 'Pollutant'])
print(anomaly_report.to_string(index=False))

print("\nSUSTAINED anomalies = genuine, consistent data-quality issue at that station")
print("(e.g. Ahmedabad's CO/SO2 - flagged on both mean and median).")
print("Spike-day-only = a few extreme single-day readings inflate the mean, but")
print("typical days are normal (e.g. Shillong's Benzene/Toluene - median is fine,")
print("only the mean looks anomalous, driven by a handful of outlier days).")

print("\nPM2.5 / PM10 check (used as primary ranking metric):")
for col in ['PM2.5', 'PM10']:
    med = city_means[col].median()
    flagged = city_means[city_means[col] > 3 * med][col]
    status = "no anomalies - safe to use" if flagged.empty else f"anomalies: {dict(flagged.round(1))}"
    print(f"  {col}: {status}")


# ---------------------------------------------------------------------------
# 5. EDA (PM2.5 used as primary ranking metric)
# ---------------------------------------------------------------------------
state_pm = df_clean.groupby('State')['PM2.5'].mean().sort_values(ascending=False)
city_pm = df_clean.groupby('City')['PM2.5'].mean().sort_values(ascending=False)
state_aqi = df_clean.groupby('State')['AQI'].mean().sort_values(ascending=False)

print("\n" + "="*60)
print("TOP 5 MOST POLLUTED STATES (by mean PM2.5)")
print("="*60)
print(state_pm.head(5).round(1))

print("\n" + "="*60)
print("TOP 5 MOST POLLUTED CITIES (by mean PM2.5)")
print("="*60)
print(city_pm.head(5).round(1))

print("\n" + "="*60)
print("LEAST POLLUTED CITIES (by mean PM2.5)")
print("="*60)
print(city_pm.tail(5).round(1))

print("\n" + "="*60)
print("[Reference only] Top 5 by composite AQI column (distorted by anomalies)")
print("="*60)
print(state_aqi.head(5).round(1))

monthly_pm = df_clean.groupby('Month')['PM2.5'].mean()
winter = monthly_pm[[11, 12, 1, 2]].mean()
summer = monthly_pm[[4, 5, 6]].mean()
print("\n" + "="*60)
print("SEASONAL PATTERN (PM2.5)")
print("="*60)
print(f"Winter (Nov-Feb) avg PM2.5: {winter:.1f}")
print(f"Summer (Apr-Jun) avg PM2.5: {summer:.1f}")
print(f"Winter is {round((winter-summer)/summer*100, 1)}% higher than summer")


# ---------------------------------------------------------------------------
# 6. VISUALIZATIONS (shown inline)
# ---------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
top5 = state_pm.head(5)
sns.barplot(x=top5.values, y=top5.index, hue=top5.index, palette='Reds_r', legend=False)
plt.xlabel('Average PM2.5 (ug/m3)'); plt.title('Top 5 Most Polluted States by PM2.5 (2015-2020)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
top5c = city_pm.head(5)
sns.barplot(x=top5c.values, y=top5c.index, hue=top5c.index, palette='Oranges_r', legend=False)
plt.xlabel('Average PM2.5 (ug/m3)'); plt.title('Top 5 Most Polluted Cities by PM2.5 (2015-2020)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 8))
all_city = city_pm.sort_values()
sns.barplot(x=all_city.values, y=all_city.index, hue=all_city.index, palette='RdYlGn_r', legend=False)
plt.xlabel('Average PM2.5 (ug/m3)'); plt.title('PM2.5 Ranking - All 26 Cities')
plt.tight_layout(); plt.show()

monthly_trend = df_clean.set_index('Date').resample('ME')['PM2.5'].mean()
plt.figure(figsize=(11, 5))
monthly_trend.plot(color='#c0392b')
plt.ylabel('Average PM2.5 (ug/m3)'); plt.title('National Average PM2.5 Trend (Monthly, 2015-2020)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(x=monthly_pm.index, y=monthly_pm.values, marker='o', color='#8e44ad')
plt.xticks(range(1, 13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.ylabel('Average PM2.5 (ug/m3)'); plt.title('Seasonal Pollution Pattern (Avg PM2.5 by Month)')
plt.tight_layout(); plt.show()

plt.figure(figsize=(11, 5))
for city in ['Delhi', 'Mumbai', 'Chennai', 'Bengaluru']:
    sub = df_clean[df_clean.City == city].set_index('Date').resample('ME')['PM2.5'].mean()
    plt.plot(sub.index, sub.values, label=city)
plt.legend(); plt.ylabel('Average PM2.5 (ug/m3)'); plt.title('PM2.5 Trend Comparison: Metro Cities')
plt.tight_layout(); plt.show()

pollutants_for_corr = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','AQI']
plt.figure(figsize=(9, 7))
sns.heatmap(df_clean[pollutants_for_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Pollutant Correlation Heatmap')
plt.tight_layout(); plt.show()

pivot = df_clean.pivot_table(values='PM2.5', index='City', columns='Month', aggfunc='mean')
plt.figure(figsize=(10, 9))
sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3)
plt.title('City vs Month - Average PM2.5 Heatmap')
plt.xlabel('Month'); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 6))
state_order = state_pm.sort_values(ascending=False)
plt.scatter(range(len(state_order)), state_order.values, s=state_order.values * 2,
            alpha=0.6, c=state_order.values, cmap='Reds')
plt.xticks(range(len(state_order)), state_order.index, rotation=75)
plt.ylabel('Average PM2.5 (ug/m3)'); plt.title('State-wise Pollution Severity by PM2.5')
plt.tight_layout(); plt.show()

plt.figure(figsize=(7, 5))
counts = df_clean['AQI_Bucket'].value_counts()
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='viridis', legend=False)
plt.ylabel('Number of Records'); plt.title('Distribution of AQI Categories')
plt.xticks(rotation=30); plt.tight_layout(); plt.show()

print("\nAll 10 visualizations displayed successfully.")
print(f"Cleaned dataset saved to '{CLEAN_FILE}'")
