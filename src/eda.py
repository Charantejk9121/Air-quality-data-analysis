"""
eda.py
------
Generates and saves the exploratory visualizations as PNG files
(instead of only plt.show(), so they can be embedded in the README
and viewed without re-running the script).
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/CI
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

OUT_DIR = "visuals"


def _save(fig_name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def top_polluted_states(df_clean, n=5):
    state_pm = df_clean.groupby("State")["PM2.5"].mean().sort_values(ascending=False)
    top = state_pm.head(n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="Reds_r", legend=False)
    plt.xlabel("Average PM2.5 (ug/m3)")
    plt.title(f"Top {n} Most Polluted States by PM2.5 (2015-2020)")
    _save("01_top_states.png")
    return state_pm


def top_polluted_cities(df_clean, n=5):
    city_pm = df_clean.groupby("City")["PM2.5"].mean().sort_values(ascending=False)
    top = city_pm.head(n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="Oranges_r", legend=False)
    plt.xlabel("Average PM2.5 (ug/m3)")
    plt.title(f"Top {n} Most Polluted Cities by PM2.5 (2015-2020)")
    _save("02_top_cities.png")
    return city_pm


def all_cities_ranking(city_pm):
    all_city = city_pm.sort_values()
    plt.figure(figsize=(9, 8))
    sns.barplot(x=all_city.values, y=all_city.index, hue=all_city.index, palette="RdYlGn_r", legend=False)
    plt.xlabel("Average PM2.5 (ug/m3)")
    plt.title("PM2.5 Ranking - All Cities")
    _save("03_all_cities_ranking.png")


def national_monthly_trend(df_clean):
    trend = df_clean.set_index("Date").resample("ME")["PM2.5"].mean()
    plt.figure(figsize=(11, 5))
    trend.plot(color="#c0392b")
    plt.ylabel("Average PM2.5 (ug/m3)")
    plt.title("National Average PM2.5 Trend (Monthly, 2015-2020)")
    _save("04_national_trend.png")


def seasonal_pattern(df_clean):
    monthly_pm = df_clean.groupby("Month")["PM2.5"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=monthly_pm.index, y=monthly_pm.values, marker="o", color="#8e44ad")
    plt.xticks(range(1, 13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.ylabel("Average PM2.5 (ug/m3)")
    plt.title("Seasonal Pollution Pattern (Avg PM2.5 by Month)")
    _save("05_seasonal_pattern.png")
    return monthly_pm


def metro_comparison(df_clean, cities=("Delhi", "Mumbai", "Chennai", "Bengaluru")):
    plt.figure(figsize=(11, 5))
    for city in cities:
        sub = df_clean[df_clean.City == city].set_index("Date").resample("ME")["PM2.5"].mean()
        plt.plot(sub.index, sub.values, label=city)
    plt.legend()
    plt.ylabel("Average PM2.5 (ug/m3)")
    plt.title("PM2.5 Trend Comparison: Metro Cities")
    _save("06_metro_comparison.png")


def correlation_heatmap(df_clean):
    cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "AQI"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df_clean[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pollutant Correlation Heatmap")
    _save("07_correlation_heatmap.png")


def city_month_heatmap(df_clean):
    pivot = df_clean.pivot_table(values="PM2.5", index="City", columns="Month", aggfunc="mean")
    plt.figure(figsize=(10, 9))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3)
    plt.title("City vs Month - Average PM2.5 Heatmap")
    plt.xlabel("Month")
    _save("08_city_month_heatmap.png")


def state_severity_scatter(state_pm):
    order = state_pm.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(order)), order.values, s=order.values * 2,
                alpha=0.6, c=order.values, cmap="Reds")
    plt.xticks(range(len(order)), order.index, rotation=75)
    plt.ylabel("Average PM2.5 (ug/m3)")
    plt.title("State-wise Pollution Severity by PM2.5")
    _save("09_state_severity_scatter.png")


def aqi_bucket_distribution(df_clean):
    plt.figure(figsize=(7, 5))
    counts = df_clean["AQI_Bucket"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.ylabel("Number of Records")
    plt.title("Distribution of AQI Categories")
    plt.xticks(rotation=30)
    _save("10_aqi_bucket_distribution.png")


def run_all(df_clean):
    """Generate and save all 10 visualizations."""
    state_pm = top_polluted_states(df_clean)
    city_pm = top_polluted_cities(df_clean)
    all_cities_ranking(city_pm)
    national_monthly_trend(df_clean)
    seasonal_pattern(df_clean)
    metro_comparison(df_clean)
    correlation_heatmap(df_clean)
    city_month_heatmap(df_clean)
    state_severity_scatter(state_pm)
    aqi_bucket_distribution(df_clean)
    print("All 10 visualizations saved to visuals/")


if __name__ == "__main__":
    from data_pipeline import load_and_clean
    df_clean = load_and_clean()
    run_all(df_clean)
