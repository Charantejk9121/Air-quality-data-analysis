"""
app.py
------
Streamlit dashboard for the Air Quality Analysis project.

Run:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from data_pipeline import load_and_clean, POLLUTANT_COLS
from model import FEATURES, predict_bucket

st.set_page_config(page_title="India Air Quality Explorer", layout="wide")

BUCKET_COLORS = {
    "Good": "#2ecc71", "Satisfactory": "#a3d977", "Moderate": "#f1c40f",
    "Poor": "#e67e22", "Very Poor": "#e74c3c", "Severe": "#8b0000",
}


@st.cache_data
def get_data():
    return load_and_clean("data/city_day.csv")


@st.cache_resource
def get_model():
    path = "outputs/aqi_model.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None


df = get_data()
model_bundle = get_model()

st.title("🌿 India Air Quality Explorer (2015–2020)")
st.caption("CPCB city-day data across 26 cities and 21 states/UTs.")

tab1, tab2, tab3 = st.tabs(["City Trends", "State Comparison", "AQI Predictor"])

# ---------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        city = st.selectbox("Choose a city", sorted(df["City"].unique()), index=0)
        pollutant = st.selectbox("Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])

    sub = df[df["City"] == city].set_index("Date")[pollutant].resample("ME").mean()

    with col2:
        fig, ax = plt.subplots(figsize=(9, 4))
        sub.plot(ax=ax, color="#c0392b")
        ax.set_ylabel(f"{pollutant} (monthly avg)")
        ax.set_title(f"{pollutant} trend — {city}")
        st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    city_df = df[df["City"] == city]
    c1.metric("Avg AQI", f"{city_df['AQI'].mean():.0f}")
    c2.metric("Avg PM2.5", f"{city_df['PM2.5'].mean():.1f}")
    c3.metric("Most common category", city_df["AQI_Bucket"].mode().iloc[0])

# ---------------------------------------------------------------------
with tab2:
    state_pm = df.groupby("State")["PM2.5"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x=state_pm.values, y=state_pm.index, hue=state_pm.index,
                palette="RdYlGn_r", legend=False, ax=ax)
    ax.set_xlabel("Average PM2.5 (ug/m3)")
    ax.set_title("States ranked by average PM2.5")
    st.pyplot(fig)

    st.markdown("**Seasonal pattern (national average PM2.5 by month)**")
    monthly_pm = df.groupby("Month")["PM2.5"].mean()
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    sns.lineplot(x=monthly_pm.index, y=monthly_pm.values, marker="o", ax=ax2, color="#8e44ad")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax2.set_ylabel("Avg PM2.5")
    st.pyplot(fig2)

# ---------------------------------------------------------------------
with tab3:
    st.markdown("Enter pollutant readings for a day and predict the **AQI category**.")
    if model_bundle is None:
        st.warning("No trained model found. Run `python main.py` first to train and save the model.")
    else:
        c1, c2, c3 = st.columns(3)
        inputs = {}
        for i, feat in enumerate([f for f in FEATURES if f != "Month"]):
            col = [c1, c2, c3][i % 3]
            inputs[feat] = col.number_input(feat, min_value=0.0, value=float(df[feat].median()))
        inputs["Month"] = st.slider("Month", 1, 12, 6)

        if st.button("Predict AQI category"):
            pred = predict_bucket(model_bundle, inputs)
            color = BUCKET_COLORS.get(pred, "#333")
            st.markdown(
                f"### Predicted category: <span style='color:{color}'>{pred}</span>",
                unsafe_allow_html=True,
            )

st.divider()
st.caption("Data source: CPCB via Kaggle (rohanrao/air-quality-data-in-india). "
           "Built with pandas, scikit-learn, and Streamlit.")
