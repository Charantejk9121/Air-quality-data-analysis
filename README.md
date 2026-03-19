[README_air_quality.md](https://github.com/user-attachments/files/26116892/README_air_quality.md)
# 🌿 Air Quality Data Analysis

> A Python-based data analysis project exploring air pollution patterns across Indian states and cities — visualising pollutant distributions, temporal trends, and geographic hotspots.

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/Charantejk9121/Air-quality-data-analysis?style=flat-square)](https://github.com/Charantejk9121/Air-quality-data-analysis/stargazers)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)

---

## 📌 Overview

This project analyses air quality data across India using Python, producing **8 insightful visualisations** that expose pollution trends, identify the most and least polluted cities and states, and examine how different pollutants contribute to overall air quality. The analysis helps understand environmental health patterns and supports data-driven policy decisions.

---

## 🎯 Objectives

1. **Pollutant distribution** across Indian states
2. **Top & bottom cities** by average pollution level
3. **Daily pollution trend** over time
4. **Geographical scatter plot** of pollutant averages by latitude/longitude
5. **Heatmap** of average pollution by state and pollutant type
6. **Monthly pollution trend** (seasonal patterns)
7. **Pollutant contribution** share (pie chart)
8. **Top 10 most polluted states** (bar chart)

---

## 📂 Repository Structure

```
Air-quality-data-analysis/
├── main.py                          # Core analysis and visualisation script
├── air_quality (2).csv              # Source dataset
├── Screenshot 2025-04-12 211358.png # Output visualisation 1
├── Screenshot 2025-04-12 211407.png # Output visualisation 2
├── Screenshot 2025-04-12 211435.png # Output visualisation 3
├── Screenshot 2025-04-12 211448.png # Output visualisation 4
├── Screenshot 2025-04-12 211504.png # Output visualisation 5
└── Screenshot 2025-04-12 211512.png # Output visualisation 6
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading, cleaning & manipulation |
| NumPy | Numerical operations |
| Matplotlib | Charting and plotting |
| Seaborn | Statistical visualisations (boxplots, heatmaps, scatter) |

---

## 📊 Visualisations Produced

### 1. 📦 Pollutant Distribution Across States
Boxplot showing the spread and outliers of pollutant levels per state — revealing which states have the highest variance in air quality.

### 2. 🏙 Top & Bottom Polluted Cities
Ranked lists of the 5 most and 5 least polluted cities by average pollutant concentration.

### 3. 📉 Daily Average Pollution Trend
Line chart tracking how average pollution levels change day by day, highlighting spikes and improvements over time.

### 4. 🌍 Geographical Scatter Plot
A latitude/longitude scatter plot colour-coded by pollutant average — visually mapping pollution intensity across India.

### 5. 🔥 State × Pollutant Heatmap
A pivot-table heatmap showing the average concentration of each pollutant (PM2.5, PM10, NO2, SO2, CO, O3) for every state.

### 6. 📅 Monthly Pollution Trend
Line chart revealing seasonal pollution patterns — identifying months with consistently high or low air quality.

### 7. 🥧 Pollutant Contribution Pie Chart
A pie chart showing which pollutants (e.g. PM2.5, NO2) account for the largest share of total pollution.

### 8. 📊 Top 10 Polluted States
Horizontal bar chart ranking the 10 states with the highest average pollution levels.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn
```

### Run the Analysis

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Charantejk9121/Air-quality-data-analysis.git
   cd Air-quality-data-analysis
   ```

2. **Update the dataset path in `main.py`:**
   ```python
   # Line 9 — update this path to your local file location
   df = pd.read_csv(r"path/to/air_quality (2).csv")
   ```

3. **Run the script:**
   ```bash
   python main.py
   ```

All 8 visualisations will be displayed sequentially.

---

## 📁 Dataset Columns

| Column | Description |
|--------|-------------|
| `state` | Indian state name |
| `city` | City name |
| `station` | Monitoring station name |
| `last_update` | Timestamp of the reading |
| `pollutant_id` | Pollutant type (PM2.5, PM10, NO2, SO2, CO, O3) |
| `pollutant_min` | Minimum pollutant level recorded |
| `pollutant_max` | Maximum pollutant level recorded |
| `pollutant_avg` | Average pollutant level |
| `latitude` | Station latitude |
| `longitude` | Station longitude |

---

## 💡 Key Insights

- Certain northern Indian states consistently show **the highest PM2.5 and PM10 levels**
- Pollution peaks sharply in **winter months** (October–January) due to crop burning and low wind
- **Major metropolitan cities** appear in both top and bottom lists depending on pollutant type
- **PM2.5 and PM10 contribute the largest share** of total pollutant load
- Geographic scatter plots reveal a visible **north-south pollution gradient** across India

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  🌿 <strong>Air Quality Data Analysis</strong> — Breathing clarity into pollution data.
</p>
