# 📦 Demand Intelligence Engine

**An end-to-end sales forecasting & demand intelligence system** — from raw retail transactions to a deployed, interactive decision-support dashboard.

🔗 **[Live Demo](https://salesforecasting-9sfccbfx2bdpah4h3fezql.streamlit.app/)** — try the dashboard yourself, no setup required.

Built on 4 years of retail order data (~9,800 transactions, 3 categories, 17 sub-categories, 4 regions), this project takes a business through the full analytics lifecycle: exploratory analysis → time series diagnostics → multi-model forecasting → anomaly detection → unsupervised demand segmentation → a production-style Streamlit application that a Supply Chain or Finance team could actually use.

---

## 🔍 What This Project Does

| Module | What it answers |
|---|---|
| **Sales Overview** | Where is revenue coming from, and how has it trended over 4 years? |
| **Forecast Explorer** | What will sales look like over the next 3 months, by category and region? |
| **Anomaly Report** | Which weeks broke pattern — and were they good news or bad news? |
| **Product Segments** | Which products need safety stock vs. automated replenishment vs. close monitoring? |

Each module in the notebook feeds directly into a corresponding page in the Streamlit dashboard — this isn't just analysis in a notebook, it's a working tool.

---

## 🧠 Key Findings

- **Revenue concentration:** Technology is the top revenue category (~$827K), closely followed by Furniture (~$729K) and Office Supplies (~$705K) — a diversified, not top-heavy, portfolio.
- **Regional performance:** West (~$710K) and East (~$670K) significantly outperform Central (~$493K) and South (~$389K).
- **Trend:** Revenue dipped slightly in Year 2 before accelerating through Years 3 and 4 — Year 4 was the strongest year on record.
- **Seasonality:** Sales consistently peak in November–December, a clear holiday-driven pattern confirmed via seasonal decomposition and an Augmented Dickey-Fuller stationarity test.
- **Forecasting:** Three models — **SARIMA**, **Prophet**, and **XGBoost** — were trained and evaluated head-to-head on held-out months. SARIMA came out on top:

  | Model | MAE | RMSE | MAPE |
  |---|---|---|---|
  | **SARIMA** | **19,244** | 29,447 | **20.5%** |
  | Prophet | 20,296 | 29,447 | 21.9% |
  | XGBoost | 29,446 | 29,447 | 32.9% |

- **Anomaly detection:** A dual-method approach (Isolation Forest + Z-score) scanned 209 weeks of data, flagging 11 and 6 outlier weeks respectively, with the two methods agreeing on the single most extreme week in the dataset — a late-March spike reaching ~3–4x normal weekly volume.
- **Segmentation:** K-Means clustering (validated with the elbow method, visualized via PCA) split the 17 sub-categories into four actionable demand segments: **High Revenue**, **Growing Demand**, **Stable Products**, and **Volatile Products** — each with a distinct inventory strategy.

---

## 🏗️ Architecture

```
Raw Transactions (train.csv)
        │
        ▼
Data Cleaning & Feature Engineering
   (dates, seasons, shipping duration, aggregation)
        │
        ▼
┌───────────────┬────────────────────┬──────────────────┐
│  EDA & Trend  │   Time Series      │  Segmentation &   │
│   Analysis    │   Forecasting      │  Anomaly Detection│
│               │ (SARIMA/Prophet/   │ (KMeans + PCA /   │
│               │   XGBoost)         │ Isolation Forest)  │
└───────────────┴────────────────────┴──────────────────┘
        │
        ▼
Streamlit Dashboard (app.py)
  Sales Overview │ Forecast Explorer │ Anomaly Report │ Product Segments
```

---

## 🛠️ Tech Stack

- **Analysis & Modeling:** Python, pandas, NumPy, statsmodels (SARIMA, seasonal decomposition, ADF test), Prophet, XGBoost, scikit-learn (KMeans, PCA, Isolation Forest)
- **Visualization:** matplotlib, seaborn, Plotly
- **Application:** Streamlit
- **Notebook:** Jupyter (`analysis.ipynb`) — full, reproducible analytical pipeline from raw data to model outputs

---

## 📁 Project Structure

```
.
├── analysis.ipynb              # Full analytical pipeline (EDA → forecasting → anomaly detection → segmentation)
├── app.py                      # Streamlit dashboard (4 pages)
├── requirements.txt
├── data/
│   └── train.csv                # Raw retail order data
├── outputs/
│   ├── monthly_sales_processed.csv
│   ├── forecast.csv
│   ├── model_metrics.csv
│   ├── anomalies.csv
│   └── product_segments.csv
└── charts/
    ├── anomaly_detection.png
    ├── clustering_elbow.png
    ├── product_clusters.png
    └── segment_forecasts.png
```

---

## 🚀 Try It / Run It Locally

**Fastest way:** just open the [live demo](https://salesforecasting-9sfccbfx2bdpah4h3fezql.streamlit.app/) — no installation needed.

To run it locally instead:

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd demand-intelligence-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook first to generate outputs/ and charts/
jupyter notebook analysis.ipynb

# 4. Launch the dashboard
streamlit run app.py
```

---

## 📊 Dashboard Preview

The dashboard has four pages:

1. **Sales Overview** — KPI cards (total sales, order count, average order value), yearly and monthly revenue trends, and an interactive region × category sub-category breakdown.
2. **Forecast Explorer** — Toggle between category-level and region-level 3-month forecasts, with live model evaluation metrics (MAE/RMSE) displayed alongside.
3. **Anomaly Report** — Full weekly anomaly timeline with a ledger of flagged outlier weeks.
4. **Product Segments** — PCA-visualized cluster map plus a stocking-strategy lookup table by sub-category.

---

## 🔮 Possible Extensions

- Incorporate external signals (planned promotions, macroeconomic indicators) into the forecasting models
- Automate model retraining on a rolling schedule as new data arrives
- Add confidence-interval bands directly into the Forecast Explorer visualizations
- Extend anomaly detection to a real-time alerting pipeline

---

*Built as an end-to-end demonstration of applied data science for retail demand planning — from raw data to a decision-ready business tool.*
