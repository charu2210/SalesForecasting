import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 7. Use real page icons in set_page_config
st.set_page_config(page_title="Demand Intelligence Engine", page_icon="📦", layout="wide")

# 8. Better sidebar branding
st.sidebar.title("Demand Intelligence Engine")
page = st.sidebar.radio(
    "Navigation Menu:", 
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"]
)

# ----------------------------------------------------
# DATA INGESTION CACHE (Page 1 Real Data Loading)
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")

    # Handle mixed date formats matching the notebook parser cleanly
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="mixed",
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚨 Critical Error: 'data/train.csv' could not be located. Please verify your file paths.")
    st.stop()

# ----------------------------------------------------
# PAGE 1 — 📊 SALES OVERVIEW
# ----------------------------------------------------
if page == "Sales Overview":
    st.title("📊 Sales Overview")
    
    # 4. Professional High-Level KPI Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    c2.metric("Orders", f"{len(df):,}")
    c3.metric("Average Order", f"${df['Sales'].mean():,.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales by Year")
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig_year = px.bar(
            yearly,
            x="Year",
            y="Sales",
            labels={"Sales": "Revenue ($)", "Year": "Year"},
            color_discrete_sequence=["#1f77b4"]
        )
        # 6. Make charts interactive with unified hover and styling
        fig_year.update_layout(hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig_year, use_container_width=True)
        
    with col2:
        st.subheader("Monthly Trend")
        monthly = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
        fig_month = px.line(
            monthly,
            x="Order Date",
            y="Sales",
            labels={"Sales": "Revenue ($)", "Order Date": "Timeline"},
            color_discrete_sequence=["#ff7f0e"]
        )
        # 6. Make charts interactive
        fig_month.update_traces(line_width=3)
        fig_month.update_layout(hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Segment Filters")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        region = st.selectbox("Region", sorted(df["Region"].unique()))
    with col_f2:
        category = st.selectbox("Category", sorted(df["Category"].unique()))
        
    filtered = df[(df["Region"] == region) & (df["Category"] == category)]
    
    if not filtered.empty:
        subcat_summary = filtered.groupby("Sub-Category")["Sales"].sum().reset_index()
        fig_filtered = px.bar(
            subcat_summary,
            x="Sub-Category",
            y="Sales",
            labels={"Sales": "Revenue ($)"},
            title=f"Sub-Category Revenue Breakdown for {region} — {category}"
        )
        fig_filtered.update_layout(hovermode="x unified")
        st.plotly_chart(fig_filtered, use_container_width=True)
    else:
        st.info("No records match the selected slice parameters.")

# ----------------------------------------------------
# PAGE 2 — 📈 FORECAST EXPLORER
# ----------------------------------------------------
elif page == "Forecast Explorer":
    st.title("📈 Forecast Explorer")
    
    if not os.path.exists("outputs/forecast.csv") or not os.path.exists("outputs/model_metrics.csv"):
        st.warning("🚨 Model runtime assets missing inside 'outputs/'. Please run your Jupyter Notebook pipeline to generate them.")
        st.stop()
        
    forecast = pd.read_csv("outputs/forecast.csv")
    metrics = pd.read_csv("outputs/model_metrics.csv")
    
    col_p2_1, col_p2_2 = st.columns([1, 2])
    
    with col_p2_1:
        # 1. & 10. Dynamic multi-tier structural dropdown selector driven by Type/Name architecture
        option = st.radio("Forecast By", ["Category", "Region"])
        
        available = forecast[forecast["Type"] == option]["Name"].unique()
        selected = st.selectbox(f"Select {option}", sorted(available))
        
        # 2. Extract specific horizon partition using historical context matching parameters
        filtered_forecast = forecast[(forecast["Type"] == option) & (forecast["Name"] == selected)]
        
        months = st.slider("Forecast Horizon", 1, 3, 3)
        
        st.markdown("---")
        st.subheader("Model Evaluation Metrics")
        
        # Pull model metrics safely without hardcoding
        st.metric("Model Algorithm", str(metrics.loc[0, "Model"]))
        st.metric("MAE", f"{metrics.loc[0, 'MAE']:,.2f}")
        st.metric("RMSE", f"{metrics.loc[0, 'RMSE']:,.2f}")
        
    with col_p2_2:
        # 5. Clean, professional chart title modification
        st.subheader(f"Sales Forecast (XGBoost) — {selected}")
        
        # 2. Plot ONLY the filtered data cut matched to the horizon selection window
        display_df = filtered_forecast.head(months)
        
        fig_forecast = px.line(
            display_df,
            x="Date",
            y="Forecast",
            markers=True,
            labels={"Forecast": "Projected Sales ($)", "Date": "Future Timeline Target"},
            color_discrete_sequence=["#2ca02c"]
        )
        # 6. Enhance plot styling
        fig_forecast.update_traces(line_width=3, marker=dict(size=8))
        fig_forecast.update_layout(hovermode="x unified")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.dataframe(display_df[["Date", "Forecast"]], use_container_width=True)

# ----------------------------------------------------
# PAGE 3 — 🚨 ANOMALY REPORT
# ----------------------------------------------------
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")
    
    if os.path.exists("charts/anomaly_detection.png"):
        st.image(
            "charts/anomaly_detection.png", 
            caption="System Weekly Multi-Method Anomaly Profiles", 
            use_column_width=True
        )
    else:
        st.error("Missing Plot Asset: 'charts/anomaly_detection.png' not found.")
        
    st.markdown("---")
    st.subheader("Flagged Outlier Activity Data Ledger")
    
    if os.path.exists("outputs/anomalies.csv"):
        anom = pd.read_csv("outputs/anomalies.csv")
        st.dataframe(anom, use_container_width=True)
    else:
        st.warning("Ledger file 'outputs/anomalies.csv' missing. Run the anomaly section in your notebook.")

# ----------------------------------------------------
# PAGE 4 — 📦 PRODUCT SEGMENTS
# ----------------------------------------------------
elif page == "Product Segments":
    st.title("📦 Product Segments")
    
    if os.path.exists("charts/product_clusters.png"):
        st.image(
            "charts/product_clusters.png", 
            caption="Portfolio Strategic Clustering Groupings (PCA Space Map)", 
            use_column_width=True
        )
    else:
        st.error("Missing Plot Asset: 'charts/product_clusters.png' not found.")
        
    st.markdown("---")
    st.subheader("Sub-Category Stocking Strategy Matrix")
    
    if os.path.exists("outputs/product_segments.csv"):
        segments = pd.read_csv("outputs/product_segments.csv")
        
        # 3. Safe fallback matching criteria list to prevent execution schema crashes
        cols = [
            c for c in ["Sub-Category", "Demand_Segment", "Total_Sales", "Growth_Rate"]
            if c in segments.columns
        ]
        st.dataframe(segments[cols], use_container_width=True)
    else:
        st.warning("Lookup file 'outputs/product_segments.csv' missing.")

# ----------------------------------------------------
# 9. UNIVERSAL STRUCTURAL FOOTER
# ----------------------------------------------------
st.markdown("---")
st.caption("Sales Forecasting Dashboard | Built with Streamlit")
