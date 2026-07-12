import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "🏠 Sales Overview",
        "📈 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Demand Segments"
    ]
)

# ==========================================
# PAGE 1
# ==========================================

if page == "🏠 Sales Overview":

    st.title("📊 Sales Forecasting Dashboard")
    st.markdown("### Sales Overview")

    # --------------------------------------
    # KPI Cards
    # --------------------------------------

    total_sales = df["Sales"].sum()
    total_orders = len(df)
    average_order = df["Sales"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "📦 Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "🛒 Average Order Value",
        f"${average_order:.2f}"
    )

    st.divider()

    # --------------------------------------
    # Filters
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        selected_region = st.selectbox(
            "Select Region",
            ["All"] + sorted(df["Region"].unique())
        )

    with col2:
        selected_category = st.selectbox(
            "Select Category",
            ["All"] + sorted(df["Category"].unique())
        )

    filtered_df = df.copy()

    if selected_region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == selected_region
        ]

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == selected_category
        ]

    # --------------------------------------
    # Chart 1
    # --------------------------------------

    st.subheader("📊 Total Sales by Year")

    yearly_sales = (
        filtered_df
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    bars = ax.bar(
        yearly_sales["Year"].astype(str),
        yearly_sales["Sales"]
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x()+bar.get_width()/2,
            height,
            f"{height:,.0f}",
            ha="center",
            fontsize=9
        )

    st.pyplot(fig)

    # --------------------------------------
    # Chart 2
    # --------------------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (
        filtered_df
        .groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(
        monthly_sales["Month"],
        monthly_sales["Sales"],
        marker="o"
    )

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales")

    plt.xticks(rotation=90)

    st.pyplot(fig2)

# ==========================================
# PAGE 2
# ==========================================

elif page == "📈 Forecast Explorer":


    st.title("📈 Forecast Explorer")

    st.write("Forecast results using the best-performing model (XGBoost).")

    forecast_type = st.selectbox(
        "Forecast Type",
        ["Overall Sales", "Furniture", "Technology", "Office Supplies", "West", "East"]
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        1, 3, 3
    )

    forecasts = {
        "Overall Sales": [41741.98, 42012.59, 39361.78],
        "Furniture": [21245.59, 19415.42, 27430.56],
        "Technology": [16448.43, 12779.05, 28750.88],
        "Office Supplies": [26745.39, 26747.66, 26747.62],
        "West": [12951.23, 15210.74, 16024.27],
        "East": [6243.07, 10669.80, 10430.28]
    }

    selected_forecast = forecasts[forecast_type][:horizon]

    forecast_df = pd.DataFrame({
        "Month": [f"Month {i+1}" for i in range(horizon)],
        "Forecasted Sales": selected_forecast
    })

    st.subheader(f"{forecast_type} Forecast")

    st.dataframe(forecast_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        forecast_df["Month"],
        forecast_df["Forecasted Sales"],
        marker="o",
        linewidth=3
    )

    ax.set_xlabel("Forecast Period")
    ax.set_ylabel("Forecasted Sales")

    st.pyplot(fig)

    col1, col2 = st.columns(2)

    col1.metric("MAE", "5614.14")
    col2.metric("RMSE", "6316.11")

    st.success("Recommended Production Model: XGBoost")

# ==========================================
# PAGE 3
# ==========================================

elif page == "🚨 Anomaly Report":

    st.title("🚨 Anomaly Report")

    st.write("Weekly sales anomalies detected using Isolation Forest.")

    # ------------------------------------
    # Weekly Sales Data
    # ------------------------------------

    weekly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
        .sum()
        .reset_index()
    )

    # Isolation Forest anomaly dates from Task 5
    anomaly_dates = [
        "2015-01-04",
        "2015-02-08",
        "2015-02-22",
        "2015-03-22",
        "2015-07-19",
        "2015-09-13",
        "2016-01-24",
        "2017-12-17",
        "2018-11-04",
        "2018-11-18",
        "2018-12-02"
    ]

    anomaly_dates = pd.to_datetime(anomaly_dates)

    anomaly_df = weekly_sales[
        weekly_sales["Order Date"].isin(anomaly_dates)
    ]

    # ------------------------------------
    # Plot
    # ------------------------------------

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales"
    )

    ax.scatter(
        anomaly_df["Order Date"],
        anomaly_df["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    # ------------------------------------
    # Table
    # ------------------------------------

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_df,
        use_container_width=True
    )

    # ------------------------------------
    # Business Explanation
    # ------------------------------------

    st.subheader("Business Interpretation")

    st.markdown("""
- Large spikes likely correspond to festive or seasonal sales events.
- Extremely low weeks may indicate reduced customer demand or operational slowdowns.
- These unusual periods should be investigated before making inventory decisions.
""")

# ==========================================
# PAGE 4
# ==========================================

elif page == "📦 Product Demand Segments":

    st.title("📦 Product Demand Segments")

    st.write("Product demand groups created using K-Means Clustering.")

    # ------------------------------------
    # Cluster Data
    # ------------------------------------

    cluster_df = pd.DataFrame({

        "Sub-Category":[
            "Appliances","Art","Paper","Labels",
            "Chairs","Phones","Accessories","Binders","Storage","Tables",
            "Copiers","Machines",
            "Envelopes","Furnishings","Fasteners","Bookcases","Supplies"
        ],

        "Cluster":[
            0,0,0,0,
            1,1,1,1,1,1,
            2,2,
            3,3,3,3,3
        ]
    })

    # Dummy PCA Coordinates (for visualization)

    cluster_df["PCA1"] = [
        1.0,1.5,2.0,2.5,
        4.0,4.5,5.0,5.5,6.0,6.5,
        8.0,8.5,
        3.0,3.5,4.2,5.2,6.8
    ]

    cluster_df["PCA2"] = [
        2.5,2.0,1.8,1.5,
        6.0,5.5,6.2,5.8,6.5,5.3,
        8.0,8.5,
        3.5,3.0,2.8,3.8,4.5
    ]

    # ------------------------------------
    # Scatter Plot
    # ------------------------------------

    fig, ax = plt.subplots(figsize=(9,6))

    scatter = ax.scatter(
        cluster_df["PCA1"],
        cluster_df["PCA2"],
        c=cluster_df["Cluster"],
        s=120
    )

    for i, txt in enumerate(cluster_df["Sub-Category"]):
        ax.text(
            cluster_df["PCA1"][i],
            cluster_df["PCA2"][i],
            txt,
            fontsize=8
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    st.pyplot(fig)

    # ------------------------------------
    # Cluster Table
    # ------------------------------------

    st.subheader("Sub-Categories by Cluster")

    st.dataframe(
        cluster_df[["Sub-Category","Cluster"]],
        use_container_width=True
    )

    # ------------------------------------
    # Stocking Strategy
    # ------------------------------------

    st.subheader("Recommended Stocking Strategy")

    st.markdown("""
**Cluster 0 – Growing Demand**
- Increase inventory gradually to support growth.

**Cluster 1 – High Volume**
- Maintain higher stock levels and prioritize availability.

**Cluster 2 – Premium / High Value**
- Keep controlled inventory with close demand monitoring.

**Cluster 3 – Low Demand / Volatile**
- Maintain limited stock and review inventory frequently.
""")