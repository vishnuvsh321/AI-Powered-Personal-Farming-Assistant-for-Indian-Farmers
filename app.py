import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------
# PAGE SETTINGS + CUSTOM CSS
# ---------------------------------------------
st.set_page_config(page_title="Indian Farmers Assistant", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #e7f7e7;
        }
        h1, h2, h3, h4 {
            color: #1b5e20;
            font-weight: 700;
        }
        .nav-title {
            font-size: 24px;
            font-weight: 800;
            color: #1b5e20;
            padding-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
df = pd.read_csv("/mnt/data/crop_yield.csv") 

# Fix date column if required
date_col = [c for c in df.columns if "date" in c.lower()]
if date_col:
    df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors="coerce")

# ---------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------
st.sidebar.markdown("<div class='nav-title'>Indian Farmers Assistant</div>", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Select a Section",
    ["Home", "Analytics Dashboard", "Crop Recommendation AI"]
)

# ---------------------------------------------
# HOME PAGE
# ---------------------------------------------
if section == "Home":
    st.image("https://wallpapercave.com/wp/wp5627799.jpg", use_column_width=True)
    st.title("🇮🇳 Indian Farmers AI Assistant")
    st.write("""
    Welcome to the AI-powered analytics and recommendation system created to support Indian farmers.
    This platform uses **data analytics**, **machine learning**, and **visual insights** to help farmers 
    make informed decisions about yield, crops, and farming conditions.
    """)

# ---------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------
elif section == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    # ------------------------------
    # Filter Slicers
    # ------------------------------
    st.subheader("🔎 Filters (Slicers)")

    # Numeric slicers
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        num_filter_col = st.selectbox("Select Numeric Column", numeric_cols)
        num_min, num_max = st.slider(
            f"Range for {num_filter_col}",
            float(df[num_filter_col].min()),
            float(df[num_filter_col].max()),
            (float(df[num_filter_col].min()), float(df[num_filter_col].max()))
        )

    with col2:
        cat_filter_col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object']).columns)
        cat_unique = df[cat_filter_col].unique().tolist()
        cat_selection = st.multiselect("Select Categories", cat_unique, default=cat_unique)

    with col3:
        if date_cols:
            date_col_selected = st.selectbox("Select Date Column", date_cols)
            date_min, date_max = st.date_input(
                f"Select Date Range for {date_col_selected}",
                [df[date_col_selected].min(), df[date_col_selected].max()]
            )
        else:
            date_col_selected = None

    # APPLY FILTERS
    filtered_df = df[
        (df[num_filter_col] >= num_min) &
        (df[num_filter_col] <= num_max) &
        (df[cat_filter_col].isin(cat_selection))
    ]

    if date_col_selected:
        filtered_df = filtered_df[
            (filtered_df[date_col_selected] >= pd.to_datetime(date_min)) &
            (filtered_df[date_col_selected] <= pd.to_datetime(date_max))
        ]

    st.write("### Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)

    # ------------------------------
    # Dashboard Plots
    # ------------------------------
    st.subheader("📈 Visual Insights")

    colA, colB = st.columns(2)

    with colA:
        numeric_choice = st.selectbox("Select Column for Histogram", numeric_cols)
        fig1 = px.histogram(filtered_df, x=numeric_choice, nbins=30, title=f"Distribution of {numeric_choice}")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        if len(numeric_cols) >= 2:
            num_x = st.selectbox("X-axis", numeric_cols, index=0)
            num_y = st.selectbox("Y-axis", numeric_cols, index=1)
            fig2 = px.scatter(filtered_df, x=num_x, y=num_y, trendline="ols",
                              title=f"{num_x} vs {num_y}")
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------
# CROP RECOMMENDATION AI MODEL
# ---------------------------------------------
elif section == "Crop Recommendation AI":
    st.title("🌾 AI-Powered Crop Recommendation")

    st.write("This model recommends the **best crop** based on farming conditions.")

    # Auto-detect features for model
    feature_candidates = ["temperature", "humidity", "rainfall", "ph", "soil_type"]
    features = [f for f in feature_candidates if f in df.columns]

    if "crop" not in df.columns:
        st.error("❌ Dataset must include a 'crop' column for training the model.")
    else:
        # Prepare ML data
        X = df[features].copy()

        # Convert soil_type to numeric if needed
        if "soil_type" in X.columns:
            X["soil_type"] = X["soil_type"].astype("category").cat.codes

        y = df["crop"]

        # Train model
        model = RandomForestClassifier()
        model.fit(X, y)

        st.subheader("Enter Conditions:")
        col1, col2, col3 = st.columns(3)

        inputs = {}
        for f in features:
            with col1 if len(inputs) % 3 == 0 else col2 if len(inputs) % 3 == 1 else col3:
                if df[f].dtype == "O":
                    inputs[f] = st.selectbox(f, df[f].unique())
                else:
                    inputs[f] = st.number_input(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))

        # Convert soil type if applicable
        if "soil_type" in inputs:
            inputs["soil_type"] = df["soil_type"].astype("category").cat.categories.get_loc(inputs["soil_type"])

        if st.button("Predict Best Crop"):
            user_df = pd.DataFrame([inputs])
            prediction = model.predict(user_df)[0]
            st.success(f"🌱 Recommended Crop: **{prediction}**")
