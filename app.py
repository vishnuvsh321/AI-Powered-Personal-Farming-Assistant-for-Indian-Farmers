import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# -----------------------------------------------------------
# PAGE CONFIG + CSS
# -----------------------------------------------------------
st.set_page_config(page_title="AI For Indian Farmers", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #e8f5e9;
}
h1, h2, h3 {
    color: #1b5e20 !important;
}
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD CLEANED DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yields.csv")

    # Ensure numeric
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")

    df = df.dropna(subset=["Area", "Production", "Yield"])

    return df

df = load_data()

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🤖 Yield Prediction", "🌾 Crop Recommendation"]
)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "🏠 Home":
    st.header("🌱 AI Farming Assistant for Indian Farmers")
    st.image(
        "https://wallpapercave.com/wp/wp5627799.jpg",
        use_column_width=True,
    )
    st.write("""
    ### 🇮🇳 Empowering Farmers with AI  
    This platform provides:
    - 📊 Agriculture Analytics Dashboard  
    - 🤖 AI-Based Yield Prediction  
    - 🌾 Smart Crop Recommendation  
    """)

# -----------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------
if page == "📊 Dashboard":
    st.header("📊 Agriculture Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crop_f = st.selectbox("Crop", ["All"] + sorted(df["Crop"].unique()))
    with col2:
        state_f = st.selectbox("State", ["All"] + sorted(df["State"].unique()))
    with col3:
        season_f = st.selectbox("Season", ["All"] + sorted(df["Season"].unique()))
    with col4:
        year_f = st.selectbox("Year", ["All"] + sorted(df["Crop_Year"].unique()))

    df_f = df.copy()

    if crop_f != "All": df_f = df_f[df_f["Crop"] == crop_f]
    if state_f != "All": df_f = df_f[df_f["State"] == state_f]
    if season_f != "All": df_f = df_f[df_f["Season"] == season_f]
    if year_f != "All": df_f = df_f[df_f["Crop_Year"] == year_f]

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Production", f"{df_f['Production'].sum():,.0f} tonnes")
    c2.metric("Total Area", f"{df_f['Area'].sum():,.0f} ha")
    c3.metric("Unique Crops", df_f["Crop"].nunique())

    # Charts
    st.subheader("Crop-wise Production")
    fig = px.bar(df_f.groupby("Crop")["Production"].sum())
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("State-wise Production")
    fig2 = px.bar(df_f.groupby("State")["Production"].sum())
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Yearly Trend")
    fig3 = px.line(df_f.groupby("Crop_Year")["Production"].sum().reset_index(),
                   x="Crop_Year", y="Production")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------
# YIELD PREDICTION MODEL
# -----------------------------------------------------------
if page == "🤖 Yield Prediction":
    st.header("🤖 AI Model: Crop Yield Prediction")

    df_model = df.copy()

    # One-hot encode
    df_model = pd.get_dummies(df_model, columns=["State", "Season", "Crop"], drop_first=True)

    X = df_model.drop(columns=["Yield"])
    y = df_model["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    st.subheader("Enter Inputs")

    year = st.number_input("Crop Year", min_value=1990, max_value=2050, value=2024)
    area = st.number_input("Area (ha)", min_value=0.1, value=100.0)
    state = st.selectbox("State", df["State"].unique())
    season = st.selectbox("Season", df["Season"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

    if st.button("Predict Yield"):
        # Build input row
        user = {
            "Crop_Year": year,
            "Area": area,
            "Production": 0   # placeholder, not used in prediction
        }

        for s in df_model.columns:
            if s.startswith("State_"):
                user[s] = 1 if s == f"State_{state}" else 0
            elif s.startswith("Season_"):
                user[s] = 1 if s == f"Season_{season}" else 0
            elif s.startswith("Crop_"):
                user[s] = 1 if s == f"Crop_{crop}" else 0

        # Missing columns (for safety)
        for col in X.columns:
            if col not in user:
                user[col] = 0

        user_df = pd.DataFrame([user])[X.columns]

        pred = model.predict(user_df)[0]

        st.success(f"🌾 Predicted Yield: **{pred:,.2f} tonnes/ha**")

        rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
        st.info(f"Model RMSE: **{rmse:,.2f}**")

# -----------------------------------------------------------
# CROP RECOMMENDATION
# -----------------------------------------------------------
if page == "🌾 Crop Recommendation":
    st.header("🌾 Smart Crop Recommendation")

    sel_state = st.selectbox("Select State", df["State"].unique())
    sel_season = st.selectbox("Select Season", df["Season"].unique())

    df_r = df[(df["State"] == sel_state) & (df["Season"] == sel_season)]

    if df_r.empty:
        st.warning("No data available for this combination.")
    else:
        df_r["Productivity"] = df_r["Production"] / df_r["Area"]
        best = df_r.groupby("Crop")["Productivity"].mean().idxmax()
        val = df_r.groupby("Crop")["Productivity"].mean().max()

        st.success(f"🌟 Recommended Crop: **{best}**")
        st.info(f"Expected Productivity: **{val:.2f} tonnes/ha**")
