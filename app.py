import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# -----------------------------------------------------------
# PAGE CONFIG + CSS STYLING
# -----------------------------------------------------------
st.set_page_config(page_title="AI For Indian Farmers", layout="wide")

st.markdown("""
<style>
/* MAIN BACKGROUND */
body {
    background-color: #ffffff;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #e8f5e9;
}

/* HEADER */
h1, h2, h3 {
    color: #1b5e20 !important;
}

/* METRICS */
.metric {
    background-color: #f1f8e9 !important;
    padding: 15px;
    border-radius: 10px;
}

/* SELECTBOX LABEL */
.css-1pahdxg-control {
    border: 2px solid #2e7d32 !important;
}

/* BUTTON GREEN THEME */
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #1b5e20;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
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
    st.markdown("<h1>🌱 AI Farming Assistant for Indian Farmers</h1>", unsafe_allow_html=True)

    st.image(
        "https://wallpapercave.com/wp/wp5627799.jpg",
        use_column_width=True,
        caption="Agriculture • India • Sustainability"
    )

    st.markdown("""
    ### 🇮🇳 Empowering Indian Farmers with AI  
    This platform provides:
    - 📊 **Interactive Agriculture Analytics Dashboard**  
    - 🤖 **AI-Powered Yield Prediction**  
    - 🌾 **Smart Crop Recommendation System**  
    - 🧠 Data insights to support scientific decisions  
    """)

# -----------------------------------------------------------
# DASHBOARD (WITH FILTERS)
# -----------------------------------------------------------
if page == "📊 Dashboard":
    st.header("📊 Agriculture Analytics Dashboard")

    # ----- FILTERS -----
    st.subheader("🔍 Filters")
    colA, colB, colC, colD = st.columns(4)

    with colA:
        crop_filter = st.selectbox("Crop", ["All"] + sorted(df["Crop"].unique()))
    with colB:
        state_filter = st.selectbox("State", ["All"] + sorted(df["State"].unique()))
    with colC:
        season_filter = st.selectbox("Season", ["All"] + sorted(df["Season"].unique()))
    with colD:
        year_filter = st.selectbox("Year", ["All"] + sorted(df["Crop_Year"].unique()))

    df_filtered = df.copy()

    if crop_filter != "All":
        df_filtered = df_filtered[df_filtered["Crop"] == crop_filter]

    if state_filter != "All":
        df_filtered = df_filtered[df_filtered["State"] == state_filter]

    if season_filter != "All":
        df_filtered = df_filtered[df_filtered["Season"] == season_filter]

    if year_filter != "All":
        df_filtered = df_filtered[df_filtered["Crop_Year"] == year_filter]

    # ----- METRICS -----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Production (tonnes)", f"{df_filtered['Production'].sum():,.0f}")
    with col2:
        st.metric("Total Cultivated Area (ha)", f"{df_filtered['Area'].sum():,.0f}")
    with col3:
        st.metric("Unique Crops", df_filtered["Crop"].nunique())

    # ----- VISUALS -----
    st.subheader("Crop-wise Production")
    fig1 = px.bar(
        df_filtered.groupby("Crop")["Production"].sum().sort_values(ascending=False),
        labels={"value": "Production", "index": "Crop"},
        title="Crop Production by Type"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("State-wise Production")
    fig2 = px.bar(
        df_filtered.groupby("State")["Production"].sum(),
        title="Production by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Trend Over Years")
    fig3 = px.line(
        df_filtered.groupby("Crop_Year")["Production"].sum().reset_index(),
        x="Crop_Year", y="Production",
        title="Production Over Years"
    )
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------
# YIELD PREDICTION MODEL (FAST + FIXED ONE-HOT ENCODING)
# -----------------------------------------------------------
if page == "🤖 Yield Prediction":
    st.header("🤖 AI Model: Crop Yield Prediction")

    # ---------- CACHE MODEL TRAINING ----------
    @st.cache_resource
    def train_yield_model(df):
        df_m = df.copy()

        # OUTLIER REMOVAL
        Q1 = df_m["Production"].quantile(0.25)
        Q3 = df_m["Production"].quantile(0.75)
        IQR = Q3 - Q1
        df_m = df_m[(df_m["Production"] >= Q1 - 1.5 * IQR) &
                    (df_m["Production"] <= Q3 + 1.5 * IQR)]

        # FEATURE ENGINEERING
        df_m["Yield"] = df_m["Production"] / df_m["Area"]

        # ONE HOT ENCODING
        df_m = pd.get_dummies(df_m, columns=["State", "Season", "Crop"], drop_first=False)

        # FEATURES & TARGET
        X = df_m.drop(["Production"], axis=1)
        y = np.log1p(df_m["Production"])

        # SPLIT
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # RANDOM FOREST MODEL
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)

        return model, X, X_test, y_test

    # ---- train once & reuse forever ----
    model, X, X_test, y_test = train_yield_model(df)

    # -------- USER INPUT FORM --------
    st.subheader("Enter Inputs for Prediction")

    year = st.number_input("Crop Year", min_value=1990, max_value=2050, value=2024)
    area = st.number_input("Area (ha)", min_value=1.0, value=500.0)
    state = st.selectbox("State", df["State"].unique())
    season = st.selectbox("Season", df["Season"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

    # -------- BUILD USER INPUT EXACTLY MATCHING X --------
    user_input = {}

    for col in X.columns:

        # 1. Numerical columns
        if col == "Crop_Year":
            user_input[col] = year
        elif col == "Area":
            user_input[col] = area
        elif col == "Yield":
            user_input[col] = 0  # placeholder

        # 2. One-hot encoded category columns
        elif col.startswith("State_"):
            user_input[col] = 1 if col == f"State_{state}" else 0
        elif col.startswith("Season_"):
            user_input[col] = 1 if col == f"Season_{season}" else 0
        elif col.startswith("Crop_"):
            user_input[col] = 1 if col == f"Crop_{crop}" else 0

        # 3. Any other column (fallback)
        else:
            user_input[col] = 0

    # Final user dataframe (correct alignment)
    user_df = pd.DataFrame([user_input], columns=X.columns)

    # -------- PREDICT --------
    if st.button("Predict Yield"):
        log_pred = model.predict(user_df)[0]
        pred = np.expm1(log_pred)  # inverse log-transform

        st.success(f"🌾 **Predicted Yield: {pred:,.2f} tonnes**")

        # REAL RMSE
        from sklearn.metrics import mean_squared_error
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(model.predict(X_test))
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

        st.info(f"Model RMSE: {rmse:,.2f}")

# -----------------------------------------------------------
# CROP RECOMMENDATION SYSTEM
# -----------------------------------------------------------
if page == "🌾 Crop Recommendation":
    st.header("🌾 Smart Crop Recommendation")

    state_sel = st.selectbox("Select State", df["State"].unique())
    season_sel = st.selectbox("Select Season", df["Season"].unique())
    area_sel = st.number_input("Available Area (ha)", min_value=1.0, value=100.0)

    df_f = df[(df["State"] == state_sel) & (df["Season"] == season_sel)]

    if df_f.empty:
        st.warning("⚠ No data available for selected filters.")
    else:
        df_f["Productivity"] = df_f["Production"] / df_f["Area"]

        top = df_f.groupby("Crop")["Productivity"].mean().sort_values(ascending=False).head(1)

        recommended_crop = top.index[0]
        prod_val = top.values[0]

        st.success(f"🌟 **Recommended Crop: {recommended_crop}**")
        st.info(f"Expected Productivity: **{prod_val:.2f} tonnes/ha**")
