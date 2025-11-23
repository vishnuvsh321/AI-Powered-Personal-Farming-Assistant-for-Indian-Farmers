import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import re

# -----------------------------------------------------------
# PAGE CONFIG + CSS (kept simple)
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
# LOAD + CLEAN DATA
# -----------------------------------------------------------
@st.cache_data
def load_and_clean(path="/mnt/data/crop_yield.csv"):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = df.columns.str.strip()

    # utility to extract numeric (keeps decimals). removes commas and units like ' tonnes'
    def to_numeric_clean(series):
        s = series.astype(str).str.replace(",", "")
        # extract first number
        extracted = s.str.extract(r'([-+]?\d*\.?\d+)', expand=False)
        return pd.to_numeric(extracted, errors="coerce")

    # Clean common numeric columns if they exist
    for col in ["Production", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield", "Yield(tn/ha)"]:
        if col in df.columns:
            df[col] = to_numeric_clean(df[col])

    # prefer 'Yield' if available; unify names
    if "Yield(tn/ha)" in df.columns and "Yield" not in df.columns:
        df.rename(columns={"Yield(tn/ha)": "Yield"}, inplace=True)

    # If Production is missing but Yield exists and Area exists, compute Production = Yield * Area
    if ("Production" not in df.columns or df["Production"].isna().all()) and ("Yield" in df.columns and "Area" in df.columns):
        df["Production"] = df["Yield"] * df["Area"]

    # drop rows missing essential columns
    required = ["Crop", "Crop_Year", "Area", "Production", "State", "Season"]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)

    # Remove extreme outliers in Production using IQR (helps stability)
    if "Production" in df.columns:
        Q1 = df["Production"].quantile(0.25)
        Q3 = df["Production"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # if IQR is zero (bad), skip filtering
        if IQR > 0:
            df = df[(df["Production"] >= lower) & (df["Production"] <= upper)]

    # Reset index
    df = df.reset_index(drop=True)
    return df

df = load_and_clean("/mnt/data/crop_yield.csv")

# -----------------------------------------------------------
# SIDEBAR + Navigation (kept same)
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🤖 Yield Prediction", "🌾 Crop Recommendation"]
)

# -----------------------------------------------------------
# HOME
# -----------------------------------------------------------
if page == "🏠 Home":
    st.markdown("<h1>🌱 AI Farming Assistant for Indian Farmers</h1>", unsafe_allow_html=True)
    st.image("https://wallpapercave.com/wp/wp5627799.jpg", use_column_width=True, caption="Agriculture • India • Sustainability")
    st.markdown("""
    ### 🇮🇳 Empowering Indian Farmers with AI  
    This platform provides:
    - 📊 **Interactive Agriculture Analytics Dashboard**  
    - 🤖 **AI-Powered Yield/Production Prediction**  
    - 🌾 **Smart Crop Recommendation System**
    """)

# -----------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------
if page == "📊 Dashboard":
    st.header("📊 Agriculture Analytics Dashboard")

    # Filters
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Production (tonnes)", f"{df_filtered['Production'].sum():,.0f}")
    with col2:
        st.metric("Total Cultivated Area (ha)", f"{df_filtered['Area'].sum():,.0f}")
    with col3:
        st.metric("Unique Crops", df_filtered["Crop"].nunique())

    st.subheader("Crop-wise Production")
    prod_by_crop = df_filtered.groupby("Crop")["Production"].sum().sort_values(ascending=False).reset_index()
    fig1 = px.bar(prod_by_crop, x="Crop", y="Production", title="Crop Production by Type")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("State-wise Production")
    fig2 = px.bar(df_filtered.groupby("State")["Production"].sum().reset_index(), x="State", y="Production", title="Production by State")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Trend Over Years")
    fig3 = px.line(df_filtered.groupby("Crop_Year")["Production"].sum().reset_index(), x="Crop_Year", y="Production", title="Production Over Years")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------
# YIELD / PRODUCTION PREDICTION (CLEAN + CACHED)
# -----------------------------------------------------------
if page == "🤖 Yield Prediction":
    st.header("🤖 AI Model: Production Prediction (cleaned & stable)")

    # TRAINING (cached so it doesn't retrain on every rerun)
    @st.cache_resource
    def train_model(df_local):
        df_m = df_local.copy()

        # Create log features to stabilize scale
        df_m["Area_log"] = np.log1p(df_m["Area"])
        df_m["Prod_log"] = np.log1p(df_m["Production"])

        # Use additional numeric features if present (cleaned earlier)
        extras = []
        for c in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
            if c in df_m.columns:
                df_m[c] = df_m[c].fillna(df_m[c].median())
                extras.append(c)

        # Label encode categorical columns (keeps model similar to your original)
        le_state = LabelEncoder()
        le_season = LabelEncoder()
        le_crop = LabelEncoder()

        df_m["State_enc"] = le_state.fit_transform(df_m["State"].astype(str))
        df_m["Season_enc"] = le_season.fit_transform(df_m["Season"].astype(str))
        df_m["Crop_enc"] = le_crop.fit_transform(df_m["Crop"].astype(str))

        # Build feature matrix — keep it simple and robust
        feature_cols = ["Crop_Year", "Area_log", "State_enc", "Season_enc", "Crop_enc"] + extras
        X = df_m[feature_cols]
        y = df_m["Prod_log"]  # log-target

        # Train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # return model + encoders + feature list + test sets for RMSE
        return {
            "model": model,
            "feature_cols": feature_cols,
            "le_state": le_state,
            "le_season": le_season,
            "le_crop": le_crop,
            "X_test": X_test,
            "y_test": y_test
        }

    state = st.selectbox("Select model training subset (quick check)", ["Full dataset"], index=0)

    # Train (cached)
    resources = train_model(df)
    model = resources["model"]
    feature_cols = resources["feature_cols"]
    le_state = resources["le_state"]
    le_season = resources["le_season"]
    le_crop = resources["le_crop"]
    X_test = resources["X_test"]
    y_test = resources["y_test"]

    # Show training RMSE (converted back to real scale)
    y_pred_test_log = model.predict(X_test)
    y_pred_test_real = np.expm1(y_pred_test_log)
    y_test_real = np.expm1(y_test)
    train_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test_real))
    st.info(f"Model RMSE (real scale): {train_rmse:,.2f} tonnes")

    st.subheader("Enter Inputs for Prediction")

    # Input widgets
    year = st.number_input("Crop Year", min_value=int(df["Crop_Year"].min()), max_value=int(df["Crop_Year"].max()), value=int(df["Crop_Year"].max()))
    area = st.number_input("Area (ha)", min_value=0.0, value=float(df["Area"].median()))
    state_in = st.selectbox("State", df["State"].unique())
    season_in = st.selectbox("Season", df["Season"].unique())
    crop_in = st.selectbox("Crop", df["Crop"].unique())

    # If extras exist, show them
    extras_input = {}
    for c in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
        if c in df.columns:
            default = float(df[c].median())
            extras_input[c] = st.number_input(c, value=default)

    # Safe transform helper for encoders (unknown values -> fallback to most common)
    def safe_transform(le, value):
        try:
            return int(le.transform([value])[0])
        except Exception:
            # fallback: map to mode (most frequent) if unseen
            classes = list(le.classes_)
            return int(np.where(classes == classes[0], 0, 0)[0]) if len(classes) > 0 else 0

    # Build user feature vector matching feature_cols
    user_row = {}
    for col in feature_cols:
        if col == "Crop_Year":
            user_row[col] = year
        elif col == "Area_log":
            user_row[col] = np.log1p(area)
        elif col == "State_enc":
            user_row[col] = safe_transform(le_state, state_in)
        elif col == "Season_enc":
            user_row[col] = safe_transform(le_season, season_in)
        elif col == "Crop_enc":
            user_row[col] = safe_transform(le_crop, crop_in)
        elif col in extras_input:
            user_row[col] = extras_input[col]
        else:
            user_row[col] = 0

    user_df = pd.DataFrame([user_row], columns=feature_cols)

    if st.button("Predict Production"):
        # Predict in log space, convert back
        pred_log = model.predict(user_df)[0]
        pred_real = np.expm1(pred_log)
        st.success(f"🌾 **Predicted Production: {pred_real:,.2f} tonnes**")

        # Show per-hectare yield as bonus if area >0
        if area > 0:
            yield_per_ha = pred_real / area
            st.info(f"Estimated Yield: {yield_per_ha:,.4f} tonnes/ha")

        # Show feature importances (top 8)
        importances = model.feature_importances_
        fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False).head(8)
        fig = px.bar(fi, x="importance", y="feature", orientation='h', title="Top feature importances")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# CROP RECOMMENDATION (unchanged but robust)
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
