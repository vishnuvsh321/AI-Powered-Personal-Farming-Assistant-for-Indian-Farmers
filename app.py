import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# -----------------------------------------------------------
# CONFIG + STYLING
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
# LOAD DATA (uses cleaned file you provided)
# -----------------------------------------------------------
DATA_PATH = "crop_yields.csv"  # using the uploaded cleaned file path

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🤖 Yield Prediction", "🌾 Crop Recommendation", "⚙️ Data"]
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
    - 🧠 Data-driven insights to support decisions  
    """)

# -----------------------------------------------------------
# DATA PAGE (quick peek)
# -----------------------------------------------------------
if page == "⚙️ Data":
    st.header("📂 Loaded dataset")
    st.write(f"Path: `{DATA_PATH}`")
    st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
    st.dataframe(df.head(200))
    st.write("Columns:")
    st.write(list(df.columns))

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
    prod_by_crop = df_filtered.groupby("Crop")["Production"].sum().sort_values(ascending=False).reset_index()
    fig1 = px.bar(prod_by_crop, x="Crop", y="Production", title="Crop Production by Type")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("State-wise Production")
    state_prod = df_filtered.groupby("State")["Production"].sum().reset_index()
    fig2 = px.bar(state_prod, x="State", y="Production", title="Production by State")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Trend Over Years")
    trend = df_filtered.groupby("Crop_Year")["Production"].sum().reset_index()
    fig3 = px.line(trend, x="Crop_Year", y="Production", title="Production Over Years")
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------
# YIELD PREDICTION MODEL (using encoded columns from your cleaned CSV)
# -----------------------------------------------------------
if page == "🤖 Yield Prediction":
    st.header("🤖 AI Model: Yield Prediction (Random Forest, using encoded columns)")

    # Choose features — Option A: use encoded columns that already exist in your file.
    # We'll only use safe numeric features + encoded categorical columns.
    feature_candidates = []

    # numeric features we allow (if present)
    for c in ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]:
        if c in df.columns:
            feature_candidates.append(c)

    # encoded categorical columns assumed present in your cleaned CSV
    for enc in ["State_encoded", "Season_encoded", "Crop_encoded"]:
        if enc in df.columns:
            feature_candidates.append(enc)

    # Verify target exists
    target_col = "Yield"
    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` not found in dataset. Please ensure your cleaned CSV has a '{target_col}' column.")
    else:
        # Build features X and target y
        X = df[feature_candidates].copy()
        y = df[target_col].copy()

        # Basic cleaning: numeric conversion & fill
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

        y = pd.to_numeric(y, errors="coerce")
        # drop rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train cached model
        @st.cache_resource
        def train_rf(X_tr, y_tr):
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_tr, y_tr)
            return model

        model = train_rf(X_train, y_train)

        # Evaluate RMSE (manual sqrt)
        preds_test = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds_test))
        st.info(f"Model RMSE (test): {rmse:,.4f} tonnes/ha")

        st.subheader("Enter Inputs for Prediction")

        # User inputs (use original categorical names; map to encoded values)
        # Build mapping dicts from df: name -> encoded value
        mapping_state = {}
        mapping_season = {}
        mapping_crop = {}

        if "State" in df.columns and "State_encoded" in df.columns:
            mapping_state = pd.Series(df["State_encoded"].values, index=df["State"]).to_dict()
        if "Season" in df.columns and "Season_encoded" in df.columns:
            mapping_season = pd.Series(df["Season_encoded"].values, index=df["Season"]).to_dict()
        if "Crop" in df.columns and "Crop_encoded" in df.columns:
            mapping_crop = pd.Series(df["Crop_encoded"].values, index=df["Crop"]).to_dict()

        # For safety: if duplicate indexes exist, keep first occurrence mapping
        def first_mapping(series_keys, series_vals):
            m = {}
            for k, v in zip(series_keys, series_vals):
                if k not in m:
                    m[k] = v
            return m

        if mapping_state:
            mapping_state = first_mapping(df["State"], df["State_encoded"])
        if mapping_season:
            mapping_season = first_mapping(df["Season"], df["Season_encoded"])
        if mapping_crop:
            mapping_crop = first_mapping(df["Crop"], df["Crop_encoded"])

        # fallback default encoded values (mode) if user selects unseen
        default_state_enc = int(df["State_encoded"].mode().iloc[0]) if "State_encoded" in df.columns else 0
        default_season_enc = int(df["Season_encoded"].mode().iloc[0]) if "Season_encoded" in df.columns else 0
        default_crop_enc = int(df["Crop_encoded"].mode().iloc[0]) if "Crop_encoded" in df.columns else 0

        # Input widgets
        year = st.number_input("Crop Year", min_value=int(df["Crop_Year"].min()), max_value=int(df["Crop_Year"].max()), value=int(df["Crop_Year"].max()))
        area = st.number_input("Area (ha)", min_value=0.01, value=float(df["Area"].median()))
        state = st.selectbox("State", sorted(df["State"].unique()))
        season = st.selectbox("Season", sorted(df["Season"].unique()))
        crop = st.selectbox("Crop", sorted(df["Crop"].unique()))

        # optional numeric extras
        extras_vals = {}
        for col in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
            if col in X.columns:
                extras_vals[col] = st.number_input(col, value=float(df[col].median()))

        if st.button("Predict Yield"):
            # Build user feature vector using encoded columns directly
            user_row = {}
            for feat in feature_candidates:
                if feat == "Crop_Year":
                    user_row[feat] = year
                elif feat == "Area":
                    user_row[feat] = area
                elif feat == "Annual_Rainfall":
                    user_row[feat] = extras_vals.get("Annual_Rainfall", df["Annual_Rainfall"].median())
                elif feat == "Fertilizer":
                    user_row[feat] = extras_vals.get("Fertilizer", df["Fertilizer"].median()) if "Fertilizer" in X.columns else 0
                elif feat == "Pesticide":
                    user_row[feat] = extras_vals.get("Pesticide", df["Pesticide"].median()) if "Pesticide" in X.columns else 0
                elif feat == "State_encoded":
                    user_row[feat] = mapping_state.get(state, default_state_enc)
                elif feat == "Season_encoded":
                    user_row[feat] = mapping_season.get(season, default_season_enc)
                elif feat == "Crop_encoded":
                    user_row[feat] = mapping_crop.get(crop, default_crop_enc)
                else:
                    # fallback numeric
                    user_row[feat] = float(X[feat].median())

            user_df = pd.DataFrame([user_row], columns=feature_candidates)

            # Ensure types numeric
            for c in user_df.columns:
                user_df[c] = pd.to_numeric(user_df[c], errors="coerce").fillna(X[c].median())

            pred = model.predict(user_df)[0]
            st.success(f"🌾 **Predicted Yield: {pred:,.4f} tonnes/ha**")

            # optional production estimate
            if area > 0:
                est_prod = pred * area
                st.info(f"Estimated Production for {area} ha: **{est_prod:,.2f} tonnes**")

            # show feature importances (top 8)
            try:
                fi = pd.Series(model.feature_importances_, index=feature_candidates).sort_values(ascending=False).head(8)
                fi_df = fi.reset_index()
                fi_df.columns = ["feature", "importance"]
                fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Top Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Feature importances not available.")

# -----------------------------------------------------------
# CROP RECOMMENDATION SYSTEM
# -----------------------------------------------------------
if page == "🌾 Crop Recommendation":
    st.header("🌾 Smart Crop Recommendation")

    state_sel = st.selectbox("Select State", sorted(df["State"].unique()))
    season_sel = st.selectbox("Select Season", sorted(df["Season"].unique()))
    area_sel = st.number_input("Available Area (ha)", min_value=1.0, value=100.0)

    df_f = df[(df["State"] == state_sel) & (df["Season"] == season_sel)]

    if df_f.empty:
        st.warning("⚠ No data available for selected filters.")
    else:
        df_f = df_f.copy()
        # Use cleaned Production and Area
        df_f["Productivity"] = df_f["Production"] / df_f["Area"]
        top = df_f.groupby("Crop")["Productivity"].mean().sort_values(ascending=False).head(1)

        recommended_crop = top.index[0]
        prod_val = top.values[0]

        st.success(f"🌟 **Recommended Crop: {recommended_crop}**")
        st.info(f"Expected Productivity: **{prod_val:.2f} tonnes/ha**")
