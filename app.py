# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --------------------------
# CONFIG & THEME
# --------------------------
st.set_page_config(page_title="AI Farming Assistant", page_icon="🌾", layout="wide")
st.markdown(
    """
    <style>
      body { background-color: #ffffff; }
      .sidebar .sidebar-content { background-color: #e8f5e9 !important; }
      h1, h2, h3 { color: #1b5e20 !important; }
      .card { background-color: #f1f8f4; padding: 16px; border-radius: 10px; border: 1px solid #c8e6c9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# DATA LOADING
# --------------------------
FILE_PATH = "crop_yield.csv"  

@st.cache_data
def load_data(path=FILE_PATH):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure Crop_Year is numeric
    if "Crop_Year" in df.columns:
        df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce").astype("Int64")
    return df

df = load_data()

# Validate expected columns
expected_cols = {"Crop","Crop_Year","Season","State","Area","Production","Annual_Rainfall","Fertilizer","Pesticide","Yield"}
missing = expected_cols - set(df.columns)
if missing:
    st.error(f"Dataset is missing expected columns: {missing}. Please check your CSV.")
    st.stop()

# --------------------------
# Helper functions: preprocessing & models
# --------------------------
@st.cache_data
def prepare_encoders(dataframe):
    enc = {}
    for col in ["Crop","Season","State"]:
        le = LabelEncoder()
        le.fit(dataframe[col].astype(str))
        enc[col] = le
    return enc

encoders = prepare_encoders(df)

def build_recommender(df_in):
    """Train RandomForest classifier to predict Crop from features."""
    df_ml = df_in.copy()
    # encode categorical
    df_ml["Season_e"] = encoders["Season"].transform(df_ml["Season"].astype(str))
    df_ml["State_e"] = encoders["State"].transform(df_ml["State"].astype(str))
    X = df_ml[["Season_e","State_e","Annual_Rainfall","Fertilizer","Pesticide"]].fillna(0)
    y = encoders["Crop"].transform(df_ml["Crop"].astype(str))
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    return clf

@st.cache_data
def build_yield_model(df_in):
    """Train a regressor (RandomForest) to predict Yield from numeric inputs + encoded categorical"""
    df_ml = df_in.copy()
    # encode state & season & crop as numeric to use as features
    df_ml["Season_e"] = encoders["Season"].transform(df_ml["Season"].astype(str))
    df_ml["State_e"] = encoders["State"].transform(df_ml["State"].astype(str))
    df_ml["Crop_e"] = encoders["Crop"].transform(df_ml["Crop"].astype(str))
    features = ["Area","Annual_Rainfall","Fertilizer","Pesticide","Season_e","State_e","Crop_e"]
    X = df_ml[features].fillna(0)
    y = df_ml["Yield"].fillna(0)
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X, y)
    return reg, features

# Build models once and cache
with st.spinner("Training AI models..."):
    recommender = build_recommender(df)
    yield_regressor, yield_features = build_yield_model(df)

# --------------------------
# PAGE LAYOUT
# --------------------------
st.title("🌾 AI-Powered Personal Farming Assistant (Kerala-ready)")
st.markdown("Analytics + AI recommendations (crop recommender, yield prediction, fertilizer guidance, forecasting).")

st.markdown("---")

# Sidebar: global filters
st.sidebar.header("Filters & Settings")
selected_state = st.sidebar.selectbox("Select state (for analytics)", sorted(df["State"].unique()))
selected_crop_for_analytics = st.sidebar.selectbox("Select crop (for analytics)", sorted(df["Crop"].unique()))
year_min, year_max = int(df["Crop_Year"].min()), int(df["Crop_Year"].max())
selected_year_range = st.sidebar.slider("Select Crop Year range", year_min, year_max, (year_min, year_max))

# Filtered view for analytics
analytics_df = df[
    (df["State"] == selected_state) &
    (df["Crop"] == selected_crop_for_analytics) &
    (df["Crop_Year"].between(selected_year_range[0], selected_year_range[1]))
].sort_values("Crop_Year")

# --------------------------
# KPI CARDS
# --------------------------
st.subheader("Key Insights")
k1, k2, k3 = st.columns(3)
with k1:
    st.markdown("<div class='card'><h4>Total Production</h4><h2>{:,}</h2></div>".format(int(analytics_df["Production"].sum() if not analytics_df["Production"].isna().all() else df["Production"].sum())), unsafe_allow_html=True)
with k2:
    st.markdown("<div class='card'><h4>Average Yield (selected)</h4><h2>{:.3f}</h2></div>".format(float(analytics_df["Yield"].mean() if len(analytics_df)>0 else df["Yield"].mean())), unsafe_allow_html=True)
with k3:
    st.markdown("<div class='card'><h4>Average Annual Rainfall</h4><h2>{:.1f} mm</h2></div>".format(float(analytics_df["Annual_Rainfall"].mean() if len(analytics_df)>0 else df["Annual_Rainfall"].mean())), unsafe_allow_html=True)

st.markdown("----------")

# --------------------------
# ANALYTICS: Trends & Plots
# --------------------------
st.subheader("Analytics & Visualizations")

col1, col2 = st.columns([2,1])

with col1:
    # Yield trend by year for the selected crop & state
    if analytics_df.empty:
        st.info("No rows for the selected Crop/State/year range — showing state-level averages instead.")
        state_df = df[(df["State"]==selected_state)].groupby("Crop_Year").agg({"Yield":"mean"}).reset_index()
        fig_yield = px.line(state_df, x="Crop_Year", y="Yield", title=f"Average Yield across crops in {selected_state}", template="plotly_white")
    else:
        fig_yield = px.line(analytics_df, x="Crop_Year", y="Yield", markers=True, title=f"Yield over years: {selected_crop_for_analytics} in {selected_state}", template="plotly_white")
    st.plotly_chart(fig_yield, use_container_width=True)

    # Production by Crop (state-level)
    top_n = st.slider("Top N crops to show in production bar (state-level)", 3, 20, 8)
    prod_by_crop = df[df["State"]==selected_state].groupby("Crop").agg({"Production":"sum"}).reset_index().sort_values("Production", ascending=False).head(top_n)
    fig_prod = px.bar(prod_by_crop, x="Crop", y="Production", title=f"Top {top_n} crops by production in {selected_state}", template="plotly_white")
    st.plotly_chart(fig_prod, use_container_width=True)

with col2:
    st.markdown("#### Summary statistics (filtered)")
    st.write(analytics_df.describe(include="all"))

st.markdown("----------")

# --------------------------
# AI MODULES
# --------------------------
st.subheader("AI Modules")

# ---- 1) Crop Recommender (classification)
st.markdown("### 🤖 Crop Recommender — recommend a crop given your field conditions")
with st.form("recommender_form"):
    colA, colB = st.columns(2)
    # Season and state options come from dataset
    season_input = colA.selectbox("Season", options=sorted(df["Season"].unique()))
    state_input = colB.selectbox("State", options=sorted(df["State"].unique()))
    rain_input = colA.number_input("Annual Rainfall (mm)", value=float(df["Annual_Rainfall"].median()))
    fert_input = colB.number_input("Fertilizer use (kg)", value=float(df["Fertilizer"].median()))
    pest_input = colA.number_input("Pesticide use (kg)", value=float(df["Pesticide"].median()))
    rec_submit = st.form_submit_button("Recommend Best Crop")

if rec_submit:
    try:
        s_e = encoders["Season"].transform([season_input])[0]
        st_e = encoders["State"].transform([state_input])[0]
        Xq = np.array([[s_e, st_e, rain_input, fert_input, pest_input]])
        pred_idx = recommender.predict(Xq)[0]
        crop_name = encoders["Crop"].inverse_transform([pred_idx])[0]
        st.success(f"🌱 Recommended Crop: **{crop_name}**")
        # show probabilities for top 3
        probs = recommender.predict_proba(Xq)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        proba_table = [{"Crop": encoders["Crop"].inverse_transform([i])[0], "Probability": float(probs[i])} for i in top3_idx]
        st.table(pd.DataFrame(proba_table))
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("----------")

# ---- 2) Yield Prediction (regression)
st.markdown("### 📈 Yield Predictor — predict expected yield for given inputs")
with st.form("yield_form"):
    crop_choice = st.selectbox("Crop", options=sorted(df["Crop"].unique()))
    state_choice = st.selectbox("State for yield prediction", options=sorted(df["State"].unique()))
    area_input = st.number_input("Area (ha)", min_value=0.0, value=1.0, step=0.1)
    rain_input2 = st.number_input("Annual Rainfall (mm)", value=float(df["Annual_Rainfall"].median()))
    fert_input2 = st.number_input("Fertilizer (kg)", value=float(df["Fertilizer"].median()))
    pest_input2 = st.number_input("Pesticide (kg)", value=float(df["Pesticide"].median()))
    yield_submit = st.form_submit_button("Predict Yield")

if yield_submit:
    try:
        crop_e = encoders["Crop"].transform([crop_choice])[0]
        season_dummy = encoders["Season"].transform([df[df["Crop"]==crop_choice]["Season"].mode()[0]])[0] if not df[df["Crop"]==crop_choice].empty else 0
        state_e2 = encoders["State"].transform([state_choice])[0]
        Xq = np.array([[area_input, rain_input2, fert_input2, pest_input2, season_dummy, state_e2, crop_e]])
        pred_yield = yield_regressor.predict(Xq)[0]
        st.success(f"🌾 Predicted Yield: **{pred_yield:.3f} (units matching dataset)**")
    except Exception as e:
        st.error(f"Yield prediction error: {e}")

st.markdown("----------")

# ---- 3) Fertilizer & Pesticide Optimizer
st.markdown("### 🧪 Fertilizer & Pesticide Guidance (historical averages)")
crop_opt = st.selectbox("Choose crop to see recommended average inputs", options=sorted(df["Crop"].unique()), index=0)
opt_row = df[df["Crop"]==crop_opt].agg({"Fertilizer":"mean","Pesticide":"mean","Yield":"mean"}).to_dict()
st.metric(label=f"Avg Fertilizer for {crop_opt} (historical)", value=f"{opt_row['Fertilizer']:.2f} kg")
st.metric(label=f"Avg Pesticide for {crop_opt} (historical)", value=f"{opt_row['Pesticide']:.2f} kg")
st.write(f"Typical historical yield for {crop_opt}: {opt_row['Yield']:.3f}")

st.markdown("---")

# ---- 4) Forecast Future Yield for selected crop & state
st.subheader("🔮 Forecast future yield (model-based projection)")
st.write("This uses a RandomForest regressor trained on historical years for the selected crop & state to project future Crop_Year yields.")

f_crop = st.selectbox("Choose crop to forecast", options=sorted(df["Crop"].unique()), index=0, key="fcrop")
f_state = st.selectbox("Choose state to forecast", options=sorted(df["State"].unique()), index=0, key="fstate")
f_horizon = st.slider("Forecast horizon (years)", 1, 10, 3)

# Prepare training data for forecasting (filtered by crop & state to capture local pattern)
train_df = df[(df["Crop"]==f_crop) & (df["State"]==f_state)].sort_values("Crop_Year")
if len(train_df) < 3:
    st.info("Not enough historical rows for this crop/state to produce a reliable forecast. Try wider filters or choose another crop/state.")
else:
    # use Crop_Year numeric + inputs to predict Yield
    X_train = train_df[["Crop_Year","Area","Annual_Rainfall","Fertilizer","Pesticide"]].fillna(0)
    y_train = train_df["Yield"].fillna(0)
    fr = RandomForestRegressor(n_estimators=200, random_state=42)
    fr.fit(X_train, y_train)
    last_year = int(train_df["Crop_Year"].max())
    future_years = np.arange(last_year+1, last_year+1+f_horizon)
    # For forecasting inputs we take median of historical Area/Rain/Fert/Pest
    med_area = float(train_df["Area"].median())
    med_rain = float(train_df["Annual_Rainfall"].median())
    med_fert = float(train_df["Fertilizer"].median())
    med_pest = float(train_df["Pesticide"].median())
    X_future = pd.DataFrame({
        "Crop_Year": future_years,
        "Area": [med_area]*f_horizon,
        "Annual_Rainfall": [med_rain]*f_horizon,
        "Fertilizer": [med_fert]*f_horizon,
        "Pesticide": [med_pest]*f_horizon
    })
    y_future = fr.predict(X_future)
    forecast_df = pd.DataFrame({"Crop_Year": future_years, "Forecast_Yield": y_future})
    fig_forecast = px.line(pd.concat([train_df[["Crop_Year","Yield"]].rename(columns={"Yield":"Value"}), forecast_df.rename(columns={"Forecast_Yield":"Value"})]),
                           x="Crop_Year", y="Value",
                           title=f"Historical + Forecast Yield - {f_crop} in {f_state}",
                           template="plotly_white")
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.write(forecast_df)

st.markdown("---")

# --------------------------
# Simple Smart Chat Assistant
# --------------------------
st.subheader("🧠 Farming Chat Assistant")
st.write("Ask simple questions about recommendations, or type 'recommend' to get a crop suggestion after entering your conditions.")

chat_input = st.text_input("Ask a question (e.g., 'Which crop should I grow with 1500 mm rain?')")

if chat_input:
    q = chat_input.lower()
    answered = False
    # Rule-based quick answers
    if "recommend" in q or "which crop" in q or "what to grow" in q:
        st.write("To recommend, please provide conditions in the Recommender section (Season, State, Annual Rainfall, Fertilizer, Pesticide) and press Recommend. Or ask 'recommend crop' with values like 'recommend crop 1500mm 100kg 5kg'.")
        # try to parse a quick inline recommendation like "recommend crop 1500 200 10"
        tokens = [t for t in q.split() if t.replace('.','',1).isdigit()]
        if len(tokens) >= 3:
            try:
                rain_q = float(tokens[0])
                fert_q = float(tokens[1])
                pest_q = float(tokens[2])
                # use state & season from sidebar defaults
                s_e = encoders["Season"].transform([season_input])[0] if 'season_input' in locals() else encoders["Season"].transform([df["Season"].mode()[0]])[0]
                st_e = encoders["State"].transform([state_input])[0] if 'state_input' in locals() else encoders["State"].transform([df["State"].mode()[0]])[0]
                Xq = np.array([[s_e, st_e, rain_q, fert_q, pest_q]])
                pred_idx = recommender.predict(Xq)[0]
                crop_name = encoders["Crop"].inverse_transform([pred_idx])[0]
                st.success(f"Suggestion (quick parse): **{crop_name}**")
                answered = True
            except Exception:
                pass
    if not answered:
        # Topic-based generic answers
        if "fertilizer" in q or "pesticide" in q:
            st.write("Fertilizer and pesticide needs vary by crop, soil and season. See the Fertilizer & Pesticide Guidance section for historical averages per crop.")
        elif "rain" in q or "rainfall" in q:
            st.write("Annual rainfall strongly influences crop suitability. Use the Recommender to set annual rainfall and obtain suggested crops.")
        elif "yield" in q:
            st.write("You can predict yield in the Yield Predictor section by providing Area, Annual Rainfall, Fertilizer and Pesticide values.")
        else:
            st.write("I can help with crop recommendations, yield prediction, fertilizer guidance, and forecasting. Try asking about those topics.")

st.markdown("----------")

# --------------------------
# Data download & repo hint
# --------------------------
st.subheader("Download data (filtered)")
st.download_button("Download filtered CSV", analytics_df.to_csv(index=False).encode('utf-8'), file_name="filtered_crop_data.csv", mime="text/csv")

st.caption("Dataset path used in this app: `/mnt/data/crop_yield.csv`")

st.markdown("Made with ❤️ — you can extend the models (XGBoost, hyperparam tuning, LSTM) or integrate sensors / weather API for more dynamic recommendations.")
