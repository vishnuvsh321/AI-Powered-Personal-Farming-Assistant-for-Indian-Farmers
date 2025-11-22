import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(page_title="Indian Farmers AI Assistant", layout="wide")

# ----------------------------------------------------
# Custom CSS for stylish dropdown navigation
# ----------------------------------------------------
st.markdown("""
<style>
.navbar {
    background-color: #2E7D32;
    padding: 15px;
    border-radius: 10px;
    width: 100%;
}

.nav-title {
    color: white;
    font-size: 26px;
    font-weight: bold;
    padding-left: 10px;
}

.dropdown {
    position: relative;
    display: inline-block;
    float: right;
    margin-right: 20px;
}

.dropbtn {
    background-color: white;
    color: #2E7D32;
    padding: 12px 20px;
    font-size: 16px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
}

.dropbtn:hover {
    background-color: #E8F5E9;
}

.dropdown-content {
    display: none;
    position: absolute;
    background-color: white;
    min-width: 180px;
    box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
    border-radius: 8px;
    z-index: 5;
}

.dropdown-content a {
    color: #2E7D32;
    padding: 12px 16px;
    text-decoration: none;
    display: block;
    font-weight: 500;
}

.dropdown-content a:hover {
    background-color: #E8F5E9;
}

.dropdown:hover .dropdown-content {
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Navigation Bar
# ----------------------------------------------------
st.markdown("""
<div class="navbar">
    <span class="nav-title">🌾 Indian Farmers AI Dashboard</span>
    <div class="dropdown">
        <button class="dropbtn">Navigate ▼</button>
        <div class="dropdown-content">
            <a href="#home">Home</a>
            <a href="#dashboard">Analytics Dashboard</a>
            <a href="#predictor">Crop Yield Predictor</a>
            <a href="#recommender">Crop Recommendation</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ----------------------------------------------------
# Dropdown menu logic
# ----------------------------------------------------
menu = st.selectbox(
    "Choose a section",
    ["Home", "Analytics Dashboard", "Crop Yield Predictor", "Crop Recommendation"]
)

# ----------------------------------------------------
# HOME
# ----------------------------------------------------
if menu == "Home":
    st.markdown("<h1 id='home' style='text-align:center;'>🌾 Welcome to the Indian Farmers AI Dashboard</h1>", unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?q=80&w=2070",
        use_column_width=True,
        caption="Empowering Indian Agriculture with AI"
    )

    st.markdown("""
        ### 🌱 About This Project  
        This platform supports **Indian Farmers** by providing:  
        - 📊 Crop production analytics  
        - 🤖 AI-powered yield prediction  
        - 🌾 Intelligent crop recommendations  
        - 📈 State & season-wise insights  
        ---
    """)

# ----------------------------------------------------
# ANALYTICS DASHBOARD
# ----------------------------------------------------
elif menu == "Analytics Dashboard":
    st.markdown("<h2 id='dashboard'>📊 Agriculture Analytics Dashboard</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Production (tonnes)", f"{df['Production'].sum():,.0f}")

    with col2:
        st.metric("Total Area (ha)", f"{df['Area'].sum():,.0f}")

    with col3:
        st.metric("Unique Crops", df["Crop"].nunique())

    st.markdown("### 🌾 Production by Crop")
    fig_crop = px.bar(
        df.groupby("Crop")["Production"].sum().sort_values(ascending=False),
        labels={"value": "Production (tonnes)", "index": "Crop"}
    )
    st.plotly_chart(fig_crop, use_container_width=True)

    st.markdown("### 🗺️ Production by State")
    fig_state = px.bar(
        df.groupby("State")["Production"].sum().sort_values(ascending=False),
        labels={"value": "Production (tonnes)", "index": "State"}
    )
    st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("### 📈 Yearly Production Trend")
    fig_year = px.line(
        df.groupby("Crop_Year")["Production"].sum().reset_index(),
        x="Crop_Year", y="Production"
    )
    st.plotly_chart(fig_year, use_container_width=True)

# ----------------------------------------------------
# CROP YIELD PREDICTOR (AI MODEL)
# ----------------------------------------------------
elif menu == "Crop Yield Predictor":
    st.markdown("<h2 id='predictor'>🤖 AI Crop Yield Predictor</h2>", unsafe_allow_html=True)

    df_model = df.copy()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    df_model["State_enc"] = le_state.fit_transform(df["State"])
    df_model["Season_enc"] = le_season.fit_transform(df["Season"])
    df_model["Crop_enc"] = le_crop.fit_transform(df["Crop"])

    X = df_model[["Crop_Year", "Area", "State_enc", "Season_enc", "Crop_enc"]]
    y = df_model["Production"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    st.markdown("### Enter Details for Prediction")

    year = st.number_input("Year", min_value=1980, max_value=2050, value=2025)
    area = st.number_input("Area (ha)", min_value=1.0, value=500.0)
    state = st.selectbox("State", df["State"].unique())
    season = st.selectbox("Season", df["Season"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

    if st.button("Predict Production"):
        input_data = pd.DataFrame({
            "Crop_Year": [year],
            "Area": [area],
            "State_enc": [le_state.transform([state])[0]],
            "Season_enc": [le_season.transform([season])[0]],
            "Crop_enc": [le_crop.transform([crop])[0]]
        })

        prediction = model.predict(input_data)[0]

        st.success(f"🌾 Predicted Production: **{prediction:,.2f} tonnes**")

# ----------------------------------------------------
# CROP RECOMMENDATION SYSTEM
# ----------------------------------------------------
elif menu == "Crop Recommendation":
    st.markdown("<h2 id='recommender'>🌾 AI-Powered Crop Recommendation</h2>", unsafe_allow_html=True)

    state_choice = st.selectbox("Select State", df["State"].unique())
    season_choice = st.selectbox("Select Season", df["Season"].unique())
    area_choice = st.number_input("Enter Area (ha)", min_value=1.0, value=100.0)

    df_filt = df[(df["State"] == state_choice) & (df["Season"] == season_choice)]

    if not df_filt.empty:
        df_filt["Productivity"] = df_filt["Production"] / df_filt["Area"]
        best_crop = df_filt.groupby("Crop")["Productivity"].mean().idxmax()
        best_prod = df_filt.groupby("Crop")["Productivity"].mean().max()

        st.success(f"🌟 Recommended Crop: **{best_crop}**")
        st.write(f"Expected Productivity: **{best_prod:.2f} tonnes/ha**")
    else:
        st.warning("No data available for the selected conditions.")

