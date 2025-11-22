import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Green-white theme
st.markdown("""
<style>
body { background-color: #f7fff7; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")   # <-- file in your repo root
    return df

df = load_data()

st.title("🌾 AI-Powered Personal Farming Assistant for Kerala Farmers")
st.write("A smart dashboard for analytics, predictions, crop recommendations and decision support.")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", [
    "📊 Data Overview",
    "📈 Analytics Dashboard",
    "🤖 Crop Recommendation (AI)",
    "📉 Yield Prediction (AI)",
    "🧪 Fertilizer–Pesticide Optimizer",
    "💬 Smart Farming Chatbot"
])

# -------------------------------------------------------------
# 1. DATA OVERVIEW
# -------------------------------------------------------------
if menu == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    st.write(df.head())
    st.write("### Dataset Summary")
    st.write(df.describe())
    st.write("### Columns:", df.columns.tolist())


# -------------------------------------------------------------
# 2. ANALYTICS DASHBOARD
# -------------------------------------------------------------
elif menu == "📈 Analytics Dashboard":
    st.header("📈 Analytics Dashboard")

    # Crop production over years
    st.subheader("🌾 Crop Production Over Years")
    crop = st.selectbox("Select Crop", df["Crop"].unique())
    crop_df = df[df["Crop"] == crop]

    fig = px.line(
        crop_df,
        x="Crop_Year",
        y="Production",
        markers=True,
        title=f"Production Trend of {crop}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # State wise production
    st.subheader("🏞 State-wise Production")
    fig2 = px.bar(
        df.groupby("State")["Production"].sum().reset_index(),
        x="State", y="Production",
        title="Total Production by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Rainfall vs Yield
    st.subheader("☔ Rainfall vs Yield")
    fig3 = px.scatter(
        df,
        x="Annual_Rainfall",
        y="Yield",
        color="Crop",
        title="Relationship Between Rainfall & Yield"
    )
    st.plotly_chart(fig3, use_container_width=True)


# -------------------------------------------------------------
# 3. CROP RECOMMENDATION (AI)
# -------------------------------------------------------------
elif menu == "🤖 Crop Recommendation (AI)":
    st.header("🤖 AI Crop Recommendation System")

    df_ml = df.copy()
    label_cols = ["Crop", "Season", "State"]
    encoders = {}

    for col in label_cols:
        enc = LabelEncoder()
        df_ml[col] = enc.fit_transform(df_ml[col])
        encoders[col] = enc

    X = df_ml[["Season", "State", "Annual_Rainfall", "Fertilizer", "Pesticide"]]
    y = df_ml["Crop"]

    model = RandomForestClassifier()
    model.fit(X, y)

    season_in = st.selectbox("Season", df["Season"].unique())
    state_in = st.selectbox("State", df["State"].unique())
    rain_in = st.number_input("Annual Rainfall (mm)")
    fert_in = st.number_input("Fertilizer (kg/ha)")
    pest_in = st.number_input("Pesticide (kg/ha)")

    if st.button("Recommend Crop"):
        input_data = [[
            encoders["Season"].transform([season_in])[0],
            encoders["State"].transform([state_in])[0],
            rain_in, fert_in, pest_in
        ]]
        pred = model.predict(input_data)[0]
        st.success(f"🌱 Recommended Crop: **{encoders['Crop'].inverse_transform([pred])[0]}**")


# -------------------------------------------------------------
# 4. YIELD PREDICTION (AI)
# -------------------------------------------------------------
elif menu == "📉 Yield Prediction (AI)":
    st.header("📉 AI Yield Prediction")

    X = df[["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]]
    y = df["Yield"]

    reg_model = RandomForestRegressor()
    reg_model.fit(X, y)

    area_in = st.number_input("Area (ha)")
    rain_in = st.number_input("Annual Rainfall (mm)")
    fert_in = st.number_input("Fertilizer (kg/ha)")
    pest_in = st.number_input("Pesticide (kg/ha)")

    if st.button("Predict Yield"):
        pred = reg_model.predict([[area_in, rain_in, fert_in, pest_in]])[0]
        st.success(f"🌾 Expected Yield: **{pred:.2f} tons/ha**")


# -------------------------------------------------------------
# 5. OPTIMIZER
# -------------------------------------------------------------
elif menu == "🧪 Fertilizer–Pesticide Optimizer":
    st.header("🧪 Fertilizer & Pesticide Optimization")

    opt = df.groupby("Crop")[["Fertilizer", "Pesticide", "Yield"]].mean().reset_index()
    best_crop = st.selectbox("Select Crop", opt["Crop"].unique())
    row = opt[opt["Crop"] == best_crop].iloc[0]

    st.success(f"""
    ### Optimal Inputs for {best_crop}
    - 🌱 **Fertilizer:** {row['Fertilizer']:.2f} kg/ha  
    - 🐛 **Pesticide:** {row['Pesticide']:.2f} kg/ha  
    - 📈 **Average Yield:** {row['Yield']:.2f} tons/ha  
    """)


# -------------------------------------------------------------
# 6. SMART CHATBOT
# -------------------------------------------------------------
elif menu == "💬 Smart Farming Chatbot":
    st.header("💬 Smart Farming Chat Assistant")

    query = st.text_input("Ask a crop, farming, fertilizer, or weather question:")

    if query:
        q = query.lower()
        if "rain" in q:
            st.write("Rainfall affects yield strongly. Choose crops tolerant to high rainfall like Rice or Coconut.")
        elif "fertilizer" in q:
            st.write("Optimal fertilizer values vary by crop. See the Optimizer tab for best results.")
        elif "pest" in q:
            st.write("Pesticide levels greatly influence yield. Moderate use improves productivity.")
        elif "yield" in q:
            st.write("Yield is influenced by area, rainfall, fertilizer, and pesticide usage.")
        else:
            st.write("I'm still learning! Try asking about rainfall, fertilizer, pests, or yield.")
