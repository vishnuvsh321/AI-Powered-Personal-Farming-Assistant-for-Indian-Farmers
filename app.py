import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(
    page_title="AI Crop Yield Analytics",
    layout="wide",
    page_icon="🌾"
)

st.markdown("""
<style>
h1, h2, h3 {
    color: #2e7d32 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    return df

df = load_data()

st.title("🌾 AI-Powered Crop Yield Analytics Dashboard")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("🔧 Filters & Options")

selected_crop = st.sidebar.selectbox("Select Crop", df["Crop"].unique())
selected_state = st.sidebar.selectbox("Select State", df["State"].unique())
show_stats = st.sidebar.checkbox("Show Summary Statistics", True)
show_trend = st.sidebar.checkbox("Show Trend Line", False)
forecast_years = st.sidebar.slider("Forecast Years", 1, 10, 5)

# ------------------------------
# Filter Data
# ------------------------------
filtered = df[(df["Crop"] == selected_crop) & (df["State"] == selected_state)]
filtered = filtered.sort_values("Crop_Year")

# ------------------------------
# Main Graph – Yield over Years
# ------------------------------
st.subheader(f"📈 Yield Trend for {selected_crop} in {selected_state}")

fig = px.line(filtered, x="Crop_Year", y="Yield",
              markers=True,
              title=f"Yield Over Time ({selected_crop}, {selected_state})",
              template="plotly_white",
              color_discrete_sequence=["green"])
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Summary Statistics
# ------------------------------
if show_stats:
    st.subheader("📊 Summary Statistics")
    st.write(filtered.describe())

# ------------------------------
# Trend Line (Linear Regression)
# ------------------------------
if show_trend:
    st.subheader("📈 Trend Line (Regression)")
    X = filtered["Crop_Year"].values.reshape(-1, 1)
    y = filtered["Yield"].values
    model = LinearRegression()
    model.fit(X, y)
    filtered["Trend"] = model.predict(X)

    fig2 = px.line(filtered, x="Crop_Year", y=["Yield", "Trend"],
                   title="Yield Trend Line",
                   template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# Forecasting (Linear Regression)
# ------------------------------
st.subheader("🔮 Forecasting Future Yield")

X = filtered["Crop_Year"].values.reshape(-1, 1)
y = filtered["Yield"].values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(filtered["Crop_Year"].max() + 1,
                         filtered["Crop_Year"].max() + 1 + forecast_years)

future_pred = model.predict(future_years.reshape(-1, 1))

forecast_df = pd.DataFrame({
    "Crop_Year": future_years,
    "Forecast_Yield": future_pred
})

fig3 = px.line(forecast_df, x="Crop_Year", y="Forecast_Yield",
               markers=True,
               title=f"Forecasted Yield for Next {forecast_years} Years",
               template="plotly_white",
               color_discrete_sequence=["green"])
st.plotly_chart(fig3, use_container_width=True)

st.write(forecast_df)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Built with ❤️ for Kerala Farmers using AI & Streamlit.")

