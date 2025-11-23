import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# -------------------------------
# Load Cleaned Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crop_yields.csv")
    return df

df = load_data()

st.title("🌾 Crop Yield Prediction – Kerala Farmers Assistant")

st.write("Using cleaned dataset with Random Forest")

# -------------------------------
# Preprocessing
# -------------------------------
# Identify feature columns
categorical_cols = ["State", "Crop", "Season"]
numeric_cols = ["Year", "Area", "Fertilizer", "Pesticide"]

# Target variable
target_col = "Yield"

df = df.dropna(subset=[target_col])

X = df[categorical_cols + numeric_cols]
y = df[target_col]

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Build Pipeline
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluate RMSE (manual, safe)
# -------------------------------
preds = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

st.write(f"### Model RMSE: **{rmse:,.2f}**")

# -------------------------------
# User Input Form
# -------------------------------
st.subheader("🔧 Enter Crop Details to Predict Yield")

with st.form("predict"):
    year = st.number_input("Year", min_value=1960, max_value=2050, value=2024)

    state = st.selectbox("State", sorted(df["State"].unique()))
    crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
    season = st.selectbox("Season", sorted(df["Season"].unique()))

    area = st.number_input("Area (hectares)", min_value=0.1, value=1.0)
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.1, value=100.0)
    pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.1, value=10.0)

    submit = st.form_submit_button("Predict Yield")

if submit:
    input_df = pd.DataFrame([{
        "Year": year,
        "State": state,
        "Crop": crop,
        "Season": season,
        "Area": area,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])

    predicted_yield = pipeline.predict(input_df)[0]

    st.success(f"🌾 **Predicted Yield: {predicted_yield:.2f} tonnes**")
