import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import itertools

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
FILE_PATH = "/mnt/data/crop_yield.csv"   # uploaded dataset path

@st.cache_data
def load_data(path=FILE_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # ensure Crop_Year is numeric if present
    if "Crop_Year" in df.columns:
        df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce").astype("Int64")
    return df

df = load_data()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🌿 Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["🏠 Home", "📊 Dashboard", "🤖 Yield Prediction", "🌾 Crop Recommendation", "🌱 Crop Combinations"]
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
        year_filter = st.selectbox("Year", ["All"] + sorted(df["Crop_Year"].dropna().unique().astype(int).tolist()))

    df_filtered = df.copy()

    if crop_filter != "All":
        df_filtered = df_filtered[df_filtered["Crop"] == crop_filter]

    if state_filter != "All":
        df_filtered = df_filtered[df_filtered["State"] == state_filter]

    if season_filter != "All":
        df_filtered = df_filtered[df_filtered["Season"] == season_filter]

    if year_filter != "All":
        df_filtered = df_filtered[df_filtered["Crop_Year"] == int(year_filter)]

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
        df_filtered.groupby("Crop")["Production"].sum().sort_values(ascending=False).reset_index(),
        x="Crop", y="Production",
        labels={"Production": "Production (tonnes)"},
        title="Crop Production by Type"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("State-wise Production")
    fig2 = px.bar(
        df_filtered.groupby("State")["Production"].sum().reset_index().sort_values("Production", ascending=False),
        x="State", y="Production",
        labels={"Production": "Production (tonnes)"},
        title="Production by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Trend Over Years")
    # handle missing Crop_Year gracefully
    if "Crop_Year" in df_filtered.columns:
        fig3 = px.line(
            df_filtered.groupby("Crop_Year")["Production"].sum().reset_index().sort_values("Crop_Year"),
            x="Crop_Year", y="Production",
            title="Production Over Years"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No Crop_Year column available for trend plot.")

# -----------------------------------------------------------
# YIELD PREDICTION MODEL (OPTIMIZED)
# -----------------------------------------------------------
if page == "🤖 Yield Prediction":
    st.header("🤖 AI Model: Crop Yield Prediction")

    # -------- CLEAN DATA --------
    df_model = df.copy()

    # REMOVE OUTLIERS USING IQR
    Q1 = df_model["Production"].quantile(0.25)
    Q3 = df_model["Production"].quantile(0.75)
    IQR = Q3 - Q1
    df_model = df_model[(df_model["Production"] >= Q1 - 1.5 * IQR) &
                        (df_model["Production"] <= Q3 + 1.5 * IQR)]

    # FEATURE ENGINEERING
    df_model["Yield"] = df_model["Production"] / df_model["Area"]

    # ONE-HOT ENCODE CATEGORICALS
    df_model = pd.get_dummies(df_model, columns=["State", "Season", "Crop"], drop_first=True)

    # FEATURES & TARGET
    X = df_model.drop(["Production"], axis=1)
    y = np.log1p(df_model["Production"])       # LOG-TRANSFORM TARGET

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------- TRAIN XGBOOST MODEL --------
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------- USER INPUT FORM --------
    st.subheader("Enter Inputs for Prediction")

    year = st.number_input("Crop Year", min_value=1990, max_value=2050, value=2024)
    area = st.number_input("Area (ha)", min_value=1.0, value=500.0)
    state = st.selectbox("State", df["State"].unique())
    season = st.selectbox("Season", df["Season"].unique())
    crop = st.selectbox("Crop", df["Crop"].unique())

    # BUILD USER INPUT ROW
    user_input = {
        "Crop_Year": year,
        "Area": area,
        "Yield": 0   # dummy, model will compute log-production anyway
    }

    # ADD ONE-HOT COLUMNS
    for col in X.columns:
        if col.startswith("State_"):
            user_input[col] = 1 if col == f"State_{state}" else 0
        elif col.startswith("Season_"):
            user_input[col] = 1 if col == f"Season_{season}" else 0
        elif col.startswith("Crop_"):
            user_input[col] = 1 if col == f"Crop_{crop}" else 0

    # Add any missing columns (safety)
    for col in X.columns:
        if col not in user_input:
            user_input[col] = 0

    user_df = pd.DataFrame([user_input])[X.columns]

    # -------- PREDICTION BUTTON --------
    if st.button("Predict Yield"):
        log_pred = model.predict(user_df)[0]          # prediction in log space
        pred = np.expm1(log_pred)                     # convert back to tonnes

        st.success(f"🌾 **Predicted Yield: {pred:,.2f} tonnes**")

        # RMSE in real scale
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

# -----------------------------------------------------------
# HYBRID: RULE-BASED + DATA-DRIVEN CROP COMBINATIONS
# -----------------------------------------------------------
if page == "🌱 Crop Combinations":
    st.header("🌱 Crop Combination & Intercropping Advisor (Hybrid)")

    st.markdown("This page combines **rule-based companion cropping knowledge** with **data-driven co-occurrence analysis** from your uploaded dataset.")

    # ----------------------
    # Rule-based companion list (India-focused examples)
    # ----------------------
    rule_pairs = {
        "Maize": ["Beans", "Cowpea"],
        "Paddy": ["Green Gram", "Sesbania"],
        "Coconut": ["Banana", "Black Pepper"],
        "Cotton": ["Groundnut", "Cowpea"],
        "Wheat": ["Mustard"],
        "Sugarcane": ["Legumes"],
        "Millet": ["Legumes", "Pulses"],
        "Tea": ["Black Pepper"],
        "Banana": ["Coconut", "Papaya"]
    }

    st.subheader("Rule-based Companion Crops (quick reference)")
    st.write("These are commonly recommended companion/ intercrop choices (agronomy-backed).")
    for main, companions in rule_pairs.items():
        st.write(f"**{main}** → {', '.join(companions)}")

    st.markdown("---")

    # ----------------------
    # Data-driven co-occurrence analysis
    # ----------------------
    st.subheader("Data-driven Co-occurrence Analysis")

    # Build transactions: group rows by (State, Season, Crop_Year) and collect all crops present in that group
    group_cols = []
    if "State" in df.columns:
        group_cols.append("State")
    if "Season" in df.columns:
        group_cols.append("Season")
    if "Crop_Year" in df.columns:
        group_cols.append("Crop_Year")
    else:
        # fallback: group only by State+Season if year missing
        if "State" in df.columns and "Season" in df.columns:
            group_cols = ["State", "Season"]

    if not group_cols:
        st.info("Dataset doesn't have State/Season/Crop_Year to infer co-cultivation groups. Data-driven analysis unavailable.")
    else:
        transactions = df.groupby(group_cols)["Crop"].agg(lambda x: list(x.unique())).reset_index()
        # create binary one-hot matrix for crops
        all_crops = sorted(df["Crop"].unique())
        one_hot = pd.DataFrame(0, index=transactions.index, columns=all_crops)
        for i, crops in enumerate(transactions["Crop"]):
            for c in crops:
                one_hot.at[i, c] = 1

        # compute support for single crops
        support = one_hot.sum() / len(one_hot)

        # compute pairwise co-occurrence counts & metrics
        pairs = []
        for a, b in itertools.combinations(all_crops, 2):
            both = ((one_hot[a] == 1) & (one_hot[b] == 1)).sum()
            support_ab = both / len(one_hot)
            if support_ab == 0:
                continue
            # confidence a->b = support(ab)/support(a)
            conf_a_b = support_ab / (support[a] if support[a] > 0 else 1e-9)
            conf_b_a = support_ab / (support[b] if support[b] > 0 else 1e-9)
            # lift = support(ab)/(support(a)*support(b))
            lift = support_ab / (support[a] * support[b]) if (support[a] * support[b])>0 else 0
            pairs.append({
                "crop_a": a,
                "crop_b": b,
                "support": support_ab,
                "confidence_a_to_b": conf_a_b,
                "confidence_b_to_a": conf_b_a,
                "lift": lift,
                "count": both
            })

        pairs_df = pd.DataFrame(pairs).sort_values("lift", ascending=False)

        st.write(f"Found **{len(pairs_df)}** co-occurring crop pairs in grouped observations.")
        st.markdown("Top pairs by lift:")
        st.dataframe(pairs_df.head(10))

        st.markdown("---")
        st.subheader("Recommend companion crops (data + rules)")

        sel_crop = st.selectbox("Select main crop", all_crops)
        sel_state = None
        sel_season = None
        # optional filters for state/season context to compute uplift
        if "State" in df.columns:
            sel_state = st.selectbox("Optional: restrict to State (for uplift calc)", ["All"] + sorted(df["State"].unique()))
        if "Season" in df.columns:
            sel_season = st.selectbox("Optional: restrict to Season (for uplift calc)", ["All"] + sorted(df["Season"].unique()))

        # rule-based companions
        rule_suggestions = rule_pairs.get(sel_crop, [])

        # data-driven companions from pairs_df
        df_pairs_sel = pairs_df[(pairs_df["crop_a"] == sel_crop) | (pairs_df["crop_b"] == sel_crop)].copy()
        # create companion name and score
        df_pairs_sel["companion"] = df_pairs_sel.apply(lambda r: r["crop_b"] if r["crop_a"] == sel_crop else r["crop_a"], axis=1)
        df_pairs_sel = df_pairs_sel.sort_values(["lift","support"], ascending=[False, False])

        # top companions from data
        top_companions = df_pairs_sel["companion"].unique().tolist()[:5]

        st.markdown("### Suggestions")
        if rule_suggestions:
            st.write("**Rule-based:** " + ", ".join(rule_suggestions))
        else:
            st.write("**Rule-based:** (no rule found)")

        if len(top_companions) > 0:
            st.write("**Data-driven top companions:** " + ", ".join(top_companions))
        else:
            st.write("**Data-driven:** Not enough co-occurrence data to suggest companions.")

        # ----------------------
        # Estimate combined productivity uplift when both crops are present historically
        # ----------------------
        st.markdown("---")
        st.subheader("Estimated Historical Combined Productivity (uplift)")

        def calc_uplift(main_crop, companion_crop, state_filter=None, season_filter=None):
            # rows where main is present (in same group) -> use transactions index mapping
            # find transactions where both present
            mask = []
            for crops in transactions["Crop"]:
                mask.append((main_crop in crops) and (companion_crop in crops))
            df_both_idx = transactions[pd.Series(mask)].index

            # transactions where main is present without companion
            mask_main_only = []
            for crops in transactions["Crop"]:
                mask_main_only.append((main_crop in crops) and (companion_crop not in crops))
            df_main_only_idx = transactions[pd.Series(mask_main_only)].index

            # map back to original df rows filtered by group_cols
            if len(df_both_idx) == 0:
                return None  # not available

            # build filters for original rows matching the selected transactions
            # get the group tuples for both and main_only
            both_groups = transactions.loc[df_both_idx, group_cols].to_dict(orient="records")
            main_only_groups = transactions.loc[df_main_only_idx, group_cols].to_dict(orient="records")

            def rows_for_groups(groups):
                frames = []
                for g in groups:
                    q = pd.Series([True] * len(df))
                    for k, v in g.items():
                        q &= (df[k] == v)
                    frames.append(df[q])
                if frames:
                    return pd.concat(frames, ignore_index=True)
                else:
                    return pd.DataFrame(columns=df.columns)

            df_both_rows = rows_for_groups(both_groups)
            df_main_only_rows = rows_for_groups(main_only_groups)

            # optionally restrict by state/season
            if state_filter and state_filter != "All" and "State" in df.columns:
                df_both_rows = df_both_rows[df_both_rows["State"] == state_filter]
                df_main_only_rows = df_main_only_rows[df_main_only_rows["State"] == state_filter]
            if season_filter and season_filter != "All" and "Season" in df.columns:
                df_both_rows = df_both_rows[df_both_rows["Season"] == season_filter]
                df_main_only_rows = df_main_only_rows[df_main_only_rows["Season"] == season_filter]

            # compute mean productivity for main crop rows when both present vs main only
            # productivity = production/area for rows where Crop == main_crop
            both_main_rows = df_both_rows[df_both_rows["Crop"] == main_crop]
            main_only_rows = df_main_only_rows[df_main_only_rows["Crop"] == main_crop]

            if both_main_rows.empty or main_only_rows.empty:
                return None

            both_prod = (both_main_rows["Production"] / both_main_rows["Area"]).mean()
            main_only_prod = (main_only_rows["Production"] / main_only_rows["Area"]).mean()

            uplift_pct = ((both_prod - main_only_prod) / main_only_prod) * 100 if main_only_prod != 0 else None

            return {
                "both_prod": both_prod,
                "main_only_prod": main_only_prod,
                "uplift_pct": uplift_pct,
                "both_count": len(both_main_rows),
                "main_only_count": len(main_only_rows)
            }

        # show estimates for top data-driven companions and rule-suggestions
        companions_to_check = list(dict.fromkeys(rule_suggestions + top_companions))[:6]  # unique preserve order limit 6
        results = []
        for comp in companions_to_check:
            uplift = calc_uplift(sel_crop, comp, state_filter=sel_state, season_filter=sel_season)
            results.append({"companion": comp, "uplift": uplift})

        # display table
        rows_display = []
        for r in results:
            comp = r["companion"]
            info = r["uplift"]
            if info is None:
                rows_display.append({"Companion": comp, "Historical uplift (%)": "N/A", "Main-prod": "N/A", "Both-prod": "N/A", "Notes": "Insufficient historical co-occurrence data"})
            else:
                rows_display.append({
                    "Companion": comp,
                    "Historical uplift (%)": f"{info['uplift_pct']:.2f}" if info['uplift_pct'] is not None else "N/A",
                    "Main-prod": f"{info['main_only_prod']:.3f}",
                    "Both-prod": f"{info['both_prod']:.3f}",
                    "Notes": f"Based on {info['both_count']} combined rows"
                })
        st.table(pd.DataFrame(rows_display))

        st.markdown("""
        **Notes & guidance:**  
        - Rule-based suggestions come from agronomy heuristics and are a safe starting point.  
        - Data-driven suggestions show crops historically grown together in the same State/Season/Year groups in your dataset — this indicates co-cultivation, not necessarily causation.  
        - Historical uplift is only an estimate; conduct small-scale trials before scaling intercropping changes.
        """)

# -----------------------------------------------------------
# END
# -----------------------------------------------------------
st.markdown("---")
st.caption(f"Dataset path used: {FILE_PATH}")
