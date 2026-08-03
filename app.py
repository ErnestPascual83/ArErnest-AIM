"""
Tropang Foodie — Restaurant Popularity & Demand Explorer
A Streamlit app that showcases:
  1. An EDA dashboard over the cleaned restaurant dataset
  2. A live classifier: will this restaurant be a "High" popularity performer?
  3. A live regressor: predicted annual demand score (0-100)

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path("data/restaurants_clean.csv")
MODELS_DIR = Path("models")

st.set_page_config(
    page_title="Tropang Foodie | Popularity & Demand Explorer",
    page_icon="🍽️",
    layout="wide",
)


# ---------- Cached loaders ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    clf = joblib.load(MODELS_DIR / "classifier_pipeline.pkl")
    reg = joblib.load(MODELS_DIR / "regressor_pipeline.pkl")
    with open(MODELS_DIR / "ui_options.json") as f:
        options = json.load(f)
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return clf, reg, options, metrics


df = load_data()
clf_pipe, reg_pipe, ui_options, metrics = load_models()

# ---------- Header ----------
st.title("🍽️ Tropang Foodie — Popularity & Demand Explorer")
st.caption(
    "Explore BGC/Makati-area restaurant data and predict popularity + demand "
    "for a hypothetical new listing."
)

tab_explore, tab_predict, tab_about = st.tabs(
    ["📊 Explore the Data", "🔮 Predict a Restaurant", "ℹ️ About This Model"]
)

# ============================================================
# TAB 1 — EDA
# ============================================================
with tab_explore:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restaurants", len(df))
    col2.metric("Avg. Rating", f"{df['weighted_public_rating'].mean():.2f} ⭐")
    col3.metric("High Popularity", int(df["is_high_popularity"].sum()))
    col4.metric("Cuisines Tracked", df["cuisine_type"].nunique())

    st.divider()

    left, right = st.columns(2)
    with left:
        district_filter = st.multiselect(
            "Filter by district", sorted(df["district"].unique()), default=None
        )
    with right:
        cuisine_filter = st.multiselect(
            "Filter by cuisine", sorted(df["cuisine_type"].unique()), default=None
        )

    filtered = df.copy()
    if district_filter:
        filtered = filtered[filtered["district"].isin(district_filter)]
    if cuisine_filter:
        filtered = filtered[filtered["cuisine_type"].isin(cuisine_filter)]

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.box(
            filtered,
            x="popularity_class",
            y="annual_demand_proxy_score_0_100",
            color="popularity_class",
            category_orders={"popularity_class": ["Low", "Medium", "High"]},
            title="Demand Score by Popularity Class",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            filtered,
            x="total_public_reviews",
            y="weighted_public_rating",
            color="popularity_class",
            size="annual_demand_proxy_score_0_100",
            hover_name="restaurant_name",
            title="Reviews vs. Rating (bubble size = demand score)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        filtered["district"].value_counts().reset_index(),
        x="district",
        y="count",
        title="Restaurant Count by District",
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("View filtered raw data"):
        st.dataframe(filtered, use_container_width=True)

# ============================================================
# TAB 2 — Predictions
# ============================================================
with tab_predict:
    st.subheader("Describe a restaurant")
    st.caption(
        "Fill in the details below to get a predicted popularity tier and demand score."
    )

    p1, p2 = st.columns(2)
    with p1:
        cuisine = st.selectbox("Cuisine group", ui_options["cuisine_group"])
        district = st.selectbox("District", ui_options["district"])
    with p2:
        rating = st.slider("Weighted public rating", 3.0, 5.0, 4.5, 0.1)
        reviews = st.number_input("Total public reviews", min_value=0, value=500, step=10)

    input_row = pd.DataFrame(
        [
            {
                "cuisine_group": cuisine,
                "district": district,
                "weighted_public_rating": rating,
                "total_public_reviews": reviews,
            }
        ]
    )

    if st.button("Predict", type="primary"):
        pred_class = clf_pipe.predict(input_row)[0]
        pred_proba = clf_pipe.predict_proba(input_row)[0][1]
        pred_demand = reg_pipe.predict(input_row)[0]

        r1, r2 = st.columns(2)
        with r1:
            label = "🔥 High Popularity" if pred_class == 1 else "🙂 Not (Yet) High Popularity"
            st.metric("Predicted Popularity Tier", label)
            st.progress(float(pred_proba))
            st.caption(f"Model confidence it's High: {pred_proba:.0%}")
        with r2:
            st.metric("Predicted Annual Demand Score", f"{pred_demand:.1f} / 100")

# ============================================================
# TAB 3 — About / model card
# ============================================================
with tab_about:
    st.subheader("Model card")
    st.markdown(
        f"""
- **Data source:** {len(df)} restaurants scraped from public Google Places
  listings across BGC, Taguig, Pasig, and Makati.
- **Classifier:** Random Forest predicting `is_high_popularity`
  (High vs. Medium/Low). Trained on cuisine, district, rating, and review count.
  Test accuracy **{metrics['classifier']['accuracy']:.0%}**, F1 **{metrics['classifier']['f1']}**,
  ROC-AUC **{metrics['classifier']['roc_auc']}** (n={metrics['classifier']['n_test']} test rows).
- **Regressor:** Random Forest predicting `annual_demand_proxy_score_0_100`.
  Test MAE **{metrics['regressor']['mae']}** points, R² **{metrics['regressor']['r2']}**
  (n={metrics['regressor']['n_test']} test rows).
- **Known limitations:** the dataset is small (127 rows) and geographically
  narrow, and `popularity_class` looks like it was thresholded directly from
  the demand score, so the classifier's near-perfect score reflects a small,
  cleanly-separated dataset rather than a guarantee it'll generalize to new
  restaurants or neighborhoods. `price_level_1_4` was dropped as a feature
  because it was missing for ~81% of rows.
"""
    )
