
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Make src folder importable
sys.path.append("src")
from recommender import recommend_items

# Load data
@st.cache_data
def load_data():
    seg = pd.read_csv("data/processed/customer_segments.csv")
    rules = pd.read_csv("data/processed/business_ready_rules.csv")
    top_rules = pd.read_csv("data/processed/top_rules_per_item.csv")
    return seg, rules, top_rules

seg_df, rules_df, top_rules_df = load_data()

# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    layout="wide",
    page_icon="🛒"
)

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", 
     "👥 Customer Segmentation",
     "🔗 Association Rules",
     "🛒 Recommendation Engine"]
)

# ---------------------
# HOME PAGE
# ---------------------
if page == "🏠 Home":
    st.title("Retail Analytics & Recommendation System")
    st.markdown("### Built by DSTI Group 14 | MSc AI & Data Science")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", f"{seg_df['user_id'].nunique():,}")
    c2.metric("Total Segments", seg_df["cluster"].nunique())
    c3.metric("Total Rules", f"{rules_df.shape[0]:,}")

    st.subheader("📈 Customer Segment Distribution")
    fig = px.histogram(seg_df, x="cluster", title="Customers per Segment")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------
# CUSTOMER SEGMENTATION PAGE
# ---------------------
elif page == "👥 Customer Segmentation":
    st.title("Customer Segmentation Explorer")

    seg_choice = st.selectbox("Select Feature for X-axis", seg_df.columns)
    seg_choice2 = st.selectbox("Select Feature for Y-axis", seg_df.columns)

    fig = px.scatter(
        seg_df, 
        x=seg_choice, 
        y=seg_choice2, 
        color="cluster",
        title="Customer Feature Scatterplot"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(seg_df.head())

# ---------------------
# ASSOCIATION RULES PAGE
# ---------------------
elif page == "🔗 Association Rules":
    st.title("Association Rules Explorer")

    metric = st.selectbox("Sort rules by:", 
        ["expected_revenue", "lift", "confidence", "support"])

    filtered = rules_df.sort_values(metric, ascending=False).head(50)

    st.dataframe(filtered)

    fig = px.scatter(
        filtered,
        x="support",
        y="confidence",
        size="expected_revenue",
        color="lift",
        hover_data=["antecedents_str", "consequents_str"],
        title="Rule Strength Visualization"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------
# RECOMMENDATION ENGINE PAGE
# ---------------------
elif page == "🛒 Recommendation Engine":
    st.title("Product Recommendation Engine")

    user_input = st.text_input(
        "Enter cart items (comma-separated):",
        placeholder="banana, organic strawberries"
    )

    top_k = st.slider("Number of recommendations:", 1, 10, 5)

    if st.button("Recommend"):
        cart = [x.strip() for x in user_input.split(",")]

        recs = recommend_items(
            cart_items=cart,
            top_k=top_k,
            min_lift=1.0,
            min_conf=0.1,
            avoid_similar=True
        )

        st.subheader("Recommended Products")
        if recs:
            for item in recs:
                st.success(f"🟢 {item}")
        else:
            st.warning("No recommendations found. Try different items.")
