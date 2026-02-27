
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os


instacart_css = """
<style>
/* Full-width layout */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    max-width: 95% !important;
}

/* Reduce space between blocks */
div[data-testid="stVerticalBlock"] > div {
    background: white;
    padding: 12px 18px;
    border-radius: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 12px !important;
}

/* Reduce internal spacing */
.css-1lcbmhc, .css-18e3th9 {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}

/* Compress header spacing */
h1, h2, h3 {
    margin-top: 3px !important;
    margin-bottom: 6px !important;
    font-family: "Inter", sans-serif;
    color: #1B5E20 !important;
    font-weight: 700 !important;
}

/* Re-style buttons */
.stButton > button {
    background-color: #43A047 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 6px 16px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* Apply compact styling to dataframes */
.dataframe {
    font-size: 0.85rem;
}

/* Shrink slider area */
.stSlider > div {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}

/* Compact sidebar */
section[data-testid="stSidebar"] {
    padding-top: 0rem !important;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
    background-attachment: fixed;
    font-family: "Inter", sans-serif;
}
</style>
"""
st.markdown(instacart_css, unsafe_allow_html=True)


# Make src importable
sys.path.append("src")
from recommender import recommend_items

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Instacart - Retail Analytics & Recommendations",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_processed():
    seg = pd.read_csv("data/processed/customer_segments.csv")
    rules = pd.read_csv("data/processed/business_ready_rules.csv")
    top_rules = pd.read_csv("data/processed/top_rules_per_item.csv")
    return seg, rules, top_rules

seg_df, rules_df, top_rules_df = load_processed()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "👥 Customer Segmentation",
        "🔗 Product Associations",
        "💰 Revenue Simulation",
        "📉 Promotion Efficiency",
        "🛒 Recommendation Engine",
    ]
)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "🏠 Home":
    st.title("🛒 Instacart - Retail Analytics & Recommendation Dashboard")
    st.markdown("#### Built by DSTI Group 14 — MSc AI & Data Science")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{seg_df['user_id'].nunique():,}")
    c2.metric("Customer Segments", seg_df['cluster'].nunique())
    c3.metric("Association Rules", f"{rules_df.shape[0]:,}")

    st.write("### Dataset Overview")
    st.write("""
    This dashboard provides insights into customer behavior, product associations, 
    revenue impact, promotion strategy, and personalized product recommendations.
    """)

# -----------------------------------------------------------
# CUSTOMER SEGMENTATION
# -----------------------------------------------------------
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Analysis")

    st.subheader("Cluster Distribution")
    fig = px.histogram(seg_df, x="cluster", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Basket Size", round(seg_df["avg_basket_size"].mean(), 2))
    c2.metric("Avg Orders / Customer", round(seg_df["total_orders"].mean(), 2))
    c3.metric("Basket Variability (Std)", round(seg_df["order_gap_std"].mean(), 2))

    # Buyer type classification
    st.subheader("Buyer Type: Frequent vs Irregular")
    seg_df["buyer_type"] = seg_df["total_orders"].apply(
        lambda x: "Frequent Buyer" if x >= 50 else "Irregular Buyer"
    )
    fig = px.pie(seg_df, names="buyer_type", title="Buyer Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Comparison")
    x_feature = st.selectbox("X-axis feature", seg_df.columns)
    y_feature = st.selectbox("Y-axis feature", seg_df.columns)
    fig = px.scatter(seg_df, x=x_feature, y=y_feature, color="cluster")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# PRODUCT ASSOCIATIONS
# -----------------------------------------------------------
elif page == "🔗 Product Associations":
    st.title("🔗 Product Associations (Frequent Bundles & Co‑Purchases)")

    st.subheader("Top 20 Bundles by Expected Revenue")
    top_bundles = rules_df.sort_values("expected_revenue", ascending=False).head(20)

    st.dataframe(top_bundles[[
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift",
        "expected_revenue",
    ]])

    fig = px.bar(
        top_bundles,
        x="expected_revenue",
        y="antecedents_str",
        color="lift",
        orientation="h",
        title="Top Revenue‑Generating Bundles",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rule Strength Visualization")
    fig = px.scatter(
        rules_df.head(500),
        x="support",
        y="confidence",
        size="expected_revenue",
        color="lift",
        hover_data=["antecedents_str", "consequents_str"],
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "💰 Revenue Simulation":
    st.title("💰 Revenue Simulation Tool")

    st.write("""
    Evaluate how much **additional revenue** can be gained by applying any association rule
    across a selected number of monthly customers.
    """)

    # ----------------------------------------------------------
    # 1) Side-by-side product selectors
    # ----------------------------------------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Antecedent Product")
        antecedent_options = sorted(rules_df["antecedents_str"].unique())
        selected_antecedent = st.selectbox(
            "Select a product:",
            antecedent_options,
            key="rev_antecedent"
        )

    # Filter rules for selected antecedent
    filtered_rules = rules_df[rules_df["antecedents_str"] == selected_antecedent]

    with colB:
        st.subheader("Consequent (Recommended) Product")
        consequent_options = filtered_rules["consequents_str"].unique().tolist()
        if not consequent_options:
            st.warning("No rules found for this antecedent. Try another product.")
            st.stop()

        selected_consequent = st.selectbox(
            "Select recommended product:",
            consequent_options,
            key="rev_consequent"
        )

    # ----------------------------------------------------------
    # 2) Rule info + expected revenue
    # ----------------------------------------------------------
    selected_rule = filtered_rules[
        filtered_rules["consequents_str"] == selected_consequent
    ].iloc[0]

    expected_rev = float(selected_rule["expected_revenue"])

    st.markdown(f"### Selected Rule: **{selected_antecedent} → {selected_consequent}**")
    st.info(f"Expected revenue per occurrence (support × utility): **${expected_rev:.4f}**")

    # ----------------------------------------------------------
    # 3) Inputs (side-by-side): monthly orders + (optional) adoption rate
    # ----------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        monthly_orders = st.slider("Monthly Orders", 500, 50000, 5000, key="rev_orders")

    with c2:
        adoption_rate = st.slider(
            "Adoption / Trigger Rate (%)",
            min_value=1, max_value=100, value=100,
            help="Share of monthly orders where this rule is actually triggered/used.",
            key="rev_adoption"
        ) / 100.0

    # ----------------------------------------------------------
    # 4) Result (side-by-side KPIs)
    # ----------------------------------------------------------
    effective_orders = monthly_orders * adoption_rate
    estimated_monthly_gain = effective_orders * expected_rev

    k1, k2 = st.columns(2)
    k1.metric("Effective Orders", f"{int(effective_orders):,}")
    k2.metric("Estimated Monthly Revenue Impact", f"${estimated_monthly_gain:,.2f}")


# -----------------------------------------------------------
# PROMOTION EFFICIENCY
# -----------------------------------------------------------

elif page == "📉 Promotion Efficiency":
    st.title("📉 Promotion Efficiency (ROI Calculator)")

    st.write("""
    Compare the profitability of a **targeted discount** versus a **broad, untargeted discount**
    for any association rule from the model.
    """)

    # ----------------------------------------------------------
    # 1. Side-by-side product selectors
    # ----------------------------------------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Antecedent Product")
        antecedent_options = sorted(rules_df["antecedents_str"].unique())
        selected_antecedent = st.selectbox(
            "Select a product:",
            antecedent_options,
            key="promo_antecedent"
        )

    # Filter rules for chosen antecedent
    filtered_rules = rules_df[rules_df["antecedents_str"] == selected_antecedent]

    with colB:
        st.subheader("Consequent (Recommended) Product")
        consequent_options = filtered_rules["consequents_str"].unique().tolist()

        selected_consequent = st.selectbox(
            "Select recommended product:",
            consequent_options,
            key="promo_consequent"
        )

    # ----------------------------------------------------------
    # 2. Get selected rule information
    # ----------------------------------------------------------
    rule_row = filtered_rules[
        filtered_rules["consequents_str"] == selected_consequent
    ].iloc[0]

    base_price = rule_row["rule_utility"]

    st.markdown(
        f"### Selected Rule: **{selected_antecedent} → {selected_consequent}**"
    )
    st.info(
        f"Base price (utility value) for the recommended item: **${base_price:.2f}**"
    )

    # ----------------------------------------------------------
    # 3. Side-by-side discount + targeted customers
    # ----------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        discount = st.slider(
            "Discount (%)",
            min_value=1,
            max_value=50,
            value=10,
            key="promo_discount"
        ) / 100

    with col2:
        targeted_customers = st.slider(
            "Targeted Customers / Month",
            min_value=100,
            max_value=10000,
            value=2000,
            key="promo_targeted"
        )

    # ----------------------------------------------------------
    # 4. Untargeted customer input
    # ----------------------------------------------------------
    untargeted_customers = st.slider(
        "Untargeted Customer Base",
        min_value=1000,
        max_value=50000,
        value=20000,
        key="promo_untargeted"
    )

    # ----------------------------------------------------------
    # 5. ROI Calculations
    # ----------------------------------------------------------
    targeted_revenue = targeted_customers * base_price * (1 - discount)
    untargeted_revenue = untargeted_customers * base_price * (1 - discount)

    roi_targeted = targeted_revenue - (targeted_customers * base_price)
    roi_untargeted = untargeted_revenue - (untargeted_customers * base_price)

    # ----------------------------------------------------------
    # 6. ROI Output (side-by-side)
    # ----------------------------------------------------------
    col3, col4 = st.columns(2)

    col3.metric(
        "ROI: Targeted Promotion",
        f"${roi_targeted:,.0f}",
        help="Targeted discount applied only to customers triggering the rule."
    )

    col4.metric(
        "ROI: Untargeted Promotion",
        f"${roi_untargeted:,.0f}",
        help="Discount applied broadly to all customers."
    )

    st.info("Positive ROI = profitable promotion. Negative ROI = loss.")

# -----------------------------------------------------------
# RECOMMENDATION ENGINE
# -----------------------------------------------------------
elif page == "🛒 Recommendation Engine":
    st.title("🛒 Product Recommendation Engine")

    user_input = st.text_input(
        "Enter cart items (comma-separated):",
        placeholder="banana, organic strawberries"
    )

    top_k = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Recommend"):
        cart = [x.strip() for x in user_input.split(",") if x.strip()]

        if not cart:
            st.warning("Please enter at least one item.")
        else:
            recs = recommend_items(
                cart_items=cart,
                top_k=top_k,
                min_lift=1.0,
                min_conf=0.1,
                avoid_similar=True
            )

            st.subheader("Recommended Products")
            if recs:
                for r in recs:
                    st.success(f"🟢 {r}")
            else:
                st.warning("No recommendations found. Try different items.")
