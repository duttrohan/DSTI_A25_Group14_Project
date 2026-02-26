
import os
import pandas as pd
from difflib import get_close_matches
from config import DATA_PROCESSED

_rules_cache = None


# -------------------------------------------------------------
# LOAD RULES (cached)
# -------------------------------------------------------------
def load_business_rules():
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    path = os.path.join(DATA_PROCESSED, "business_ready_rules.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"business_ready_rules.csv not found in {DATA_PROCESSED}; "
            "run scripts/run_association_rules.py first."
        )

    rules = pd.read_csv(path)
    _rules_cache = rules
    return rules


# -------------------------------------------------------------
# UTILITY: fuzzy match item name
# -------------------------------------------------------------
def _fuzzy_match_item(user_item: str, all_items):
    """
    Return the closest matching product name (case-insensitive),
    or None if nothing close enough.
    """
    lowered = [x.lower() for x in all_items]
    matches = get_close_matches(user_item.lower(), lowered, n=1, cutoff=0.6)
    if matches:
        idx = lowered.index(matches[0])
        return all_items[idx]
    return None


# -------------------------------------------------------------
# UTILITY: simple similarity filter
# -------------------------------------------------------------
def _too_similar(a: str, b: str) -> bool:
    a_low = str(a).lower()
    b_low = str(b).lower()
    return (a_low in b_low) or (b_low in a_low)


# -------------------------------------------------------------
# MAIN RECOMMENDER (IMPROVED)
# -------------------------------------------------------------
def recommend_items(
    cart_items,
    top_k: int = 5,
    min_lift: float = 1.0,
    min_conf: float = 0.1,
    avoid_similar: bool = True,
):
    """
    Robust recommendation engine with:
    - fuzzy matching
    - substring matching
    - fallback rules
    - rule ranking: expected_revenue > lift > confidence
    """

    if not cart_items:
        return []

    rules = load_business_rules()

    # Build a list of all product names appearing anywhere in rules
    all_products = list(
        set(rules["antecedents_str"].tolist()) |
        set(rules["consequents_str"].tolist())
    )

    # ---------------------------------------------------------
    # 1) Normalize cart items (fuzzy matching)
    # ---------------------------------------------------------
    normalized_cart = []
    for it in cart_items:
        fm = _fuzzy_match_item(it, all_products)
        normalized_cart.append(fm if fm else it)

    # ---------------------------------------------------------
    # 2) Match rules by:
    #    (A) exact antecedent match
    #    (B) substring match
    # ---------------------------------------------------------
    candidate_rules = pd.DataFrame()

    for item in normalized_cart:
        # Exact match
        exact = rules[rules["antecedents_str"].str.lower() == item.lower()]

        # Substring fuzzy match (ex: "strawberries" -> "Organic Strawberries")
        contains = rules[rules["antecedents_str"].str.contains(item, case=False, na=False)]

        candidate_rules = pd.concat([candidate_rules, exact, contains], ignore_index=True)

    candidate_rules = candidate_rules.drop_duplicates()

    # ---------------------------------------------------------
    # 3) Apply rule strength filters
    # ---------------------------------------------------------
    candidate_rules = candidate_rules[
        (candidate_rules["lift"] >= min_lift) &
        (candidate_rules["confidence"] >= min_conf)
    ]

    # ---------------------------------------------------------
    # 4) If candidates exist, rank and return top-k
    # ---------------------------------------------------------
    if not candidate_rules.empty:

        # Sort by revenue, then lift, then confidence
        candidate_rules = candidate_rules.sort_values(
            by=["expected_revenue", "lift", "confidence"],
            ascending=False
        )

        recs = []
        for _, row in candidate_rules.iterrows():
            c = row["consequents_str"]

            # Skip if same as cart or too similar
            if c in normalized_cart:
                continue
            if avoid_similar and any(_too_similar(c, x) for x in normalized_cart):
                continue

            # Unique & top_k limit
            if c not in recs:
                recs.append(c)
            if len(recs) >= top_k:
                break

        if recs:
            return recs

    # ---------------------------------------------------------
    # 5) FALLBACK: If no rules matched
    # ---------------------------------------------------------
    # Recommend top revenue consequences globally
    fallback_rules = rules.sort_values(
        by=["expected_revenue", "lift", "confidence"],
        ascending=False
    )

    fallback_recs = []
    for _, row in fallback_rules.iterrows():
        c = row["consequents_str"]
        if c not in normalized_cart and c not in fallback_recs:
            fallback_recs.append(c)
        if len(fallback_recs) >= top_k:
            break

    return fallback_recs
