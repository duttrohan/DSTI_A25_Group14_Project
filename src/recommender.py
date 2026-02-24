import os
import pandas as pd
from config import DATA_PROCESSED

_rules_cache = None


def load_business_rules():
    """
    Lazy-load business_ready_rules.csv once and cache it.
    """
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


def _too_similar(a: str, b: str) -> bool:
    """
    Simple heuristic: if one name is contained in the other (case-insensitive),
    treat them as too similar (e.g. 'Banana' vs 'Bag of Organic Bananas').
    """
    a_low = str(a).lower()
    b_low = str(b).lower()
    return (a_low in b_low) or (b_low in a_low)


def recommend_items(
    cart_items,
    top_k: int = 5,
    min_lift: float = 1.0,
    min_conf: float = 0.1,
    avoid_similar: bool = True,
):
    """
    cart_items: list of product names currently in the cart.
    top_k: number of recommendations to return.
    min_lift, min_conf: thresholds to filter rules used for recommendation.
    avoid_similar: if True, avoid consequents that look too similar to items in cart.

    Returns:
      list of recommended product names.
    """
    if not cart_items:
        return []

    rules = load_business_rules()

    # 1) Take 1->1 rules whose antecedent is in the cart
    candidate_rules = rules[
        rules["antecedents_str"].isin(cart_items)
    ]

    # 2) Filter by rule strength
    candidate_rules = candidate_rules[
        (candidate_rules["lift"] >= min_lift) &
        (candidate_rules["confidence"] >= min_conf)
    ]

    if candidate_rules.empty:
        return []

    # 3) Sort by expected_revenue, then lift, then confidence
    candidate_rules = candidate_rules.sort_values(
        by=["expected_revenue", "lift", "confidence"],
        ascending=False
    )

    # 4) Extract consequents and clean them
    recs = candidate_rules["consequents_str"].tolist()

    seen = set()
    unique_recs = []
    for r in recs:
        # Skip items already in the cart
        if r in cart_items:
            continue

        # Skip if too similar to any cart item
        if avoid_similar and any(_too_similar(r, c) for c in cart_items):
            continue

        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

        if len(unique_recs) >= top_k:
            break

    return unique_recs