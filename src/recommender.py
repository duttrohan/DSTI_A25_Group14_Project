
import os
import pandas as pd
from config import DATA_INTERIM, DATA_PROCESSED
from difflib import get_close_matches
from config import DATA_PROCESSED

_business_rules_cache = None
_product_meta_cache = None


# -------------------------------------------------------------
# LOAD RULES (cached)
# -------------------------------------------------------------
def load_business_rules():
    """
    Load 1->1 cleaned rules (business_ready_rules.csv).
    """
    global _business_rules_cache
    if _business_rules_cache is not None:
        return _business_rules_cache

    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    path = os.path.join(DATA_PROCESSED, "business_ready_rules.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_association_rules.py first."
        )

    rules = pd.read_csv(path)
    _business_rules_cache = rules
    return rules


def load_product_meta():
    """
    Build product -> {department, aisle, popularity} mapping
    from order_products_full_with_price.csv.
    """
    global _product_meta_cache
    if _product_meta_cache is not None:
        return _product_meta_cache

    full_path = os.path.join(DATA_INTERIM, "order_products_full_with_price.csv")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"{full_path} not found. Run scripts/run_enrichment.py first."
        )

    df = pd.read_csv(full_path)

    # popularity (how often each product is bought)
    pop = df["product_name"].value_counts()
    pop.name = "popularity"

    meta = (
        df[["product_name", "department", "aisle"]]
        .drop_duplicates()
        .set_index("product_name")
    )
    meta = meta.join(pop, how="left").fillna({"popularity": 0})

    _product_meta_cache = meta
    return meta


def _too_similar(a: str, b: str) -> bool:
    """
    Simple heuristic: treat names as similar if one contains the other.
    E.g. 'Banana' vs 'Bag of Organic Bananas'.
    """
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
    Final recommender strategy:

    1) For each item in cart, try 1->1 rules where antecedent == that item
       (from business_ready_rules.csv).
       - Filter by lift and confidence.
       - Sort by expected_revenue, then lift, then confidence.
       - Collect consequents not in cart (and not too similar).

       If we find at least one such rule for any cart item, we:
         - Use these rule-based recommendations (possibly padded with popular items).

    2) If there are NO matching rules for ANY cart item:
       - Ignore rules completely.
       - Recommend popular products from the SAME department(s) as the items
         in the cart (based on order_products_full_with_price.csv).

    3) If still not enough, fall back to globally popular products.

    This way:
      - Items with strong rules (e.g. Bananas, Organic Strawberries)
        get rule-based recommendations.
      - Items with no rules (e.g. some specific chips or niche products)
        get department-consistent popularity-based suggestions instead of
        random organic produce.

    Robust recommendation engine with:
    - fuzzy matching
    - substring matching
    - fallback rules
    - rule ranking: expected_revenue > lift > confidence
    """

    if not cart_items:
        return []

    rules = load_business_rules()
    meta = load_product_meta()

    cart_set = set(cart_items)
    recs = []
    seen = set()

    # --- 1) Rule-based recommendations, per cart item ---
    found_any_rule = False

    for cart_item in cart_items:
        # Rules with this item as antecedent
        r = rules[
            (rules["antecedents_str"] == cart_item) &
            (rules["lift"] >= min_lift) &
            (rules["confidence"] >= min_conf)
        ].copy()

        if r.empty:
            continue

        found_any_rule = True

        r = r.sort_values(
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

        for _, row in r.iterrows():
            c = row["consequents_str"]
            if c in cart_set:
                continue
            if avoid_similar and any(_too_similar(c, ci) for ci in cart_items):
                continue
            if c not in seen:
                seen.add(c)
                recs.append(c)
            if len(recs) >= top_k:
                break
        if len(recs) >= top_k:
            break

    # If we found at least one rule-based recommendation, return or pad with popularity
    if found_any_rule:
        if len(recs) < top_k:
            pop = meta.sort_values("popularity", ascending=False).index.tolist()
            for p in pop:
                if p in cart_set or p in seen:
                    continue
                if avoid_similar and any(_too_similar(p, ci) for ci in cart_items):
                    continue
                seen.add(p)
                recs.append(p)
                if len(recs) >= top_k:
                    break
        return recs[:top_k]

    # --- 2) No rules at all: department-based popularity fallback ---

    # Determine departments present in cart
    depts = set()
    for ci in cart_items:
        try:
            depts.add(str(meta.loc[ci]["department"]))
        except KeyError:
            continue

    pop_df = meta.sort_values("popularity", ascending=False)

    for p, row in pop_df.iterrows():
        if p in cart_set or p in seen:
            continue
        dept_p = str(row.get("department", ""))
        if depts and dept_p not in depts:
            continue  # restrict to departments found in cart

        if avoid_similar and any(_too_similar(p, ci) for ci in cart_items):
            continue

        seen.add(p)
        recs.append(p)
        if len(recs) >= top_k:
            break

    # --- 3) Final fallback: global popularity if still not enough ---
    if len(recs) < top_k:
        pop_all = meta.sort_values("popularity", ascending=False).index.tolist()
        for p in pop_all:
            if p in cart_set or p in seen:
                continue
            if avoid_similar and any(_too_similar(p, ci) for ci in cart_items):
                continue
            seen.add(p)
            recs.append(p)
            if len(recs) >= top_k:
                break

    return recs[:top_k]

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
