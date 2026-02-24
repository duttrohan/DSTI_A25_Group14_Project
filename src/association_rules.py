import os
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from config import DATA_PROCESSED
from transactions import build_transactions, encode_transactions


def is_similar_name(a: str, b: str) -> bool:
    """
    Heuristic to detect very similar product names.
    Used to remove rules like 'Banana -> Bag of Organic Bananas'.
    """
    a_low = str(a).lower()
    b_low = str(b).lower()
    return (a_low in b_low) or (b_low in a_low)


def mine_fp_growth_with_utility(
    top_n_products: int = 500,
    min_support: float = 0.002,   # more permissive than before
    min_conf: float = 0.1,        # more permissive
    min_lift: float = 1.0,        # more permissive
):
    """
    Run FP-Growth with utility (prices) and build:
      - all rules
      - business-ready 1->1 rules
      - top rules per item

    Returns:
      rules, business_rules, top_rules_per_item (DataFrames)
    """
    #  Build transactions and basket
    print(f"[rules] Building transactions (top_n_products={top_n_products})...")
    transactions, full = build_transactions(
        top_n_products=top_n_products,
        use_price=True
    )
    basket = encode_transactions(transactions)
    print(f"[rules] Basket shape: {basket.shape}")

    # Frequent itemsets with FP-Growth
    print(
        f"[rules] Running FP-Growth (min_support={min_support})..."
    )
    freq = fpgrowth(basket, min_support=min_support, use_colnames=True)
    print(f"[rules] Frequent itemsets found: {freq.shape[0]}")

    # Build product_name -> price map
    price_map = (
        full[["product_name", "price"]]
        .dropna()
        .drop_duplicates()
        .set_index("product_name")["price"]
    )

    def itemset_utility(items):
        return sum(price_map.get(item, 0.0) for item in items)

    # Utility and expected revenue for itemsets
    freq["itemset_utility"] = freq["itemsets"].apply(lambda s: itemset_utility(list(s)))
    freq["expected_revenue"] = freq["support"] * freq["itemset_utility"]

    # Association rules
    print("[rules] Generating association rules from frequent itemsets...")
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    print(f"[rules] Raw rules: {rules.shape[0]}")

    # Filter by support/conf/lift thresholds
    rules = rules[
        (rules["confidence"] >= min_conf) &
        (rules["lift"] >= min_lift)
    ].copy()
    print(f"[rules] After filtering by conf/lift: {rules.shape[0]}")

    # Nicely formatted columns
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)

    # Rule-level utility and expected revenue
    def union_utility(row):
        items = set(row["antecedents"]) | set(row["consequents"])
        return itemset_utility(items)

    rules["rule_utility"] = rules.apply(union_utility, axis=1)
    rules["expected_revenue"] = rules["support"] * rules["rule_utility"]

    # Saving all rules
    all_path = os.path.join(DATA_PROCESSED, "association_rules_fp_all.csv")
    rules.to_csv(all_path, index=False)
    print(f"[rules] All FP-Growth rules saved → {all_path}")

    # 1->1 business rules, remove "similar name" pairs
    business_rules = rules[
        (rules["antecedent_len"] == 1) &
        (rules["consequent_len"] == 1)
    ].copy()

    # Remove rules where antecedent and consequent names are too similar
    before_sim = business_rules.shape[0]
    business_rules = business_rules[
        ~business_rules.apply(
            lambda row: is_similar_name(
                row["antecedents_str"],
                row["consequents_str"],
            ),
            axis=1,
        )
    ]
    after_sim = business_rules.shape[0]
    print(f"[rules] 1->1 rules before similar-name filter: {before_sim}")
    print(f"[rules] 1->1 rules after similar-name filter:  {after_sim}")

    # Sort by expected_revenue, lift, confidence
    business_rules = business_rules.sort_values(
        by=["expected_revenue", "lift", "confidence"],
        ascending=False
    )

    business_path = os.path.join(DATA_PROCESSED, "business_ready_rules.csv")
    business_rules.to_csv(business_path, index=False)
    print(f"[rules] business_ready_rules saved → {business_path}")

    # Top 3 rules per antecedent
    top_rules_per_item = (
        business_rules
        .groupby("antecedents_str")
        .head(3)
        .reset_index(drop=True)
    )
    top_path = os.path.join(DATA_PROCESSED, "top_rules_per_item.csv")
    top_rules_per_item.to_csv(top_path, index=False)
    print(f"[rules] top_rules_per_item saved → {top_path}")

    return rules, business_rules, top_rules_per_item