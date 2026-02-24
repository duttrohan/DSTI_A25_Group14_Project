import os
import time
from mlxtend.frequent_patterns import apriori, association_rules
from config import DATA_PROCESSED
from transactions import build_transactions, encode_transactions

def run_apriori_comparison(min_support=0.01, min_conf=0.2):
    """
    Run Apriori on a sampled basket to compare with FP-Growth.
    """
    # Build baskets
    transactions, _ = build_transactions(top_n_products=200, use_price=True)
    basket = encode_transactions(transactions)

    # Optionally sample rows to keep Apriori fast
    basket_sample = basket.sample(n=min(5000, basket.shape[0]), random_state=42)

    start = time.time()
    freq_ap = apriori(
        basket_sample,
        min_support=min_support,
        use_colnames=True,
        low_memory=True
    )
    apriori_time = time.time() - start

    rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_conf)

    # Save for report
    out_path = os.path.join(DATA_PROCESSED, "apriori_rules_sample.csv")
    rules_ap.to_csv(out_path, index=False)

    print(f"[apriori] Frequent itemsets: {freq_ap.shape[0]}")
    print(f"[apriori] Rules: {rules_ap.shape[0]}")
    print(f"[apriori] Runtime (sample): {apriori_time:.2f} seconds")
    print(f"[apriori] Rules saved → {out_path}")

    return freq_ap, rules_ap, apriori_time