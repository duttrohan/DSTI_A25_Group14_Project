import os
import pandas as pd
from config import DATA_PROCESSED
from transactions import build_transactions, encode_transactions
from eclat_simple import eclat_from_basket


def run_eclat_demo(
    top_n_products: int = 100,
    min_support: float = 0.01,
    sample_size: int = 10000,
):
    """
    Run a simple Eclat implementation on a smaller sample of the basket.
    """
    print(f"[eclat_demo] Building transactions with top {top_n_products} products...")
    transactions, _ = build_transactions(
        top_n_products=top_n_products,
        use_price=True
    )
    print(f"[eclat_demo] Total transactions (before sampling): {len(transactions)}")

    # Sample for speed
    if len(transactions) > sample_size:
        import random
        random.seed(42)
        transactions_sample = random.sample(transactions, sample_size)
    else:
        transactions_sample = transactions

    print(f"[eclat_demo] Using {len(transactions_sample)} transactions for Eclat demo")

    # Build one-hot basket from sample
    basket = encode_transactions(transactions_sample)
    print(f"[eclat_demo] Basket shape (sample): {basket.shape}")

    # Run Eclat
    print(f"[eclat_demo] Running simple Eclat (min_support={min_support})...")
    itemsets = eclat_from_basket(basket, min_support=min_support)
    print(f"[eclat_demo] Found {len(itemsets)} frequent itemsets")

    # Convert to DataFrame
    rows = []
    for itemset, support, count in itemsets:
        rows.append(
            {
                "itemset": ", ".join(sorted(itemset)),
                "support": support,
                "support_count": count,
            }
        )
    df_itemsets = pd.DataFrame(rows).sort_values(by="support", ascending=False)

    out_path = os.path.join(DATA_PROCESSED, "eclat_itemsets_demo.csv")
    df_itemsets.to_csv(out_path, index=False)
    print(f"[eclat_demo] Eclat itemsets saved → {out_path}")
    print(f"[eclat_demo] Total itemsets: {df_itemsets.shape[0]}")

    return df_itemsets