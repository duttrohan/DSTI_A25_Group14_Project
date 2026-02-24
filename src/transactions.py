import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from config import DATA_INTERIM

def build_transactions(top_n_products: int = 500, use_price: bool = True):
    """
    Build list of products per order (transactions).
    Returns (transactions, full_df with price).
    """
    filename = "order_products_full_with_price.csv" if use_price else "order_products_full.csv"
    full_path = os.path.join(DATA_INTERIM, filename)
    full = pd.read_csv(full_path)

    # focus on top N products to keep basket matrix manageable
    top_products = (
        full["product_name"]
        .value_counts()
        .head(top_n_products)
        .index
    )

    filtered = full[full["product_name"].isin(top_products)]

    transactions = (
        filtered.groupby("order_id")["product_name"]
        .apply(list)
        .tolist()
    )

    print(f"[transactions] Built {len(transactions)} transactions, {len(top_products)} products.")
    return transactions, full


def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_).astype(bool)
    print(f"[transactions] Basket shape: {basket.shape}")
    return basket