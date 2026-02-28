import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from config import DATA_INTERIM


def build_transactions(top_n_products: int = 500):
    """
    Build transactions as list of product_name per order, restricted
    to the top_n_products most frequent products.

    Returns:
      transactions: list of lists
      full_price: full merged DataFrame with price
    """
    full_path = os.path.join(DATA_INTERIM, "order_products_full_with_price.csv")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"{full_path} not found. Run run_enrichment.py first."
        )

    full = pd.read_csv(full_path)

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
    """
    One-hot encode list-of-lists transactions into a basket DataFrame.
    """
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_).astype(bool)
    print(f"[transactions] Basket shape: {basket.shape}")
    return basket