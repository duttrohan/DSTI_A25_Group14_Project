import os
import pandas as pd
from config import DATA_RAW, DATA_INTERIM

def load_product_prices():

    path = os.path.join(DATA_RAW, "products_with_prices.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"products_with_prices.csv not found in {DATA_RAW}")
    df = pd.read_csv(path)
    if "product_id" not in df.columns or "price" not in df.columns:
        raise ValueError("products_with_prices.csv must have 'product_id' and 'price' columns")
    return df[["product_id", "price"]]


def add_prices_to_full_orders():
    """
    Merge prices into the full order-product table.
    """
    full_path = os.path.join(DATA_INTERIM, "order_products_full.csv")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            "order_products_full.csv not found; run build_full_order_products() first."
        )

    full = pd.read_csv(full_path)
    prices = load_product_prices()

    full_price = full.merge(prices, on="product_id", how="left")

    out_path = os.path.join(DATA_INTERIM, "order_products_full_with_price.csv")
    full_price.to_csv(out_path, index=False)
    print(f"[pricing] order_products_full_with_price saved → {out_path}")
    return full_price