import os
import pandas as pd
from config import DATA_RAW, DATA_INTERIM


def attach_prices():
    """
    Attach synthetic prices from products_with_prices_synthetic.csv
    to the merged table order_products_full.csv.

    Outputs:
      data/interim/order_products_full_with_price.csv
    """
    full_path = os.path.join(DATA_INTERIM, "order_products_full.csv")
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"{full_path} not found. Run run_enrichment.py (build_full_order_products) first."
        )

    products_price_path = os.path.join(DATA_RAW, "products_with_prices_synthetic.csv")
    if not os.path.exists(products_price_path):
        raise FileNotFoundError(
            f"{products_price_path} not found. Run generate_synthetic_prices.py first."
        )

    print("[pricing] Loading merged orders...")
    full = pd.read_csv(full_path)

    print("[pricing] Loading synthetic prices...")
    prod_price = pd.read_csv(products_price_path)[["product_id", "price"]]

    print("[pricing] Merging prices into full table...")
    full_price = full.merge(prod_price, on="product_id", how="left")

    out_path = os.path.join(DATA_INTERIM, "order_products_full_with_price.csv")
    full_price.to_csv(out_path, index=False)
    print(f"[pricing] Saved order_products_full_with_price → {out_path}")
    print("[pricing] Shape:", full_price.shape)
    return full_price