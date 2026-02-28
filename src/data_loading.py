import os
import pandas as pd
from config import DATA_RAW, DATA_INTERIM


def load_raw_data():
    """
    Load original Instacart CSVs from data/raw.
    Requires:
      - orders.csv
      - order_products__prior.csv
      - products.csv
      - aisles.csv
      - departments.csv
    """
    orders = pd.read_csv(os.path.join(DATA_RAW, "orders.csv"))
    order_products = pd.read_csv(os.path.join(DATA_RAW, "order_products__prior.csv"))
    products = pd.read_csv(os.path.join(DATA_RAW, "products.csv"))
    aisles = pd.read_csv(os.path.join(DATA_RAW, "aisles.csv"))
    departments = pd.read_csv(os.path.join(DATA_RAW, "departments.csv"))
    return orders, order_products, products, aisles, departments


def build_full_order_products():
    """
    Build main merged table:
      one row = one product in one order, with order + product + aisle + dept info.

    Output:
      data/interim/order_products_full.csv
    """
    print("[data_loading] Loading raw CSVs...")
    orders, order_products, products, aisles, departments = load_raw_data()

    print("[data_loading] Merging tables...")
    full = (
        order_products
        .merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(aisles, on="aisle_id", how="left")
        .merge(departments, on="department_id", how="left")
    )

    out_path = os.path.join(DATA_INTERIM, "order_products_full.csv")
    full.to_csv(out_path, index=False)
    print(f"[data_loading] Saved merged order_products_full → {out_path}")
    print("[data_loading] Shape:", full.shape)
    return full