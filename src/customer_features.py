import os
import pandas as pd
from config import DATA_INTERIM

def build_customer_features(use_price: bool = True):

    filename = "order_products_full_with_price.csv" if use_price else "order_products_full.csv"
    full_path = os.path.join(DATA_INTERIM, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{filename} not found in {DATA_INTERIM}")

    full = pd.read_csv(full_path)

    # Features per user
    total_orders = full.groupby("user_id")["order_id"].nunique()
    total_products = full.groupby("user_id")["product_id"].count()
    avg_basket_size = total_products / total_orders

    order_gap_std = (
        full.groupby("user_id")["days_since_prior_order"]
        .std()
        .fillna(0)
    )

    unique_products = full.groupby("user_id")["product_id"].nunique()

    data = {
        "total_orders": total_orders,
        "total_products": total_products,
        "avg_basket_size": avg_basket_size,
        "order_gap_std": order_gap_std,
        "unique_products": unique_products,
    }

    if use_price and "price" in full.columns:
        total_spent = full.groupby("user_id")["price"].sum()
        data["total_spent"] = total_spent

    features = pd.DataFrame(data)
    return features