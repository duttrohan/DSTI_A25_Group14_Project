import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import DATA_RAW, DATA_INTERIM


def load_products_with_dept():
    products_path = os.path.join(DATA_RAW, "products.csv")
    aisles_path = os.path.join(DATA_RAW, "aisles.csv")
    depts_path = os.path.join(DATA_RAW, "departments.csv")

    products = pd.read_csv(products_path)
    aisles = pd.read_csv(aisles_path)
    depts = pd.read_csv(depts_path)

    # Merge to get department names for each product
    prod_full = (
        products
        .merge(aisles, on="aisle_id", how="left")
        .merge(depts, on="department_id", how="left")
    )

    return prod_full


def estimate_popularity():
    """
    Estimate product popularity (how often each product appears in orders)
    from order_products__prior.csv.
    """
    op_path = os.path.join(DATA_RAW, "order_products__prior.csv")
    op = pd.read_csv(op_path)

    freq = op["product_id"].value_counts()
    freq.name = "frequency"
    return freq


def assign_price_ranges(department_name: str) -> tuple[float, float]:
    """
    Assign a base (min_price, max_price) range depending on department.
    You can adjust these ranges to your liking.
    """
    d = (department_name or "").lower()

    # Basic heuristic ranges (in euros)
    if "produce" in d:
        return 1.0, 4.0      # fruits & veggies
    if "dairy" in d:
        return 1.0, 4.0
    if "meat" in d or "seafood" in d:
        return 4.0, 15.0
    if "bakery" in d:
        return 1.5, 6.0
    if "beverages" in d:
        return 1.0, 6.0
    if "snacks" in d:
        return 1.0, 5.0
    if "canned" in d or "dry goods" in d or "pantry" in d:
        return 1.0, 4.0
    if "frozen" in d:
        return 2.0, 8.0
    if "personal care" in d or "household" in d:
        return 2.0, 10.0

    # Default range if we don't recognize the department
    return 1.0, 5.0


def generate_synthetic_prices():
    """
    Generate a synthetic price (in euros) for each product_id based on:
      - department (base price range)
      - popularity (slight adjustment)
      - random noise within the range
    """
    print("[prices] Loading products and departments...")
    prod_full = load_products_with_dept()

    print("[prices] Estimating product popularity...")
    freq = estimate_popularity()    # index=product_id

    prod = prod_full.merge(
        freq, left_on="product_id", right_index=True, how="left"
    )
    prod["frequency"] = prod["frequency"].fillna(0)

    # Normalize frequency to [0, 1] (for slight adjustments)
    max_freq = prod["frequency"].max() or 1.0
    prod["freq_norm"] = prod["frequency"] / max_freq

    rng = np.random.default_rng(seed=42)

    prices = []
    for _, row in prod.iterrows():
        dept = row.get("department", "")
        min_price, max_price = assign_price_ranges(dept)

        # Random base price in range
        base_price = rng.uniform(min_price, max_price)

        # Slight discount for very popular items (up to -20%)
        discount_factor = 1.0 - 0.2 * row["freq_norm"]
        price = base_price * discount_factor

        # Round to 2 decimals
        price = round(float(price), 2)

        prices.append(price)

    prod["price"] = prices

    # Keep only the relevant columns for downstream use
    out_cols = ["product_id", "product_name", "aisle_id", "department_id", "price"]
    out = prod[out_cols].copy()

    out_path = os.path.join(DATA_RAW, "products_with_prices_synthetic.csv")
    out.to_csv(out_path, index=False)
    print(f"[prices] Synthetic prices saved → {out_path}")
    print("[prices] Example rows:")
    print(out.head())

    return out


if __name__ == "__main__":
    generate_synthetic_prices()