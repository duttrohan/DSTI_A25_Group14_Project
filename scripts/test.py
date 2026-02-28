import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from config import DATA_INTERIM

full = pd.read_csv(os.path.join(DATA_INTERIM, "order_products_full_with_price.csv"))

print("Top 50 most frequent products:")
print(full["product_name"].value_counts().head(50))

print("\nProducts containing 'chips':")
print(full[full["product_name"].str.contains("chips", case=False, na=False)][
    ["product_name", "aisle", "department"]
].drop_duplicates().head(20))

print("\nProducts containing 'soda':")
print(full[full["product_name"].str.contains("soda", case=False, na=False)][
    ["product_name", "aisle", "department"]
].drop_duplicates().head(20))

print("\nProducts containing 'yogurt':")
print(full[full["product_name"].str.contains("yogurt", case=False, na=False)][
    ["product_name", "aisle", "department"]
].drop_duplicates().head(20))

print("\nProducts containing 'milk':")
print(full[full["product_name"].str.contains("milk", case=False, na=False)][
    ["product_name", "aisle", "department"]
].drop_duplicates().head(20))