import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print("[run_enrichment] PROJECT_ROOT:", PROJECT_ROOT)
print("[run_enrichment] SRC_PATH:", SRC_PATH)
print("[run_enrichment] sys.path head:", sys.path[:3])

from data_loading import build_full_order_products
from pricing import add_prices_to_full_orders


if __name__ == "__main__":
    print("[run_enrichment] Starting enrichment pipeline...")
    full = build_full_order_products()
    print(f"[run_enrichment] order_products_full shape: {full.shape}")
    full_with_price = add_prices_to_full_orders()
    print(f"[run_enrichment] order_products_full_with_price shape: {full_with_price.shape}")
    print("[run_enrichment] Done.")