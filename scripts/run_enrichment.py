import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("[run_enrichment] PROJECT_ROOT:", PROJECT_ROOT)
print("[run_enrichment] SRC_PATH:", SRC_PATH)

from data_loading import build_full_order_products
from pricing import attach_prices


if __name__ == "__main__":
    print("[run_enrichment] Starting enrichment (merge + synthetic prices)...")
    full = build_full_order_products()
    full_price = attach_prices()
    print("[run_enrichment] order_products_full shape:", full.shape)
    print("[run_enrichment] order_products_full_with_price shape:", full_price.shape)
    print("[run_enrichment] Done.")