import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("[run_eclat_demo] PROJECT_ROOT:", PROJECT_ROOT)
print("[run_eclat_demo] SRC_PATH:", SRC_PATH)
print("[run_eclat_demo] sys.path head:", sys.path[:3])

from eclat_demo import run_eclat_demo

if __name__ == "__main__":
    print("[run_eclat_demo] Starting Eclat demo...")
    df_itemsets = run_eclat_demo(
        top_n_products=100,
        min_support=0.01,   # can raise to 0.02 if too many itemsets
        sample_size=10000,  # sampled transactions
    )
    print("[run_eclat_demo] Result shape:", df_itemsets.shape)
    print("[run_eclat_demo] Done.")