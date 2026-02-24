import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("[run_association_rules] PROJECT_ROOT:", PROJECT_ROOT)
print("[run_association_rules] SRC_PATH:", SRC_PATH)
print("[run_association_rules] sys.path head:", sys.path[:3])

from association_rules import mine_fp_growth_with_utility


if __name__ == "__main__":
    print("[run_association_rules] Starting FP-Growth with utility...")
    rules, business_rules, top_rules_per_item = mine_fp_growth_with_utility(
        top_n_products=500,   # adjust if needed
        min_support=0.002,
        min_conf=0.1,
        min_lift=1.0,
    )
    print("[run_association_rules] Total rules:", rules.shape)
    print("[run_association_rules] Business rules:", business_rules.shape)
    print("[run_association_rules] Done.")