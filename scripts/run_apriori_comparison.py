import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from apriori_comparison import run_apriori_comparison

if __name__ == "__main__":
    print("[run_apriori_comparison] Starting...")
    freq_ap, rules_ap, t_ap = run_apriori_comparison(min_support=0.01, min_conf=0.2)
    print("[run_apriori_comparison] Done.")