import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("[run_clustering] PROJECT_ROOT:", PROJECT_ROOT)
print("[run_clustering] SRC_PATH:", SRC_PATH)
print("[run_clustering] sys.path head:", sys.path[:3])

from clustering import cluster_customers

if __name__ == "__main__":
    print("[run_clustering] Starting customer clustering...")
    features_clustered, model, scaler = cluster_customers(n_clusters=4, use_price=True)
    print("[run_clustering] Result shape:", features_clustered.shape)
    print("[run_clustering] Done.")