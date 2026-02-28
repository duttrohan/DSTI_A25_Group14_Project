import os
import sys

# Make src importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from recommender import recommend_items

if __name__ == "__main__":
    print("[test_recommender] Testing recommender...")

    test_cases = [
        ["Banana"],
        ["Organic Strawberries"],
        ["Banana", "Organic Strawberries"],
        ["Plain Greek Yogurt"],      # from 'yogurt' list
        ["Organic Whole Milk"],      # from 'milk' list
        ["Original Potato Chips"],   # from 'chips' list
        ["Soda"],                    # from 'soda' list
    ]

    for cart in test_cases:
        recs = recommend_items(
            cart_items=cart,
            top_k=5,
            min_lift=1.0,
            min_conf=0.1,
            avoid_similar=True,
        )
        print("\nCart:", cart)
        print("Recommendations:", recs)