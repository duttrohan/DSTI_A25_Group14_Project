
# DSTI_A25_Group14_Project – Retail Analytics & Recommendation

This repository contains the work of **DSTI Group 14** for the Retail Analytics
and Recommendation project. The goal is to analyse grocery transaction data,
build customer segments, mine association rules (including **utility‑aware**
patterns using prices), and expose a **recommendation interface** that the web
application can use.

The project uses the Instacart dataset (extended with product prices) plus
Python‑based data science and a simple web API.

---

## 1. Repository Structure

```text
Retail_Project/
├─ streamlit_app.py
├─ data/
│  ├─ raw/        # Original Instacart CSVs + products_with_prices.csv (NOT in Git)
│  ├─ interim/    # Merged / enriched data (generated)
│  └─ processed/  # Final ML outputs: segments, rules, etc. (generated)
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py               # Paths configuration
│  ├─ data_loading.py         # Load & merge Instacart tables
│  ├─ pricing.py              # Attach prices to products/orders
│  ├─ customer_features.py    # Build customer‑level features
│  ├─ clustering.py           # Customer segmentation (MiniBatchKMeans)
│  ├─ transactions.py         # Build transactions and basket (one‑hot) data
│  ├─ association_rules.py    # FP‑Growth + utility, rule generation
│  ├─ apriori_comparison.py   # Apriori on a sample (comparison only)
│  ├─ eclat_simple.py         # Simple Eclat (TID‑list) implementation
│  ├─ eclat_demo.py           # Eclat demo on a sample
│  ├─ recommender.py          # `recommend_items(...)` function for web app
│  └─ ...
│
├─ scripts/
│  ├─ run_enrichment.py       # Build merged data + price enrichment
│  ├─ run_clustering.py       # Run customer segmentation
│  ├─ run_association_rules.py# Run FP‑Growth + utility, export rules
│  ├─ run_apriori_comparison.py # Run Apriori on sample (for report)
│  ├─ run_eclat_demo.py       # Run Eclat demo (for report)
│  └─ test_recommender.py     # Quick CLI tests of recommendation function
│
├─ notebooks/
│  └─ eda.ipynb               # Exploratory Data Analysis notebook
│
├─ app.py                     # (Optional) Flask API exposing /recommend
├─ .gitignore                 # Ignore large data, caches, venv, etc.
└─ README.md                  # This file

```
# 2. Data & Requirements

## 2.1. Input data (not included in Git)
Place the following CSV files under data/raw/:

aisles.csv
departments.csv
order_products__prior.csv
orders.csv
products.csv
products_with_prices.csv ← extended file with at least:
product_id
price

## 2.2. Python environment
Example with conda:
```bash

conda c reate -n retail_env python=3.11
conda activate retail_env

# Install main dependencies
pip install pandas numpy scikit-learn mlxtend flask
```

# 3. Configuration
Paths are defined in src/config.py. By default:

```python
PROJECT_ROOT = ""  # adapt the path

DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_INTERIM   = os.path.join(PROJECT_ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
```
# 4. How to Run the ML Pipeline
All commands below are executed from the project root:

## 4.1. Step 1 – Data enrichment (merge + prices)
Build the main merged table and add product prices.

```bash
python scripts/run_enrichment.py
```

Outputs:
- data/interim/order_products_full.csv
  → merged Instacart tables (orders, products, aisles, departments, etc.)
- data/interim/order_products_full_with_price.csv
  → same as above, plus a price column per product in each order line.
  
This is the base dataset for both EDA and all ML steps..

## 4.2. Step 2 – Customer features & segmentation
Aggregate orders to customer level and cluster customers.

```bash

python scripts/run_clustering.py
```

Outputs:
- data/processed/customer_segments.csv
  -> Each row = one user_id with features such as:

- total_orders
- total_products
-avg_basket_size
-order_gap_std
- unique_products
- total_spent (using prices)
- cluster (MiniBatchKMeans segment label)

These segments can be used by the business team and in dashboards.

## 4.3. Step 3 – Association rules with FP‑Growth + utility
Mine frequent itemsets and association rules using FP‑Growth, then compute utility and expected revenue using prices.

```bash
python scripts/run_association_rules.py
```

Outputs:

- data/processed/association_rules_fp_all.csv
→ all mined rules (any size) with:
  - support, confidence, lift
  - itemset & rule utility
  - expected revenue
- data/processed/business_ready_rules.csv
→ filtered 1→1 rules only, with:
  - antecedents_str, consequents_str
  - support, confidence, lift
  - rule_utility, expected_revenue

Similar products (e.g. “Banana → Bag of Organic Bananas”) are filtered out.
- data/processed/top_rules_per_item.csv
→ up to 3 best rules per antecedent, convenient for some UIs.

The mining function implements utility‑aware pattern mining by computing monetary utility and expected revenue for each itemset/rule, which serves the same business purpose as UP‑Tree (high‑utility itemset mining).

## 4.4. Step 4 – Apriori and Eclat (for comparison)
Used mainly for experiments and the report.

Apriori (sample):
```bash
python scripts/run_apriori_comparison.py
```
Output:
- data/processed/apriori_rules_sample.csv
→ rules mined from a sample of the basket with Apriori; used to compare runtime and rule count vs FP‑Growth.

Eclat (sample):
```bash
python scripts/run_eclat_demo.py
```
Outputs:

- data/processed/eclat_itemsets_demo.csv
→ frequent itemsets mined with a simple Eclat implementation on a sample; used to illustrate TID‑list, depth‑first mining.

# 5. EDA – Notebook
The notebook notebooks/eda.ipynb performs Exploratory Data Analysis on the merged dataset. It uses the same build_full_order_products() logic as run_enrichment.py, ensuring that the ML steps are a continuation of the EDA.

Typical analyses include:

- Basket size distribution
- Top products / aisles / departments
- Customer ordering frequency
- Price distributions

# 6. Recommender Interface (for Web Application)
The web application should not re‑implement the ML logic. Instead, it calls a simple function defined in src/recommender.py:

```python
from recommender import recommend_items

cart = ["Banana", "Organic Strawberries"]
recommendations = recommend_items(
    cart_items=cart,
    top_k=5,
    min_lift=1.0,
    min_conf=0.1,
    avoid_similar=True,
)
```
## 6.1. How recommend_items works
1. Loads data/processed/business_ready_rules.csv.
2. Filters rules to those where antecedents_str matches items in the current cart.
3. Applies thresholds on lift and confidence to keep strong rules.
4. Ranks rules by:
   - expected_revenue (utility × support),
   - then lift,
   - then confidence.
5. Returns up to top_k distinct consequents, excluding:
   - items already in the cart,
   - items that are “too similar” (e.g. “Banana” vs “Bag of Organic Bananas”).

This gives business‑friendly, high‑value cross‑sell suggestions.

# 7. Files Needed by the Web Team
For the web application, the main artifacts are:

- data/processed/customer_segments.csv
→ for customer segmentation dashboards and analytics.

-  data/processed/business_ready_rules.csv
→ for product‑level recommendations and bundles.

- data/processed/top_rules_per_item.csv (optional)
→ simplified view with top 3 rules per product.

And the code/API:

- src/recommender.py
→ provides recommend_items(cart_items, ...).

# 8. Methods Summary (for Report Mapping)
- Association models:

  - Apriori (sample)
  - Eclat (sample, TID‑list based)
  - FP‑Growth (full data, main pipeline)
- Utility‑aware mining:

  - Products are enriched with prices (products_with_prices.csv).
  - For each itemset/rule:
    - itemset_utility / rule_utility (sum of prices),
    - expected_revenue = support × utility.
  - FP‑Growth + utility ranking is used as a practical high‑utility itemset mining approach (inspired by UP‑Tree).
- Predictive/segmentation models:

  - Customer segmentation via MiniBatchKMeans on behavioral and monetary features.
  - Association rules used predictively for next‑item recommendations.
  
# 9. Contributors
- Data Analysis: 
- Data Science / ML: Rahiba Shereef
- Web / Cloud / Front‑end: 
- Supervisor / Course: 

