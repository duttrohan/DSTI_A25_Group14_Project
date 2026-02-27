# Data-Driven Retail Insights for Cost Savings & Revenue Growth
**Team:** DSTI Group 14  
**Course:** Applied MSc in DS & AI  
**Date:** February 2026

## 1. Executive Summary
- What business problem we solve
- Key wins: e.g., +$X/month with top bundles, actionable segments, promo ROI insights
- High-level architecture: data → ML → dashboards → recommendations

## 2. Dataset & Enrichment
- Instacart tables (orders, products, aisles, departments, order_products__prior)
- **Price enrichment** via `products_with_prices.csv`
- Final merged lines: `order_products_full_with_price.csv`
- Data volume (rows, users, products); quality checks & handling

## 3. Methodology
### 3.1 Feature Engineering (Customers)
- Behavioral metrics: total_orders, unique_products, avg_basket_size, order_gap_std
- Monetary metrics: total_spent
### 3.2 Segmentation
- MiniBatchKMeans (why, how many clusters, silhouette or elbow)
- Segment profiles (budget vs premium; frequent vs irregular)
### 3.3 Association Rules (Utility-Aware)
- FP-Growth (support/confidence/lift)
- **Utility**: item prices → `rule_utility`, `expected_revenue = support × utility`
- Business-ready 1→1 rule filtering + similar-item exclusion
- Safe-mode: product popularity filter + order cap (to control RAM)
### 3.4 Recommender
- Input cart → filter rules → rank by expected_revenue > lift > confidence
- Robustness: fuzzy matching, substring match, global fallbacks

## 4. Results
- Segmentation: distribution, KPIs (avg basket, variability); business personas
- Rules: top bundles by revenue; lift/confidence chart; examples
- Recommender quality: examples with/without fuzzy match
- Promo ROI vs Untargeted ROI (show scenario)

## 5. Dashboard
- Pages overview (Segmentation, Associations, Revenue Simulation, Promo Efficiency, Recommender)
- Key interactions (side-by-side selectors, adoption slider)
- Screenshots

## 6. Business Impact
- **Revenue simulation:** example scenario with chosen rule and monthly orders
- **Promotion efficiency:** when targeted promos beat blanket discounts
- Prioritized actions for a retail manager (Top 5)

## 7. Limitations & Risks
- Sampling (safe-mode) vs full dataset
- Rule stability over time, seasonal effects
- Price accuracy & margin assumptions (if any)
- Cold-start for unseen items/users

## 8. Future Work
- Full high-utility itemset mining (UP-Tree), time-aware rules, personalization by cluster
- A/B test harness in app
- Margin-aware ROI (costs of goods, promo cannibalization)

## 9. Reproducibility
- Exact commands (see Appendix A)
- Environment & versions

## 10. References
- Instacart dataset (Kaggle) and association rule mining literature

 ## 11. Rubric Coverage Matrix 
<img width="551" height="281" alt="image" src="https://github.com/user-attachments/assets/4d5ed77d-c2e9-4f12-801f-081905362623" />


### Appendix A — Reproduction Commands
```bash
python scripts/run_enrichment.py
python scripts/run_clustering.py
python scripts/run_association_rules.py
streamlit run streamlit_app.py
