# Customer Segmentation Using K-Means Clustering

## Overview
Segmented 1,000 e-commerce customers into 5 meaningful groups using K-Means Clustering and RFM Analysis (Recency, Frequency, Monetary). Results were visualized using PCA and translated into actionable marketing strategies.

---

## Results

| Segment | Customers | Avg Spend | Action |
|---|---|---|---|
| Champions | 135 | 13,454 | Reward and retain |
| Loyal Customers | 272 | 5,416 | Upsell |
| At Risk | 266 | 2,887 | Win-back campaigns |
| Lost Customers | 169 | 754 | Strong discount offers |
| New Customers | 158 | 1,156 | Onboarding support |

Silhouette Score: 0.50 — good cluster separation
PCA Variance Explained: 94.4%

---

## What's Inside

| File | Description |
|---|---|
| `customer_segmentation.py` | Full project code |
| `cs_1_eda.png` | RFM distributions, correlation heatmap, scatter plots |
| `cs_2_elbow.png` | Elbow method and Silhouette score to find optimal K |
| `cs_3_pca_clusters.png` | PCA 2D visualization of clusters and segment pie chart |
| `cs_4_rfm_profiles.png` | RFM metrics compared across all 5 segments |
| `cs_5_recommendations.png` | Business strategy table for each segment |

---

## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (KMeans, PCA, StandardScaler)

---

## Key Concepts Covered
- RFM feature engineering (Recency, Frequency, Monetary)
- Feature scaling with StandardScaler (essential for distance-based algorithms)
- Elbow Method and Silhouette Score to determine optimal K
- PCA for dimensionality reduction and 2D cluster visualization
- Translating ML output into real business recommendations

---

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python customer_segmentation.py
```

---

## Dataset
Synthetic e-commerce dataset with 1,000 customers across 5 behavioral archetypes mimicking real-world UCI Online Retail data patterns.
