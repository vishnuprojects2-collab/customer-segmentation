"""
Customer Segmentation Using K-Means Clustering
===============================================
Dataset: Synthetic E-Commerce dataset (RFM Analysis)
Tech: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
"""

# ─── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.style.use("seaborn-v0_8-whitegrid")
COLORS  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
LABELS  = ["Champions", "Loyal Customers", "At Risk", "Lost Customers", "New Customers"]

# ─── 2. GENERATE REALISTIC E-COMMERCE DATA ────────────────────
np.random.seed(42)
N = 1000

customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, N+1)]

# Simulate different customer types naturally
segments_true = np.random.choice(["champion","loyal","at_risk","lost","new"],
                                  size=N, p=[0.15, 0.25, 0.25, 0.20, 0.15])

recency, frequency, monetary = [], [], []
for s in segments_true:
    if s == "champion":
        recency.append(np.random.randint(1, 15))
        frequency.append(np.random.randint(20, 50))
        monetary.append(np.random.uniform(5000, 20000))
    elif s == "loyal":
        recency.append(np.random.randint(10, 40))
        frequency.append(np.random.randint(10, 25))
        monetary.append(np.random.uniform(2000, 8000))
    elif s == "at_risk":
        recency.append(np.random.randint(60, 120))
        frequency.append(np.random.randint(5, 15))
        monetary.append(np.random.uniform(1000, 5000))
    elif s == "lost":
        recency.append(np.random.randint(150, 365))
        frequency.append(np.random.randint(1, 5))
        monetary.append(np.random.uniform(100, 1500))
    else:  # new
        recency.append(np.random.randint(1, 30))
        frequency.append(np.random.randint(1, 5))
        monetary.append(np.random.uniform(200, 2000))

df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Recency":    recency,
    "Frequency":  frequency,
    "Monetary":   monetary
})
print(f"Dataset: {len(df)} customers")
print(df.describe().round(2))

# ─── 3. EDA ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Customer Segmentation — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Distributions
for i, (col, color) in enumerate(zip(["Recency","Frequency","Monetary"], COLORS)):
    axes[0, i].hist(df[col], bins=30, color=color, alpha=0.8, edgecolor="white")
    axes[0, i].set_title(f"{col} Distribution")
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel("Count")
    mean_val = df[col].mean()
    axes[0, i].axvline(mean_val, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.0f}")
    axes[0, i].legend()

# Correlation heatmap
corr = df[["Recency","Frequency","Monetary"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[1, 0], linewidths=0.5, center=0)
axes[1, 0].set_title("RFM Correlation Heatmap")

# Frequency vs Monetary
scatter = axes[1, 1].scatter(df["Frequency"], df["Monetary"],
                              c=df["Recency"], cmap="RdYlGn_r",
                              alpha=0.6, s=20)
plt.colorbar(scatter, ax=axes[1, 1], label="Recency (days)")
axes[1, 1].set_title("Frequency vs Monetary\n(color = Recency)")
axes[1, 1].set_xlabel("Frequency"); axes[1, 1].set_ylabel("Monetary (₹)")

# Recency vs Monetary
scatter2 = axes[1, 2].scatter(df["Recency"], df["Monetary"],
                               c=df["Frequency"], cmap="YlOrRd",
                               alpha=0.6, s=20)
plt.colorbar(scatter2, ax=axes[1, 2], label="Frequency")
axes[1, 2].set_title("Recency vs Monetary\n(color = Frequency)")
axes[1, 2].set_xlabel("Recency (days)"); axes[1, 2].set_ylabel("Monetary (₹)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/cs_1_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ EDA plot saved")

# ─── 4. SCALING ───────────────────────────────────────────────
rfm = df[["Recency", "Frequency", "Monetary"]].copy()
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
print("\nData scaled using StandardScaler ✅")

# ─── 5. ELBOW METHOD + SILHOUETTE ─────────────────────────────
wcss        = []
sil_scores  = []
K_range     = range(2, 11)

for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Choosing Optimal K — Elbow Method & Silhouette Score", fontsize=14, fontweight="bold")

axes[0].plot(K_range, wcss, marker="o", color=COLORS[0], linewidth=2, markersize=8)
axes[0].axvline(5, color=COLORS[1], linestyle="--", linewidth=2, label="Optimal K=5")
axes[0].set_title("Elbow Method (WCSS vs K)")
axes[0].set_xlabel("Number of Clusters (K)"); axes[0].set_ylabel("WCSS")
axes[0].legend()

axes[1].plot(K_range, sil_scores, marker="s", color=COLORS[2], linewidth=2, markersize=8)
axes[1].axvline(5, color=COLORS[1], linestyle="--", linewidth=2, label="Optimal K=5")
axes[1].set_title("Silhouette Score vs K\n(higher = better separation)")
axes[1].set_xlabel("Number of Clusters (K)"); axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/cs_2_elbow.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Elbow + Silhouette plot saved")

# ─── 6. FIT FINAL K-MEANS (K=5) ───────────────────────────────
K_OPTIMAL = 5
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(rfm_scaled)

sil = silhouette_score(rfm_scaled, df["Cluster"])
print(f"\nFinal K-Means (K={K_OPTIMAL}) → Silhouette Score: {sil:.4f}")

# ─── 7. LABEL CLUSTERS BY BEHAVIOR ───────────────────────────
cluster_summary = df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
cluster_summary["Size"] = df.groupby("Cluster").size()
print("\nCluster Profiles (raw means):")
print(cluster_summary.round(2))

# Auto-assign meaningful labels based on Frequency + Recency
def assign_label(row):
    if row["Frequency"] >= 20 and row["Recency"] <= 20:
        return "Champions"
    elif row["Frequency"] >= 10 and row["Recency"] <= 45:
        return "Loyal Customers"
    elif row["Recency"] >= 150:
        return "Lost Customers"
    elif row["Recency"] >= 60:
        return "At Risk"
    else:
        return "New Customers"

cluster_summary["Label"] = cluster_summary.apply(assign_label, axis=1)
label_map = cluster_summary["Label"].to_dict()
df["Segment"] = df["Cluster"].map(label_map)
print("\nCluster → Segment mapping:")
print(label_map)

# ─── 8. PCA VISUALIZATION ─────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(rfm_scaled)
df["PCA1"] = pca_coords[:, 0]
df["PCA2"] = pca_coords[:, 1]
var_explained = pca.explained_variance_ratio_ * 100

segment_colors = {
    "Champions":       COLORS[0],
    "Loyal Customers": COLORS[2],
    "At Risk":         COLORS[4],
    "Lost Customers":  COLORS[1],
    "New Customers":   COLORS[3],
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Customer Segments — PCA 2D Visualization", fontsize=15, fontweight="bold")

for seg, color in segment_colors.items():
    mask = df["Segment"] == seg
    axes[0].scatter(df.loc[mask, "PCA1"], df.loc[mask, "PCA2"],
                    c=color, label=seg, alpha=0.7, s=25)
axes[0].set_title(f"K-Means Clusters (PCA)\nVariance explained: {sum(var_explained):.1f}%")
axes[0].set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
axes[0].set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
axes[0].legend(loc="upper right", fontsize=9)

# Cluster size pie chart
seg_counts = df["Segment"].value_counts()
colors_pie = [segment_colors.get(s, "#999999") for s in seg_counts.index]
axes[1].pie(seg_counts.values, labels=seg_counts.index, autopct="%1.1f%%",
            colors=colors_pie, startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Customer Segment Distribution")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/cs_3_pca_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ PCA cluster plot saved")

# ─── 9. RFM SEGMENT PROFILES ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("RFM Profile by Customer Segment", fontsize=15, fontweight="bold")

metrics = ["Recency", "Frequency", "Monetary"]
ylabels = ["Avg Recency (days — lower is better)", "Avg Frequency (orders)", "Avg Monetary (₹)"]

for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
    seg_avg = df.groupby("Segment")[metric].mean().sort_values(
        ascending=(metric == "Recency"))
    bar_colors = [segment_colors.get(s, "#999") for s in seg_avg.index]
    bars = axes[i].bar(seg_avg.index, seg_avg.values, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[i].set_title(f"Avg {metric} by Segment")
    axes[i].set_ylabel(ylabel)
    axes[i].tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, seg_avg.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(seg_avg)*0.01,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/cs_4_rfm_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ RFM profiles plot saved")

# ─── 10. BUSINESS RECOMMENDATIONS ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

recommendations = {
    "Champions":       "Reward them. Ask for reviews. Offer early access to new products.",
    "Loyal Customers": "Upsell higher value products. Engage with loyalty programs.",
    "At Risk":         "Send win-back campaigns. Offer limited-time discounts.",
    "Lost Customers":  "Reactivate with strong discount offers or survey why they left.",
    "New Customers":   "Provide onboarding support. Offer first-purchase discount.",
}

table_data = [[seg, recommendations[seg], f"{df[df['Segment']==seg].shape[0]} customers"]
              for seg in recommendations]
col_labels = ["Segment", "Marketing Strategy", "Size"]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellLoc="left", loc="center", colWidths=[0.22, 0.58, 0.18])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)

# Style header
for j in range(3):
    tbl[(0, j)].set_facecolor("#2196F3")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Style rows
row_colors = [segment_colors.get(row[0], "#f0f0f0") for row in table_data]
for i, color in enumerate(row_colors, start=1):
    for j in range(3):
        tbl[(i, j)].set_facecolor(color + "33")  # light tint

ax.set_title("Business Recommendations by Segment", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/cs_5_recommendations.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Business recommendations table saved")

# ─── 11. SUMMARY ─────────────────────────────────────────────
print("\n" + "="*60)
print("         CUSTOMER SEGMENTATION — SUMMARY")
print("="*60)
print(f"Total Customers Analysed : {len(df)}")
print(f"Algorithm                : K-Means (K={K_OPTIMAL})")
print(f"Silhouette Score         : {sil:.4f}  (max=1.0)")
print(f"PCA Variance Explained   : {sum(var_explained):.1f}%")
print("-"*60)
for seg in recommendations:
    count = df[df["Segment"] == seg].shape[0]
    avg_m = df[df["Segment"] == seg]["Monetary"].mean()
    print(f"  {seg:<22} → {count:>4} customers | Avg Spend: ₹{avg_m:,.0f}")
print("="*60)
print("\n✅ All outputs saved to /mnt/user-data/outputs/")
