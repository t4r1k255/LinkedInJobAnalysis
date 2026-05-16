import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
from utils_progress import ProgressBar, StepTracker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter

# ── AYARLAR ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
RANDOM_SEED = 42
N_CLUSTERS  = 8
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=8, script_name="analysis_07_title_clustering.py — Title Clustering")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=1, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "title", "normalized_salary",
             "formatted_experience_level", "remote_allowed"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")
_bar.finish()

postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

# Başlık temizleme
postings["title_clean"] = (postings["title"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-z\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip())

title_df = postings[postings["title_clean"].str.len() > 2].copy().reset_index(drop=True)
print(f"Unvan analizi için {len(title_df):,} ilan hazır.")
tracker.done(1)

tracker.start(2, "TF-IDF vectorization")
# ── TF-IDF + K-MEANS ──────────────────────────────────────────────────────────
print("TF-IDF vektörizasyonu yapılıyor...")
tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5
)
X = tfidf.fit_transform(title_df["title_clean"])
print(f"TF-IDF matris boyutu: {X.shape}")
tracker.done(2)

tracker.start(3, "K-Means clustering")
print(f"K-Means kümeleme yapılıyor (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
title_df["cluster"] = kmeans.fit_predict(X)
print("Kümeleme tamamlandı.")
tracker.done(3)

# Her küme için en yaygın unvan terimlerini bul
feature_names = tfidf.get_feature_names_out()
cluster_labels = {}
for i in range(N_CLUSTERS):
    center = kmeans.cluster_centers_[i]
    top_terms = [feature_names[j] for j in center.argsort()[-5:][::-1]]
    cluster_labels[i] = " / ".join(top_terms[:3])

title_df["cluster_label"] = title_df["cluster"].map(cluster_labels)
print("Küme etiketleri:")
for k, v in cluster_labels.items():
    count = (title_df["cluster"] == k).sum()
    print(f"  Küme {k}: {v}  ({count:,} ilan)")

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(4, "Plot 1 — Cluster sizes")
# GRAFİK 1 — Küme büyüklükleri
# ══════════════════════════════════════════════════════════════════════════════
cluster_counts = title_df["cluster_label"].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(cluster_counts.index, cluster_counts.values,
               color=sns.color_palette("Blues_d", len(cluster_counts)))
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title(f"İş Unvanı Kümelerine Göre İlan Dağılımı (k={N_CLUSTERS})",
             fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "32_cluster_sizes.png"), dpi=150)
plt.close()
print("32_cluster_sizes.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 2 — PCA 2D visualization")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — PCA ile 2D görselleştirme (10k örneklem)
# ══════════════════════════════════════════════════════════════════════════════
print("PCA görselleştirmesi hazırlanıyor...")
sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(title_df), size=10_000, replace=False)
X_sample   = X[sample_idx]
labels_sample = title_df["cluster"].iloc[sample_idx].values

pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_sample.toarray())

colors = sns.color_palette("Blues_d", N_CLUSTERS)
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(N_CLUSTERS):
    mask = labels_sample == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[colors[i]], label=cluster_labels[i],
               alpha=0.5, s=8, edgecolors="none")

ax.set_title("İş Unvanı Kümeleri — PCA 2D Görselleştirme (10k örnek)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel(f"PCA Bileşen 1 (Varyans: {pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PCA Bileşen 2 (Varyans: {pca.explained_variance_ratio_[1]:.1%})")
ax.legend(title="Küme", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
          markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "33_cluster_pca.png"), dpi=150)
plt.close()
print("33_cluster_pca.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 3 — Cluster salary")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Kümeye göre medyan maaş
# ══════════════════════════════════════════════════════════════════════════════
cluster_salary = (title_df[title_df["normalized_salary"].notna()]
                  .groupby("cluster_label")["normalized_salary"]
                  .median()
                  .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(cluster_salary.index, cluster_salary.values,
               color=sns.color_palette("Blues_d", len(cluster_salary)))
ax.bar_label(bars,
             labels=[f"${v:,.0f}" for v in cluster_salary.values],
             padding=4, fontsize=9)
ax.set_xlabel("Medyan Yıllık Maaş (USD)")
ax.set_title("İş Unvanı Kümesine Göre Medyan Maaş",
             fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "34_cluster_salary.png"), dpi=150)
plt.close()
print("34_cluster_salary.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 4 — Cluster experience")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Kümeye göre deneyim seviyesi dağılımı
# ══════════════════════════════════════════════════════════════════════════════
exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
cluster_exp = (title_df[title_df["formatted_experience_level"].isin(exp_order)]
               .groupby(["cluster_label", "formatted_experience_level"])
               .size()
               .unstack(fill_value=0)[exp_order])
cluster_exp_pct = cluster_exp.div(cluster_exp.sum(axis=1), axis=0).mul(100)

fig, ax = plt.subplots(figsize=(13, 6))
cluster_exp_pct.plot(kind="bar", stacked=True, ax=ax,
                     color=sns.color_palette("Blues_d", 5),
                     edgecolor="white", width=0.65)
ax.set_xlabel("")
ax.set_ylabel("Oran (%)")
ax.set_title("İş Unvanı Kümesine Göre Deneyim Seviyesi Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(title="Deneyim", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "35_cluster_experience.png"), dpi=150)
plt.close()
print("35_cluster_experience.png kaydedildi.")
tracker.done(7)

tracker.start(8, "Plot 5 — Cluster remote rate")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Kümeye göre remote oranı
# ══════════════════════════════════════════════════════════════════════════════
cluster_remote = (title_df.groupby("cluster_label")["remote_allowed"]
                  .mean()
                  .mul(100)
                  .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(cluster_remote.index, cluster_remote.values,
               color=sns.color_palette("Blues_d", len(cluster_remote)))
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_xlabel("Remote İlan Oranı (%)")
ax.set_title("İş Unvanı Kümesine Göre Remote İlan Oranı",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "36_cluster_remote.png"), dpi=150)
plt.close()
print("36_cluster_remote.png kaydedildi.")
tracker.done(8)
tracker.finish()

print("\nTüm kümeleme grafikleri outputs/ klasörüne kaydedildi.")