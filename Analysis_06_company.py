import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── AYARLAR ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
# SAMPLE_SIZE = 100_000  # tüm veri kullanılıyor
RANDOM_SEED = 42  # clustering/shuffle için kullanılıyor
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

# ── VERİ YÜKLEME ──────────────────────────────────────────────────────────────
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "company_id", "normalized_salary",
             "formatted_experience_level", "remote_allowed", "formatted_work_type"],
    low_memory=False
).reset_index(drop=True)

companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                             usecols=["company_id", "name", "company_size", "state"])
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
employee_counts= pd.read_csv(os.path.join(DATA_PATH, "employee_counts.csv"))

# Temizleme
postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

# Şirket büyüklük etiketi
size_map = {1:"1-10", 2:"11-50", 3:"51-200", 4:"201-500",
            5:"501-1K", 6:"1K-5K", 7:"5K-10K", 8:"10K+"}
companies["size_label"] = companies["company_size"].map(size_map)

# Join
postings = postings.merge(companies, on="company_id", how="left")

sample_ids   = set(postings["job_id"])
job_ind_f    = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full = job_ind_f.merge(industries, on="industry_id", how="left")

print("Veri hazır.\n")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — En çok ilan veren 15 şirket
# ══════════════════════════════════════════════════════════════════════════════
top_companies = (postings[postings["name"].notna()]
                 .groupby("name")["job_id"]
                 .count()
                 .sort_values(ascending=False)
                 .head(15))

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top_companies.index[::-1], top_companies.values[::-1],
               color=sns.color_palette("Blues_d", 15)[::-1])
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title("En Çok İlan Veren 15 Şirket", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "27_top_companies.png"), dpi=150)
plt.close()
print("27_top_companies.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Şirket büyüklüğüne göre medyan maaş
# ══════════════════════════════════════════════════════════════════════════════
size_order = ["1-10","11-50","51-200","201-500","501-1K","1K-5K","5K-10K","10K+"]
size_sal = (postings[postings["normalized_salary"].notna() &
                     postings["size_label"].notna()]
            .groupby("size_label")["normalized_salary"]
            .median()
            .reindex(size_order)
            .dropna())

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(size_sal.index, size_sal.values,
              color=sns.color_palette("Blues_d", len(size_sal)),
              edgecolor="white")
ax.bar_label(bars,
             labels=[f"${v:,.0f}" for v in size_sal.values],
             padding=4, fontsize=9)
ax.set_xlabel("Şirket Büyüklüğü")
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Şirket Büyüklüğüne Göre Medyan Maaş", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "28_company_size_salary.png"), dpi=150)
plt.close()
print("28_company_size_salary.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Şirket büyüklüğü × Sektör yoğunluk haritası
# ══════════════════════════════════════════════════════════════════════════════
top8_ind = job_ind_full["industry_name"].value_counts().head(8).index.tolist()
size_ind = (postings[["job_id","size_label"]]
            .merge(job_ind_full[["job_id","industry_name"]], on="job_id")
            .query("industry_name in @top8_ind and size_label.notna()", engine="python"))

size_ind_pivot = (size_ind.groupby(["size_label","industry_name"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(size_order))
size_ind_pct = size_ind_pivot.div(size_ind_pivot.sum(axis=1), axis=0).mul(100).round(1)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(size_ind_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Şirket Büyüklüğü × Sektör Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Şirket Büyüklüğü")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "29_company_size_industry.png"), dpi=150)
plt.close()
print("29_company_size_industry.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — En yüksek medyan maaş ödeyen 15 şirket (min 20 ilan)
# ══════════════════════════════════════════════════════════════════════════════
company_sal = (postings[postings["normalized_salary"].notna() &
                        postings["name"].notna()]
               .groupby("name")["normalized_salary"]
               .agg(["median","count"])
               .query("count >= 20")
               .sort_values("median", ascending=False)
               .head(15))

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(company_sal.index[::-1], company_sal["median"][::-1],
               color=sns.color_palette("Blues_d", 15)[::-1])
ax.bar_label(bars,
             labels=[f"${v:,.0f}" for v in company_sal["median"][::-1]],
             padding=4, fontsize=9)
ax.set_xlabel("Medyan Yıllık Maaş (USD)")
ax.set_title("En Yüksek Maaş Ödeyen 15 Şirket\n(min. 20 ilan)",
             fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "30_top_paying_companies.png"), dpi=150)
plt.close()
print("30_top_paying_companies.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Şirket büyüklüğü × Deneyim seviyesi dağılımı
# ══════════════════════════════════════════════════════════════════════════════
exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
size_exp = (postings[postings["size_label"].notna() &
                     postings["formatted_experience_level"].isin(exp_order)]
            .groupby(["size_label","formatted_experience_level"])
            .size()
            .unstack(fill_value=0)
            .reindex(size_order)[exp_order])
size_exp_pct = size_exp.div(size_exp.sum(axis=1), axis=0).mul(100)

fig, ax = plt.subplots(figsize=(13, 6))
size_exp_pct.plot(kind="bar", stacked=True, ax=ax,
                  color=sns.color_palette("Blues_d", 5),
                  edgecolor="white", width=0.65)
ax.set_xlabel("Şirket Büyüklüğü")
ax.set_ylabel("Oran (%)")
ax.set_title("Şirket Büyüklüğüne Göre Deneyim Seviyesi Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(title="Deneyim", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "31_company_size_experience.png"), dpi=150)
plt.close()
print("31_company_size_experience.png kaydedildi.")

print("\nTüm şirket analizi grafikleri outputs/ klasörüne kaydedildi.")