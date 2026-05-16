import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
# SAMPLE_SIZE = 100_000  # tüm veri kullanılıyor
RANDOM_SEED = 42  # clustering/shuffle için kullanılıyor
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "company_id", "normalized_salary",
             "formatted_experience_level", "remote_allowed"],
    low_memory=False
).reset_index(drop=True)

benefits       = pd.read_csv(os.path.join(DATA_PATH, "benefits.csv"))
companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                             usecols=["company_id", "company_size"])
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))

postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

size_map = {1:"1-10", 2:"11-50", 3:"51-200", 4:"201-500",
            5:"501-1K", 6:"1K-5K", 7:"5K-10K", 8:"10K+"}
companies["size_label"] = companies["company_size"].map(size_map)
postings = postings.merge(companies, on="company_id", how="left")

sample_ids   = set(postings["job_id"])
benefits_f   = benefits[benefits["job_id"].isin(sample_ids)].copy()
job_ind_f    = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full = job_ind_f.merge(industries, on="industry_id", how="left")

print(f"Benefits verisi: {len(benefits_f):,} kayit")
print(f"Benzersiz benefit turu: {benefits_f['type'].nunique()}")
print(f"\nTop 20 benefit:\n{benefits_f['type'].value_counts().head(20).to_string()}\n")

# GRAFİK 1
top_benefits = benefits_f["type"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top_benefits.index[::-1], top_benefits.values[::-1],
               color=sns.color_palette("Blues_d", 20)[::-1])
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayisi")
ax.set_title("En Yaygin 20 Yan Hak (Benefit)", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "37_top_benefits.png"), dpi=150)
plt.close()
print("37_top_benefits.png kaydedildi.")

# GRAFİK 2
size_order  = ["1-10","11-50","51-200","201-500","501-1K","1K-5K","5K-10K","10K+"]
ben_per_job = benefits_f.groupby("job_id").size().reset_index(name="benefit_count")
ben_per_job = ben_per_job.merge(postings[["job_id","size_label"]], on="job_id", how="left")
size_ben = (ben_per_job[ben_per_job["size_label"].notna()]
            .groupby("size_label")["benefit_count"]
            .mean().reindex(size_order).dropna())
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(size_ben.index, size_ben.values,
              color=sns.color_palette("Blues_d", len(size_ben)), edgecolor="white")
ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
ax.set_xlabel("Sirket Buyuklugu")
ax.set_ylabel("Ortalama Benefit Sayisi")
ax.set_title("Sirket Buyuklugune Gore Ortalama Benefit Sayisi",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "38_benefits_by_company_size.png"), dpi=150)
plt.close()
print("38_benefits_by_company_size.png kaydedildi.")

# GRAFİK 3
top5_ind  = job_ind_full["industry_name"].value_counts().head(5).index.tolist()
top10_ben = benefits_f["type"].value_counts().head(10).index.tolist()
ben_ind = (benefits_f[benefits_f["type"].isin(top10_ben)]
           .merge(job_ind_full[["job_id","industry_name"]], on="job_id", how="inner")
           .query("industry_name in @top5_ind"))
ben_ind_pivot = (ben_ind.groupby(["industry_name","type"])
                 .size().unstack(fill_value=0).reindex(top5_ind))
ben_ind_pct = ben_ind_pivot.div(ben_ind_pivot.sum(axis=1), axis=0).mul(100).round(1)
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(ben_ind_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Top 5 Sectorde Top 10 Benefit Dagilimi (%)",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "39_benefits_by_industry.png"), dpi=150)
plt.close()
print("39_benefits_by_industry.png kaydedildi.")

# GRAFİK 4
ben_sal = (ben_per_job
           .merge(postings[["job_id","normalized_salary"]], on="job_id", how="left")
           .dropna(subset=["normalized_salary"]))
ben_sal["benefit_group"] = ben_sal["benefit_count"].clip(upper=10).astype(str)
ben_sal.loc[ben_sal["benefit_count"] >= 10, "benefit_group"] = "10+"
order = [str(i) for i in range(1, 10)] + ["10+"]
ben_sal_grouped = (ben_sal.groupby("benefit_group")["normalized_salary"]
                   .median().reindex(order).dropna())
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(ben_sal_grouped.index, ben_sal_grouped.values,
              color=sns.color_palette("Blues_d", len(ben_sal_grouped)), edgecolor="white")
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in ben_sal_grouped.values],
             padding=4, fontsize=8)
ax.set_xlabel("Benefit Sayisi")
ax.set_ylabel("Medyan Yillik Maas (USD)")
ax.set_title("Benefit Sayisina Gore Medyan Maas",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "40_benefit_count_salary.png"), dpi=150)
plt.close()
print("40_benefit_count_salary.png kaydedildi.")

# GRAFİK 5
exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
top8_ben  = benefits_f["type"].value_counts().head(8).index.tolist()
ben_exp = (benefits_f[benefits_f["type"].isin(top8_ben)]
           .merge(postings[["job_id","formatted_experience_level"]], on="job_id", how="left")
           .query("formatted_experience_level in @exp_order"))
ben_exp_pivot = (ben_exp.groupby(["formatted_experience_level","type"])
                 .size().unstack(fill_value=0).reindex(exp_order))
ben_exp_pct = ben_exp_pivot.div(ben_exp_pivot.sum(axis=1), axis=0).mul(100).round(1)
fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(ben_exp_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Deneyim Seviyesine Gore Top 8 Benefit Dagilimi (%)",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "41_benefits_by_experience.png"), dpi=150)
plt.close()
print("41_benefits_by_experience.png kaydedildi.")

print("\nTum benefit grafikleri outputs/ klasorune kaydedildi.")