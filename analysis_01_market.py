import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from utils_progress import ProgressBar, StepTracker

# ── AYARLAR ───────────────────────────────────────────────────────────────────
DATA_PATH    = "data/"
OUTPUT_PATH  = "outputs/"
RANDOM_SEED  = 42
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=8, script_name="analysis_01_market.py — Market Overview")

# ── VERİ YÜKLEME VE TEMİZLEME ─────────────────────────────────────────────────
tracker.start(1, "Loading data")
_bar = ProgressBar(total=2, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id","company_id","title","location","remote_allowed",
             "formatted_experience_level","formatted_work_type","normalized_salary","listed_time"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")
companies = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                        usecols=["company_id","name","company_size"])
_bar.step("companies.csv")
_bar.finish()
tracker.done(1)

tracker.start(2, "Cleaning & merging")
postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings["listed_date"] = pd.to_datetime(postings["listed_time"], unit="ms", errors="coerce")
postings["state"] = postings["location"].str.extract(r",\s*([A-Z]{2})\s*$")
postings.loc[postings["normalized_salary"] < 10_000,  "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None
postings = postings.merge(companies.rename(columns={"name":"company_name"}),
                          on="company_id", how="left")
size_map = {1:"1-10",2:"11-50",3:"51-200",4:"201-500",
            5:"501-1K",6:"1K-5K",7:"5K-10K",8:"10K+"}
postings["company_size_label"] = postings["company_size"].map(size_map)
tracker.done(2, f"{len(postings):,} rows ready")

tracker.start(3, "Plot 1 — Experience distribution")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — Deneyim seviyesine göre ilan sayısı
# ══════════════════════════════════════════════════════════════════════════════
exp_order = ["Entry level","Associate","Mid-Senior level","Director","Executive","Internship","Unknown"]
exp_counts = (postings["formatted_experience_level"]
              .value_counts()
              .reindex(exp_order)
              .dropna())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(exp_counts.index, exp_counts.values, color=sns.color_palette("Blues_d", len(exp_counts)))
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title("Deneyim Seviyesine Göre İlan Dağılımı", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "01_experience_distribution.png"), dpi=150)
plt.close()
print("01_experience_distribution.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — Work type distribution")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Çalışma tipine göre dağılım (pasta)
# ══════════════════════════════════════════════════════════════════════════════
wt_counts = postings["formatted_work_type"].value_counts()
colors = sns.color_palette("Blues_d", len(wt_counts))

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    wt_counts.values,
    labels=wt_counts.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=140,
    pctdistance=0.82
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Çalışma Tipine Göre İlan Dağılımı", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "02_work_type_distribution.png"), dpi=150)
plt.close()
print("02_work_type_distribution.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — Remote vs Office")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Remote vs Ofis
# ══════════════════════════════════════════════════════════════════════════════
remote_counts = postings["remote_allowed"].map({1:"Remote", 0:"Ofis / Hibrit"}).value_counts()

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(remote_counts.index, remote_counts.values,
              color=["#1F4E79","#AED6F1"], edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=10)
ax.set_ylabel("İlan Sayısı")
ax.set_title("Remote vs. Ofis/Hibrit İlan Dağılımı", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "03_remote_vs_office.png"), dpi=150)
plt.close()
print("03_remote_vs_office.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — Top states")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — En çok ilan veren 15 eyalet
# ══════════════════════════════════════════════════════════════════════════════
top_states = postings["state"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(top_states.index, top_states.values,
              color=sns.color_palette("Blues_d", 15))
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=8)
ax.set_xlabel("Eyalet")
ax.set_ylabel("İlan Sayısı")
ax.set_title("En Çok İlan Veren 15 Eyalet", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "04_top_states.png"), dpi=150)
plt.close()
print("04_top_states.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — Salary by experience")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Deneyim seviyesine göre medyan maaş
# ══════════════════════════════════════════════════════════════════════════════
sal_data = postings[postings["normalized_salary"].notna()].copy()
exp_order_sal = ["Entry level","Associate","Mid-Senior level","Director","Executive"]
sal_exp = (sal_data[sal_data["formatted_experience_level"].isin(exp_order_sal)]
           .groupby("formatted_experience_level")["normalized_salary"]
           .median()
           .reindex(exp_order_sal)
           .dropna())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(sal_exp.index, sal_exp.values,
              color=sns.color_palette("Blues_d", len(sal_exp)))
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in sal_exp.values], padding=4, fontsize=9)
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Deneyim Seviyesine Göre Medyan Yıllık Maaş", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "05_salary_by_experience.png"), dpi=150)
plt.close()
print("05_salary_by_experience.png kaydedildi.")
tracker.done(7)

tracker.start(8, "Plot 6 — Remote rate by company size")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 6 — Şirket büyüklüğüne göre remote oran
# ══════════════════════════════════════════════════════════════════════════════
size_order = ["1-10","11-50","51-200","201-500","501-1K","1K-5K","5K-10K","10K+"]
remote_by_size = (postings.groupby("company_size_label")["remote_allowed"]
                  .mean()
                  .mul(100)
                  .reindex(size_order)
                  .dropna())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(remote_by_size.index, remote_by_size.values,
              color=sns.color_palette("Blues_d", len(remote_by_size)))
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlabel("Şirket Büyüklüğü (Çalışan Sayısı)")
ax.set_ylabel("Remote İlan Oranı (%)")
ax.set_title("Şirket Büyüklüğüne Göre Remote İlan Oranı", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "06_remote_by_company_size.png"), dpi=150)
plt.close()
print("06_remote_by_company_size.png kaydedildi.")
tracker.done(8)

tracker.finish()