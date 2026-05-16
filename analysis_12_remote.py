import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
from utils_progress import ProgressBar, StepTracker

DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
RANDOM_SEED = 42
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=7, script_name="analysis_12_remote.py — Remote Work Analysis")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=6, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "company_id", "normalized_salary", "formatted_experience_level",
             "remote_allowed", "formatted_work_type", "location"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")
companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                             usecols=["company_id","company_size"])
_bar.step("companies.csv")
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
_bar.step("job_industries.csv")
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
_bar.step("industries.csv")
job_skills     = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
_bar.step("job_skills.csv")
skills         = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
_bar.step("skills.csv")

postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None
postings["state"] = postings["location"].str.extract(r",\s*([A-Z]{2})\s*$")

size_map = {1:"1-10", 2:"11-50", 3:"51-200", 4:"201-500",
            5:"501-1K", 6:"1K-5K", 7:"5K-10K", 8:"10K+"}
companies["size_label"] = companies["company_size"].map(size_map)
postings = postings.merge(companies, on="company_id", how="left")

sample_ids      = set(postings["job_id"])
job_ind_f       = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full    = job_ind_f.merge(industries, on="industry_id", how="left")
job_skills_f    = job_skills[job_skills["job_id"].isin(sample_ids)]
job_skills_full = job_skills_f.merge(skills, on="skill_abr", how="left")

remote_df   = postings[postings["remote_allowed"] == 1]
office_df   = postings[postings["remote_allowed"] == 0]
sal_df      = postings[postings["normalized_salary"].notna()]

print(f"Remote ilan: {len(remote_df):,} ({len(remote_df)/len(postings)*100:.1f}%)")
print(f"Ofis/Hibrit: {len(office_df):,} ({len(office_df)/len(postings)*100:.1f}%)")
_bar.finish()
tracker.done(1)

tracker.start(2, "Cleaning & joins")
tracker.done(2)

print(f"Maaş verisi olan: {len(sal_df):,}")

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(3, "Plot 1 — Remote salary premium by industry")
# GRAFİK 1 — Remote maaş primi: sektör bazında remote vs ofis maaş farkı
# ══════════════════════════════════════════════════════════════════════════════
top10_ind = job_ind_full["industry_name"].value_counts().head(10).index.tolist()
ind_remote_sal = (sal_df[["job_id","normalized_salary","remote_allowed"]]
                  .merge(job_ind_full[["job_id","industry_name"]], on="job_id", how="inner")
                  .query("industry_name in @top10_ind"))

remote_premium = []
for ind in top10_ind:
    sub = ind_remote_sal[ind_remote_sal["industry_name"] == ind]
    r = sub[sub["remote_allowed"]==1]["normalized_salary"].median()
    o = sub[sub["remote_allowed"]==0]["normalized_salary"].median()
    if pd.notna(r) and pd.notna(o):
        remote_premium.append({
            "industry": ind,
            "remote_median": r,
            "office_median": o,
            "premium": r - o,
            "premium_pct": (r - o) / o * 100
        })

rp_df = pd.DataFrame(remote_premium).sort_values("premium")

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#1F4E79" if v >= 0 else "#AED6F1" for v in rp_df["premium"]]
bars = ax.barh(rp_df["industry"], rp_df["premium"], color=colors)
ax.bar_label(bars,
             labels=[f"+${v:,.0f}" if v >= 0 else f"${v:,.0f}" for v in rp_df["premium"]],
             padding=4, fontsize=9)
ax.axvline(0, color="white", linewidth=1, alpha=0.5)
ax.set_xlabel("Remote Maaş Primi (USD)")
ax.set_title("Sektöre Göre Remote Çalışma Maaş Primi\n(Remote Medyan − Ofis Medyan)",
             fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "57_remote_salary_premium_by_industry.png"), dpi=150)
plt.close()
print("57_remote_salary_premium_by_industry.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — Remote vs Office skills")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Remote ilanlarında en çok istenen skill'ler vs ofis ilanları
# ══════════════════════════════════════════════════════════════════════════════
top15_skills = job_skills_full["skill_name"].value_counts().head(15).index.tolist()

remote_skills = (job_skills_full[job_skills_full["job_id"].isin(set(remote_df["job_id"])) &
                                  job_skills_full["skill_name"].isin(top15_skills)]
                 ["skill_name"].value_counts()
                 .div(len(remote_df)) * 100)

office_skills = (job_skills_full[job_skills_full["job_id"].isin(set(office_df["job_id"])) &
                                  job_skills_full["skill_name"].isin(top15_skills)]
                 ["skill_name"].value_counts()
                 .div(len(office_df)) * 100)

skill_compare = pd.DataFrame({"Remote": remote_skills, "Ofis/Hibrit": office_skills}).fillna(0)
skill_compare["diff"] = skill_compare["Remote"] - skill_compare["Ofis/Hibrit"]
skill_compare = skill_compare.sort_values("diff")

fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(skill_compare))
w = 0.35
ax.barh(x + w/2, skill_compare["Remote"], w, label="Remote", color="#1F4E79")
ax.barh(x - w/2, skill_compare["Ofis/Hibrit"], w, label="Ofis/Hibrit", color="#AED6F1")
ax.set_yticks(x)
ax.set_yticklabels(skill_compare.index, fontsize=9)
ax.set_xlabel("İlan Başına Skill Oranı (%)")
ax.set_title("Remote vs. Ofis/Hibrit İlanlarında Skill Dağılımı",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "58_remote_vs_office_skills.png"), dpi=150)
plt.close()
print("58_remote_vs_office_skills.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — Remote rate × Company size")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Remote oran × şirket büyüklüğü × sektör (çift eksenli)
# ══════════════════════════════════════════════════════════════════════════════
size_order = ["1-10","11-50","51-200","201-500","501-1K","1K-5K","5K-10K","10K+"]
size_remote = (postings[postings["size_label"].notna()]
               .groupby("size_label")["remote_allowed"]
               .agg(["mean","count"])
               .reindex(size_order)
               .dropna())
size_remote["remote_pct"] = size_remote["mean"] * 100

# Sadece mevcut size kategorilerini kullan
valid_sizes = size_remote.index.tolist()

size_sal_remote = (sal_df[sal_df["size_label"].notna()]
                   .groupby(["size_label","remote_allowed"])["normalized_salary"]
                   .median()
                   .unstack()
                   .reindex(valid_sizes))
size_sal_remote.columns = ["office_sal", "remote_sal"]

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
x = np.arange(len(valid_sizes))

ax1.bar(x, size_remote["remote_pct"], color="#AED6F1", alpha=0.7, label="Remote Oran (%)")
ax1.set_ylabel("Remote İlan Oranı (%)", color="#1F4E79")
ax1.set_xticks(x)
ax1.set_xticklabels(valid_sizes, rotation=15)

if "remote_sal" in size_sal_remote.columns:
    ax2.plot(x, size_sal_remote["remote_sal"].reindex(valid_sizes),
             color="#1F4E79", linewidth=2.5, marker="o", markersize=7, label="Remote Maaş")
    ax2.plot(x, size_sal_remote["office_sal"].reindex(valid_sizes),
             color="#2E75B6", linewidth=2, marker="s", markersize=6,
             linestyle="--", label="Ofis Maaş")
ax2.set_ylabel("Medyan Maaş (USD)", color="#1F4E79")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title("Şirket Büyüklüğüne Göre Remote Oran ve Maaş",
              fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "59_remote_by_size_salary.png"), dpi=150)
plt.close()
print("59_remote_by_size_salary.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — State remote scatter")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Eyalet bazında remote oran ve maaş scatter (top 20 eyalet)
# ══════════════════════════════════════════════════════════════════════════════
top20_states = postings["state"].value_counts().head(20).index.tolist()
state_analysis = (postings[postings["state"].isin(top20_states)]
                  .groupby("state")
                  .agg(
                      remote_pct=("remote_allowed", "mean"),
                      median_salary=("normalized_salary", "median"),
                      count=("job_id", "count")
                  )
                  .dropna(subset=["median_salary"]))
state_analysis["remote_pct"] *= 100

fig, ax = plt.subplots(figsize=(11, 7))
scatter = ax.scatter(
    state_analysis["remote_pct"],
    state_analysis["median_salary"],
    s=state_analysis["count"] / 8,
    c=state_analysis["median_salary"],
    cmap="Blues",
    alpha=0.85,
    edgecolors="#1F4E79",
    linewidth=0.8
)
for state, row in state_analysis.iterrows():
    ax.annotate(state, (row["remote_pct"], row["median_salary"]),
                fontsize=9, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points")
plt.colorbar(scatter, ax=ax, label="Medyan Maaş (USD)")
ax.set_xlabel("Remote İlan Oranı (%)")
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Eyalet: Remote Oran vs. Medyan Maaş\n(Daire = ilan sayısı)",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))

# Trend çizgisi
z = np.polyfit(state_analysis["remote_pct"], state_analysis["median_salary"], 1)
p = np.poly1d(z)
x_line = np.linspace(state_analysis["remote_pct"].min(), state_analysis["remote_pct"].max(), 100)
ax.plot(x_line, p(x_line), "--", color="#1F4E79", alpha=0.5, linewidth=1.5, label="Trend")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "60_remote_state_scatter.png"), dpi=150)
plt.close()
print("60_remote_state_scatter.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — Remote × Experience salary")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Remote × Deneyim × Maaş (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
exp_remote_sal = (sal_df[sal_df["formatted_experience_level"].isin(exp_order)]
                  .groupby(["formatted_experience_level","remote_allowed"])["normalized_salary"]
                  .median()
                  .unstack()
                  .reindex(exp_order))
exp_remote_sal.columns = ["Ofis/Hibrit", "Remote"]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(exp_order))
w = 0.35
bars1 = ax.bar(x - w/2, exp_remote_sal["Ofis/Hibrit"], w, label="Ofis/Hibrit",
               color="#AED6F1", edgecolor="white")
bars2 = ax.bar(x + w/2, exp_remote_sal["Remote"], w, label="Remote",
               color="#1F4E79", edgecolor="white")
ax.bar_label(bars1, labels=[f"${v:,.0f}" if pd.notna(v) else "" for v in exp_remote_sal["Ofis/Hibrit"]],
             padding=3, fontsize=7, rotation=45)
ax.bar_label(bars2, labels=[f"${v:,.0f}" if pd.notna(v) else "" for v in exp_remote_sal["Remote"]],
             padding=3, fontsize=7, rotation=45)
ax.set_xticks(x)
ax.set_xticklabels(exp_order, rotation=15, ha="right")
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Deneyim Seviyesi × Çalışma Türü Maaş Karşılaştırması",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "61_remote_exp_salary_grouped.png"), dpi=150)
plt.close()
print("61_remote_exp_salary_grouped.png kaydedildi.")
tracker.done(7)
tracker.finish()

print("\nTüm remote analiz grafikleri outputs/ klasörüne kaydedildi.")