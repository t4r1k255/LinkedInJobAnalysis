import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
from utils_progress import ProgressBar, StepTracker
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ── AYARLAR ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
RANDOM_SEED = 42
os.makedirs(OUTPUT_PATH, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=7, script_name="analysis_04_crosssection.py — Cross-Section")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=6, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "location", "remote_allowed", "normalized_salary",
             "formatted_experience_level", "formatted_work_type"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")

companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                             usecols=["company_id", "name", "company_size"])
_bar.step("companies.csv")
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
_bar.step("job_industries.csv")
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
_bar.step("industries.csv")
job_skills     = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
_bar.step("job_skills.csv")
skills         = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
_bar.step("skills.csv")

# Temizleme
postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None
postings["state"] = postings["location"].str.extract(r",\s*([A-Z]{2})\s*$")

# Maaş aralığı sütunu
bins   = [0, 40_000, 70_000, 100_000, 140_000, 200_000, float("inf")]
labels = ["<$40K", "$40K-70K", "$70K-100K", "$100K-140K", "$140K-200K", "$200K+"]
postings["salary_band"] = pd.cut(postings["normalized_salary"], bins=bins, labels=labels)

# Join'ler
sample_ids      = set(postings["job_id"])
job_ind_f       = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full    = job_ind_f.merge(industries, on="industry_id", how="left")
job_skills_f    = job_skills[job_skills["job_id"].isin(sample_ids)]
job_skills_full = job_skills_f.merge(skills, on="skill_abr", how="left")

_bar.finish()
tracker.done(1)

tracker.start(2, "Cleaning, salary bands & joins")
# cleaning...
tracker.done(2)

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(3, "Plot 1 — Salary band × Experience")
# GRAFİK 1 — Maaş bandı × Deneyim seviyesi (ısı haritası)
# ══════════════════════════════════════════════════════════════════════════════
exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
sal_exp = (postings[postings["salary_band"].notna() &
                    postings["formatted_experience_level"].isin(exp_order)]
           .groupby(["formatted_experience_level", "salary_band"], observed=True)
           .size()
           .unstack(fill_value=0)
           .reindex(exp_order))

sal_exp_pct = sal_exp.div(sal_exp.sum(axis=1), axis=0).mul(100).round(1)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(sal_exp_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Deneyim Seviyesi × Maaş Bandı Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Maaş Bandı")
ax.set_ylabel("")
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "17_salary_band_experience.png"), dpi=150)
plt.close()
print("17_salary_band_experience.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — Sector × Salary band")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Sektör × Maaş bandı (ısı haritası, top 10 sektör)
# ══════════════════════════════════════════════════════════════════════════════
top10_ind = job_ind_full["industry_name"].value_counts().head(10).index.tolist()
ind_sal = (postings[["job_id", "salary_band"]]
           .merge(job_ind_full[["job_id", "industry_name"]], on="job_id")
           .query("industry_name in @top10_ind")
           .dropna(subset=["salary_band"]))

ind_sal_pivot = (ind_sal.groupby(["industry_name", "salary_band"], observed=True)
                 .size()
                 .unstack(fill_value=0)
                 .reindex(top10_ind))
ind_sal_pct = ind_sal_pivot.div(ind_sal_pivot.sum(axis=1), axis=0).mul(100).round(1)

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(ind_sal_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Sektör × Maaş Bandı Dağılımı — Top 10 Sektör (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Maaş Bandı")
ax.set_ylabel("")
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "18_salary_band_industry.png"), dpi=150)
plt.close()
print("18_salary_band_industry.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — State bubble chart")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Eyalet × Remote oran + Medyan maaş (bubble chart)
# ══════════════════════════════════════════════════════════════════════════════
top15_states = postings["state"].value_counts().head(15).index.tolist()
state_df = (postings[postings["state"].isin(top15_states)]
            .groupby("state")
            .agg(
                count=("job_id", "count"),
                remote_pct=("remote_allowed", "mean"),
                median_salary=("normalized_salary", "median")
            ).reset_index())
state_df["remote_pct"] = state_df["remote_pct"] * 100
state_df = state_df.dropna(subset=["median_salary"])

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(
    state_df["remote_pct"],
    state_df["median_salary"],
    s=state_df["count"] / 10,
    c=state_df["median_salary"],
    cmap="Blues",
    alpha=0.8,
    edgecolors="#1F4E79",
    linewidth=0.8
)
for _, row in state_df.iterrows():
    ax.annotate(row["state"],
                (row["remote_pct"], row["median_salary"]),
                fontsize=9, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points")

plt.colorbar(scatter, ax=ax, label="Medyan Maaş (USD)")
ax.set_xlabel("Remote İlan Oranı (%)")
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Eyalet Bazında Remote Oran vs. Medyan Maaş\n(Daire boyutu = ilan sayısı)",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "19_state_remote_salary_bubble.png"), dpi=150)
plt.close()
print("19_state_remote_salary_bubble.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — Work type × Experience")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Çalışma tipi × Deneyim seviyesi (yığılmış bar)
# ══════════════════════════════════════════════════════════════════════════════
exp_order2 = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
work_types = ["Full-time", "Contract", "Part-time"]

wt_exp = (postings[postings["formatted_experience_level"].isin(exp_order2) &
                   postings["formatted_work_type"].isin(work_types)]
          .groupby(["formatted_experience_level", "formatted_work_type"])
          .size()
          .unstack(fill_value=0)
          .reindex(exp_order2))
wt_exp_pct = wt_exp.div(wt_exp.sum(axis=1), axis=0).mul(100)

fig, ax = plt.subplots(figsize=(11, 5))
wt_exp_pct.plot(kind="bar", stacked=True, ax=ax,
                color=["#1F4E79", "#2E75B6", "#AED6F1"],
                edgecolor="white", width=0.6)
ax.set_xlabel("")
ax.set_ylabel("Oran (%)")
ax.set_title("Deneyim Seviyesine Göre Çalışma Tipi Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(title="Çalışma Tipi", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "20_worktype_experience.png"), dpi=150)
plt.close()
print("20_worktype_experience.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — Skill salary band heatmap")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Maaş bandına göre skill profili (ısı haritası)
# ══════════════════════════════════════════════════════════════════════════════
top8_skills = job_skills_full["skill_name"].value_counts().head(8).index.tolist()
skill_sal = (postings[["job_id", "salary_band"]]
             .merge(job_skills_full[["job_id", "skill_name"]], on="job_id")
             .query("skill_name in @top8_skills")
             .dropna(subset=["salary_band"]))

skill_sal_pivot = (skill_sal.groupby(["salary_band", "skill_name"], observed=True)
                   .size()
                   .unstack(fill_value=0)
                   .reindex(labels))
skill_sal_pct = skill_sal_pivot.div(skill_sal_pivot.sum(axis=1), axis=0).mul(100).round(1)

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(skill_sal_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Maaş Bandına Göre Top 8 Skill Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Maaş Bandı")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "21_skill_salary_band.png"), dpi=150)
plt.close()
print("21_skill_salary_band.png kaydedildi.")
tracker.done(7)
tracker.finish()

print("\nTüm kesitsel analiz grafikleri outputs/ klasörüne kaydedildi.")