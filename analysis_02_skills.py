import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from utils_progress import ProgressBar, StepTracker

# ── AYARLAR ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
RANDOM_SEED = 42
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=7, script_name="analysis_02_skills.py — Skill Demand")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=5, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "formatted_experience_level", "normalized_salary"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")

postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

job_skills     = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
_bar.step("job_skills.csv")
skills         = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
_bar.step("skills.csv")
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
_bar.step("job_industries.csv")
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
_bar.step("industries.csv")

# Sample'daki job_id'lerle filtrele
sample_ids = set(postings["job_id"])
job_skills_f = job_skills[job_skills["job_id"].isin(sample_ids)]
job_skills_full = job_skills_f.merge(skills, on="skill_abr", how="left")

job_ind_f = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full = job_ind_f.merge(industries, on="industry_id", how="left")

_bar.finish()
tracker.done(1)

tracker.start(2, "Filtering & joining skill/industry data")


# ══════════════════════════════════════════════════════════════════════════════
tracker.done(2)

tracker.start(3, "Plot 1 — Top 20 skills")
# GRAFİK 1 — En çok istenen 20 skill
# ══════════════════════════════════════════════════════════════════════════════
top_skills = job_skills_full["skill_name"].value_counts().head(20)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top_skills.index[::-1], top_skills.values[::-1],
               color=sns.color_palette("Blues_d", 20)[::-1])
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title("En Çok İstenen 20 Skill", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "07_top_20_skills.png"), dpi=150)
plt.close()
print("07_top_20_skills.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — Top industries")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — En çok ilan veren 15 sektör
# ══════════════════════════════════════════════════════════════════════════════
top_industries = job_ind_full["industry_name"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top_industries.index[::-1], top_industries.values[::-1],
               color=sns.color_palette("Blues_d", 15)[::-1])
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title("En Çok İlan Veren 15 Sektör", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "08_top_industries.png"), dpi=150)
plt.close()
print("08_top_industries.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — Skill by experience heatmap")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Deneyim seviyesine göre en popüler 10 skill (ısı haritası)
# ══════════════════════════════════════════════════════════════════════════════
exp_skill = (job_skills_full
             .merge(postings[["job_id","formatted_experience_level"]], on="job_id", how="left"))

exp_levels = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
exp_skill_f = exp_skill[exp_skill["formatted_experience_level"].isin(exp_levels)]

top10_skills = job_skills_full["skill_name"].value_counts().head(10).index.tolist()
exp_skill_top = exp_skill_f[exp_skill_f["skill_name"].isin(top10_skills)]

heatmap_data = (exp_skill_top
                .groupby(["formatted_experience_level", "skill_name"])
                .size()
                .unstack(fill_value=0)
                .reindex(exp_levels))

# Normalize: her seviyenin toplam ilanına göre yüzde
heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0).mul(100).round(1)

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(heatmap_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Deneyim Seviyesine Göre Top 10 Skill Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "09_skill_by_experience_heatmap.png"), dpi=150)
plt.close()
print("09_skill_by_experience_heatmap.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — Top paying skills")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Maaşla en çok ilişkili 15 skill (medyan maaş)
# ══════════════════════════════════════════════════════════════════════════════
skill_salary = (job_skills_full
                .merge(postings[["job_id","normalized_salary"]], on="job_id", how="left"))
skill_salary = skill_salary[skill_salary["normalized_salary"].notna()]

# En az 100 ilanda geçen skill'ler
skill_counts = skill_salary["skill_name"].value_counts()
common_skills = skill_counts[skill_counts >= 100].index
skill_salary_f = skill_salary[skill_salary["skill_name"].isin(common_skills)]

top_paying = (skill_salary_f
              .groupby("skill_name")["normalized_salary"]
              .median()
              .sort_values(ascending=False)
              .head(15))

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top_paying.index[::-1], top_paying.values[::-1],
               color=sns.color_palette("Blues_d", 15)[::-1])
ax.bar_label(bars,
             labels=[f"${v:,.0f}" for v in top_paying.values[::-1]],
             padding=4, fontsize=9)
ax.set_xlabel("Medyan Yıllık Maaş (USD)")
ax.set_title("En Yüksek Maaşla İlişkili 15 Skill", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "10_top_paying_skills.png"), dpi=150)
plt.close()
print("10_top_paying_skills.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — Skill by industry stacked")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Top 5 sektörde skill dağılımı (yığılmış bar)
# ══════════════════════════════════════════════════════════════════════════════
top5_ind = job_ind_full["industry_name"].value_counts().head(5).index.tolist()
ind_skill = (job_skills_full
             .merge(job_ind_full[["job_id","industry_name"]], on="job_id", how="inner"))
ind_skill_f = ind_skill[ind_skill["industry_name"].isin(top5_ind)]

top8_skills = job_skills_full["skill_name"].value_counts().head(8).index.tolist()
ind_skill_top = ind_skill_f[ind_skill_f["skill_name"].isin(top8_skills)]

stack_data = (ind_skill_top
              .groupby(["industry_name","skill_name"])
              .size()
              .unstack(fill_value=0)
              .reindex(top5_ind))
stack_pct = stack_data.div(stack_data.sum(axis=1), axis=0).mul(100)

fig, ax = plt.subplots(figsize=(13, 6))
stack_pct.plot(kind="bar", stacked=True, ax=ax,
               colormap="Blues", edgecolor="white", width=0.6)
ax.set_xlabel("")
ax.set_ylabel("Skill Oranı (%)")
ax.set_title("Top 5 Sektörde Top 8 Skill Dağılımı (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(title="Skill", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "11_skill_by_industry_stacked.png"), dpi=150)
plt.close()
print("11_skill_by_industry_stacked.png kaydedildi.")
tracker.done(7)
tracker.finish()

print("\nTüm skill grafikleri outputs/ klasörüne kaydedildi.")