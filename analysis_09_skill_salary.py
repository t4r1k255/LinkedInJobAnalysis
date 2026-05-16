import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os
from scipy import stats

DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "normalized_salary", "formatted_experience_level", "remote_allowed"],
    low_memory=False
).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

job_skills     = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
skills         = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))

postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

sample_ids      = set(postings["job_id"])
job_skills_f    = job_skills[job_skills["job_id"].isin(sample_ids)]
job_skills_full = job_skills_f.merge(skills, on="skill_abr", how="left")
job_ind_f       = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full    = job_ind_f.merge(industries, on="industry_id", how="left")

# Maaş verisi olan ilanlar
sal_df = postings[postings["normalized_salary"].notna()].copy()
skill_sal = job_skills_full.merge(sal_df[["job_id","normalized_salary"]], on="job_id", how="inner")

print(f"Maaş verisi olan ilan: {len(sal_df):,}")
print(f"Skill-maaş eşleşme: {len(skill_sal):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — Skill premium: skill olan vs olmayan maaş farkı
# ══════════════════════════════════════════════════════════════════════════════
common_skills = job_skills_full["skill_name"].value_counts()
common_skills = common_skills[common_skills >= 200].index.tolist()

premiums = []
for skill in common_skills:
    has_skill    = sal_df[sal_df["job_id"].isin(
                      job_skills_full[job_skills_full["skill_name"]==skill]["job_id"]
                   )]["normalized_salary"].dropna()
    no_skill     = sal_df[~sal_df["job_id"].isin(
                      job_skills_full[job_skills_full["skill_name"]==skill]["job_id"]
                   )]["normalized_salary"].dropna()
    if len(has_skill) >= 50 and len(no_skill) >= 50:
        premium = has_skill.median() - no_skill.median()
        _, pval = stats.mannwhitneyu(has_skill, no_skill, alternative="two-sided")
        premiums.append({
            "skill": skill,
            "premium": premium,
            "median_with": has_skill.median(),
            "median_without": no_skill.median(),
            "n": len(has_skill),
            "pval": pval
        })

premium_df = pd.DataFrame(premiums)
premium_df = premium_df[premium_df["pval"] < 0.05].sort_values("premium", ascending=False)

top15_pos = premium_df.head(15)
top15_neg = premium_df.tail(15).sort_values("premium")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

bars = axes[0].barh(top15_pos["skill"][::-1], top15_pos["premium"][::-1],
                    color=sns.color_palette("Blues_d", 15)[::-1])
axes[0].bar_label(bars, labels=[f"+${v:,.0f}" for v in top15_pos["premium"][::-1]],
                  padding=4, fontsize=8)
axes[0].set_title("En Yüksek Maaş Primi Veren 15 Skill", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Maaş Primi (USD)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))

bars2 = axes[1].barh(top15_neg["skill"], top15_neg["premium"],
                     color=sns.color_palette("Blues_d", 15))
axes[1].bar_label(bars2, labels=[f"${v:,.0f}" for v in top15_neg["premium"]],
                  padding=4, fontsize=8)
axes[1].set_title("En Düşük Maaş İlişkili 15 Skill", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Maaş Farkı (USD)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))

fig.suptitle("Skill Varlığının Maaş Üzerindeki Etkisi (Mann-Whitney U, p<0.05)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "42_skill_salary_premium.png"), dpi=150, bbox_inches="tight")
plt.close()
print("42_skill_salary_premium.png kaydedildi.")
print(f"İstatistiksel anlamlı skill sayısı: {len(premium_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Skill başına medyan maaş (boxplot, top 10)
# ══════════════════════════════════════════════════════════════════════════════
top10_skills = job_skills_full["skill_name"].value_counts().head(10).index.tolist()
box_data = skill_sal[skill_sal["skill_name"].isin(top10_skills)]
skill_medians = box_data.groupby("skill_name")["normalized_salary"].median().sort_values()

fig, ax = plt.subplots(figsize=(13, 6))
skill_order = skill_medians.index.tolist()
bp = ax.boxplot(
    [box_data[box_data["skill_name"]==s]["normalized_salary"].dropna().values
     for s in skill_order],
    labels=skill_order,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    flierprops=dict(marker="o", markersize=2, alpha=0.3),
    whiskerprops=dict(linewidth=1),
    capprops=dict(linewidth=1)
)
colors = sns.color_palette("Blues_d", len(skill_order))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

ax.set_ylabel("Yıllık Maaş (USD)")
ax.set_title("Top 10 Skill'e Göre Maaş Dağılımı (Boxplot)",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "43_skill_salary_boxplot.png"), dpi=150)
plt.close()
print("43_skill_salary_boxplot.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Deneyim seviyesi × Skill kombinasyonu maaş analizi
# ══════════════════════════════════════════════════════════════════════════════
exp_order   = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
top6_skills = job_skills_full["skill_name"].value_counts().head(6).index.tolist()

exp_skill_sal = (skill_sal
                 .merge(postings[["job_id","formatted_experience_level"]], on="job_id", how="left")
                 .query("formatted_experience_level in @exp_order and skill_name in @top6_skills"))

exp_skill_median = (exp_skill_sal.groupby(["formatted_experience_level","skill_name"])["normalized_salary"]
                    .median()
                    .unstack()
                    .reindex(exp_order))

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(exp_skill_median / 1000, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Medyan Maaş ($K)"})
ax.set_title("Deneyim × Skill Kombinasyonunda Medyan Maaş ($K)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "44_exp_skill_salary_heatmap.png"), dpi=150)
plt.close()
print("44_exp_skill_salary_heatmap.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Sektör × Skill maaş matrisi
# ══════════════════════════════════════════════════════════════════════════════
top5_ind    = job_ind_full["industry_name"].value_counts().head(5).index.tolist()
top6_skills2= job_skills_full["skill_name"].value_counts().head(6).index.tolist()

ind_skill_sal = (skill_sal
                 .merge(job_ind_full[["job_id","industry_name"]], on="job_id", how="inner")
                 .query("industry_name in @top5_ind and skill_name in @top6_skills2"))

ind_skill_median = (ind_skill_sal.groupby(["industry_name","skill_name"])["normalized_salary"]
                    .median()
                    .unstack()
                    .reindex(top5_ind))

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(ind_skill_median / 1000, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Medyan Maaş ($K)"})
ax.set_title("Sektör × Skill Kombinasyonunda Medyan Maaş ($K) — Top 5 Sektör",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "45_industry_skill_salary_heatmap.png"), dpi=150)
plt.close()
print("45_industry_skill_salary_heatmap.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Skill kombinasyon sayısı vs maaş
# ══════════════════════════════════════════════════════════════════════════════
skill_count_per_job = (job_skills_full.groupby("job_id")["skill_name"]
                       .count()
                       .reset_index(name="skill_count"))
skill_count_sal = skill_count_per_job.merge(
    sal_df[["job_id","normalized_salary"]], on="job_id", how="inner")

skill_count_sal["skill_group"] = skill_count_sal["skill_count"].clip(upper=8).astype(str)
skill_count_sal.loc[skill_count_sal["skill_count"] >= 8, "skill_group"] = "8+"
order = [str(i) for i in range(1, 8)] + ["8+"]

sc_grouped = (skill_count_sal.groupby("skill_group")["normalized_salary"]
              .agg(["median","count"])
              .reindex(order)
              .dropna())

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(sc_grouped.index, sc_grouped["median"],
              color=sns.color_palette("Blues_d", len(sc_grouped)), edgecolor="white")
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in sc_grouped["median"]],
             padding=4, fontsize=9)
ax.set_xlabel("İlanda İstenen Skill Sayısı")
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("İstenen Skill Sayısına Göre Medyan Maaş",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))

ax2 = ax.twinx()
ax2.plot(range(len(sc_grouped)), sc_grouped["count"], color="#1F4E79",
         linewidth=2, marker="o", markersize=5, linestyle="--")
ax2.set_ylabel("İlan Sayısı", color="#1F4E79")
ax2.tick_params(axis="y", labelcolor="#1F4E79")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "46_skill_count_salary.png"), dpi=150)
plt.close()
print("46_skill_count_salary.png kaydedildi.")

print("\nTüm skill-maaş grafikleri outputs/ klasörüne kaydedildi.")
