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

tracker = StepTracker(total_steps=7, script_name="analysis_11_career_ladder.py — Career Ladder")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=4, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "company_id", "normalized_salary",
             "formatted_experience_level", "remote_allowed", "formatted_work_type"],
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

postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

size_map = {1:"1-10", 2:"11-50", 3:"51-200", 4:"201-500",
            5:"501-1K", 6:"1K-5K", 7:"5K-10K", 8:"10K+"}
companies["size_label"] = companies["company_size"].map(size_map)
postings = postings.merge(companies, on="company_id", how="left")

sample_ids   = set(postings["job_id"])
job_ind_f    = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full = job_ind_f.merge(industries, on="industry_id", how="left")

sal_df   = postings[postings["normalized_salary"].notna()].copy()
exp_order= ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]

_bar.finish()
tracker.done(1)

tracker.start(2, "Cleaning & joins")
tracker.done(2)

print(f"Maaş verisi olan ilan: {len(sal_df):,}")

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(3, "Plot 1 — Career ladder violin")
# GRAFİK 1 — Kariyer merdiveni: deneyim bazında maaş dağılımı (violin)
# ══════════════════════════════════════════════════════════════════════════════
violin_df = sal_df[sal_df["formatted_experience_level"].isin(exp_order)].copy()

fig, ax = plt.subplots(figsize=(13, 6))
parts = ax.violinplot(
    [violin_df[violin_df["formatted_experience_level"]==exp]["normalized_salary"].dropna().values
     for exp in exp_order],
    positions=range(len(exp_order)),
    showmedians=True,
    showextrema=True
)
colors = sns.color_palette("Blues_d", len(exp_order))
for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)

medians = [violin_df[violin_df["formatted_experience_level"]==exp]["normalized_salary"].median()
           for exp in exp_order]
for i, med in enumerate(medians):
    ax.text(i, med + 5000, f"${med:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(range(len(exp_order)))
ax.set_xticklabels(exp_order, rotation=15, ha="right")
ax.set_ylabel("Yıllık Maaş (USD)")
ax.set_title("Kariyer Merdiveni — Deneyim Seviyesine Göre Maaş Dağılımı (Violin)",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "52_career_ladder_violin.png"), dpi=150)
plt.close()
print("52_career_ladder_violin.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — Career ladder by industry")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Sektör bazında kariyer merdiveni (çizgi grafik, top 6 sektör)
# ══════════════════════════════════════════════════════════════════════════════
top6_ind = job_ind_full["industry_name"].value_counts().head(6).index.tolist()
ind_exp_sal = (sal_df[["job_id","normalized_salary","formatted_experience_level"]]
               .merge(job_ind_full[["job_id","industry_name"]], on="job_id", how="inner")
               .query("industry_name in @top6_ind and formatted_experience_level in @exp_order"))

ind_ladder = (ind_exp_sal.groupby(["industry_name","formatted_experience_level"])["normalized_salary"]
              .median()
              .unstack()
              .reindex(columns=exp_order))

fig, ax = plt.subplots(figsize=(13, 6))
colors = sns.color_palette("Blues_d", len(top6_ind))
for i, ind in enumerate(top6_ind):
    if ind in ind_ladder.index:
        row = ind_ladder.loc[ind].dropna()
        if len(row) >= 2:
            ax.plot(row.index, row.values, marker="o", linewidth=2.5,
                    markersize=7, color=colors[i], label=ind)
            ax.annotate(f"${row.iloc[-1]:,.0f}",
                        (row.index[-1], row.iloc[-1]),
                        fontsize=7, xytext=(5, 0), textcoords="offset points")

ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Sektör Bazında Kariyer Merdiveni — Top 6 Sektör",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.legend(fontsize=8, loc="upper left")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "53_career_ladder_by_industry.png"), dpi=150)
plt.close()
print("53_career_ladder_by_industry.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — Company size × Experience matrix")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Şirket büyüklüğü bazında kariyer merdiveni
# ══════════════════════════════════════════════════════════════════════════════
size_order = ["1-10","11-50","51-200","201-500","501-1K","1K-5K","5K-10K"]
size_exp_sal = (sal_df[sal_df["formatted_experience_level"].isin(exp_order) &
                       sal_df["size_label"].isin(size_order)]
                .groupby(["size_label","formatted_experience_level"])["normalized_salary"]
                .median()
                .unstack()
                .reindex(size_order)[exp_order])

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(size_exp_sal / 1000, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Medyan Maaş ($K)"})
ax.set_title("Şirket Büyüklüğü × Deneyim Seviyesi Maaş Matrisi ($K)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Şirket Büyüklüğü")
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "54_career_ladder_company_size.png"), dpi=150)
plt.close()
print("54_career_ladder_company_size.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — Salary growth rate")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Maaş artış hızı (entry → mid-senior, sektör bazında)
# ══════════════════════════════════════════════════════════════════════════════
growth_data = []
for ind in top6_ind:
    subset = ind_exp_sal[ind_exp_sal["industry_name"] == ind]
    entry  = subset[subset["formatted_experience_level"]=="Entry level"]["normalized_salary"].median()
    senior = subset[subset["formatted_experience_level"]=="Mid-Senior level"]["normalized_salary"].median()
    if pd.notna(entry) and pd.notna(senior) and entry > 0:
        growth_data.append({
            "industry": ind,
            "entry_salary": entry,
            "senior_salary": senior,
            "growth_pct": (senior - entry) / entry * 100,
            "growth_abs": senior - entry
        })

growth_df = pd.DataFrame(growth_data).sort_values("growth_pct", ascending=True)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(growth_df["industry"], growth_df["growth_pct"],
               color=sns.color_palette("Blues_d", len(growth_df)))
ax.bar_label(bars, fmt="+%.1f%%", padding=4, fontsize=10)
ax.set_xlabel("Maaş Artışı (Entry → Mid-Senior, %)")
ax.set_title("Sektöre Göre Entry→Mid-Senior Maaş Artış Oranı",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "55_salary_growth_rate.png"), dpi=150)
plt.close()
print("55_salary_growth_rate.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — Remote career ladder")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Remote çalışmanın kariyer merdiveni üzerindeki etkisi
# ══════════════════════════════════════════════════════════════════════════════
remote_ladder = (sal_df[sal_df["formatted_experience_level"].isin(exp_order)]
                 .groupby(["remote_allowed","formatted_experience_level"])["normalized_salary"]
                 .median()
                 .unstack()
                 .reindex(columns=exp_order))

fig, ax = plt.subplots(figsize=(12, 6))
colors_remote = ["#1F4E79", "#AED6F1"]
labels_remote = ["Ofis/Hibrit", "Remote"]
for i, (idx, row) in enumerate(remote_ladder.iterrows()):
    row = row.dropna()
    ax.plot(row.index, row.values, marker="o", linewidth=2.5,
            markersize=8, color=colors_remote[i], label=labels_remote[i])
    for x, y in zip(row.index, row.values):
        ax.annotate(f"${y:,.0f}", (x, y), fontsize=8,
                    xytext=(0, 10 if i==0 else -16), textcoords="offset points",
                    ha="center")

ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Remote vs. Ofis/Hibrit — Kariyer Merdiveni Maaş Karşılaştırması",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.legend(fontsize=10)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "56_remote_career_ladder.png"), dpi=150)
plt.close()
print("56_remote_career_ladder.png kaydedildi.")
tracker.done(7)
tracker.finish()

print("\nTüm kariyer merdiveni grafikleri outputs/ klasörüne kaydedildi.")
