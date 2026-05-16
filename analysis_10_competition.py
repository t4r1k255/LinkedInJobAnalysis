import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from utils_progress import ProgressBar, StepTracker

DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
RANDOM_SEED = 42
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=7, script_name="analysis_10_competition.py — Competition Analysis")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=5, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "title", "applies", "views", "normalized_salary",
             "formatted_experience_level", "remote_allowed", "location"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")
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

sample_ids      = set(postings["job_id"])
job_ind_f       = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full    = job_ind_f.merge(industries, on="industry_id", how="left")
job_skills_f    = job_skills[job_skills["job_id"].isin(sample_ids)]
job_skills_full = job_skills_f.merge(skills, on="skill_abr", how="left")

# Rekabet skoru: applies / views
comp_df = postings[postings["applies"].notna() & postings["views"].notna()].copy()
comp_df = comp_df[comp_df["views"] > 0]
comp_df["competition_score"] = comp_df["applies"] / comp_df["views"]
comp_df["title_clean"] = comp_df["title"].str.lower().str.strip()

print(f"Rekabet verisi olan ilan: {len(comp_df):,}")
print(f"Ortalama applies: {comp_df['applies'].mean():.1f}")
print(f"Ortalama views: {comp_df['views'].mean():.1f}")
_bar.finish()
tracker.done(1)

tracker.start(2, "Computing competition scores")
print(f"Ortalama rekabet skoru: {comp_df['competition_score'].mean():.3f}")
tracker.done(2)

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(3, "Plot 1 — Most competitive titles")
# GRAFİK 1 — En rekabetçi 15 iş unvanı (başvuru/görüntülenme)
# ══════════════════════════════════════════════════════════════════════════════
title_comp = (comp_df.groupby("title_clean")
              .agg(avg_score=("competition_score","mean"),
                   avg_applies=("applies","mean"),
                   count=("job_id","count"))
              .query("count >= 20")
              .sort_values("avg_score", ascending=False)
              .head(15))

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(title_comp.index[::-1], title_comp["avg_score"][::-1],
               color=sns.color_palette("Blues_d", 15)[::-1])
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
ax.set_xlabel("Ortalama Başvuru/Görüntülenme Oranı")
ax.set_title("En Rekabetçi 15 İş Unvanı\n(Başvuru / Görüntülenme, min. 20 ilan)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "47_most_competitive_titles.png"), dpi=150)
plt.close()
print("47_most_competitive_titles.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — Competition by industry")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Sektöre göre rekabet skoru
# ══════════════════════════════════════════════════════════════════════════════
ind_comp = (comp_df[["job_id","competition_score"]]
            .merge(job_ind_full[["job_id","industry_name"]], on="job_id", how="inner"))
top12_ind = ind_comp["industry_name"].value_counts().head(12).index.tolist()
ind_comp_f = ind_comp[ind_comp["industry_name"].isin(top12_ind)]

ind_comp_med = (ind_comp_f.groupby("industry_name")["competition_score"]
                .median().sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(ind_comp_med.index, ind_comp_med.values,
               color=sns.color_palette("Blues_d", len(ind_comp_med)))
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
ax.set_xlabel("Medyan Rekabet Skoru (Başvuru/Görüntülenme)")
ax.set_title("Sektöre Göre Medyan Rekabet Skoru",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "48_competition_by_industry.png"), dpi=150)
plt.close()
print("48_competition_by_industry.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — Competition by experience")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Deneyim seviyesine göre rekabet
# ══════════════════════════════════════════════════════════════════════════════
exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
exp_comp  = (comp_df[comp_df["formatted_experience_level"].isin(exp_order)]
             .groupby("formatted_experience_level")["competition_score"]
             .agg(["median","mean","count"])
             .reindex(exp_order))

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(exp_order))
bars = ax.bar(x, exp_comp["median"],
              color=sns.color_palette("Blues_d", len(exp_order)), edgecolor="white")
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(exp_order, rotation=15, ha="right")
ax.set_ylabel("Medyan Rekabet Skoru")
ax.set_title("Deneyim Seviyesine Göre İlan Rekabeti",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "49_competition_by_experience.png"), dpi=150)
plt.close()
print("49_competition_by_experience.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — Remote vs Office competition")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Remote vs ofis rekabet karşılaştırması
# ══════════════════════════════════════════════════════════════════════════════
comp_df["work_type"] = comp_df["remote_allowed"].map({1:"Remote", 0:"Ofis/Hibrit"})
remote_comp = comp_df.groupby("work_type")["competition_score"].agg(["median","mean","count"])

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(remote_comp.index, remote_comp["median"],
              color=["#1F4E79","#AED6F1"], edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=12)
ax.set_ylabel("Medyan Rekabet Skoru")
ax.set_title("Remote vs. Ofis/Hibrit İlan Rekabeti",
             fontsize=13, fontweight="bold", pad=12)
for i, (idx, row) in enumerate(remote_comp.iterrows()):
    ax.text(i, row["median"] * 0.5, f"n={int(row['count']):,}",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "50_competition_remote_vs_office.png"), dpi=150)
plt.close()
print("50_competition_remote_vs_office.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — Low competition opportunities")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — En az rekabetçi (fırsat) iş unvanları
# ══════════════════════════════════════════════════════════════════════════════
low_comp = (comp_df.groupby("title_clean")
            .agg(avg_score=("competition_score","mean"),
                 avg_salary=("normalized_salary","median"),
                 count=("job_id","count"))
            .query("count >= 20 and avg_salary > 60000")
            .sort_values("avg_score")
            .head(15))

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(
    low_comp["avg_score"],
    low_comp["avg_salary"],
    s=low_comp["count"] * 3,
    c=low_comp["avg_salary"],
    cmap="Blues",
    alpha=0.8,
    edgecolors="#1F4E79",
    linewidth=0.8
)
for _, row in low_comp.iterrows():
    ax.annotate(row.name[:30],
                (row["avg_score"], row["avg_salary"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points")
plt.colorbar(scatter, ax=ax, label="Medyan Maaş (USD)")
ax.set_xlabel("Rekabet Skoru (düşük = daha az rekabet)")
ax.set_ylabel("Medyan Maaş (USD)")
ax.set_title("Düşük Rekabet + Yüksek Maaş Fırsatları\n(min. 20 ilan, min. $60K)",
             fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "51_low_competition_opportunities.png"), dpi=150)
plt.close()
print("51_low_competition_opportunities.png kaydedildi.")
tracker.done(7)
tracker.finish()

print("\nTüm rekabet analizi grafikleri outputs/ klasörüne kaydedildi.")
