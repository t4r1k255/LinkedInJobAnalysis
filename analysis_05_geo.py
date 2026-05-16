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

tracker = StepTracker(total_steps=7, script_name="analysis_05_geo.py — Geographic Analysis")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=4, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "location", "remote_allowed", "normalized_salary",
             "formatted_experience_level", "formatted_work_type"],
    low_memory=False
).reset_index(drop=True)
_bar.step("postings.csv")

companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                             usecols=["company_id", "name", "company_size", "state", "city"])
_bar.step("companies.csv")
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
_bar.step("job_industries.csv")
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
_bar.step("industries.csv")

# Temizleme
postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None
postings["state"] = postings["location"].str.extract(r",\s*([A-Z]{2})\s*$")
postings["city"]  = postings["location"].str.extract(r"^([^,]+)")

# Join
sample_ids   = set(postings["job_id"])
job_ind_f    = job_industries[job_industries["job_id"].isin(sample_ids)]
job_ind_full = job_ind_f.merge(industries, on="industry_id", how="left")

_bar.finish()
tracker.done(1)

tracker.start(2, "Cleaning & joins")
tracker.done(2)

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(3, "Plot 1 — State median salary")
# GRAFİK 1 — Eyalet bazında medyan maaş (top 20)
# ══════════════════════════════════════════════════════════════════════════════
state_salary = (postings[postings["normalized_salary"].notna() &
                         postings["state"].notna()]
                .groupby("state")["normalized_salary"]
                .agg(["median", "count"])
                .query("count >= 100")
                .sort_values("median", ascending=False)
                .head(20))

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(state_salary.index[::-1], state_salary["median"][::-1],
               color=sns.color_palette("Blues_d", 20)[::-1])
ax.bar_label(bars,
             labels=[f"${v:,.0f}" for v in state_salary["median"][::-1]],
             padding=4, fontsize=8)
ax.set_xlabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Eyalet Bazında Medyan Maaş — Top 20\n(min. 100 ilan)",
             fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "22_state_median_salary.png"), dpi=150)
plt.close()
print("22_state_median_salary.png kaydedildi.")
tracker.done(3)

tracker.start(4, "Plot 2 — State remote rate")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Eyalet bazında remote oran (top 20 ilan sayısına göre)
# ══════════════════════════════════════════════════════════════════════════════
top20_states = postings["state"].value_counts().head(20).index.tolist()
state_remote = (postings[postings["state"].isin(top20_states)]
                .groupby("state")["remote_allowed"]
                .mean()
                .mul(100)
                .sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(state_remote.index[::-1], state_remote.values[::-1],
               color=sns.color_palette("Blues_d", 20)[::-1])
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
ax.set_xlabel("Remote İlan Oranı (%)")
ax.set_title("Eyalet Bazında Remote İlan Oranı — Top 20 Eyalet",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "23_state_remote_rate.png"), dpi=150)
plt.close()
print("23_state_remote_rate.png kaydedildi.")
tracker.done(4)

tracker.start(5, "Plot 3 — Top cities")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Top 15 şehir ilan sayısı
# ══════════════════════════════════════════════════════════════════════════════
top_cities = postings["city"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top_cities.index[::-1], top_cities.values[::-1],
               color=sns.color_palette("Blues_d", 15)[::-1])
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title("En Çok İlan Veren 15 Şehir", fontsize=13, fontweight="bold", pad=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "24_top_cities.png"), dpi=150)
plt.close()
print("24_top_cities.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 4 — State × Industry heatmap")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Eyalet × Sektör yoğunluk haritası (top 10 eyalet, top 8 sektör)
# ══════════════════════════════════════════════════════════════════════════════
top10_states = postings["state"].value_counts().head(10).index.tolist()
top8_ind     = job_ind_full["industry_name"].value_counts().head(8).index.tolist()

state_ind = (postings[["job_id", "state"]]
             .merge(job_ind_full[["job_id", "industry_name"]], on="job_id")
             .query("state in @top10_states and industry_name in @top8_ind"))

state_ind_pivot = (state_ind.groupby(["state", "industry_name"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(top10_states))
state_ind_pct = state_ind_pivot.div(state_ind_pivot.sum(axis=1), axis=0).mul(100).round(1)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(state_ind_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Oran (%)"})
ax.set_title("Eyalet × Sektör Dağılımı — Top 10 Eyalet, Top 8 Sektör (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "25_state_industry_heatmap.png"), dpi=150)
plt.close()
print("25_state_industry_heatmap.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 5 — State experience distribution")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Eyalet bazında deneyim seviyesi dağılımı (top 8 eyalet)
# ══════════════════════════════════════════════════════════════════════════════
top8_states  = postings["state"].value_counts().head(8).index.tolist()
exp_order    = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]

state_exp = (postings[postings["state"].isin(top8_states) &
                      postings["formatted_experience_level"].isin(exp_order)]
             .groupby(["state", "formatted_experience_level"])
             .size()
             .unstack(fill_value=0)
             .reindex(top8_states)[exp_order])
state_exp_pct = state_exp.div(state_exp.sum(axis=1), axis=0).mul(100)

fig, ax = plt.subplots(figsize=(13, 6))
state_exp_pct.plot(kind="bar", stacked=True, ax=ax,
                   color=sns.color_palette("Blues_d", 5),
                   edgecolor="white", width=0.65)
ax.set_xlabel("")
ax.set_ylabel("Oran (%)")
ax.set_title("Eyalet Bazında Deneyim Seviyesi Dağılımı — Top 8 Eyalet (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(title="Deneyim", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=20, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "26_state_experience.png"), dpi=150)
plt.close()
print("26_state_experience.png kaydedildi.")
tracker.done(7)
tracker.finish()

print("\nTüm coğrafi analiz grafikleri outputs/ klasörüne kaydedildi.")