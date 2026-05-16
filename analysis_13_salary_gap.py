"""
analysis_13_salary_gap.py
LinkedIn Job Analysis — Salary Negotiation Gap
Gap = max_salary - min_salary (yıllık USD ilanlarında müzakere aralığı)

Grafikler:
  68_salary_gap_distribution.png   — Gap dağılımı (histogram + KDE)
  69_salary_gap_by_industry.png    — Sektör bazında medyan gap
  70_salary_gap_by_exp.png         — Deneyim bazında gap kutusu
  71_salary_band_vs_gap.png        — Maaş seviyesi vs gap genişliği
  72_salary_gap_negotiation_map.py — Sektör × Deneyim müzakere haritası
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

STYLE = {
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#1A1D27",
    "axes.edgecolor":    "#2E3347",
    "axes.labelcolor":   "#C8CDD8",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.color":       "#8B92A5",
    "ytick.color":       "#8B92A5",
    "text.color":        "#C8CDD8",
    "grid.color":        "#2E3347",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#1A1D27",
    "legend.edgecolor":  "#2E3347",
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "#0F1117",
    "font.size":         10,
}

BLUE_PALETTE = ["#1E3A5F", "#1F4E79", "#2E75B6", "#5B9BD5", "#AED6F1", "#D6EAF8"]

tracker = StepTracker(
    total_steps=6,
    script_name="analysis_13_salary_gap.py — Salary Negotiation Gap"
)

# ── 1. VERİ YÜKLEME ───────────────────────────────────────────────────────────
tracker.start(1, "Loading data")
_bar = ProgressBar(total=5, title="Loading CSV files", unit="files")

salaries = pd.read_csv(os.path.join(DATA_PATH, "salaries.csv"))
_bar.step("salaries.csv")

postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "formatted_experience_level", "location"],
    low_memory=False
)
_bar.step("postings.csv")

job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
_bar.step("job_industries.csv")

industries = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
_bar.step("industries.csv")

_bar.finish()

# ── TEMİZLEME ─────────────────────────────────────────────────────────────────
# Yıllık USD, min+max mevcut, mantıklı aralıkta
sal = salaries[
    (salaries["pay_period"] == "YEARLY") &
    (salaries["currency"] == "USD") &
    salaries["min_salary"].notna() &
    salaries["max_salary"].notna()
].copy()

sal = sal[
    (sal["min_salary"] >= 10_000) &
    (sal["max_salary"] <= 500_000) &
    (sal["max_salary"] >= sal["min_salary"])
].copy()

sal["gap"]     = sal["max_salary"] - sal["min_salary"]
sal["gap_pct"] = sal["gap"] / sal["min_salary"] * 100
sal["mid_sal"] = (sal["min_salary"] + sal["max_salary"]) / 2

# Postings ile birleştir
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
postings["state"] = postings["location"].str.extract(r",\s*([A-Z]{2})\s*$")

df = sal.merge(postings[["job_id", "formatted_experience_level", "state"]], on="job_id", how="left")

# Sektör ekle
job_ind = job_industries.merge(industries, on="industry_id", how="left")
primary_ind = job_ind.groupby("job_id")["industry_name"].first().reset_index()
df = df.merge(primary_ind, on="job_id", how="left")

print(f"  Kullanılabilir satır: {len(df):,}")
print(f"  Medyan gap: ${df['gap'].median():,.0f} ({df['gap_pct'].median():.1f}%)")
tracker.done(1)

# ── 2. GRAFİK 1 — Gap dağılımı ───────────────────────────────────────────────
tracker.start(2, "Plot 1 — Gap distribution")
plt.rcParams.update(STYLE)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Maaş Müzakere Aralığı (Gap) Dağılımı", fontsize=15, fontweight="bold", y=1.02)

# Sol: Histogram
ax = axes[0]
gap_clip = df["gap"].clip(upper=200_000)
ax.hist(gap_clip, bins=50, color="#2E75B6", alpha=0.85, edgecolor="none")
ax.axvline(df["gap"].median(), color="#F59E0B", linewidth=2, linestyle="--",
           label=f"Medyan: ${df['gap'].median():,.0f}")
ax.axvline(df["gap"].mean(), color="#10B981", linewidth=2, linestyle=":",
           label=f"Ortalama: ${df['gap'].mean():,.0f}")
ax.set_xlabel("Maaş Aralığı / Gap (USD)")
ax.set_ylabel("İlan Sayısı")
ax.set_title("Gap Dağılımı (0–$200k)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Sağ: Gap % dağılımı
ax2 = axes[1]
gap_pct_clip = df["gap_pct"].clip(upper=200)
ax2.hist(gap_pct_clip, bins=50, color="#5B9BD5", alpha=0.85, edgecolor="none")
ax2.axvline(df["gap_pct"].median(), color="#F59E0B", linewidth=2, linestyle="--",
            label=f"Medyan: %{df['gap_pct'].median():.1f}")
ax2.set_xlabel("Maaş Aralığı / Gap (%)")
ax2.set_ylabel("İlan Sayısı")
ax2.set_title("Gap % Dağılımı (0–%200)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"%{int(x)}"))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "68_salary_gap_distribution.png"), dpi=150)
plt.close()
print("  68_salary_gap_distribution.png kaydedildi.")
tracker.done(2)

# ── 3. GRAFİK 2 — Sektör bazında gap ─────────────────────────────────────────
tracker.start(3, "Plot 2 — Gap by industry")
plt.rcParams.update(STYLE)

top15_ind = df["industry_name"].value_counts().head(15).index.tolist()
ind_gap = (df[df["industry_name"].isin(top15_ind)]
           .groupby("industry_name")
           .agg(
               median_gap=("gap", "median"),
               median_gap_pct=("gap_pct", "median"),
               median_sal=("mid_sal", "median"),
               count=("job_id", "count")
           )
           .sort_values("median_gap", ascending=True))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Sektör Bazında Maaş Müzakere Aralığı", fontsize=15, fontweight="bold", y=1.02)

# Sol: Mutlak gap
ax = axes[0]
colors = [BLUE_PALETTE[int(i * (len(BLUE_PALETTE)-1) / (len(ind_gap)-1))]
          for i in range(len(ind_gap))]
bars = ax.barh(ind_gap.index, ind_gap["median_gap"], color=colors, alpha=0.9)
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in ind_gap["median_gap"]],
             padding=4, fontsize=8, color="#C8CDD8")
ax.set_xlabel("Medyan Gap (USD)")
ax.set_title("Medyan Mutlak Gap (min→max farkı)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
ax.grid(axis="x", alpha=0.3)
ax.spines[["top","right","left","bottom"]].set_visible(False)

# Sağ: Yüzde gap
ax2 = axes[1]
ind_gap_sorted_pct = ind_gap.sort_values("median_gap_pct", ascending=True)
bars2 = ax2.barh(ind_gap_sorted_pct.index, ind_gap_sorted_pct["median_gap_pct"],
                 color=colors, alpha=0.9)
ax2.bar_label(bars2, labels=[f"%{v:.1f}" for v in ind_gap_sorted_pct["median_gap_pct"]],
              padding=4, fontsize=8, color="#C8CDD8")
ax2.set_xlabel("Medyan Gap (%)")
ax2.set_title("Medyan Göreli Gap (min maaşa oranı)")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"%{int(x)}"))
ax2.grid(axis="x", alpha=0.3)
ax2.spines[["top","right","left","bottom"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "69_salary_gap_by_industry.png"), dpi=150)
plt.close()
print("  69_salary_gap_by_industry.png kaydedildi.")
tracker.done(3)

# ── 4. GRAFİK 3 — Deneyim bazında gap ────────────────────────────────────────
tracker.start(4, "Plot 3 — Gap by experience")
plt.rcParams.update(STYLE)

exp_order = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
exp_df = df[df["formatted_experience_level"].isin(exp_order)].copy()

exp_stats = (exp_df.groupby("formatted_experience_level")
             .agg(
                 median_gap=("gap", "median"),
                 median_gap_pct=("gap_pct", "median"),
                 median_min=("min_salary", "median"),
                 median_max=("max_salary", "median"),
                 count=("job_id", "count")
             )
             .reindex(exp_order))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Deneyim Seviyesine Göre Maaş Müzakere Aralığı", fontsize=15, fontweight="bold", y=1.02)

# Sol: Stacked bar — min + gap
ax = axes[0]
x = np.arange(len(exp_order))
w = 0.55
bars_min = ax.bar(x, exp_stats["median_min"], w, label="Medyan Min Maaş",
                  color="#1F4E79", alpha=0.9)
bars_gap = ax.bar(x, exp_stats["median_gap"], w, bottom=exp_stats["median_min"],
                  label="Müzakere Aralığı (Gap)", color="#F59E0B", alpha=0.85)

for i, (_, row) in enumerate(exp_stats.iterrows()):
    ax.text(i, row["median_min"] + row["median_gap"] + 2000,
            f"${row['median_gap']:,.0f}",
            ha="center", va="bottom", fontsize=8, color="#F59E0B", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(exp_order, rotation=15, ha="right")
ax.set_ylabel("Yıllık Maaş (USD)")
ax.set_title("Medyan Min Maaş + Gap")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Sağ: Gap % bar
ax2 = axes[1]
colors_exp = [BLUE_PALETTE[i] for i in range(len(exp_order))]
bars2 = ax2.bar(x, exp_stats["median_gap_pct"], w, color=colors_exp, alpha=0.9)
ax2.bar_label(bars2, labels=[f"%{v:.1f}" for v in exp_stats["median_gap_pct"]],
              padding=4, fontsize=9, color="#C8CDD8")
ax2.set_xticks(x)
ax2.set_xticklabels(exp_order, rotation=15, ha="right")
ax2.set_ylabel("Gap / Min Maaş (%)")
ax2.set_title("Deneyim Seviyesine Göre Gap %")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"%{int(v)}"))
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "70_salary_gap_by_exp.png"), dpi=150)
plt.close()
print("  70_salary_gap_by_exp.png kaydedildi.")
tracker.done(4)

# ── 5. GRAFİK 4 — Maaş seviyesi vs gap ──────────────────────────────────────
tracker.start(5, "Plot 4 — Salary band vs gap")
plt.rcParams.update(STYLE)

bins   = [0, 40_000, 70_000, 100_000, 140_000, 200_000, float("inf")]
labels_band = ["<$40K", "$40K-70K", "$70K-100K", "$100K-140K", "$140K-200K", "$200K+"]
df["salary_band"] = pd.cut(df["mid_sal"], bins=bins, labels=labels_band)

band_gap = (df.groupby("salary_band", observed=True)
            .agg(
                median_gap=("gap", "median"),
                median_gap_pct=("gap_pct", "median"),
                count=("job_id", "count")
            )
            .dropna())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Maaş Seviyesi vs Müzakere Aralığı", fontsize=15, fontweight="bold", y=1.02)

ax = axes[0]
colors_band = [BLUE_PALETTE[i % len(BLUE_PALETTE)] for i in range(len(band_gap))]
bars = ax.bar(band_gap.index, band_gap["median_gap"], color=colors_band, alpha=0.9, edgecolor="none")
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in band_gap["median_gap"]],
             padding=4, fontsize=9, color="#C8CDD8")
ax.set_xlabel("Maaş Bandı (Orta Nokta)")
ax.set_ylabel("Medyan Gap (USD)")
ax.set_title("Maaş Bandına Göre Mutlak Gap")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

ax2 = axes[1]
bars2 = ax2.bar(band_gap.index, band_gap["median_gap_pct"], color=colors_band, alpha=0.9, edgecolor="none")
ax2.bar_label(bars2, labels=[f"%{v:.1f}" for v in band_gap["median_gap_pct"]],
              padding=4, fontsize=9, color="#C8CDD8")
ax2.set_xlabel("Maaş Bandı (Orta Nokta)")
ax2.set_ylabel("Medyan Gap (%)")
ax2.set_title("Maaş Bandına Göre Göreli Gap")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"%{int(v)}"))
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "71_salary_band_vs_gap.png"), dpi=150)
plt.close()
print("  71_salary_band_vs_gap.png kaydedildi.")
tracker.done(5)

# ── 6. GRAFİK 5 — Sektör × Deneyim müzakere haritası ─────────────────────────
tracker.start(6, "Plot 5 — Sector × Experience negotiation heatmap")
plt.rcParams.update(STYLE)

top8_ind = df["industry_name"].value_counts().head(8).index.tolist()
neg_df = df[
    df["industry_name"].isin(top8_ind) &
    df["formatted_experience_level"].isin(exp_order)
]

neg_pivot = (neg_df.groupby(["industry_name", "formatted_experience_level"])
             ["gap_pct"]
             .median()
             .unstack()
             .reindex(columns=exp_order)
             .reindex(top8_ind))

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    neg_pivot, annot=True, fmt=".1f", cmap="Blues",
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Medyan Gap (%)"}
)
ax.set_title("Sektör × Deneyim Seviyesi — Müzakere Aralığı Haritası (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "72_salary_gap_negotiation_map.png"), dpi=150)
plt.close()
print("  72_salary_gap_negotiation_map.png kaydedildi.")
tracker.done(6)

tracker.finish()

print(f"""
ÖZET:
  Kullanılan satır : {len(df):,}
  Medyan gap       : ${df['gap'].median():,.0f}
  Medyan gap %     : %{df['gap_pct'].median():.1f}
  En geniş sektör  : {ind_gap['median_gap'].idxmax()} (${ind_gap['median_gap'].max():,.0f})
  En dar sektör    : {ind_gap['median_gap'].idxmin()} (${ind_gap['median_gap'].min():,.0f})

Grafikler outputs/ klasörüne kaydedildi.
""")
