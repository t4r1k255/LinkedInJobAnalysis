import pandas as pd
from google import genai
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from utils_progress import ProgressBar, StepTracker
import json
import time
from dotenv import load_dotenv

# ── AYARLAR ───────────────────────────────────────────────────────────────────
load_dotenv()
DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
SAMPLE_SIZE = 100_000   # Stratified sampling havuzu (LLM için değil)
RANDOM_SEED = 42
LLM_SAMPLE  = 500
BATCH_SIZE  = 25
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams["font.family"] = "DejaVu Sans"

tracker = StepTracker(total_steps=8, script_name="analysis_03_llm.py — LLM Analysis (Gemini)")

tracker.start(1, "Loading data")
_bar = ProgressBar(total=1, title="Loading CSV files", unit="files")
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=["job_id", "title", "description", "formatted_experience_level",
             "normalized_salary", "remote_allowed"],
    low_memory=False
).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
_bar.step("postings.csv")
_bar.finish()

postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")

# ── STRATİFİED SAMPLING ───────────────────────────────────────────────────────
# Deneyim seviyesine göre orantılı örnekleme
desc_df = postings[postings["description"].notna()].copy()

level_counts = desc_df["formatted_experience_level"].value_counts()
level_ratios = level_counts / level_counts.sum()

samples = []
for level, ratio in level_ratios.items():
    n = max(1, round(ratio * LLM_SAMPLE))
    group = desc_df[desc_df["formatted_experience_level"] == level]
    n = min(n, len(group))
    samples.append(group.sample(n=n, random_state=RANDOM_SEED))

llm_df = pd.concat(samples).sample(frac=1, random_state=RANDOM_SEED).head(LLM_SAMPLE).reset_index(drop=True)

print(f"Stratified sampling dağılımı:")
print(llm_df["formatted_experience_level"].value_counts().to_string())
tracker.done(1)

tracker.start(2, "Stratified sampling")
print(f"Toplam: {len(llm_df)} ilan Gemini ile analiz edilecek.")
tracker.done(2)

# ── LLM ANALİZİ ───────────────────────────────────────────────────────────────
def analyze_batch(batch_df):
    items = []
    for _, row in batch_df.iterrows():
        desc = str(row["description"])[:800]
        items.append(f'JOB_ID:{row["job_id"]}\nTITLE:{row["title"]}\nDESC:{desc}')

    prompt = f"""
Analyze the following {len(batch_df)} job postings and return a JSON array.
For each posting return exactly this structure:
{{
  "job_id": <number>,
  "requires_degree": true/false,
  "has_soft_skills": true/false,
  "soft_skill_type": "communication" | "leadership" | "teamwork" | "analytical" | "other" | "none",
  "urgency": "high" | "medium" | "low",
  "tech_role": true/false
}}

Rules:
- requires_degree: true if bachelor/master/degree/BS/MS/PhD mentioned
- has_soft_skills: true if soft skills explicitly mentioned
- urgency: high if "immediately/urgent/ASAP" mentioned, low if "no rush/flexible", else medium
- tech_role: true if programming/software/data/engineering role

Return ONLY a valid JSON array, no explanation, no markdown.

POSTINGS:
{'---'.join(items)}
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"  Batch hatası: {e}")
        return []

# ── BATCH İŞLEME ─────────────────────────────────────────────────────────────
tracker.start(3, "Gemini API — batch processing")
results = []
total_batches = (LLM_SAMPLE + BATCH_SIZE - 1) // BATCH_SIZE
_batch_bar = ProgressBar(total=total_batches, title="Gemini API batches", unit="batches")

for i in range(0, LLM_SAMPLE, BATCH_SIZE):
    batch = llm_df.iloc[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    batch_results = analyze_batch(batch)
    results.extend(batch_results)
    _batch_bar.step(f"Batch {batch_num} — {len(batch_results)} results")
    time.sleep(13)

# ── SONUÇLARI KAYDET ─────────────────────────────────────────────────────────
_batch_bar.finish()
tracker.done(3)

tracker.start(4, "Merging results")
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_PATH, "llm_results.csv"), index=False)
print(f"LLM analizi tamamlandı. {len(results_df)} ilan işlendi.")
tracker.done(4)

merged = llm_df.merge(results_df, on="job_id", how="inner")
print(f"Birleştirilen kayıt sayısı: {len(merged)}\n")

# ══════════════════════════════════════════════════════════════════════════════
tracker.start(5, "Plot 1 — Degree requirement")
# GRAFİK 1 — Diploma gerektiren vs gerektirmeyen
# ══════════════════════════════════════════════════════════════════════════════
deg_counts = merged["requires_degree"].map({True: "Diploma Gerekli", False: "Diploma Gerekmiyor"}).value_counts()

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(deg_counts.index, deg_counts.values,
              color=["#1F4E79", "#AED6F1"], edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
ax.set_ylabel("İlan Sayısı")
ax.set_title("İlan Açıklamalarında Diploma Şartı", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "12_degree_requirement.png"), dpi=150)
plt.close()
print("12_degree_requirement.png kaydedildi.")
tracker.done(5)

tracker.start(6, "Plot 2 — Soft skill types")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Soft skill tiplerine göre dağılım
# ══════════════════════════════════════════════════════════════════════════════
soft = merged[merged["has_soft_skills"] == True]
soft_counts = soft["soft_skill_type"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(soft_counts.index[::-1], soft_counts.values[::-1],
               color=sns.color_palette("Blues_d", len(soft_counts))[::-1])
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
ax.set_xlabel("İlan Sayısı")
ax.set_title("İlanlarda En Çok Aranan Soft Skill Tipleri", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "13_soft_skills.png"), dpi=150)
plt.close()
print("13_soft_skills.png kaydedildi.")
tracker.done(6)

tracker.start(7, "Plot 3-4 — Urgency & Tech salary")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — İlan aciliyetine göre dağılım
# ══════════════════════════════════════════════════════════════════════════════
urgency_counts = merged["urgency"].value_counts().reindex(["high", "medium", "low"]).dropna()
urgency_labels = {"high": "Acil", "medium": "Normal", "low": "Esnek"}
urgency_counts.index = [urgency_labels.get(x, x) for x in urgency_counts.index]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(urgency_counts.index, urgency_counts.values,
              color=["#1F4E79", "#2E75B6", "#AED6F1"], edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=11)
ax.set_ylabel("İlan Sayısı")
ax.set_title("İlan Aciliyet Dağılımı", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "14_urgency.png"), dpi=150)
plt.close()
print("14_urgency.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Tech vs non-tech rollerde maaş karşılaştırması
# ══════════════════════════════════════════════════════════════════════════════
sal_data = merged[merged["normalized_salary"].notna()].copy()
sal_data.loc[sal_data["normalized_salary"] < 10_000, "normalized_salary"] = None
sal_data.loc[sal_data["normalized_salary"] > 1_000_000, "normalized_salary"] = None
sal_data = sal_data[sal_data["normalized_salary"].notna()]

sal_data["role_type"] = sal_data["tech_role"].map({True: "Tech Rol", False: "Non-Tech Rol"})
sal_by_type = sal_data.groupby("role_type")["normalized_salary"].median()

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(sal_by_type.index, sal_by_type.values,
              color=["#1F4E79", "#AED6F1"], edgecolor="white", width=0.5)
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in sal_by_type.values], padding=4, fontsize=11)
ax.set_ylabel("Medyan Yıllık Maaş (USD)")
ax.set_title("Tech vs. Non-Tech Rollerde Medyan Maaş", fontsize=13, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "15_tech_vs_nontech_salary.png"), dpi=150)
plt.close()
print("15_tech_vs_nontech_salary.png kaydedildi.")
tracker.done(7)

tracker.start(8, "Plot 5 — Degree by experience")
# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Deneyim seviyesine göre diploma şartı (stratified analiz)
# ══════════════════════════════════════════════════════════════════════════════
deg_by_level = merged.groupby("formatted_experience_level")["requires_degree"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(deg_by_level.index, deg_by_level.values * 100,
              color=sns.color_palette("Blues_d", len(deg_by_level)))
ax.bar_label(bars, labels=[f"%{v:.0f}" for v in deg_by_level.values * 100], padding=4, fontsize=10)
ax.set_ylabel("Diploma Şartı Oranı (%)")
ax.set_title("Deneyim Seviyesine Göre Diploma Şartı Oranı", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 110)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "16_degree_by_experience.png"), dpi=150)
plt.close()
print("16_degree_by_experience.png kaydedildi.")
tracker.done(8)
tracker.finish()

print("\nTüm LLM grafikleri outputs/ klasörüne kaydedildi.")