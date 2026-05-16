"""
shap_dependence.py
SHAP Dependence Plots — model_02 best_salary_pipeline.joblib kullanır.

Çalıştırmadan önce model_02_pipeline_v3_cv_safe.py'nin tamamlanmış olması
ve models/best_salary_pipeline.joblib dosyasının oluşturulmuş olması gerekir.
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
MODELS_PATH = "models/"
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
SHAP_SAMPLE = 2_000
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Model yükle ───────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_PATH, "best_salary_pipeline.joblib")
if not os.path.exists(model_path):
    # model_03 varsa onu dene
    model_path = os.path.join(MODELS_PATH, "best_salary_model_03_advanced_progress.joblib")

print(f"Model yükleniyor: {model_path}")
obj = joblib.load(model_path)

# Pipeline veya dict olabilir
if isinstance(obj, dict):
    pipeline    = obj.get("pipeline") or obj.get("model")
    feature_cols= obj.get("feature_cols", [])
    metrics     = obj.get("metrics", {})
else:
    pipeline     = obj
    feature_cols = []
    metrics      = {}

print(f"Model tipi: {type(pipeline)}")
if metrics:
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# ── Veri yükle ve feature engineering (model_02 ile aynı logic) ───────────────
# Model_02'nin feature_cols'unu kullanarak X'i rebuild ediyoruz.
# Bunun için postings'i yeniden yükleyip pipeline'ı transform ediyoruz.
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    low_memory=False
).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"))
job_skills     = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
skills_ref     = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
benefits       = pd.read_csv(os.path.join(DATA_PATH, "benefits.csv"))

print("Veriler yüklendi.")

# feature_cols varsa pipeline'a X besle
if feature_cols:
    postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)
    postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")
    postings.loc[postings["normalized_salary"] < 10_000,    "normalized_salary"] = None
    postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

    # Maaş verisi olan ilanlar
    sal_df = postings[postings["normalized_salary"].notna()].copy()

    # feature_cols içindeki sütunları hazırla
    for col in feature_cols:
        if col not in sal_df.columns:
            sal_df[col] = 0

    X_raw = sal_df[feature_cols].copy()
    y_log = np.log1p(sal_df["normalized_salary"].values)

    # Pipeline'dan preprocessor'ı al
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        model_step   = pipeline.named_steps.get("model") or list(pipeline.named_steps.values())[-1]
    else:
        print("Pipeline adımları bulunamadı. Ham X kullanılıyor.")
        preprocessor = None
        model_step   = pipeline

    # Sample
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X_raw), size=min(SHAP_SAMPLE, len(X_raw)), replace=False)
    X_sample_raw = X_raw.iloc[idx]

    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X_sample_raw)
            print(f"Transformed shape: {X_transformed.shape}")
        except Exception as e:
            print(f"Transform hatası: {e}")
            X_transformed = X_sample_raw.values
    else:
        X_transformed = X_sample_raw.values

    # SHAP hesapla
    print(f"\nSHAP değerleri hesaplanıyor ({SHAP_SAMPLE} örnek)...")
    try:
        explainer  = shap.TreeExplainer(model_step)
        shap_values= explainer.shap_values(X_transformed)
        print("SHAP tamamlandı.")

        # Feature isimleri
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                feat_names = preprocessor.get_feature_names_out()
            except:
                feat_names = [f"f{i}" for i in range(X_transformed.shape[1])]
        else:
            feat_names = feature_cols[:X_transformed.shape[1]] if feature_cols else [f"f{i}" for i in range(X_transformed.shape[1])]

        shap_df = pd.DataFrame(np.abs(shap_values), columns=feat_names)
        top_features = shap_df.mean().sort_values(ascending=False).head(10).index.tolist()
        print(f"Top 10 feature: {top_features[:5]}...")

    except Exception as e:
        print(f"SHAP TreeExplainer hatası: {e}")
        print("KernelExplainer deneniyor (yavaş)...")
        explainer   = shap.KernelExplainer(model_step.predict, X_transformed[:100])
        shap_values = explainer.shap_values(X_transformed[:200])
        feat_names  = feature_cols[:X_transformed.shape[1]]
        shap_df     = pd.DataFrame(np.abs(shap_values), columns=feat_names)
        top_features= shap_df.mean().sort_values(ascending=False).head(10).index.tolist()

else:
    print("feature_cols bulunamadı. Lütfen model_02 veya model_03'ün çıktısını kontrol edin.")
    exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — SHAP Summary Plot (Beeswarm)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGrafik 1: SHAP Beeswarm...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feat_names,
    max_display=15,
    show=False,
    plot_size=None
)
plt.title("SHAP Summary (Beeswarm) — En Etkili 15 Feature",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "62_shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("62_shap_beeswarm.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — SHAP Bar Plot (Mean |SHAP|)
# ══════════════════════════════════════════════════════════════════════════════
print("Grafik 2: SHAP Bar...")
plt.figure(figsize=(11, 7))
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feat_names,
    plot_type="bar",
    max_display=15,
    show=False,
    plot_size=None
)
plt.title("SHAP Feature Importance (Mean |SHAP|) — Top 15",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "63_shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("63_shap_bar.png kaydedildi.")

# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3-6 — SHAP Dependence Plots (Top 4 feature)
# ══════════════════════════════════════════════════════════════════════════════
feat_idx_map = {name: i for i, name in enumerate(feat_names)}

for plot_num, feat in enumerate(top_features[:4], start=3):
    if feat not in feat_idx_map:
        continue
    fidx = feat_idx_map[feat]
    print(f"Grafik {plot_num+62}: SHAP Dependence — {feat}...")

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_vals = X_transformed[:, fidx] if hasattr(X_transformed, "__getitem__") else X_transformed.toarray()[:, fidx]
    shap_vals = shap_values[:, fidx]

    sc = ax.scatter(feat_vals, shap_vals, c=shap_vals, cmap="coolwarm",
                    alpha=0.5, s=8, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="SHAP Değeri")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    # Trend çizgisi
    try:
        z = np.polyfit(feat_vals, shap_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(x_line, p(x_line), color="#1F4E79", linewidth=2, label="Trend")
        ax.legend(fontsize=9)
    except:
        pass

    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("SHAP Değeri (Log Maaş Etkisi)", fontsize=10)
    ax.set_title(f"SHAP Dependence Plot — {feat}",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out_name = f"{62+plot_num}_shap_dependence_{feat[:30].replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_PATH, out_name), dpi=150)
    plt.close()
    print(f"{out_name} kaydedildi.")

print("\nTüm SHAP grafikleri outputs/ klasörüne kaydedildi.")
