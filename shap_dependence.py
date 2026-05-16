"""
shap_dependence.py
SHAP Dependence Plots — model_03 best_salary_model_03_advanced_progress.joblib kullanır.

Çalıştırmadan önce:
  1. model_03_salary_advanced_progress.py tamamlanmış olmalı
  2. prepare_shap_data.py çalıştırılmış olmalı (models/shap_X.parquet vb.)
"""
import os
import re
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
import scipy.sparse as sp
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# =============================================================================
# MODEL_03'TEN KOPYALANAN CUSTOM TRANSFORMERS
# joblib.load() bu sınıfları __main__'de arar — burada tanımlı olmalı.
# =============================================================================
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.log1p(np.maximum(arr, 0))


class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    """CV-safe target encoder — model_03 ile aynı implementasyon."""
    def __init__(self, cols=None, smoothing=40.0, min_samples_leaf=8):
        self.cols = cols or []
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        y_arr = np.asarray(y, dtype=float)
        self.global_ = float(np.nanmedian(y_arr))
        self.maps_ = {}
        for col in self.cols:
            if col not in X_df.columns:
                continue
            temp = pd.DataFrame({col: X_df[col].fillna("Unknown").astype(str), "_target": y_arr})
            stats = temp.groupby(col)["_target"].agg(["median", "count"])
            weight = stats["count"] / (stats["count"] + self.smoothing)
            encoded = weight * stats["median"] + (1.0 - weight) * self.global_
            encoded = encoded.where(stats["count"] >= self.min_samples_leaf, self.global_)
            self.maps_[col] = encoded.to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.cols:
            new_col = f"te_{col}"
            if col not in X_df.columns or col not in self.maps_:
                X_df[new_col] = self.global_
                continue
            X_df[new_col] = (
                X_df[col].fillna("Unknown").astype(str)
                .map(self.maps_[col]).fillna(self.global_).astype(float)
            )
        return X_df


DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
MODELS_PATH = "models/"
RANDOM_SEED = 42
SHAP_SAMPLE = 2_000

# model_03 config ile eşleşmeli
TITLE_SVD_COMPONENTS = 25
DESC_SVD_COMPONENTS  = 100

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Model yükle ───────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_PATH, "best_salary_model_03_advanced_progress.joblib")
if not os.path.exists(model_path):
    model_path = os.path.join(MODELS_PATH, "best_salary_pipeline.joblib")

print(f"Model yükleniyor: {model_path}")
obj = joblib.load(model_path)

if isinstance(obj, dict):
    metrics = obj.get("metrics", {})

    # ── Metrics düzgün yazdır ─────────────────────────────────────────────────
    print("\nModel metrikleri:")
    for k, v in metrics.items():
        if k == "weights" and isinstance(v, dict):
            # Ensemble ağırlıkları — her model için ayrı satır
            print(f"  Ensemble weights:")
            for model_name, w in v.items():
                print(f"    {model_name}: {w:.3f}")
        elif isinstance(v, float):
            if k in ("rmse", "mae"):
                print(f"  {k}: ${v:,.0f}")
            else:
                print(f"  {k}: {v:.4f}")
        # dict olmayan diğer tipler sessizce geç

    if obj.get("type") == "ensemble":
        pipelines = obj.get("pipelines", {})
        pipeline  = pipelines.get("LightGBM") or list(pipelines.values())[0]
        print(f"\nEnsemble modeli — SHAP için kullanılan: LightGBM pipeline")
    else:
        pipeline = obj.get("pipeline") or obj.get("model")
else:
    pipeline = obj
    metrics  = {}

print(f"Pipeline tipi: {type(pipeline)}")

# ── X ve feature_cols: prepare_shap_data.py çıktısından yükle ─────────────────
x_parquet = os.path.join(MODELS_PATH, "shap_X.parquet")
fc_json   = os.path.join(MODELS_PATH, "shap_feature_cols.json")
y_npy     = os.path.join(MODELS_PATH, "shap_y.npy")

if not os.path.exists(x_parquet):
    print("\n  HATA: models/shap_X.parquet bulunamadı.")
    print("  Önce prepare_shap_data.py'yi çalıştırın (~2-3 dk):")
    print("    python prepare_shap_data.py")
    exit(1)

print(f"\nFeature matrix yükleniyor: {x_parquet}")
X_full = pd.read_parquet(x_parquet)

with open(fc_json) as f:
    raw_feature_cols = json.load(f)   # 122 girdi kolonu

y_log_full = np.load(y_npy)
print(f"X shape: {X_full.shape}  |  raw feature_cols: {len(raw_feature_cols)}")

# ── Sample + transform ────────────────────────────────────────────────────────
rng     = np.random.RandomState(RANDOM_SEED)
idx     = rng.choice(len(X_full), size=min(SHAP_SAMPLE, len(X_full)), replace=False)
X_sample = X_full.iloc[idx].copy()

print(f"\nPipeline transform uygulanıyor ({len(X_sample)} örnek)...")
try:
    X_transformed = pipeline[:-1].transform(X_sample)
    if sp.issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    X_transformed = np.array(X_transformed, dtype=float)
    print(f"Transformed shape: {X_transformed.shape}")
except Exception as e:
    print(f"Transform hatası: {e}")
    exit(1)

# ── Feature isimleri (gerçek isimler) ─────────────────────────────────────────
# Preprocessor 122 girdi → N çıktı üretir.
# Sıra: raw_cols (cat+te+log_num+num+bin+skill+benefit+spec) → title_svd_0..24 → desc_svd_0..99
n_out       = X_transformed.shape[1]
n_raw       = len(raw_feature_cols)
n_title_svd = TITLE_SVD_COMPONENTS   # 25
n_desc_svd  = DESC_SVD_COMPONENTS    # 100

# SVD sütunlarını oluştur
title_svd_names = [f"title_svd_{i}" for i in range(n_title_svd)]
desc_svd_names  = [f"desc_svd_{i}"  for i in range(n_desc_svd)]

# Tam isim listesi — transformed boyutuna kırp
all_feat_names = raw_feature_cols + title_svd_names + desc_svd_names
feat_names     = all_feat_names[:n_out]

# Eksik kalırsa generic isimle doldur
if len(feat_names) < n_out:
    feat_names += [f"f{i}" for i in range(len(feat_names), n_out)]

print(f"Feature names: {len(feat_names)} (ilk 5: {feat_names[:5]})")

# ── Model adımı ───────────────────────────────────────────────────────────────
model_step = pipeline[-1]

# ── SHAP hesapla ──────────────────────────────────────────────────────────────
print(f"\nSHAP değerleri hesaplanıyor ({len(X_sample)} örnek)...")
explainer   = shap.TreeExplainer(model_step)
shap_values = explainer.shap_values(X_transformed)
print("SHAP tamamlandı.")

shap_df      = pd.DataFrame(np.abs(shap_values), columns=feat_names)
top_features = shap_df.mean().sort_values(ascending=False).head(10).index.tolist()
print(f"Top 5 feature: {top_features[:5]}")

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
    print(f"Grafik {plot_num + 62}: SHAP Dependence — {feat}...")

    fig, ax = plt.subplots(figsize=(10, 6))
    arr       = X_transformed.toarray() if sp.issparse(X_transformed) else np.array(X_transformed)
    feat_vals = arr[:, fidx]
    shap_vals = shap_values[:, fidx]

    sc = ax.scatter(feat_vals, shap_vals, c=shap_vals, cmap="coolwarm",
                    alpha=0.5, s=8, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="SHAP Değeri")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    try:
        z      = np.polyfit(feat_vals, shap_vals, 1)
        p      = np.poly1d(z)
        x_line = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(x_line, p(x_line), color="#1F4E79", linewidth=2, label="Trend")
        ax.legend(fontsize=9)
    except Exception:
        pass

    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("SHAP Değeri (Log Maaş Etkisi)", fontsize=10)
    ax.set_title(f"SHAP Dependence Plot — {feat}",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    safe_name = re.sub(r"[^\w]", "_", feat[:35])
    out_name  = f"{62 + plot_num}_shap_dependence_{safe_name}.png"
    plt.savefig(os.path.join(OUTPUT_PATH, out_name), dpi=150)
    plt.close()
    print(f"{out_name} kaydedildi.")

print("\nTüm SHAP grafikleri outputs/ klasörüne kaydedildi.")