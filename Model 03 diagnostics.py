"""
model_03_diagnostics.py
Model 03 — Overfitting / Underfitting Diagnostik Analizi

model_03_salary_advanced_progress.py tamamlanmış ve joblib kaydedilmiş olmalı.
prepare_shap_data.py çalıştırılmış olmalı (models/shap_X.parquet, shap_y.npy).

Grafikler:
  73_diag_train_vs_oof.png       — Train vs OOF skor karşılaştırması
  74_diag_learning_curve.png     — Learning curve (underfitting testi)
  75_diag_residual_by_range.png  — Maaş aralığına göre hata analizi
  76_diag_error_distribution.png — Hata dağılımı (normallik testi)
  77_diag_fold_stability.png     — Fold bazında skor stabilitesi
"""

import os
import sys
import warnings
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, learning_curve
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Custom Transformers (joblib.load için gerekli) ────────────────────────────
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.log1p(np.maximum(arr, 0))

class MedianTargetEncoder(BaseEstimator, TransformerMixin):
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
            if col not in X_df.columns: continue
            temp = pd.DataFrame({col: X_df[col].fillna("Unknown").astype(str), "_target": y_arr})
            stats_g = temp.groupby(col)["_target"].agg(["median", "count"])
            weight = stats_g["count"] / (stats_g["count"] + self.smoothing)
            encoded = weight * stats_g["median"] + (1.0 - weight) * self.global_
            encoded = encoded.where(stats_g["count"] >= self.min_samples_leaf, self.global_)
            self.maps_[col] = encoded.to_dict()
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.cols:
            new_col = f"te_{col}"
            if col not in X_df.columns or col not in self.maps_:
                X_df[new_col] = self.global_; continue
            X_df[new_col] = (X_df[col].fillna("Unknown").astype(str)
                             .map(self.maps_[col]).fillna(self.global_).astype(float))
        return X_df

sys.modules[__name__].LogTransformer = LogTransformer
sys.modules[__name__].MedianTargetEncoder = MedianTargetEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/"
OUTPUT_PATH = "outputs/"
MODELS_PATH = "models/"
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

COLORS = {
    "train":    "#F59E0B",
    "val":      "#6366F1",
    "ensemble": "#10B981",
    "error":    "#F43F5E",
    "neutral":  "#3B82F6",
}

print("\n" + "="*70)
print("  model_03_diagnostics.py — Overfitting / Underfitting Analizi")
print("="*70)

# ── Model yükle ───────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_PATH, "best_salary_model_03_advanced_progress.joblib")
print(f"\nModel yükleniyor: {model_path}")
obj = joblib.load(model_path)

pipelines    = obj["pipelines"]
feature_cols = obj["feature_cols"]
cv_metrics   = obj["metrics"]
weights      = obj["weights"]

lgb_pipeline = pipelines["LightGBM"]
print(f"  Ensemble weights: {weights}")
print(f"  CV R² (OOF): {cv_metrics['r2']:.4f}")
print(f"  CV RMSE:     ${cv_metrics['rmse']:,.0f}")

# ── Feature matrix yükle (prepare_shap_data.py çıktısı) ──────────────────────
x_parquet = os.path.join(MODELS_PATH, "shap_X.parquet")
y_npy     = os.path.join(MODELS_PATH, "shap_y.npy")

if not os.path.exists(x_parquet):
    print("\n  HATA: models/shap_X.parquet bulunamadı.")
    print("  Önce prepare_shap_data.py'yi çalıştırın:")
    print("    python prepare_shap_data.py")
    sys.exit(1)

print(f"\nFeature matrix yükleniyor...")
X_full    = pd.read_parquet(x_parquet)
y_log     = np.load(y_npy)
y_raw     = np.expm1(y_log)
print(f"  X shape: {X_full.shape}")
print(f"  Target range: ${y_raw.min():,.0f} – ${y_raw.max():,.0f}")

# ── Train prediction (tüm veriyle) ───────────────────────────────────────────
print("\nTrain tahminleri hesaplanıyor (full-data fit)...")
train_pred_log = lgb_pipeline.predict(X_full)
train_pred_raw = np.expm1(train_pred_log)

train_r2   = r2_score(y_raw, train_pred_raw)
train_rmse = np.sqrt(mean_squared_error(y_raw, train_pred_raw))
oof_r2     = cv_metrics["r2"]
oof_rmse   = cv_metrics["rmse"]

print(f"  Train R²:   {train_r2:.4f}  |  OOF R²:   {oof_r2:.4f}")
print(f"  Train RMSE: ${train_rmse:,.0f}  |  OOF RMSE: ${oof_rmse:,.0f}")
overfit_gap = train_r2 - oof_r2
print(f"  Overfit gap (Train-OOF): {overfit_gap:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — Train vs OOF Skor Karşılaştırması
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update(STYLE)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Train vs OOF — Overfit Kontrolü", fontsize=15, fontweight="bold", y=1.02)

# R² karşılaştırması
ax = axes[0]
labels = ["Train R²\n(full-data fit)", "OOF R²\n(cross-validation)"]
values = [train_r2, oof_r2]
colors_bar = [COLORS["train"], COLORS["val"]]
bars = ax.bar(labels, values, color=colors_bar, alpha=0.85, width=0.5, edgecolor="none")
ax.bar_label(bars, labels=[f"{v:.4f}" for v in values],
             padding=6, fontsize=12, color="#E2E8F0", fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_ylabel("R² Skoru")
ax.set_title("R² Karşılaştırması")
ax.axhline(0.75, color="#4B5563", linewidth=1, linestyle="--", alpha=0.5)
gap_label = f"Gap: {overfit_gap:.4f}"
gap_color = "#F43F5E" if overfit_gap > 0.05 else "#10B981"
ax.text(0.5, max(values) * 0.5, gap_label,
        ha="center", transform=ax.get_xaxis_transform(),
        fontsize=11, color=gap_color, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# RMSE karşılaştırması
ax2 = axes[1]
rmse_vals = [train_rmse, oof_rmse]
bars2 = ax2.bar(["Train RMSE\n(full-data fit)", "OOF RMSE\n(cross-validation)"],
                rmse_vals, color=colors_bar, alpha=0.85, width=0.5, edgecolor="none")
ax2.bar_label(bars2, labels=[f"${v:,.0f}" for v in rmse_vals],
              padding=6, fontsize=12, color="#E2E8F0", fontweight="bold")
ax2.set_ylabel("RMSE (USD)")
ax2.set_title("RMSE Karşılaştırması")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

# Yorum kutusu
verdict = "✓ Sağlıklı" if overfit_gap < 0.05 else "⚠️ Overfit riski"
verdict_color = "#10B981" if overfit_gap < 0.05 else "#F59E0B"
fig.text(0.5, -0.05,
         f"Değerlendirme: {verdict}  |  Train-OOF gap = {overfit_gap:.4f}  "
         f"({'< 0.05 → normal' if overfit_gap < 0.05 else '> 0.05 → dikkat'})",
         ha="center", fontsize=11, color=verdict_color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "73_diag_train_vs_oof.png"), dpi=150)
plt.close()
print("  ✓ 73_diag_train_vs_oof.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Learning Curve (Underfitting Testi)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLearning curve hesaplanıyor (bu ~5-10 dk sürebilir)...")
plt.rcParams.update(STYLE)

# LightGBM pipeline'ı learning curve için kullan
train_sizes_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
train_sizes_n   = [int(p * len(X_full)) for p in train_sizes_pct]

train_r2_scores = []
val_r2_scores   = []
train_rmse_scores = []
val_rmse_scores   = []

kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
rng = np.random.RandomState(RANDOM_SEED)

for n in train_sizes_n:
    idx = rng.choice(len(X_full), size=n, replace=False)
    X_sub = X_full.iloc[idx].reset_index(drop=True)
    y_sub = y_log[idx]

    fold_train_r2, fold_val_r2 = [], []
    fold_train_rmse, fold_val_rmse = [], []

    for tr_idx, val_idx in kf.split(X_sub):
        X_tr, X_val = X_sub.iloc[tr_idx], X_sub.iloc[val_idx]
        y_tr, y_val = y_sub[tr_idx], y_sub[val_idx]

        lgb_pipeline.fit(X_tr, y_tr)

        # Train skor
        tr_pred = np.expm1(lgb_pipeline.predict(X_tr))
        tr_true = np.expm1(y_tr)
        fold_train_r2.append(r2_score(tr_true, tr_pred))
        fold_train_rmse.append(np.sqrt(mean_squared_error(tr_true, tr_pred)))

        # Val skor
        val_pred = np.expm1(lgb_pipeline.predict(X_val))
        val_true = np.expm1(y_val)
        fold_val_r2.append(r2_score(val_true, val_pred))
        fold_val_rmse.append(np.sqrt(mean_squared_error(val_true, val_pred)))

    train_r2_scores.append(np.mean(fold_train_r2))
    val_r2_scores.append(np.mean(fold_val_r2))
    train_rmse_scores.append(np.mean(fold_train_rmse))
    val_rmse_scores.append(np.mean(fold_val_rmse))
    print(f"  n={n:>6,}  train_R²={np.mean(fold_train_r2):.3f}  val_R²={np.mean(fold_val_r2):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Learning Curve — Underfitting / Overfitting Testi", fontsize=15, fontweight="bold", y=1.02)

sizes_k = [n/1000 for n in train_sizes_n]

ax = axes[0]
ax.plot(sizes_k, train_r2_scores, "o-", color=COLORS["train"], linewidth=2.5,
        markersize=7, label="Train R²")
ax.plot(sizes_k, val_r2_scores, "s-", color=COLORS["val"], linewidth=2.5,
        markersize=7, label="Validation R²")
ax.fill_between(sizes_k, train_r2_scores, val_r2_scores,
                alpha=0.15, color=COLORS["error"], label="Gap (overfit zone)")
ax.set_xlabel("Eğitim Verisi Boyutu (k satır)")
ax.set_ylabel("R² Skoru")
ax.set_title("R² Learning Curve")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

ax2 = axes[1]
ax2.plot(sizes_k, [v/1000 for v in train_rmse_scores], "o-", color=COLORS["train"],
         linewidth=2.5, markersize=7, label="Train RMSE")
ax2.plot(sizes_k, [v/1000 for v in val_rmse_scores], "s-", color=COLORS["val"],
         linewidth=2.5, markersize=7, label="Validation RMSE")
ax2.set_xlabel("Eğitim Verisi Boyutu (k satır)")
ax2.set_ylabel("RMSE ($k)")
ax2.set_title("RMSE Learning Curve")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

# Yorum
final_gap = train_r2_scores[-1] - val_r2_scores[-1]
plateau = abs(val_r2_scores[-1] - val_r2_scores[-2]) < 0.005
verdict_lc = "✓ Plateau'ya ulaştı — daha fazla veri skor artışı sağlamaz" if plateau else \
             "→ Daha fazla veri skor artırabilir"
fig.text(0.5, -0.04,
         f"Son noktada gap: {final_gap:.3f}  |  {verdict_lc}",
         ha="center", fontsize=10, color="#C8CDD8")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "74_diag_learning_curve.png"), dpi=150)
plt.close()
print("  ✓ 74_diag_learning_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Maaş Aralığına Göre Hata Analizi
# ══════════════════════════════════════════════════════════════════════════════
print("\nResidual analizi...")
plt.rcParams.update(STYLE)

# Full-data train pred (zaten hesaplandı)
residuals = train_pred_raw - y_raw
abs_errors = np.abs(residuals)
pct_errors = abs_errors / y_raw * 100

bins   = [0, 40_000, 70_000, 100_000, 140_000, 200_000, 300_001]
labels_band = ["<$40K", "$40K-70K", "$70K-100K", "$100K-140K", "$140K-200K", "$200K+"]
salary_band = pd.cut(y_raw, bins=bins, labels=labels_band)

band_df = pd.DataFrame({
    "band":      salary_band,
    "abs_error": abs_errors,
    "pct_error": pct_errors,
    "residual":  residuals,
    "count":     1
})

band_stats = (band_df.groupby("band", observed=True)
              .agg(
                  median_abs=("abs_error", "median"),
                  median_pct=("pct_error", "median"),
                  mean_residual=("residual", "mean"),
                  count=("count", "sum")
              )
              .dropna())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Maaş Aralığına Göre Model Hata Analizi", fontsize=15, fontweight="bold", y=1.02)

# Medyan mutlak hata
ax = axes[0]
bars = ax.bar(band_stats.index, band_stats["median_abs"],
              color=COLORS["neutral"], alpha=0.85, edgecolor="none")
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in band_stats["median_abs"]],
             padding=4, fontsize=8, color="#C8CDD8")
ax.set_ylabel("Medyan Mutlak Hata (USD)")
ax.set_title("Medyan Mutlak Hata")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Medyan % hata
ax2 = axes[1]
colors_pct = [COLORS["ensemble"] if v < 25 else COLORS["error"]
              for v in band_stats["median_pct"]]
bars2 = ax2.bar(band_stats.index, band_stats["median_pct"],
                color=colors_pct, alpha=0.85, edgecolor="none")
ax2.bar_label(bars2, labels=[f"%{v:.1f}" for v in band_stats["median_pct"]],
              padding=4, fontsize=8, color="#C8CDD8")
ax2.axhline(25, color="#F59E0B", linewidth=1.5, linestyle="--",
            alpha=0.7, label="25% eşik")
ax2.set_ylabel("Medyan % Hata")
ax2.set_title("Medyan Göreli Hata (%)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"%{int(x)}"))
ax2.legend(fontsize=8)
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

# Sistematik bias (ortalama residual)
ax3 = axes[2]
bias_colors = [COLORS["neutral"] if v >= 0 else COLORS["error"]
               for v in band_stats["mean_residual"]]
bars3 = ax3.bar(band_stats.index, band_stats["mean_residual"],
                color=bias_colors, alpha=0.85, edgecolor="none")
ax3.bar_label(bars3, labels=[f"${v:+,.0f}" for v in band_stats["mean_residual"]],
              padding=4, fontsize=8, color="#C8CDD8")
ax3.axhline(0, color="#6B7280", linewidth=1.2, linestyle="--")
ax3.set_ylabel("Ortalama Residual (USD)")
ax3.set_title("Sistematik Bias\n(+ = aşırı tahmin, - = eksik tahmin)")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k"))
plt.setp(ax3.get_xticklabels(), rotation=20, ha="right")
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "75_diag_residual_by_range.png"), dpi=150)
plt.close()
print("  ✓ 75_diag_residual_by_range.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Hata Dağılımı (Normallik Testi)
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update(STYLE)

# Log-space residuals (daha anlamlı)
log_pred = lgb_pipeline.predict(X_full)
log_residuals = log_pred - y_log

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hata Dağılımı Analizi", fontsize=15, fontweight="bold", y=1.02)

# Histogram
ax = axes[0]
ax.hist(log_residuals, bins=80, color=COLORS["neutral"], alpha=0.85, edgecolor="none",
        density=True)
# Normal fit
mu, sigma = log_residuals.mean(), log_residuals.std()
x_norm = np.linspace(log_residuals.min(), log_residuals.max(), 200)
ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
        color=COLORS["train"], linewidth=2.5, label=f"Normal fit\nμ={mu:.3f}, σ={sigma:.3f}")
ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--")
ax.set_xlabel("Residual (log-space)")
ax.set_ylabel("Yoğunluk")
ax.set_title("Residual Dağılımı")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Q-Q Plot
ax2 = axes[1]
sample = log_residuals[np.random.choice(len(log_residuals), min(3000, len(log_residuals)), replace=False)]
(osm, osr), (slope, intercept, r) = stats.probplot(sample, dist="norm")
ax2.scatter(osm, osr, alpha=0.3, s=5, color=COLORS["val"], label="Residuals")
x_qq = np.array([osm.min(), osm.max()])
ax2.plot(x_qq, slope * x_qq + intercept, color=COLORS["train"],
         linewidth=2, label=f"Normal çizgi (r={r:.3f})")
ax2.set_xlabel("Teorik Quantile")
ax2.set_ylabel("Örneklem Quantile")
ax2.set_title("Q-Q Plot (Normallik Testi)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

# Predicted vs Residual scatter
ax3 = axes[2]
sample_idx = np.random.choice(len(log_pred), min(5000, len(log_pred)), replace=False)
ax3.scatter(np.expm1(log_pred[sample_idx])/1000,
            log_residuals[sample_idx],
            alpha=0.2, s=5, color=COLORS["neutral"])
ax3.axhline(0, color="#F59E0B", linewidth=1.5, linestyle="--")
# Rolling mean
order = np.argsort(log_pred[sample_idx])
xs = np.expm1(log_pred[sample_idx][order]) / 1000
rs = log_residuals[sample_idx][order]
window = max(len(xs)//20, 50)
rolling = pd.Series(rs).rolling(window, center=True).mean()
ax3.plot(xs, rolling.values, color=COLORS["error"], linewidth=2, label="Trend")
ax3.set_xlabel("Tahmin Edilen Maaş ($k)")
ax3.set_ylabel("Residual (log-space)")
ax3.set_title("Predicted vs Residual")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

# Shapiro-Wilk normallik testi (örneklem)
stat, pval = stats.shapiro(sample[:500])
fig.text(0.5, -0.04,
         f"Shapiro-Wilk normallik testi (n=500): W={stat:.4f}, p={pval:.4f}  |  "
         f"{'Normal dağılım kabul edilemez' if pval < 0.05 else 'Normal dağılım reddedilemez'} (α=0.05)",
         ha="center", fontsize=10, color="#C8CDD8")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "76_diag_error_distribution.png"), dpi=150)
plt.close()
print("  ✓ 76_diag_error_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 5 — Fold Stabilitesi
# ══════════════════════════════════════════════════════════════════════════════
print("\nFold stabilitesi analizi...")
plt.rcParams.update(STYLE)

fold_r2s, fold_rmses, fold_maes = [], [], []
kf5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for fold, (tr_idx, val_idx) in enumerate(kf5.split(X_full)):
    X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]

    lgb_pipeline.fit(X_tr, y_tr)
    pred_log = lgb_pipeline.predict(X_val)
    pred_raw = np.expm1(pred_log)
    true_raw = np.expm1(y_val)

    fold_r2s.append(r2_score(true_raw, pred_raw))
    fold_rmses.append(np.sqrt(mean_squared_error(true_raw, pred_raw)))
    fold_maes.append(mean_absolute_error(true_raw, pred_raw))
    print(f"  Fold {fold+1}: R²={fold_r2s[-1]:.4f}  RMSE=${fold_rmses[-1]:,.0f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("5-Fold CV Stabilitesi — Tutarlılık Analizi",
             fontsize=15, fontweight="bold", y=1.02)

folds = [f"Fold {i+1}" for i in range(5)]
metrics_data = [
    (fold_r2s, "R² Skoru", "R²", COLORS["val"], "{:.4f}"),
    (fold_rmses, "RMSE (USD)", "RMSE", COLORS["train"], "${:,.0f}"),
    (fold_maes, "MAE (USD)", "MAE", COLORS["ensemble"], "${:,.0f}"),
]

for ax, (vals, ylabel, title, color, fmt) in zip(axes, metrics_data):
    bars = ax.bar(folds, vals, color=color, alpha=0.85, edgecolor="none", width=0.6)
    ax.bar_label(bars, labels=[fmt.format(v) for v in vals],
                 padding=4, fontsize=8, color="#C8CDD8")
    ax.axhline(np.mean(vals), color="#F59E0B", linewidth=1.5,
               linestyle="--", label=f"Ortalama: {fmt.format(np.mean(vals))}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Fold Bazında {title}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

std_r2 = np.std(fold_r2s)
verdict_stab = "✓ Stabil" if std_r2 < 0.01 else "⚠️ Yüksek varyans"
fig.text(0.5, -0.04,
         f"R² std: {std_r2:.4f}  |  {verdict_stab}  "
         f"({'< 0.01 → tutarlı' if std_r2 < 0.01 else '> 0.01 → dikkat'})",
         ha="center", fontsize=10, color="#10B981" if std_r2 < 0.01 else "#F59E0B")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "77_diag_fold_stability.png"), dpi=150)
plt.close()
print("  ✓ 77_diag_fold_stability.png")


# ── Özet ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  DIAGNOSTIC ÖZET")
print("="*70)
print(f"  Train R²:      {train_r2:.4f}")
print(f"  OOF R²:        {oof_r2:.4f}")
print(f"  Overfit gap:   {overfit_gap:.4f}  {'✓ Normal' if overfit_gap < 0.05 else '⚠️ Dikkat'}")
print(f"  Fold R² std:   {np.std(fold_r2s):.4f}  {'✓ Stabil' if np.std(fold_r2s) < 0.01 else '⚠️ Yüksek'}")
print(f"  Shapiro p-val: {pval:.4f}  {'Normal' if pval > 0.05 else 'Non-normal'} dağılım")
print("="*70)
print("\n  Grafikler outputs/ klasörüne kaydedildi.")
print("  73_diag_train_vs_oof.png")
print("  74_diag_learning_curve.png")
print("  75_diag_residual_by_range.png")
print("  76_diag_error_distribution.png")
print("  77_diag_fold_stability.png")