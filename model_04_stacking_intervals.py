"""
model_04_stacking_intervals.py

LinkedIn Job Analysis — Model 04
Stacking Ensemble + Prediction Intervals + Segment Error Analysis

Purpose
-------
Model 04 uses the already selected Model 03 pipelines/joblib and the prepared
Model 03 feature matrix to build an additional Ridge stacking layer.

It adds:
  1. Ridge meta-learner over Model 03 base model predictions
  2. Three prediction interval methods:
       A) Residual quantile interval
       B) Quantile regression interval
       C) Conformal interval
  3. Segment-level error analysis:
       - experience level
       - industry
       - state
       - remote flag
       - salary band
       - title family
  4. Model 03 vs Model 04 comparison
  5. Saved Model 04 joblib artifact for dashboard/report usage

Run these first:
    python model_03_salary_advanced_progress.py
    python prepare_shap_data.py

Expected files:
    models/best_salary_model_03_advanced_progress.joblib
    models/shap_X.parquet
    models/shap_y.npy
    models/shap_feature_cols.json

Run:
    python model_04_stacking_intervals.py
"""

import os
import sys
import json
import time
import types
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

try:
    from utils_progress import ProgressBar, StepTracker
except Exception:
    class ProgressBar:
        def __init__(self, total=1, title="", unit="steps"):
            self.total = total
            self.i = 0
            self.unit = unit
            if title:
                print(f"  {title}", flush=True)
        def step(self, label=""):
            self.i += 1
            print(f"    {self.i}/{self.total} {self.unit} {label}", flush=True)
        def finish(self, label="Done"):
            print(f"    {self.total}/{self.total} {self.unit} {label}", flush=True)
    class StepTracker:
        def __init__(self, total_steps=1, script_name=""):
            self.total_steps = total_steps
            self.t0 = time.time()
            print("\n" + "=" * 82)
            print(f"  {script_name}")
            print("=" * 82)
        def start(self, step, label):
            self._step_t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step {step}/{self.total_steps} {label}", flush=True)
        def done(self, step, label=None):
            msg = f"  ✓ Step {step}/{self.total_steps} completed in {time.time()-self._step_t0:.1f}s"
            if label:
                msg += f"  {label}"
            print(msg, flush=True)
        def finish(self):
            print("=" * 82)
            print(f"  ✓ All steps completed in {(time.time()-self.t0)/60:.2f} min")
            print("=" * 82)


# =============================================================================
# Custom transformers required for loading Model 03 joblib
# =============================================================================
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

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
            if col not in X_df.columns:
                continue
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
                X_df[new_col] = self.global_
                continue
            X_df[new_col] = (
                X_df[col]
                .fillna("Unknown")
                .astype(str)
                .map(self.maps_[col])
                .fillna(self.global_)
                .astype(float)
            )
        return X_df


for module_name in ["__main__", "model_03_salary_advanced_progress", "model_03_salary_advanced", "model_02_pipeline_v3_cv_safe"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)
    setattr(sys.modules[module_name], "LogTransformer", LogTransformer)
    setattr(sys.modules[module_name], "MedianTargetEncoder", MedianTargetEncoder)


# =============================================================================
# Config
# =============================================================================
OUTPUT_PATH = Path("outputs")
MODELS_PATH = Path("models")
OUTPUT_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
INTERVAL_ALPHA = 0.10
CALIBRATION_SIZE = 0.20

MODEL03_PATH = MODELS_PATH / "best_salary_model_03_advanced_progress.joblib"
X_PATH = MODELS_PATH / "shap_X.parquet"
Y_PATH = MODELS_PATH / "shap_y.npy"
FEATURE_COLS_PATH = MODELS_PATH / "shap_feature_cols.json"

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
plt.rcParams.update(STYLE)
sns.set_theme(style="whitegrid")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics_dict(y_true_raw, y_pred_raw):
    return {
        "rmse": rmse(y_true_raw, y_pred_raw),
        "mae": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "r2": float(r2_score(y_true_raw, y_pred_raw)),
        "mape": safe_mape(y_true_raw, y_pred_raw),
    }


def clip_salary(values):
    return np.clip(np.asarray(values, dtype=float), 0, None)


def savefig(name):
    path = OUTPUT_PATH / name
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {name}")


def model03_predict_log(obj, X):
    if obj.get("type") == "ensemble":
        weights = obj.get("weights", {})
        pipelines = obj.get("pipelines", {})
        pred = np.zeros(len(X), dtype=float)
        total_w = 0.0
        for name, pipe in pipelines.items():
            w = float(weights.get(name, 0.0))
            if w == 0:
                continue
            pred += w * np.asarray(pipe.predict(X), dtype=float)
            total_w += w
        if total_w == 0:
            raise ValueError("Model 03 ensemble weights are missing or zero.")
        return pred / total_w
    if "pipeline" in obj:
        return np.asarray(obj["pipeline"].predict(X), dtype=float)
    raise ValueError("Unsupported Model 03 object format.")


def get_base_pipelines(obj):
    if obj.get("type") == "ensemble":
        return obj["pipelines"]
    if "pipeline" in obj:
        return {obj.get("model_name", "Model03"): obj["pipeline"]}
    raise ValueError("Unsupported model object. Expected ensemble or single pipeline.")


def make_meta_features(preds_dict):
    names = list(preds_dict.keys())
    meta = np.column_stack([preds_dict[n] for n in names])
    if len(names) >= 2:
        meta_std = np.std(meta, axis=1).reshape(-1, 1)
        meta_mean = np.mean(meta, axis=1).reshape(-1, 1)
        meta_min = np.min(meta, axis=1).reshape(-1, 1)
        meta_max = np.max(meta, axis=1).reshape(-1, 1)
        meta = np.hstack([meta, meta_mean, meta_std, meta_min, meta_max])
        feature_names = names + ["base_mean", "base_std", "base_min", "base_max"]
    else:
        feature_names = names
    return meta, feature_names


def interval_scores(y_true, pred, lower, upper):
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)) * 100)
    avg_width = float(np.mean(upper - lower))
    median_width = float(np.median(upper - lower))
    alpha = INTERVAL_ALPHA
    width = upper - lower
    penalty_low = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_high = (2 / alpha) * (y_true - upper) * (y_true > upper)
    winkler = float(np.mean(width + penalty_low + penalty_high))
    return {
        "coverage_pct": coverage,
        "avg_width": avg_width,
        "median_width": median_width,
        "winkler_score": winkler,
    }


def build_residual_quantile_interval(y_true, pred, alpha=0.10):
    residuals = np.asarray(y_true) - np.asarray(pred)
    return {
        "method": "residual_quantile",
        "q_low": float(np.quantile(residuals, alpha / 2)),
        "q_high": float(np.quantile(residuals, 1 - alpha / 2)),
    }


def apply_residual_quantile_interval(pred, params):
    lower = clip_salary(np.asarray(pred) + params["q_low"])
    upper = clip_salary(np.asarray(pred) + params["q_high"])
    return lower, upper


def build_conformal_interval(y_cal, pred_cal, alpha=0.10):
    residual_abs = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
    n = len(residual_abs)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(max(q_level, 0), 1)
    q_hat = float(np.quantile(residual_abs, q_level, method="higher"))
    return {"method": "conformal_absolute", "q_hat": q_hat, "alpha": alpha}


def apply_conformal_interval(pred, params):
    q = params["q_hat"]
    return clip_salary(np.asarray(pred) - q), clip_salary(np.asarray(pred) + q)


def build_quantile_interval(meta_train, y_train_raw, alpha=0.10):
    lower_model = GradientBoostingRegressor(
        loss="quantile", alpha=alpha / 2, n_estimators=350,
        max_depth=3, learning_rate=0.035, random_state=RANDOM_STATE,
    )
    upper_model = GradientBoostingRegressor(
        loss="quantile", alpha=1 - alpha / 2, n_estimators=350,
        max_depth=3, learning_rate=0.035, random_state=RANDOM_STATE,
    )
    lower_model.fit(meta_train, y_train_raw)
    upper_model.fit(meta_train, y_train_raw)
    return {"method": "quantile_regression", "lower_model": lower_model, "upper_model": upper_model, "alpha": alpha}


def apply_quantile_interval(meta, params, point_pred=None):
    lower = np.asarray(params["lower_model"].predict(meta), dtype=float)
    upper = np.asarray(params["upper_model"].predict(meta), dtype=float)
    if point_pred is not None:
        point_pred = np.asarray(point_pred, dtype=float)
        lower = np.minimum(lower, point_pred)
        upper = np.maximum(upper, point_pred)
    return clip_salary(lower), clip_salary(upper)


def segment_metrics(df, group_col, min_n=80, top_n=15):
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, x in df.dropna(subset=[group_col]).groupby(group_col):
        if len(x) < min_n:
            continue
        rows.append({
            group_col: key,
            "n": len(x),
            "rmse": rmse(x["actual_salary"], x["pred_salary"]),
            "mae": float(mean_absolute_error(x["actual_salary"], x["pred_salary"])),
            "r2": float(r2_score(x["actual_salary"], x["pred_salary"])) if len(x) >= 3 else np.nan,
            "bias": float(np.mean(x["pred_salary"] - x["actual_salary"])),
            "median_actual": float(np.median(x["actual_salary"])),
            "median_pred": float(np.median(x["pred_salary"])),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).head(top_n)


def plot_barh(df, y_col, x_col, title, xlabel, filename):
    if df.empty:
        print(f"  ! Skipped {filename}: no data")
        return
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(df))))
    data = df.copy().sort_values(x_col, ascending=True)
    bars = ax.barh(data[y_col].astype(str), data[x_col], color="#6366F1")
    labels = [f"${v:,.0f}" if ("RMSE" in xlabel or "MAE" in xlabel or "USD" in xlabel) else f"{v:.3f}" for v in data[x_col].values]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    if "RMSE" in xlabel or "MAE" in xlabel or "USD" in xlabel:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    savefig(filename)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    tracker = StepTracker(total_steps=9, script_name="model_04_stacking_intervals.py — Stacking + Prediction Intervals")

    tracker.start(1, "Loading Model 03 and feature matrix")
    if not MODEL03_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL03_PATH}. Run model_03_salary_advanced_progress.py first.")
    if not X_PATH.exists() or not Y_PATH.exists() or not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError("Missing shap_X/shap_y/shap_feature_cols. Run prepare_shap_data.py first.")

    model03_obj = joblib.load(MODEL03_PATH)
    base_pipelines = get_base_pipelines(model03_obj)
    base_model_names = list(base_pipelines.keys())
    X = pd.read_parquet(X_PATH)
    y_log = np.load(Y_PATH)
    y_raw = np.expm1(y_log)
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    print(f"  Model 03 type: {model03_obj.get('type')}")
    print(f"  Base models: {base_model_names}")
    print(f"  X shape: {X.shape}")
    print(f"  y range: ${y_raw.min():,.0f} - ${y_raw.max():,.0f}")
    tracker.done(1)

    tracker.start(2, "Computing Model 03 full-data predictions")
    model03_pred_log_full = model03_predict_log(model03_obj, X)
    model03_pred_raw_full = clip_salary(np.expm1(model03_pred_log_full))
    model03_full_metrics = metrics_dict(y_raw, model03_pred_raw_full)
    print(f"  Model 03 full-data R²: {model03_full_metrics['r2']:.4f}")
    print(f"  Model 03 full-data RMSE: ${model03_full_metrics['rmse']:,.0f}")
    print("  Note: full-data metrics are optimistic; OOF/CV metrics remain the reliable score.")
    tracker.done(2)

    tracker.start(3, f"Building {N_FOLDS}-fold OOF base predictions")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_base = {name: np.zeros(len(X), dtype=float) for name in base_model_names}
    fold_rows = []
    fold_bar = ProgressBar(total=N_FOLDS * len(base_model_names), title="OOF base model training", unit="fits")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y_log[tr_idx]
        y_val_raw = y_raw[val_idx]
        for name in base_model_names:
            pipe = clone(base_pipelines[name])
            pipe.fit(X_tr, y_tr)
            pred_log = np.asarray(pipe.predict(X_val), dtype=float)
            oof_base[name][val_idx] = pred_log
            pred_raw = clip_salary(np.expm1(pred_log))
            fold_rows.append({
                "fold": fold,
                "model": name,
                "rmse": rmse(y_val_raw, pred_raw),
                "mae": float(mean_absolute_error(y_val_raw, pred_raw)),
                "r2": float(r2_score(y_val_raw, pred_raw)),
            })
            fold_bar.step(f"Fold {fold} — {name}")
    fold_bar.finish()
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUTPUT_PATH / "model_04_base_oof_fold_metrics.csv", index=False)
    meta_oof, meta_feature_names = make_meta_features(oof_base)
    print(f"  Meta feature shape: {meta_oof.shape}")
    print(f"  Meta features: {meta_feature_names}")
    tracker.done(3)

    tracker.start(4, "Training Ridge stacking meta-learner")
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
    ridge.fit(meta_oof, y_log)
    stack_pred_log_oof = ridge.predict(meta_oof)
    stack_pred_raw_oof = clip_salary(np.expm1(stack_pred_log_oof))

    weights = model03_obj.get("weights", {name: 1 / len(base_model_names) for name in base_model_names})
    model03_oof_log = np.zeros(len(X), dtype=float)
    total_w = 0.0
    for name in base_model_names:
        w = float(weights.get(name, 0.0))
        model03_oof_log += w * oof_base[name]
        total_w += w
    model03_oof_log = model03_oof_log / max(total_w, 1e-12)
    model03_oof_raw = clip_salary(np.expm1(model03_oof_log))

    model03_oof_metrics = metrics_dict(y_raw, model03_oof_raw)
    model04_oof_metrics = metrics_dict(y_raw, stack_pred_raw_oof)
    print("\nOOF comparison:")
    print(f"  Model 03 weighted OOF: RMSE=${model03_oof_metrics['rmse']:,.0f}  MAE=${model03_oof_metrics['mae']:,.0f}  R²={model03_oof_metrics['r2']:.4f}")
    print(f"  Model 04 Ridge stack:  RMSE=${model04_oof_metrics['rmse']:,.0f}  MAE=${model04_oof_metrics['mae']:,.0f}  R²={model04_oof_metrics['r2']:.4f}")
    print(f"  Ridge alpha: {ridge.alpha_}")
    coef_df = pd.DataFrame({"feature": meta_feature_names, "coefficient": ridge.coef_}).sort_values("coefficient", ascending=False)
    coef_df.to_csv(OUTPUT_PATH / "model_04_ridge_coefficients.csv", index=False)
    tracker.done(4)

    tracker.start(5, "Building three prediction interval methods")
    idx_all = np.arange(len(y_raw))
    try:
        strat = pd.qcut(y_raw, q=8, labels=False, duplicates="drop")
    except Exception:
        strat = None
    train_idx, cal_idx = train_test_split(idx_all, test_size=CALIBRATION_SIZE, random_state=RANDOM_STATE, stratify=strat)
    meta_train = meta_oof[train_idx]
    y_train_raw = y_raw[train_idx]
    y_cal_raw = y_raw[cal_idx]
    pred_train = stack_pred_raw_oof[train_idx]
    pred_cal = stack_pred_raw_oof[cal_idx]

    residual_params = build_residual_quantile_interval(y_train_raw, pred_train, alpha=INTERVAL_ALPHA)
    residual_lower, residual_upper = apply_residual_quantile_interval(stack_pred_raw_oof, residual_params)

    quantile_params = build_quantile_interval(meta_train, y_train_raw, alpha=INTERVAL_ALPHA)
    quantile_lower, quantile_upper = apply_quantile_interval(meta_oof, quantile_params, point_pred=stack_pred_raw_oof)

    conformal_params = build_conformal_interval(y_cal_raw, pred_cal, alpha=INTERVAL_ALPHA)
    conformal_lower, conformal_upper = apply_conformal_interval(stack_pred_raw_oof, conformal_params)

    interval_methods = {
        "Residual Quantile": (residual_lower, residual_upper, residual_params),
        "Quantile Regression": (quantile_lower, quantile_upper, {"method": "quantile_regression"}),
        "Conformal": (conformal_lower, conformal_upper, conformal_params),
    }
    interval_rows = []
    for method, (lower, upper, params) in interval_methods.items():
        interval_rows.append({"method": method, **interval_scores(y_raw, stack_pred_raw_oof, lower, upper)})
    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(OUTPUT_PATH / "model_04_interval_method_scores.csv", index=False)
    print("\nInterval methods:")
    for _, row in interval_df.iterrows():
        print(f"  {row['method']:<20} coverage={row['coverage_pct']:.1f}%  avg_width=${row['avg_width']:,.0f}  winkler=${row['winkler_score']:,.0f}")
    tracker.done(5)

    tracker.start(6, "Creating error analysis dataframe")
    result_df = X.copy()
    result_df["actual_salary"] = y_raw
    result_df["pred_salary"] = stack_pred_raw_oof
    result_df["pred_log"] = stack_pred_log_oof
    result_df["residual"] = result_df["pred_salary"] - result_df["actual_salary"]
    result_df["abs_error"] = result_df["residual"].abs()
    result_df["ape"] = np.where(result_df["actual_salary"] > 0, result_df["abs_error"] / result_df["actual_salary"] * 100, np.nan)
    bins = [0, 50_000, 75_000, 100_000, 130_000, 160_000, 200_000, 300_000]
    labels = ["<$50k", "$50k–75k", "$75k–100k", "$100k–130k", "$130k–160k", "$160k–200k", "$200k+"]
    result_df["salary_band"] = pd.cut(result_df["actual_salary"], bins=bins, labels=labels, include_lowest=True)
    for col in ["formatted_experience_level", "primary_industry", "state_final", "title_family", "pay_period"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna("Unknown").astype(str)
    if "remote_flag" in result_df.columns:
        result_df["remote_label"] = result_df["remote_flag"].map({1: "Remote", 0: "Non-remote / unknown"}).fillna("Unknown")
    else:
        result_df["remote_label"] = "Unknown"
    result_df.sort_values("residual", ascending=False).head(25).to_csv(OUTPUT_PATH / "model_04_top_overpredicted.csv", index=False)
    result_df.sort_values("residual", ascending=True).head(25).to_csv(OUTPUT_PATH / "model_04_top_underpredicted.csv", index=False)
    segments = {
        "experience": "formatted_experience_level",
        "industry": "primary_industry",
        "state": "state_final",
        "remote": "remote_label",
        "salary_band": "salary_band",
        "title_family": "title_family",
        "pay_period": "pay_period",
    }
    segment_tables = {}
    for name, col in segments.items():
        min_n = 40 if name in ("remote", "salary_band", "pay_period") else 80
        tbl = segment_metrics(result_df, col, min_n=min_n, top_n=20)
        segment_tables[name] = tbl
        tbl.to_csv(OUTPUT_PATH / f"model_04_error_by_{name}.csv", index=False)
    tracker.done(6)

    tracker.start(7, "Generating model comparison plots")
    comparison = pd.DataFrame([
        {"model": "Model 03 Weighted OOF", "RMSE": model03_oof_metrics["rmse"], "MAE": model03_oof_metrics["mae"], "Raw R²": model03_oof_metrics["r2"]},
        {"model": "Model 04 Ridge Stacking", "RMSE": model04_oof_metrics["rmse"], "MAE": model04_oof_metrics["mae"], "Raw R²": model04_oof_metrics["r2"]},
    ])
    comparison.to_csv(OUTPUT_PATH / "model_04_comparison_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "Raw R²"]):
        bars = ax.bar(comparison["model"], comparison[metric], color=["#6366F1", "#10B981"])
        if metric == "Raw R²":
            labels_bar = [f"{v:.4f}" for v in comparison[metric]]
        else:
            labels_bar = [f"${v:,.0f}" for v in comparison[metric]]
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.bar_label(bars, labels=labels_bar, padding=4, fontsize=9)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Model 03 Weighted Ensemble vs Model 04 Ridge Stacking", fontsize=14, fontweight="bold")
    savefig("78_model04_stacking_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    coef_plot = coef_df.copy()
    bars = ax.barh(coef_plot["feature"][::-1], coef_plot["coefficient"][::-1], color="#10B981")
    ax.axvline(0, color="#C8CDD8", linewidth=1)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in coef_plot["coefficient"][::-1]], padding=4, fontsize=8)
    ax.set_title("Model 04 Ridge Meta-Learner Coefficients")
    ax.set_xlabel("Coefficient")
    ax.spines[["top", "right"]].set_visible(False)
    savefig("79_model04_ridge_coefficients.png")
    tracker.done(7)

    tracker.start(8, "Generating interval and calibration plots")
    interval_plot = interval_df.copy()
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(interval_plot))
    width = 0.35
    bars1 = ax1.bar(x - width/2, interval_plot["coverage_pct"], width, label="Coverage (%)", color="#6366F1")
    ax1.axhline((1 - INTERVAL_ALPHA) * 100, color="#F59E0B", linestyle="--", linewidth=1.5, label="Target coverage")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(interval_plot["method"], rotation=15, ha="right")
    ax1.set_ylim(0, max(100, interval_plot["coverage_pct"].max() + 5))
    ax1.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=8)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, interval_plot["avg_width"], width, label="Avg width", color="#10B981")
    ax2.set_ylabel("Average interval width (USD)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax2.bar_label(bars2, labels=[f"${v/1000:.0f}k" for v in interval_plot["avg_width"]], padding=3, fontsize=8)
    ax1.set_title("Prediction Interval Methods — Coverage vs Width")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    savefig("80_model04_interval_methods.png")

    rng = np.random.RandomState(RANDOM_STATE)
    sample_n = min(80, len(result_df))
    sample_idx = rng.choice(len(result_df), size=sample_n, replace=False)
    sample = pd.DataFrame({
        "actual": y_raw[sample_idx],
        "pred": stack_pred_raw_oof[sample_idx],
        "conformal_low": conformal_lower[sample_idx],
        "conformal_high": conformal_upper[sample_idx],
        "quantile_low": quantile_lower[sample_idx],
        "quantile_high": quantile_upper[sample_idx],
    }).sort_values("actual").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    sx = np.arange(len(sample))
    ax.scatter(sx, sample["actual"], s=18, color="#F8FAFC", label="Actual", zorder=4)
    ax.scatter(sx, sample["pred"], s=18, color="#10B981", label="Prediction", zorder=4)
    ax.fill_between(sx, sample["conformal_low"], sample["conformal_high"], color="#6366F1", alpha=0.18, label="Conformal interval")
    ax.fill_between(sx, sample["quantile_low"], sample["quantile_high"], color="#F59E0B", alpha=0.12, label="Quantile interval")
    ax.set_title("Model 04 Example Prediction Intervals")
    ax.set_xlabel("Random sample sorted by actual salary")
    ax.set_ylabel("Salary (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("81_model04_prediction_intervals_sample.png")

    cal_df = result_df[["actual_salary", "pred_salary"]].copy()
    cal_df["pred_decile"] = pd.qcut(cal_df["pred_salary"], q=10, labels=False, duplicates="drop")
    dec = cal_df.groupby("pred_decile").agg(avg_pred=("pred_salary", "mean"), avg_actual=("actual_salary", "mean"), n=("actual_salary", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(dec["avg_pred"], dec["avg_actual"], s=70, color="#10B981")
    for _, row in dec.iterrows():
        ax.annotate(f"D{int(row['pred_decile'])+1}", (row["avg_pred"], row["avg_actual"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    mn = min(dec["avg_pred"].min(), dec["avg_actual"].min())
    mx = max(dec["avg_pred"].max(), dec["avg_actual"].max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="#F59E0B", label="Perfect calibration")
    ax.set_title("Model 04 Calibration by Prediction Decile")
    ax.set_xlabel("Average predicted salary")
    ax.set_ylabel("Average actual salary")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend()
    ax.grid(alpha=0.3)
    savefig("82_model04_calibration_deciles.png")
    tracker.done(8)

    tracker.start(9, "Generating segment error analysis plots and saving artifact")
    if not segment_tables["experience"].empty:
        plot_barh(segment_tables["experience"], "formatted_experience_level", "rmse", "Model 04 RMSE by Experience Level", "RMSE (USD)", "83_model04_error_by_experience.png")
    if not segment_tables["industry"].empty:
        plot_barh(segment_tables["industry"], "primary_industry", "rmse", "Model 04 Highest RMSE Industries", "RMSE (USD)", "84_model04_error_by_industry.png")
    if not segment_tables["state"].empty:
        plot_barh(segment_tables["state"], "state_final", "rmse", "Model 04 Highest RMSE States", "RMSE (USD)", "85_model04_error_by_state.png")
    if not segment_tables["salary_band"].empty:
        plot_barh(segment_tables["salary_band"], "salary_band", "rmse", "Model 04 RMSE by Salary Band", "RMSE (USD)", "86_model04_error_by_salary_band.png")
    if not segment_tables["title_family"].empty:
        plot_barh(segment_tables["title_family"], "title_family", "rmse", "Model 04 Highest RMSE Title Families", "RMSE (USD)", "87_model04_error_by_title_family.png")

    bias_tbl = segment_tables["industry"].copy()
    if not bias_tbl.empty:
        low_bias = bias_tbl.sort_values("bias").head(8)
        high_bias = bias_tbl.sort_values("bias").tail(8)
        bias_tbl = pd.concat([low_bias, high_bias], ignore_index=True).drop_duplicates(subset=["primary_industry"])
        fig, ax = plt.subplots(figsize=(12, 7))
        colors_bias = ["#F43F5E" if v < 0 else "#10B981" for v in bias_tbl["bias"]]
        bars = ax.barh(bias_tbl["primary_industry"], bias_tbl["bias"], color=colors_bias)
        ax.axvline(0, color="#C8CDD8", linewidth=1)
        ax.bar_label(bars, labels=[f"${v:,.0f}" for v in bias_tbl["bias"]], padding=4, fontsize=8)
        ax.set_title("Model 04 Bias by Industry (Prediction - Actual)")
        ax.set_xlabel("Average bias (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        savefig("88_model04_bias_by_industry.png")

    title_col = "title_text" if "title_text" in result_df.columns else None
    if title_col:
        title_errors = (
            result_df.groupby(title_col)
            .agg(n=("actual_salary", "size"), mae=("abs_error", "mean"), bias=("residual", "mean"))
            .query("n >= 3")
            .sort_values("mae", ascending=False)
            .head(15)
            .reset_index()
        )
        title_errors.to_csv(OUTPUT_PATH / "model_04_top_error_titles.csv", index=False)
        plot_barh(title_errors, title_col, "mae", "Model 04 Highest-MAE Job Titles", "MAE (USD)", "89_model04_top_error_titles.png")

    result_df["model03_oof_pred_salary"] = model03_oof_raw
    result_df["residual_interval_low"] = residual_lower
    result_df["residual_interval_high"] = residual_upper
    result_df["quantile_interval_low"] = quantile_lower
    result_df["quantile_interval_high"] = quantile_upper
    result_df["conformal_interval_low"] = conformal_lower
    result_df["conformal_interval_high"] = conformal_upper
    result_df.to_parquet(OUTPUT_PATH / "model_04_oof_predictions_intervals.parquet", index=False)

    final_ridge = Ridge(alpha=float(ridge.alpha_))
    final_ridge.fit(meta_oof, y_log)

    model04_obj = {
        "type": "stacking_ridge_with_intervals",
        "base_model_path": str(MODEL03_PATH),
        "base_model_names": base_model_names,
        "model03_object": model03_obj,
        "ridge_meta_model": final_ridge,
        "ridge_alpha": float(ridge.alpha_),
        "meta_feature_names": meta_feature_names,
        "feature_cols": feature_cols,
        "log_target": True,
        "interval_alpha": INTERVAL_ALPHA,
        "intervals": {
            "residual_quantile": residual_params,
            "conformal": conformal_params,
            "quantile_regression": quantile_params,
        },
        "metrics": {
            "model03_weighted_oof": model03_oof_metrics,
            "model04_ridge_oof": model04_oof_metrics,
            "interval_scores": interval_df.to_dict(orient="records"),
        },
        "notes": {
            "prediction_scale": "Meta model predicts log salary; convert with expm1.",
            "interval_methods": "Residual quantile, quantile regression, and conformal intervals are computed on OOF predictions.",
        },
    }
    model04_path = MODELS_PATH / "best_salary_model_04_stacking_intervals.joblib"
    joblib.dump(model04_obj, model04_path)
    metrics_path = OUTPUT_PATH / "model_04_metrics.json"
    metrics_json = {
        "model03_weighted_oof": model03_oof_metrics,
        "model04_ridge_oof": model04_oof_metrics,
        "ridge_alpha": float(ridge.alpha_),
        "ridge_coefficients": coef_df.to_dict(orient="records"),
        "interval_scores": interval_df.to_dict(orient="records"),
        "files": {
            "model": str(model04_path),
            "predictions": str(OUTPUT_PATH / "model_04_oof_predictions_intervals.parquet"),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    print("\nSaved:")
    print(f"  ✓ {model04_path}")
    print(f"  ✓ {metrics_path}")
    print("  ✓ outputs/model_04_oof_predictions_intervals.parquet")
    print("\nFinal Model 04 summary:")
    print(f"  Model 04 OOF RMSE: ${model04_oof_metrics['rmse']:,.0f}")
    print(f"  Model 04 OOF MAE:  ${model04_oof_metrics['mae']:,.0f}")
    print(f"  Model 04 OOF R²:   {model04_oof_metrics['r2']:.4f}")
    tracker.done(9)
    tracker.finish()
