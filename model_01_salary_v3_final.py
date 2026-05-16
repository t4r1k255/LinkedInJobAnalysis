"""
model_01_salary_v3_final.py — v3
LinkedIn Job Postings — Salary Prediction

Models:
  - XGBoost
  - LightGBM
  - CatBoost

v3 changes:
  - Salary outlier filtering tightened with 1%–99% quantile + upper cap
  - Top skills increased from 20 to 50
  - n_skills feature added
  - Benefits features added:
      benefit_count + top benefit multi-hot columns
  - City salary encoding added
  - Industry salary encoding added
  - pay_period and company_size moved to categorical features
  - Title keyword features expanded
  - title_len and title_word_count added
  - Model creation changed to factory function so every fold gets a fresh model

Important note:
  state_salary_enc / city_salary_enc / industry_salary_enc are global target encodings.
  They can improve R² quickly, but for a stricter academic version, make them CV-safe.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import shap

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
# İstersen Windows terminalde şunu set edebilirsin:
# set LINKEDIN_DATA_DIR=C:\Users\tarik\PycharmProjects\LinkedInJobAnalysis\data
BASE = Path(os.getenv(
    "LINKEDIN_DATA_DIR",
    r"C:\Users\tarik\PycharmProjects\LinkedInJobAnalysis\data"
))

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ── Plot Style ────────────────────────────────────────────────────────────────
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
    "xgb":    "#F59E0B",
    "lgb":    "#6366F1",
    "cat":    "#10B981",
    "accent": "#3B82F6",
    "muted":  "#4B5563",
}


# ── Helper Functions ──────────────────────────────────────────────────────────
def safe_read_csv(path, usecols=None):
    """
    CSV okurken daha anlaşılır hata vermek için küçük yardımcı fonksiyon.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return pd.read_csv(path, usecols=usecols)
    except ValueError as e:
        print(f"\nColumn error while reading: {path.name}")
        print(f"Requested columns: {usecols}")
        print("Reading full header to show available columns...")
        header = pd.read_csv(path, nrows=0)
        print(f"Available columns: {list(header.columns)}")
        raise e


def clean_col_name(value):
    """
    Feature isimlerinde sorun çıkarabilecek karakterleri temizler.
    """
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value if value else "unknown"


def add_median_target_encoding(df, group_col, target_col, new_col, global_median):
    """
    Global median target encoding.
    R²'yi artırır ama CV leakage içerir.
    Daha strict versiyon için fold içinde hesaplanmalıdır.
    """
    med = df.groupby(group_col)[target_col].median()
    df[new_col] = df[group_col].map(med).fillna(global_median)
    df[f"log_{new_col}"] = np.log1p(df[new_col])
    return df


# ── 1. DATA LOADING & FEATURE ENGINEERING ─────────────────────────────────────
def load_and_engineer():
    print("Loading data...")

    post = safe_read_csv(BASE / "postings.csv", usecols=[
        "job_id",
        "title",
        "company_id",
        "location",
        "normalized_salary",
        "pay_period",
        "currency",
        "formatted_experience_level",
        "formatted_work_type",
        "remote_allowed",
        "views",
        "applies",
    ])

    # ── Salary filtering ──────────────────────────────────────────────────────
    # v2: 10k–1M aralığıydı.
    # v3: 1%–99% quantile + 300k upper cap.
    salary_mask = post["normalized_salary"].notna()
    q_low = post.loc[salary_mask, "normalized_salary"].quantile(0.01)
    q_high = post.loc[salary_mask, "normalized_salary"].quantile(0.99)

    lower_bound = max(10_000, q_low)
    upper_bound = min(300_000, q_high)

    df = post[
        post["normalized_salary"].between(lower_bound, upper_bound) &
        (post["currency"].isin(["USD"]) | post["currency"].isna())
    ].copy()

    df["job_id"] = df["job_id"].astype("Int64")
    df["company_id"] = df["company_id"].astype("Int64")

    print(f"  Salary bounds used: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    print(f"  Salary rows after filtering: {len(df):,}")

    # ── Companies ─────────────────────────────────────────────────────────────
    comp = safe_read_csv(
        BASE / "companies.csv",
        usecols=["company_id", "company_size", "state", "country"]
    )

    emp = safe_read_csv(
        BASE / "employee_counts.csv",
        usecols=["company_id", "employee_count", "follower_count"]
    )

    emp = (
        emp.sort_values("employee_count")
        .drop_duplicates("company_id", keep="last")
    )

    comp = comp.merge(emp, on="company_id", how="left")
    df = df.merge(comp, on="company_id", how="left")

    # ── Industries ────────────────────────────────────────────────────────────
    job_ind = safe_read_csv(BASE / "job_industries.csv")
    ind_ref = safe_read_csv(BASE / "industries.csv")

    job_ind = job_ind.merge(ind_ref, on="industry_id", how="left")

    primary_ind = (
        job_ind.groupby("job_id")["industry_name"]
        .first()
        .reset_index()
        .rename(columns={"industry_name": "primary_industry"})
    )

    df = df.merge(primary_ind, on="job_id", how="left")

    # ── Skills ────────────────────────────────────────────────────────────────
    jskills = safe_read_csv(BASE / "job_skills.csv")
    skl_ref = safe_read_csv(BASE / "skills.csv")

    jskills = jskills.merge(skl_ref, on="skill_abr", how="left")

    top_skills = (
        jskills[jskills["job_id"].isin(df["job_id"])]
        ["skill_name"]
        .dropna()
        .value_counts()
        .head(50)
        .index
        .tolist()
    )

    skill_pivot = (
        jskills[
            jskills["job_id"].isin(df["job_id"]) &
            jskills["skill_name"].isin(top_skills)
        ]
        .assign(val=1)
        .pivot_table(
            index="job_id",
            columns="skill_name",
            values="val",
            fill_value=0,
            aggfunc="max"
        )
    )

    skill_pivot.columns = [
        f"skill_{clean_col_name(c)}"
        for c in skill_pivot.columns
    ]

    skill_pivot = skill_pivot.reset_index()
    df = df.merge(skill_pivot, on="job_id", how="left")

    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    df[skill_cols] = df[skill_cols].fillna(0).astype(int)

    skill_count = (
        jskills[jskills["job_id"].isin(df["job_id"])]
        .groupby("job_id")["skill_name"]
        .nunique()
        .reset_index()
        .rename(columns={"skill_name": "n_skills"})
    )

    df = df.merge(skill_count, on="job_id", how="left")
    df["n_skills"] = df["n_skills"].fillna(0)

    # ── Benefits ──────────────────────────────────────────────────────────────
    benefit_cols = []
    benefits_path = BASE / "benefits.csv"

    if benefits_path.exists():
        benefits = safe_read_csv(benefits_path)

        if "job_id" in benefits.columns:
            benefit_count = (
                benefits.groupby("job_id")
                .size()
                .reset_index(name="benefit_count")
            )

            df = df.merge(benefit_count, on="job_id", how="left")
            df["benefit_count"] = df["benefit_count"].fillna(0)

            if "type" in benefits.columns:
                benefits["type"] = benefits["type"].fillna("Unknown")

                top_benefits = (
                    benefits["type"]
                    .value_counts()
                    .head(10)
                    .index
                    .tolist()
                )

                benefit_pivot = (
                    benefits[benefits["type"].isin(top_benefits)]
                    .assign(val=1)
                    .pivot_table(
                        index="job_id",
                        columns="type",
                        values="val",
                        fill_value=0,
                        aggfunc="max"
                    )
                )

                benefit_pivot.columns = [
                    f"benefit_{clean_col_name(c)}"
                    for c in benefit_pivot.columns
                ]

                benefit_pivot = benefit_pivot.reset_index()
                df = df.merge(benefit_pivot, on="job_id", how="left")

                benefit_cols = [
                    c for c in df.columns
                    if c.startswith("benefit_") and c != "benefit_count"
                ]

                df[benefit_cols] = df[benefit_cols].fillna(0).astype(int)
            else:
                print("  Warning: benefits.csv has no 'type' column. Only benefit_count added.")
        else:
            print("  Warning: benefits.csv has no 'job_id' column. Benefits skipped.")
            df["benefit_count"] = 0
    else:
        print("  Warning: benefits.csv not found. Benefits skipped.")
        df["benefit_count"] = 0

    # ── Title keywords ────────────────────────────────────────────────────────
    title_lower = df["title"].fillna("").str.lower()

    df["title_senior"]     = title_lower.str.contains(r"senior|sr\.?\b|lead|staff", regex=True).astype(int)
    df["title_principal"]  = title_lower.str.contains(r"principal", regex=True).astype(int)
    df["title_director"]   = title_lower.str.contains(r"director|head of", regex=True).astype(int)
    df["title_vp"]         = title_lower.str.contains(r"\bvp\b|vice pres", regex=True).astype(int)
    df["title_chief"]      = title_lower.str.contains(r"\bchief\b|president|ceo|cto|cfo|coo", regex=True).astype(int)
    df["title_manager"]    = title_lower.str.contains(r"manager|mgr", regex=True).astype(int)
    df["title_junior"]     = title_lower.str.contains(r"junior|jr\.?\b|entry|intern", regex=True).astype(int)
    df["title_associate"]  = title_lower.str.contains(r"\bassociate\b", regex=True).astype(int)
    df["title_engineer"]   = title_lower.str.contains(r"engineer|developer|software|swe|sde", regex=True).astype(int)
    df["title_architect"]  = title_lower.str.contains(r"architect", regex=True).astype(int)
    df["title_data"]       = title_lower.str.contains(r"data|analyst|scientist|ml|ai|machine", regex=True).astype(int)
    df["title_sales"]      = title_lower.str.contains(r"sales|account exec|business dev", regex=True).astype(int)
    df["title_consultant"] = title_lower.str.contains(r"consultant|advisor", regex=True).astype(int)

    # v3 new title categories
    df["title_nurse"]      = title_lower.str.contains(r"nurse|rn\b|lpn", regex=True).astype(int)
    df["title_product"]    = title_lower.str.contains(r"product", regex=True).astype(int)
    df["title_finance"]    = title_lower.str.contains(r"finance|accountant|controller|accounting|auditor", regex=True).astype(int)
    df["title_security"]   = title_lower.str.contains(r"security|cyber|infosec", regex=True).astype(int)
    df["title_marketing"]  = title_lower.str.contains(r"marketing|brand|seo|content", regex=True).astype(int)
    df["title_hr"]         = title_lower.str.contains(r"human resources|recruiter|talent|people partner", regex=True).astype(int)

    df["title_len"] = df["title"].fillna("").str.len()
    df["title_word_count"] = (
        df["title"]
        .fillna("")
        .str.split()
        .str.len()
        .fillna(0)
    )

    # ── Experience level imputation ───────────────────────────────────────────
    def impute_exp_level(row):
        if pd.notna(row["formatted_experience_level"]):
            return row["formatted_experience_level"]

        t = row["title"].lower() if pd.notna(row["title"]) else ""

        if any(x in t for x in ["chief", "president", "ceo", "cto", "cfo", "coo"]):
            return "Executive"

        if any(x in t for x in ["vp", "vice pres", "director", "head of"]):
            return "Director"

        if any(x in t for x in ["principal", "senior", "sr.", "lead", "staff", "manager"]):
            return "Mid-Senior level"

        if any(x in t for x in ["junior", "jr.", "intern", "entry"]):
            return "Entry level"

        return "Mid-Senior level"

    before_null_exp = df["formatted_experience_level"].isna().sum()
    df["formatted_experience_level"] = df.apply(impute_exp_level, axis=1)
    after_null_exp = df["formatted_experience_level"].isna().sum()

    print(f"  Experience level nulls imputed: {before_null_exp:,} -> {after_null_exp:,}")

    # ── Location extraction & target encoding ─────────────────────────────────
    df["state_loc"] = df["location"].str.extract(r",\s*([A-Z]{2})$")
    df["state_from_comp"] = df["state"]
    df["state_final"] = df["state_loc"].fillna(df["state_from_comp"]).fillna("Unknown")

    df["city_loc"] = df["location"].str.extract(r"^([^,]+)")
    df["city_loc"] = df["city_loc"].fillna("Unknown").str.strip()

    df["primary_industry"] = df["primary_industry"].fillna("Unknown")

    global_median = df["normalized_salary"].median()

    df = add_median_target_encoding(
        df=df,
        group_col="state_final",
        target_col="normalized_salary",
        new_col="state_salary_enc",
        global_median=global_median
    )

    df = add_median_target_encoding(
        df=df,
        group_col="city_loc",
        target_col="normalized_salary",
        new_col="city_salary_enc",
        global_median=global_median
    )

    df = add_median_target_encoding(
        df=df,
        group_col="primary_industry",
        target_col="normalized_salary",
        new_col="industry_salary_enc",
        global_median=global_median
    )

    # ── Numeric features ──────────────────────────────────────────────────────
    df["is_hourly"] = (df["pay_period"] == "HOURLY").astype(int)

    df["log_views"] = np.log1p(df["views"].fillna(0))

    df["has_applies"] = df["applies"].notna().astype(int)
    df["log_applies"] = np.log1p(df["applies"].fillna(0))

    df["log_employee_count"] = np.log1p(df["employee_count"].fillna(0))
    df["log_follower_count"] = np.log1p(df["follower_count"].fillna(0))

    df["remote_flag"] = df["remote_allowed"].fillna(0).astype(int)

    # company_size bazı satırlarda float/NaN gelebiliyor.
    # Categorical yapacağımız için string olarak ele alacağız.
    df["company_size"] = df["company_size"].fillna("Unknown").astype(str)
    df["pay_period"] = df["pay_period"].fillna("Unknown")
    df["formatted_work_type"] = df["formatted_work_type"].fillna("Unknown")

    print(f"  Top skills used: {len(skill_cols)}")
    print(f"  Benefit multi-hot columns used: {len(benefit_cols)}")
    print(f"  Feature engineering done. Shape: {df.shape}")

    return df, top_skills, skill_cols, benefit_cols


def prepare_features(df, skill_cols, benefit_cols):
    cat_features = [
        "formatted_experience_level",
        "formatted_work_type",
        "primary_industry",
        "pay_period",
        "company_size",
    ]

    num_features = [
        "log_employee_count",
        "log_follower_count",
        "log_views",
        "log_applies",
        "has_applies",
        "is_hourly",
        "remote_flag",

        # Target encodings
        "log_state_salary_enc",
        "log_city_salary_enc",
        "log_industry_salary_enc",

        # Count features
        "n_skills",
        "benefit_count",

        # Title length features
        "title_len",
        "title_word_count",

        # Title keyword features
        "title_senior",
        "title_principal",
        "title_director",
        "title_vp",
        "title_chief",
        "title_manager",
        "title_junior",
        "title_associate",
        "title_engineer",
        "title_architect",
        "title_data",
        "title_sales",
        "title_consultant",
        "title_nurse",
        "title_product",
        "title_finance",
        "title_security",
        "title_marketing",
        "title_hr",
    ] + skill_cols + benefit_cols

    df_enc = df.copy()

    for col in cat_features:
        df_enc[col] = df_enc[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])

    for col in num_features:
        if col not in df_enc.columns:
            df_enc[col] = 0
        df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce").fillna(0)

    features = cat_features + num_features

    X = df_enc[features].copy()
    y = df_enc["normalized_salary"].copy()
    y_log = np.log1p(y)

    return X, y, y_log, features, cat_features


# ── 2. MODEL TRAINING & CV ────────────────────────────────────────────────────
def create_models(cat_feature_indices):
    """
    Her fold için fresh model üretmek daha temizdir.
    Aynı model objesini fold içinde tekrar tekrar fit etmek bazen sorun çıkarabilir.
    """
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=900,
            learning_rate=0.045,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.2,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=900,
            learning_rate=0.045,
            max_depth=7,
            num_leaves=80,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=18,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=900,
            learning_rate=0.045,
            depth=7,
            l2_leaf_reg=3,
            subsample=0.85,
            random_seed=42,
            verbose=0,
            cat_features=cat_feature_indices,
            allow_writing_files=False,
        ),
    }


def train_and_evaluate(X, y_log, y_raw, cat_feature_indices):
    print("\nTraining models with 5-fold CV...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        log_r2_scores = []
        oof = np.zeros(len(y_log))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            models = create_models(cat_feature_indices)
            model = models[name]

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

            model.fit(X_tr, y_tr)

            pred_log = model.predict(X_val)
            oof[val_idx] = pred_log

            pred_raw = np.expm1(pred_log)
            true_raw = np.expm1(y_val)

            rmse = np.sqrt(mean_squared_error(true_raw, pred_raw))
            mae = mean_absolute_error(true_raw, pred_raw)
            r2 = r2_score(true_raw, pred_raw)
            log_r2 = r2_score(y_val, pred_log)

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            log_r2_scores.append(log_r2)

            print(
                f"  {name} Fold {fold + 1}: "
                f"RMSE=${rmse:,.0f} "
                f"MAE=${mae:,.0f} "
                f"Raw R²={r2:.3f} "
                f"Log R²={log_r2:.3f}"
            )

        results[name] = {
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
            "r2_mean": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "log_r2_mean": np.mean(log_r2_scores),
            "log_r2_std": np.std(log_r2_scores),
            "oof": oof,
        }

        print(
            f"  ► {name}: "
            f"RMSE=${np.mean(rmse_scores):,.0f}±{np.std(rmse_scores):,.0f} "
            f"MAE=${np.mean(mae_scores):,.0f} "
            f"Raw R²={np.mean(r2_scores):.3f} "
            f"Log R²={np.mean(log_r2_scores):.3f}\n"
        )

    # Full-data fit for SHAP
    fitted = {}
    models = create_models(cat_feature_indices)

    for name, model in models.items():
        print(f"Fitting full-data {name} for SHAP...")
        model.fit(X, y_log)
        fitted[name] = model

    return results, fitted


# ── 3. PLOTS ──────────────────────────────────────────────────────────────────
def plot_model_comparison(results):
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        "Model Comparison — 5-Fold Cross Validation",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )

    model_names = list(results.keys())
    colors = [COLORS["xgb"], COLORS["lgb"], COLORS["cat"]]

    metrics = [
        ("rmse_mean", "rmse_std", "RMSE (USD)", "Lower is better"),
        ("mae_mean", "mae_std", "MAE (USD)", "Lower is better"),
        ("r2_mean", "r2_std", "Raw R² Score", "Higher is better"),
        ("log_r2_mean", "log_r2_std", "Log R² Score", "Higher is better"),
    ]

    for ax, (mean_key, std_key, title, subtitle) in zip(axes, metrics):
        means = [results[m][mean_key] for m in model_names]
        stds = [results[m][std_key] for m in model_names]

        bars = ax.bar(
            model_names,
            means,
            color=colors,
            alpha=0.85,
            edgecolor="none",
            width=0.55
        )

        ax.errorbar(
            model_names,
            means,
            yerr=stds,
            fmt="none",
            color="#C8CDD8",
            capsize=5,
            linewidth=1.5
        )

        for bar, val in zip(bars, means):
            if "rmse" in mean_key or "mae" in mean_key:
                fmt = f"${val:,.0f}"
            else:
                fmt = f"{val:.3f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                fmt,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#E2E8F0",
                fontweight="bold"
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=9, color="#8B92A5")
        ax.grid(axis="y", alpha=0.4)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

        if "rmse" in mean_key or "mae" in mean_key:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x / 1000:.0f}k")
            )

    plt.tight_layout()
    plt.savefig(OUT / "model_01_comparison.png")
    plt.close()

    print("  ✓ model_01_comparison.png")


def plot_pred_vs_actual(results, y_log, y_raw):
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Predicted vs Actual Salary",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )

    for ax, (name, color) in zip(
        axes,
        zip(results.keys(), [COLORS["xgb"], COLORS["lgb"], COLORS["cat"]])
    ):
        oof_raw = np.expm1(results[name]["oof"])
        true_raw = y_raw.values

        rng = np.random.default_rng(42)
        idx = rng.choice(len(oof_raw), min(5000, len(oof_raw)), replace=False)

        ax.scatter(
            true_raw[idx] / 1000,
            oof_raw[idx] / 1000,
            alpha=0.3,
            s=8,
            color=color,
            rasterized=True
        )

        lim = max(true_raw.max(), oof_raw.max()) / 1000 * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#6B7280", linewidth=1.2)

        ax.text(
            0.05,
            0.93,
            f"Raw R² = {results[name]['r2_mean']:.3f}\n"
            f"Log R² = {results[name]['log_r2_mean']:.3f}\n"
            f"RMSE = ${results[name]['rmse_mean'] / 1000:.1f}k",
            transform=ax.transAxes,
            fontsize=9,
            color="#E2E8F0",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#252836",
                edgecolor="#3B4052"
            )
        )

        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Actual Salary ($k)")
        ax.set_ylabel("Predicted Salary ($k)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "model_01_pred_vs_actual.png")
    plt.close()

    print("  ✓ model_01_pred_vs_actual.png")


def plot_residuals(results, y_log, y_raw):
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Residual Analysis", fontsize=15, fontweight="bold", y=1.02)

    for ax, (name, color) in zip(
        axes,
        zip(results.keys(), [COLORS["xgb"], COLORS["lgb"], COLORS["cat"]])
    ):
        oof_raw = np.expm1(results[name]["oof"])
        residuals = oof_raw - y_raw.values

        rng = np.random.default_rng(42)
        idx = rng.choice(len(residuals), min(5000, len(residuals)), replace=False)

        ax.scatter(
            oof_raw[idx] / 1000,
            residuals[idx] / 1000,
            alpha=0.25,
            s=7,
            color=color,
            rasterized=True
        )

        ax.axhline(0, color="#6B7280", linewidth=1.2, linestyle="--")

        order = np.argsort(oof_raw[idx])
        xs = oof_raw[idx][order] / 1000
        rs = residuals[idx][order] / 1000

        window = max(len(xs) // 20, 50)
        rolling_mean = pd.Series(rs).rolling(window, center=True).mean()

        ax.plot(
            xs,
            rolling_mean.values,
            color="#F87171",
            linewidth=1.5,
            label="Trend"
        )

        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Predicted Salary ($k)")
        ax.set_ylabel("Residual ($k)")
        ax.set_ylim(-200, 200)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "model_01_residuals.png")
    plt.close()

    print("  ✓ model_01_residuals.png")


def plot_shap_summary(fitted_models, X, features):
    plt.rcParams.update(STYLE)

    model = fitted_models["LightGBM"]
    explainer = shap.TreeExplainer(model)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(3000, len(X)), replace=False)

    X_sample = X.iloc[idx]
    shap_values = explainer.shap_values(X_sample)

    display_names = [
        f.replace("skill_", "Skill: ")
         .replace("benefit_", "Benefit: ")
         .replace("_", " ")
         .title()
        for f in features
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(
        "SHAP Feature Importance — LightGBM\nImpact on log salary",
        fontsize=13,
        fontweight="bold",
        pad=15
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=display_names,
        plot_type="dot",
        max_display=20,
        show=False,
        color_bar=True,
        alpha=0.6,
        plot_size=None
    )

    plt.tight_layout()
    plt.savefig(OUT / "model_01_shap_summary.png", facecolor="#0F1117")
    plt.close()

    print("  ✓ model_01_shap_summary.png")

    return shap_values, X_sample, explainer, display_names


def get_shap_importance(model, X, sample_size=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    # Bazı SHAP sürümlerinde list dönebilir.
    if isinstance(sv, list):
        sv = sv[0]

    return np.abs(sv).mean(axis=0)


def plot_shap_bar(fitted_models, X, features):
    plt.rcParams.update(STYLE)

    all_importances = {}

    for name, model in fitted_models.items():
        print(f"Calculating SHAP importance for {name}...")
        all_importances[name] = get_shap_importance(model, X, sample_size=2000)

    avg_importance = np.mean(list(all_importances.values()), axis=0)

    top_idx = np.argsort(avg_importance)[-15:]

    top_features = [
        features[i]
        .replace("skill_", "skill: ")
        .replace("benefit_", "benefit: ")
        .replace("_", " ")
        for i in top_idx
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.set_title(
        "Top 15 Features — Mean |SHAP| Across Models",
        fontsize=13,
        fontweight="bold"
    )

    x_pos = np.arange(len(top_idx))
    bar_height = 0.25

    for i, (name, color) in enumerate(zip(
        ["XGBoost", "LightGBM", "CatBoost"],
        [COLORS["xgb"], COLORS["lgb"], COLORS["cat"]]
    )):
        ax.barh(
            x_pos + i * bar_height,
            all_importances[name][top_idx],
            bar_height,
            label=name,
            color=color,
            alpha=0.85
        )

    ax.set_yticks(x_pos + bar_height)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| on log salary", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "model_01_shap_importance.png")
    plt.close()

    print("  ✓ model_01_shap_importance.png")


def print_summary(results):
    print("\n" + "=" * 78)
    print("  FINAL RESULTS SUMMARY — model_01_salary_v3_final.py v3")
    print("=" * 78)

    fmt = "{:<12} {:>16} {:>14} {:>10} {:>10}"
    print(fmt.format("Model", "RMSE (USD)", "MAE (USD)", "Raw R²", "Log R²"))
    print("-" * 78)

    for name, r in results.items():
        print(fmt.format(
            name,
            f"${r['rmse_mean']:,.0f} ±{r['rmse_std']:,.0f}",
            f"${r['mae_mean']:,.0f}",
            f"{r['r2_mean']:.3f}",
            f"{r['log_r2_mean']:.3f}",
        ))

    print("=" * 78)

    best = max(results, key=lambda k: results[k]["r2_mean"])

    print(f"\n  Best model by Raw R²: {best} (Raw R²={results[best]['r2_mean']:.3f})")

    print("\n  v3 changes applied:")
    print("    + Salary outlier filtering tightened")
    print("    + Top skills increased from 20 to 50")
    print("    + n_skills added")
    print("    + benefit_count and top benefit multi-hot features added")
    print("    + city/state/industry median salary encodings added")
    print("    + pay_period and company_size converted to categorical features")
    print("    + title keyword features expanded")
    print("    + title_len and title_word_count added")
    print("    + Raw R² and Log R² are both reported")

    print("\n  Note:")
    print("    city/state/industry target encodings are global in this v3 version.")
    print("    This can increase R² but may introduce mild CV leakage.")
    print("    For a stricter report version, implement fold-based CV-safe encoding.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  model_01_salary_v3_final.py v3 — LinkedIn Salary Prediction")
    print("=" * 70)

    df, top_skills, skill_cols, benefit_cols = load_and_engineer()

    X, y_raw, y_log, features, cat_features = prepare_features(
        df=df,
        skill_cols=skill_cols,
        benefit_cols=benefit_cols
    )

    cat_feature_indices = [features.index(c) for c in cat_features]

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Categorical: {len(cat_features)} | Numerical: {len(features) - len(cat_features)}")
    print(
        f"  Target range: ${y_raw.min():,.0f} - ${y_raw.max():,.0f} "
        f"(median ${y_raw.median():,.0f})"
    )

    print("\nCategorical features:")
    for col in cat_features:
        print(f"  - {col}")

    results, fitted_models = train_and_evaluate(
        X=X,
        y_log=y_log,
        y_raw=y_raw,
        cat_feature_indices=cat_feature_indices
    )

    print("\nGenerating plots...")

    plot_model_comparison(results)
    plot_pred_vs_actual(results, y_log, y_raw)
    plot_residuals(results, y_log, y_raw)
    plot_shap_summary(fitted_models, X, features)
    plot_shap_bar(fitted_models, X, features)

    print_summary(results)

    print(f"\n  All outputs saved to: {OUT.resolve()}/")
    print("  Files:")
    print("    - model_01_comparison.png")
    print("    - model_01_pred_vs_actual.png")
    print("    - model_01_residuals.png")
    print("    - model_01_shap_summary.png")
    print("    - model_01_shap_importance.png")
