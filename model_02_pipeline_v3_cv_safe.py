"""
model_02_pipeline.py — v3_cv_safe
LinkedIn Job Postings — Pipeline + Hyperparameter Tuning

Bu sürüm model_01_salary.py v3 sonrası için hazırlanmıştır.
- model_01 v3'e yakın feature engineering
- sklearn Pipeline
- Optuna tuning
- CV-safe target encoding
- title TF-IDF + SVD
- best pipeline kaydı: models/best_salary_pipeline.joblib

Not:
Target encoding Pipeline içinde yapıldığı için CV sırasında validation fold'un maaşı encoding hesabına karışmaz.
Bu nedenle model_01 v3'teki global target encoding'e göre daha sağlıklıdır.
"""

import os
import re
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import optuna

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(os.getenv("LINKEDIN_DATA_DIR", "data"))
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_TRIALS = 15          # İlk deneme için 15. Çalışınca 40/70 yapabilirsin.
N_TUNE_FOLDS = 2       # Optuna içinde hız için 2-fold
N_CV_FOLDS = 5         # Final değerlendirme

USE_TITLE_TEXT_FEATURES = True
TITLE_TFIDF_MAX_FEATURES = 2500
TITLE_SVD_COMPONENTS = 30

STYLE = {
    "figure.facecolor": "#0F1117", "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#2E3347", "axes.labelcolor": "#C8CDD8",
    "xtick.color": "#8B92A5", "ytick.color": "#8B92A5",
    "text.color": "#C8CDD8", "grid.color": "#2E3347",
    "grid.linestyle": "--", "grid.alpha": 0.6,
    "savefig.facecolor": "#0F1117", "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 10,
}
COLORS = {"xgb": "#F59E0B", "lgb": "#6366F1", "cat": "#10B981", "base": "#4B5563", "tuned": "#3B82F6"}


def safe_read_csv(path, usecols=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path, usecols=usecols)
    except ValueError as e:
        print(f"\nColumn error while reading {path.name}")
        print(f"Requested columns: {usecols}")
        header = pd.read_csv(path, nrows=0)
        print(f"Available columns: {list(header.columns)}")
        raise e


def clean_col_name(value):
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value if value else "unknown"


def find_column(df, possible_names):
    lower = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_city(location):
    if pd.isna(location):
        return "Unknown"
    loc = str(location).strip()
    low = loc.lower()
    if not loc:
        return "Unknown"
    if "remote" in low:
        return "Remote"

    metro_map = {
        "new york city metropolitan area": "New York",
        "greater new york city area": "New York",
        "san francisco bay area": "San Francisco",
        "greater seattle area": "Seattle",
        "greater boston": "Boston",
        "greater chicago area": "Chicago",
        "los angeles metropolitan area": "Los Angeles",
        "washington dc-baltimore area": "Washington",
        "washington dc metropolitan area": "Washington",
    }
    for key, val in metro_map.items():
        if key in low:
            return val
    if "," in loc:
        return loc.split(",")[0].strip()
    return loc


def extract_state(location, company_state=None):
    if pd.notna(location):
        loc = str(location)
        m = re.search(r",\s*([A-Z]{2})(?:\s|$)", loc)
        if m:
            return m.group(1)
        low = loc.lower()
        if "san francisco bay area" in low or "los angeles" in low:
            return "CA"
        if "new york" in low:
            return "NY"
        if "greater seattle" in low:
            return "WA"
        if "greater boston" in low:
            return "MA"
        if "greater chicago" in low:
            return "IL"
        if "washington dc" in low:
            return "DC"
    if pd.notna(company_state):
        return str(company_state)
    return "Unknown"


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.log1p(np.maximum(X, 0))


class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    """CV-safe median target encoder. y log salary olduğu için output da log salary ölçeğindedir."""
    def __init__(self, cols=None, smoothing=30.0, min_samples_leaf=5):
        self.cols = cols or []
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=float)
        self.global_ = float(np.nanmedian(y))
        self.maps_ = {}
        for col in self.cols:
            if col not in X.columns:
                continue
            temp = pd.DataFrame({col: X[col].fillna("Unknown").astype(str), "_y": y})
            stats = temp.groupby(col)["_y"].agg(["median", "count"])
            weight = stats["count"] / (stats["count"] + self.smoothing)
            enc = weight * stats["median"] + (1 - weight) * self.global_
            enc = enc.where(stats["count"] >= self.min_samples_leaf, self.global_)
            self.maps_[col] = enc.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.cols:
            new_col = f"te_{col}"
            if col not in X.columns or col not in self.maps_:
                X[new_col] = self.global_
            else:
                X[new_col] = X[col].fillna("Unknown").astype(str).map(self.maps_[col]).fillna(self.global_).astype(float)
        return X


# ── 1. Data + feature engineering ─────────────────────────────────────────────
def load_and_engineer():
    print("Loading data...")
    post = safe_read_csv(BASE / "postings.csv", usecols=[
        "job_id", "title", "company_id", "location", "normalized_salary",
        "pay_period", "currency", "formatted_experience_level",
        "formatted_work_type", "remote_allowed", "views", "applies",
    ])

    salary_mask = post["normalized_salary"].notna()
    q_low = post.loc[salary_mask, "normalized_salary"].quantile(0.01)
    q_high = post.loc[salary_mask, "normalized_salary"].quantile(0.99)
    lower_bound = max(10_000, q_low)
    upper_bound = min(300_000, q_high)

    df = post[
        post["normalized_salary"].between(lower_bound, upper_bound)
        & (post["currency"].isin(["USD"]) | post["currency"].isna())
    ].copy()
    df["job_id"] = df["job_id"].astype("Int64")
    df["company_id"] = df["company_id"].astype("Int64")
    print(f"  Salary bounds used: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    print(f"  Salary rows after filtering: {len(df):,}")

    # Companies
    comp = safe_read_csv(BASE / "companies.csv", usecols=["company_id", "company_size", "state", "country"])
    emp = safe_read_csv(BASE / "employee_counts.csv", usecols=["company_id", "employee_count", "follower_count"])
    emp = emp.sort_values("employee_count").drop_duplicates("company_id", keep="last")
    comp = comp.merge(emp, on="company_id", how="left")
    df = df.merge(comp, on="company_id", how="left")

    df["company_job_count"] = df.groupby("company_id")["job_id"].transform("count").fillna(0)
    df["has_employee_count"] = df["employee_count"].notna().astype(int)
    df["has_follower_count"] = df["follower_count"].notna().astype(int)
    df["follower_per_employee"] = df["follower_count"].fillna(0) / (df["employee_count"].fillna(0) + 1)

    # Company industries count
    cind_path = BASE / "company_industries.csv"
    if cind_path.exists():
        cind = safe_read_csv(cind_path)
        if "company_id" in cind.columns:
            cind_count = cind.groupby("company_id").size().reset_index(name="company_industry_count")
            df = df.merge(cind_count, on="company_id", how="left")
        else:
            df["company_industry_count"] = 0
    else:
        df["company_industry_count"] = 0
    df["company_industry_count"] = df["company_industry_count"].fillna(0)

    # Company specialities
    speciality_cols = []
    cspec_path = BASE / "company_specialities.csv"
    if cspec_path.exists():
        cspec = safe_read_csv(cspec_path)
        company_col = find_column(cspec, ["company_id"])
        spec_col = find_column(cspec, ["speciality", "specialty", "specialities", "specialties"])
        if company_col and spec_col:
            cspec = cspec.rename(columns={company_col: "company_id", spec_col: "speciality"})
            cspec["speciality"] = cspec["speciality"].fillna("Unknown").astype(str)
            spec_count = cspec.groupby("company_id")["speciality"].nunique().reset_index(name="company_speciality_count")
            df = df.merge(spec_count, on="company_id", how="left")
            top_specs = cspec["speciality"].value_counts().head(20).index.tolist()
            sp = (cspec[cspec["speciality"].isin(top_specs)]
                  .assign(val=1)
                  .pivot_table(index="company_id", columns="speciality", values="val", fill_value=0, aggfunc="max"))
            sp.columns = [f"company_spec_{clean_col_name(c)}" for c in sp.columns]
            sp = sp.reset_index()
            speciality_cols = [c for c in sp.columns if c != "company_id"]
            df = df.merge(sp, on="company_id", how="left")
            for col in speciality_cols:
                df[col] = df[col].fillna(0).astype("int8")
        else:
            df["company_speciality_count"] = 0
    else:
        df["company_speciality_count"] = 0
    df["company_speciality_count"] = df["company_speciality_count"].fillna(0)

    # Primary industry
    job_ind = safe_read_csv(BASE / "job_industries.csv")
    ind_ref = safe_read_csv(BASE / "industries.csv")
    job_ind = job_ind.merge(ind_ref, on="industry_id", how="left")
    primary = (job_ind.groupby("job_id")["industry_name"].first().reset_index()
               .rename(columns={"industry_name": "primary_industry"}))
    df = df.merge(primary, on="job_id", how="left")
    df["primary_industry"] = df["primary_industry"].fillna("Unknown")

    # Skills top 50
    jsk = safe_read_csv(BASE / "job_skills.csv")
    sref = safe_read_csv(BASE / "skills.csv")
    jsk = jsk.merge(sref, on="skill_abr", how="left")
    top_skills = (jsk[jsk["job_id"].isin(df["job_id"])]
                  ["skill_name"].dropna().value_counts().head(50).index.tolist())
    skill_pivot = (jsk[jsk["job_id"].isin(df["job_id"]) & jsk["skill_name"].isin(top_skills)]
                   .assign(val=1)
                   .pivot_table(index="job_id", columns="skill_name", values="val", fill_value=0, aggfunc="max"))
    skill_pivot.columns = [f"skill_{clean_col_name(c)}" for c in skill_pivot.columns]
    skill_pivot = skill_pivot.reset_index()
    skill_cols = [c for c in skill_pivot.columns if c != "job_id"]
    df = df.merge(skill_pivot, on="job_id", how="left")
    for col in skill_cols:
        df[col] = df[col].fillna(0).astype("int8")

    skill_count = (jsk[jsk["job_id"].isin(df["job_id"])]
                   .groupby("job_id")["skill_name"].nunique().reset_index(name="n_skills"))
    df = df.merge(skill_count, on="job_id", how="left")
    df["n_skills"] = df["n_skills"].fillna(0)

    # Benefits
    benefit_cols = []
    ben_path = BASE / "benefits.csv"
    if ben_path.exists():
        ben = safe_read_csv(ben_path)
        if "job_id" in ben.columns:
            bc = ben.groupby("job_id").size().reset_index(name="benefit_count")
            df = df.merge(bc, on="job_id", how="left")
            df["benefit_count"] = df["benefit_count"].fillna(0)
            if "type" in ben.columns:
                ben["type"] = ben["type"].fillna("Unknown").astype(str)
                top_ben = ben["type"].value_counts().head(10).index.tolist()
                bp = (ben[ben["type"].isin(top_ben)].assign(val=1)
                      .pivot_table(index="job_id", columns="type", values="val", fill_value=0, aggfunc="max"))
                bp.columns = [f"benefit_{clean_col_name(c)}" for c in bp.columns]
                bp = bp.reset_index()
                benefit_cols = [c for c in bp.columns if c != "job_id"]
                df = df.merge(bp, on="job_id", how="left")
                for col in benefit_cols:
                    df[col] = df[col].fillna(0).astype("int8")
        else:
            df["benefit_count"] = 0
    else:
        df["benefit_count"] = 0
    df["benefit_count"] = df["benefit_count"].fillna(0)

    # Title features
    title = df["title"].fillna("").str.lower()
    df["title_senior"] = title.str.contains(r"senior|sr\.?\b|lead|staff", regex=True).astype(int)
    df["title_principal"] = title.str.contains(r"principal", regex=True).astype(int)
    df["title_director"] = title.str.contains(r"director|head of", regex=True).astype(int)
    df["title_vp"] = title.str.contains(r"\bvp\b|vice pres", regex=True).astype(int)
    df["title_chief"] = title.str.contains(r"\bchief\b|president|ceo|cto|cfo|coo", regex=True).astype(int)
    df["title_manager"] = title.str.contains(r"manager|mgr", regex=True).astype(int)
    df["title_junior"] = title.str.contains(r"junior|jr\.?\b|entry|intern", regex=True).astype(int)
    df["title_associate"] = title.str.contains(r"\bassociate\b", regex=True).astype(int)
    df["title_engineer"] = title.str.contains(r"engineer|developer|software|swe|sde", regex=True).astype(int)
    df["title_architect"] = title.str.contains(r"architect", regex=True).astype(int)
    df["title_data"] = title.str.contains(r"data|analyst|scientist|ml|ai|machine", regex=True).astype(int)
    df["title_sales"] = title.str.contains(r"sales|account exec|business dev", regex=True).astype(int)
    df["title_consultant"] = title.str.contains(r"consultant|advisor", regex=True).astype(int)
    df["title_nurse"] = title.str.contains(r"nurse|rn\b|lpn", regex=True).astype(int)
    df["title_product"] = title.str.contains(r"product", regex=True).astype(int)
    df["title_finance"] = title.str.contains(r"finance|accountant|controller|accounting|auditor", regex=True).astype(int)
    df["title_security"] = title.str.contains(r"security|cyber|infosec", regex=True).astype(int)
    df["title_marketing"] = title.str.contains(r"marketing|brand|seo|content", regex=True).astype(int)
    df["title_hr"] = title.str.contains(r"human resources|recruiter|talent|people partner", regex=True).astype(int)
    df["title_len"] = df["title"].fillna("").str.len()
    df["title_word_count"] = df["title"].fillna("").str.split().str.len().fillna(0)
    df["title_text"] = df["title"].fillna("").astype(str)

    # Experience imputation
    def impute_exp(row):
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

    before = df["formatted_experience_level"].isna().sum()
    df["formatted_experience_level"] = df.apply(impute_exp, axis=1)
    after = df["formatted_experience_level"].isna().sum()
    print(f"  Experience level nulls imputed: {before:,} -> {after:,}")

    # Location cleanup
    df["city_loc"] = df["location"].apply(normalize_city)
    df["state_final"] = df.apply(lambda r: extract_state(r["location"], r.get("state", None)), axis=1)

    # Flags / cleanup
    df["is_hourly"] = (df["pay_period"] == "HOURLY").astype(int)
    df["remote_flag"] = df["remote_allowed"].fillna(0).astype(int)
    df["has_applies"] = df["applies"].notna().astype(int)
    df["has_views"] = df["views"].notna().astype(int)

    df["formatted_work_type"] = df["formatted_work_type"].fillna("Unknown")
    df["pay_period"] = df["pay_period"].fillna("Unknown")
    df["company_size"] = df["company_size"].fillna("Unknown").astype(str)
    df["company_id"] = df["company_id"].fillna(-1).astype(str)

    print(f"  Top skills used: {len(skill_cols)}")
    print(f"  Benefit multi-hot columns used: {len(benefit_cols)}")
    print(f"  Company speciality columns used: {len(speciality_cols)}")
    print(f"  Feature engineering done. Shape: {df.shape}")
    return df, skill_cols, benefit_cols, speciality_cols


# Feature groups
CAT_COLS = ["formatted_experience_level", "formatted_work_type", "primary_industry", "pay_period", "company_size", "state_final"]
TARGET_ENCODE_COLS = ["state_final", "city_loc", "primary_industry", "company_id"]
TE_NUM_COLS = [f"te_{c}" for c in TARGET_ENCODE_COLS]
LOG_NUM_COLS = ["views", "applies", "employee_count", "follower_count", "company_job_count"]
NUM_COLS = ["n_skills", "benefit_count", "company_industry_count", "company_speciality_count", "follower_per_employee", "title_len", "title_word_count"]
BINARY_COLS = [
    "is_hourly", "remote_flag", "has_applies", "has_views", "has_employee_count", "has_follower_count",
    "title_senior", "title_principal", "title_director", "title_vp", "title_chief", "title_manager",
    "title_junior", "title_associate", "title_engineer", "title_architect", "title_data", "title_sales",
    "title_consultant", "title_nurse", "title_product", "title_finance", "title_security", "title_marketing", "title_hr",
]


def build_feature_columns(skill_cols, benefit_cols, speciality_cols):
    cols = CAT_COLS + TARGET_ENCODE_COLS + LOG_NUM_COLS + NUM_COLS + BINARY_COLS + skill_cols + benefit_cols + speciality_cols + ["title_text"]
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_X_y(df, skill_cols, benefit_cols, speciality_cols):
    feature_cols = build_feature_columns(skill_cols, benefit_cols, speciality_cols)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].copy()
    y_log = np.log1p(df["normalized_salary"].values)
    return X, y_log, feature_cols


def build_preprocessor(skill_cols, benefit_cols, speciality_cols):
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
    ])
    log_num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", LogTransformer()),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    bin_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))])

    transformers = [
        ("cat", cat_pipe, CAT_COLS),
        ("te", num_pipe, TE_NUM_COLS),
        ("log_num", log_num_pipe, LOG_NUM_COLS),
        ("num", num_pipe, NUM_COLS),
        ("bin", bin_pipe, BINARY_COLS + skill_cols + benefit_cols + speciality_cols),
    ]

    if USE_TITLE_TEXT_FEATURES:
        title_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=TITLE_TFIDF_MAX_FEATURES, ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents="unicode", lowercase=True)),
            ("svd", TruncatedSVD(n_components=TITLE_SVD_COMPONENTS, random_state=RANDOM_STATE)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("title_text", title_pipe, "title_text"))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def build_model(model_name, params=None):
    params = params or {}
    if model_name == "XGBoost":
        defaults = dict(n_estimators=700, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.2, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        defaults.update(params)
        return xgb.XGBRegressor(**defaults)
    if model_name == "LightGBM":
        defaults = dict(n_estimators=700, learning_rate=0.05, max_depth=7, num_leaves=80, subsample=0.85, colsample_bytree=0.85, min_child_samples=18, reg_alpha=0.05, reg_lambda=1.0, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
        defaults.update(params)
        return lgb.LGBMRegressor(**defaults)
    if model_name == "CatBoost":
        defaults = dict(iterations=700, learning_rate=0.05, depth=7, l2_leaf_reg=3, subsample=0.85, random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False)
        defaults.update(params)
        return CatBoostRegressor(**defaults)
    raise ValueError(model_name)


def build_pipeline(model_name, skill_cols, benefit_cols, speciality_cols, params=None):
    return Pipeline([
        ("target_encoder", MedianTargetEncoder(cols=TARGET_ENCODE_COLS, smoothing=30.0, min_samples_leaf=5)),
        ("preprocessor", build_preprocessor(skill_cols, benefit_cols, speciality_cols)),
        ("model", build_model(model_name, params)),
    ])


def evaluate_cv(model_name, X, y_log, skill_cols, benefit_cols, speciality_cols, params=None, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_list, mae_list, r2_list, log_r2_list = [], [], [], []
    for fold, (tr, val) in enumerate(kf.split(X), 1):
        pipe = build_pipeline(model_name, skill_cols, benefit_cols, speciality_cols, params)
        pipe.fit(X.iloc[tr], y_log[tr])
        pred_log = pipe.predict(X.iloc[val])
        true_raw = np.expm1(y_log[val])
        pred_raw = np.expm1(pred_log)
        rmse = np.sqrt(mean_squared_error(true_raw, pred_raw))
        mae = mean_absolute_error(true_raw, pred_raw)
        r2 = r2_score(true_raw, pred_raw)
        log_r2 = r2_score(y_log[val], pred_log)
        rmse_list.append(rmse); mae_list.append(mae); r2_list.append(r2); log_r2_list.append(log_r2)
        print(f"  {model_name} Fold {fold}: RMSE=${rmse:,.0f} MAE=${mae:,.0f} Raw R²={r2:.3f} Log R²={log_r2:.3f}")
    return {
        "rmse": float(np.mean(rmse_list)), "rmse_std": float(np.std(rmse_list)),
        "mae": float(np.mean(mae_list)), "mae_std": float(np.std(mae_list)),
        "r2": float(np.mean(r2_list)), "r2_std": float(np.std(r2_list)),
        "log_r2": float(np.mean(log_r2_list)), "log_r2_std": float(np.std(log_r2_list)),
    }


def baseline_cv(X, y_log, skill_cols, benefit_cols, speciality_cols):
    print("Running baseline CV (3-fold)...")
    results = {}
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        print(f"\nBaseline {name}:")
        res = evaluate_cv(name, X, y_log, skill_cols, benefit_cols, speciality_cols, params=None, n_splits=3)
        results[name] = res
        print(f"  ► {name}: RMSE=${res['rmse']:,.0f} MAE=${res['mae']:,.0f} Raw R²={res['r2']:.3f} Log R²={res['log_r2']:.3f}")
    return results


def suggest_params(trial, model_name):
    if model_name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 1000),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 8.0, log=True),
        )
    if model_name == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 1000),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            num_leaves=trial.suggest_int("num_leaves", 31, 140),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 8.0, log=True),
        )
    return dict(
        iterations=trial.suggest_int("iterations", 400, 900),
        depth=trial.suggest_int("depth", 4, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        subsample=trial.suggest_float("subsample", 0.65, 1.0),
    )


def tune_model(model_name, X, y_log, skill_cols, benefit_cols, speciality_cols):
    kf = KFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    def objective(trial):
        params = suggest_params(trial, model_name)
        scores = []
        for tr, val in kf.split(X):
            pipe = build_pipeline(model_name, skill_cols, benefit_cols, speciality_cols, params)
            pipe.fit(X.iloc[tr], y_log[tr])
            pred_log = pipe.predict(X.iloc[val])
            rmse = np.sqrt(mean_squared_error(np.expm1(y_log[val]), np.expm1(pred_log)))
            scores.append(rmse)
        return float(np.mean(scores))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study


def plot_tuning_curves(studies):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Hyperparameter Optimization History — Optuna", fontsize=15, fontweight="bold", y=1.02)
    colors = [COLORS["xgb"], COLORS["lgb"], COLORS["cat"]]
    for ax, (name, study), color in zip(axes, studies.items(), colors):
        values = study.trials_dataframe()["value"].values
        ax.scatter(range(len(values)), values, color=color, s=18, alpha=0.55, label="Trial RMSE")
        ax.plot(np.minimum.accumulate(values), color="#F8FAFC", linewidth=2, label="Best so far")
        ax.axhline(study.best_value, linestyle="--", color=color, linewidth=1.2)
        ax.set_title(name, color=color, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV RMSE")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "model_02_tuning_curves.png")
    plt.close()
    print("  ✓ model_02_tuning_curves.png")


def plot_baseline_vs_tuned(baseline, tuned):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Baseline vs Tuned — CV-safe Pipeline", fontsize=15, fontweight="bold", y=1.02)
    metrics = [("rmse", "RMSE", "Lower is better"), ("mae", "MAE", "Lower is better"), ("r2", "Raw R²", "Higher is better"), ("log_r2", "Log R²", "Higher is better")]
    names = list(baseline.keys())
    for ax, (key, title, subtitle) in zip(axes, metrics):
        x = np.arange(len(names)); w = 0.35
        base_vals = [baseline[n][key] for n in names]
        tuned_vals = [tuned[n][key] for n in names]
        ax.bar(x - w/2, base_vals, w, label="Baseline", color=COLORS["base"], alpha=0.85)
        ax.bar(x + w/2, tuned_vals, w, label="Tuned", color=COLORS["tuned"], alpha=0.85)
        for i, (bv, tv) in enumerate(zip(base_vals, tuned_vals)):
            if key in ("rmse", "mae"):
                delta = (bv - tv) / bv * 100; sign = "▼"
            else:
                delta = (tv - bv) / abs(bv) * 100 if bv else 0; sign = "▲"
            ax.text(i, max(bv, tv) * 1.025, f"{sign}{abs(delta):.1f}%", ha="center", fontsize=8, color="#86EFAC", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=9, color="#8B92A5")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        if key in ("rmse", "mae"):
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT / "model_02_baseline_vs_tuned.png")
    plt.close()
    print("  ✓ model_02_baseline_vs_tuned.png")


def plot_param_importance(studies):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Hyperparameter Importance — Optuna", fontsize=15, fontweight="bold", y=1.02)
    colors = [COLORS["xgb"], COLORS["lgb"], COLORS["cat"]]
    for ax, (name, study), color in zip(axes, studies.items(), colors):
        try:
            imp = optuna.importance.get_param_importances(study)
            params = list(imp.keys())[:8]
            vals = [imp[p] for p in params]
            short = [p.replace("learning_rate", "lr").replace("n_estimators", "n_est").replace("colsample_bytree", "col_sample").replace("min_child_samples", "min_child_samp").replace("min_child_weight", "min_child_w") for p in params]
            y = np.arange(len(params))
            ax.barh(y, vals, color=color, alpha=0.85)
            ax.set_yticks(y); ax.set_yticklabels(short, fontsize=9)
            ax.set_xlabel("Importance")
            ax.set_title(name, color=color, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
        except Exception:
            ax.text(0.5, 0.5, "Not enough\ntrials", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, color=color, fontweight="bold")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "model_02_param_importance.png")
    plt.close()
    print("  ✓ model_02_param_importance.png")


def plot_final_summary(tuned):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    fig.suptitle("Final Tuned Model Performance Summary", fontsize=15, fontweight="bold", y=1.02)
    names = list(tuned.keys())
    best = max(tuned, key=lambda n: tuned[n]["r2"])
    cols = ["Model", "RMSE", "MAE", "Raw R²", "Log R²", "Best"]
    rows = [[n, f"${tuned[n]['rmse']:,.0f} ±{tuned[n]['rmse_std']:,.0f}", f"${tuned[n]['mae']:,.0f}", f"{tuned[n]['r2']:.4f}", f"{tuned[n]['log_r2']:.4f}", "★" if n == best else ""] for n in names]
    cmap = {"XGBoost": COLORS["xgb"], "LightGBM": COLORS["lgb"], "CatBoost": COLORS["cat"]}
    colors = [[cmap[n]] + ["#1A1D27"] * (len(cols) - 1) for n in names]
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center", cellColours=colors)
    table.auto_set_font_size(False); table.set_fontsize(10.5); table.scale(1.2, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2E3347")
        if row == 0:
            cell.set_facecolor("#252836"); cell.set_text_props(color="#E2E8F0", fontweight="bold")
        else:
            cell.set_text_props(color="#E2E8F0")
    ax.text(0.5, -0.08, "Target encoding is CV-safe: mappings are fitted only on train folds.", ha="center", transform=ax.transAxes, fontsize=8, color="#8B92A5")
    plt.tight_layout()
    plt.savefig(OUT / "model_02_summary_table.png")
    plt.close()
    print("  ✓ model_02_summary_table.png")


def save_best_pipeline(tuned, best_params, X, y_log, skill_cols, benefit_cols, speciality_cols, feature_cols):
    best_name = max(tuned, key=lambda n: tuned[n]["r2"])
    print(f"\nSaving best pipeline: {best_name} (Raw R²={tuned[best_name]['r2']:.4f})")
    pipe = build_pipeline(best_name, skill_cols, benefit_cols, speciality_cols, best_params[best_name])
    pipe.fit(X, y_log)
    path = MODELS / "best_salary_pipeline.joblib"
    joblib.dump({
        "pipeline": pipe,
        "model_name": best_name,
        "best_params": best_params[best_name],
        "feature_cols": feature_cols,
        "cat_cols": CAT_COLS,
        "target_encode_cols": TARGET_ENCODE_COLS,
        "log_num_cols": LOG_NUM_COLS,
        "num_cols": NUM_COLS,
        "binary_cols": BINARY_COLS,
        "skill_cols": skill_cols,
        "benefit_cols": benefit_cols,
        "speciality_cols": speciality_cols,
        "metrics": tuned[best_name],
        "log_target": True,
        "note": "Prediction is log salary. Use np.expm1(obj['pipeline'].predict(X_new)).",
    }, path)
    print(f"  ✓ Pipeline saved → {path}")
    print("  Usage:")
    print("    obj = joblib.load('models/best_salary_pipeline.joblib')")
    print("    pred_salary = np.expm1(obj['pipeline'].predict(X_new))")
    return pipe, best_name, path


if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("  model_02_pipeline.py v3_cv_safe — Pipeline + Hyperparameter Tuning")
    print("=" * 78)

    df, skill_cols, benefit_cols, speciality_cols = load_and_engineer()
    X, y_log, feature_cols = build_X_y(df, skill_cols, benefit_cols, speciality_cols)

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} input features")
    print(f"  Skill cols: {len(skill_cols)} | Benefit cols: {len(benefit_cols)} | Company speciality cols: {len(speciality_cols)}")
    print(f"  Title TF-IDF/SVD: {'ON' if USE_TITLE_TEXT_FEATURES else 'OFF'}")
    print(f"  Target salary range: ${df['normalized_salary'].min():,.0f} - ${df['normalized_salary'].max():,.0f}")
    print(f"  Target salary median: ${df['normalized_salary'].median():,.0f}")

    print("\n" + "-" * 78)
    print("BASELINE CV")
    print("-" * 78)
    baseline = baseline_cv(X, y_log, skill_cols, benefit_cols, speciality_cols)

    print("\n" + "-" * 78)
    print(f"OPTUNA TUNING ({N_TRIALS} trials × 3 models × {N_TUNE_FOLDS}-fold CV)")
    print("-" * 78)
    studies, best_params = {}, {}
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        print(f"\nTuning {name}...", flush=True)
        study = tune_model(name, X, y_log, skill_cols, benefit_cols, speciality_cols)
        studies[name] = study
        best_params[name] = study.best_params
        print(f"  Best tuning RMSE: ${study.best_value:,.0f}")
        print(f"  Best params: {study.best_params}")

    print("\n" + "-" * 78)
    print(f"FINAL {N_CV_FOLDS}-FOLD CV WITH BEST PARAMS")
    print("-" * 78)
    tuned = {}
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        print(f"\nFinal CV {name}:")
        res = evaluate_cv(name, X, y_log, skill_cols, benefit_cols, speciality_cols, best_params[name], N_CV_FOLDS)
        tuned[name] = res
        print(f"  ► {name}: RMSE=${res['rmse']:,.0f}±{res['rmse_std']:,.0f} MAE=${res['mae']:,.0f} Raw R²={res['r2']:.4f} Log R²={res['log_r2']:.4f}")

    print("\nGenerating plots...")
    plot_tuning_curves(studies)
    plot_baseline_vs_tuned(baseline, tuned)
    plot_param_importance(studies)
    plot_final_summary(tuned)

    pipe, best_name, pipeline_path = save_best_pipeline(tuned, best_params, X, y_log, skill_cols, benefit_cols, speciality_cols, feature_cols)

    print("\n" + "=" * 78)
    print("  TUNING RESULTS SUMMARY — model_02_pipeline.py v3_cv_safe")
    print("=" * 78)
    fmt = "{:<12} {:>18} {:>14} {:>10} {:>10}"
    print(fmt.format("Model", "RMSE (USD)", "MAE (USD)", "Raw R²", "Log R²"))
    print("-" * 78)
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        r, b = tuned[name], baseline[name]
        delta = (b["rmse"] - r["rmse"]) / b["rmse"] * 100
        print(fmt.format(name, f"${r['rmse']:,.0f} ±{r['rmse_std']:,.0f}", f"${r['mae']:,.0f}", f"{r['r2']:.4f}", f"{r['log_r2']:.4f}"))
        print(f"  {'':10} RMSE improved {delta:+.1f}% vs baseline")
    print("=" * 78)
    print(f"\n  Best model: {best_name}")
    print(f"  Saved pipeline: {pipeline_path}")
    print("  Plots:")
    print("    - outputs/model_02_tuning_curves.png")
    print("    - outputs/model_02_baseline_vs_tuned.png")
    print("    - outputs/model_02_param_importance.png")
    print("    - outputs/model_02_summary_table.png")
    print("\n  Note: Target encoding is CV-safe in this version.")
