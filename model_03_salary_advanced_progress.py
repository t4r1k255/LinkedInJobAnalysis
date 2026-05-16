"""
model_03_salary_advanced.py
LinkedIn Job Postings — Advanced Salary Prediction

Amaç:
  Model 2'deki CV-safe pipeline yapısını koruyup Raw R² değerini sağlıklı şekilde artırmak.

Yeni eklenenler:
  - YEARLY-only opsiyonu
  - description_clean text feature
  - description içinden salary/pay range ifadelerini temizleme
  - title TF-IDF + SVD
  - description TF-IDF + SVD
  - title_cluster
  - interaction features
  - CV-safe target encoding
  - XGBoost + LightGBM tuning
  - OOF ensemble
"""

import os
import re
import json
import warnings
import time
import joblib
from pathlib import Path

from tqdm import tqdm  # CV fold barları için hâlâ kullanılıyor

import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import optuna

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
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
from utils_progress import ProgressBar, StepTracker, ChunkProgressBar



try:
    from tqdm import tqdm  # CV fold barları için hâlâ kullanılıyor
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


def log_step(message):
    """Saatli ve flush'lı log basar; uzun işlemlerde kodun nerede olduğunu gösterir."""
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def log_duration(start_time, message):
    elapsed = time.time() - start_time
    print(f"      ✓ {message} completed in {elapsed/60:.2f} min", flush=True)


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIG
# =============================================================================
BASE = Path(os.getenv("LINKEDIN_DATA_DIR", "data"))
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42

# 0.80'e yaklaşmak için ilk sağlıklı deney: yalnızca yıllık maaş ilanları.
# False yaparsan tüm pay_period türleriyle karşılaştırma yapar.
RUN_YEARLY_ONLY = False

# İlk çalıştırmada düşük kalsın. Çalışınca 20/30/50 yapılabilir.
N_TRIALS = 30

N_TUNE_FOLDS = 2
N_FINAL_FOLDS = 5

MODEL_NAMES = ["LightGBM", "XGBoost"]

USE_TITLE_TEXT_FEATURES = True
USE_DESCRIPTION_TEXT_FEATURES = True
USE_TITLE_CLUSTER = True

TITLE_TFIDF_MAX_FEATURES = 2500
TITLE_SVD_COMPONENTS = 25

DESC_TFIDF_MAX_FEATURES = 6000
DESC_SVD_COMPONENTS = 100

N_TITLE_CLUSTERS = 35


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
    "XGBoost": "#F59E0B",
    "LightGBM": "#6366F1",
    "CatBoost": "#10B981",
    "Ensemble": "#8B5CF6",
}


# =============================================================================
# HELPERS
# =============================================================================
def safe_read_csv(path, usecols=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return pd.read_csv(path, usecols=usecols)
    except ValueError as e:
        print(f"\nColumn error while reading: {path.name}")
        print(f"Requested columns: {usecols}")
        header = pd.read_csv(path, nrows=0)
        print(f"Available columns: {list(header.columns)}")
        raise e


def clean_col_name(value):
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value if value else "unknown"


def find_column(df, names):
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
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
        "dallas-fort worth metroplex": "Dallas",
        "greater houston": "Houston",
        "greater phoenix area": "Phoenix",
        "greater atlanta area": "Atlanta",
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
        low = loc.lower()

        m = re.search(r",\s*([A-Z]{2})(?:\s|$)", loc)
        if m:
            return m.group(1)

        metro_state_map = {
            "san francisco bay area": "CA",
            "new york": "NY",
            "greater seattle": "WA",
            "greater boston": "MA",
            "greater chicago": "IL",
            "los angeles": "CA",
            "washington dc": "DC",
            "dallas": "TX",
            "houston": "TX",
            "phoenix": "AZ",
            "atlanta": "GA",
        }

        for key, val in metro_state_map.items():
            if key in low:
                return val

    if pd.notna(company_state):
        return str(company_state)

    return "Unknown"


def clean_salary_text(text):
    """
    Maaş tahmininde leakage olmaması için description içinden açık maaş bilgisini temizler.
    Örn:
      $120,000 - $160,000
      $55/hr
      salary range is ...
      compensation range ...
    """
    if pd.isna(text):
        return ""

    s = str(text)
    s = re.sub(r"\s+", " ", s)

    # $120,000, $120k, $50/hr, $120,000 - $160,000
    s = re.sub(
        r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|K)?\s*(?:-|to|–|—)?\s*\$?\s?\d{0,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|K)?",
        " ",
        s,
        flags=re.IGNORECASE,
    )

    # 50/hr, 50 per hour, 40-60 hourly
    s = re.sub(
        r"\b\d{1,3}(?:\.\d+)?\s*(?:-|to|–|—)?\s*\d{0,3}(?:\.\d+)?\s*(?:per hour|/hour|/hr|hourly|hr)\b",
        " ",
        s,
        flags=re.IGNORECASE,
    )

    # Compensation/salary sentence removal
    keywords = [
        "salary range", "compensation range", "pay range", "base salary",
        "annual salary", "hourly rate", "expected salary", "salary",
        "compensation", "pay rate", "bonus eligible"
    ]

    for kw in keywords:
        pattern = rf"[^.]*\b{re.escape(kw)}\b[^.]*\."
        s = re.sub(pattern, " ", s, flags=re.IGNORECASE)

    s = re.sub(r"\b(?:usd|dollars|per year|yearly|annually)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def make_title_family(title_lower):
    t = title_lower or ""

    if re.search(r"nurse|rn\b|lpn|medical assistant|physician|therapist|clinical", t):
        return "healthcare"
    if re.search(r"machine learning|data scientist|data engineer|data analyst|analytics|bi\b|ai\b|ml\b", t):
        return "data_ai"
    if re.search(r"software|engineer|developer|swe|sde|backend|frontend|full stack|devops|cloud|architect", t):
        return "software_engineering"
    if re.search(r"product manager|product owner|product lead", t):
        return "product"
    if re.search(r"sales|account executive|business development|customer success", t):
        return "sales"
    if re.search(r"marketing|brand|seo|content|growth", t):
        return "marketing"
    if re.search(r"finance|accountant|accounting|controller|auditor|tax", t):
        return "finance"
    if re.search(r"human resources|recruiter|talent|people partner", t):
        return "hr_recruiting"
    if re.search(r"security|cyber|infosec", t):
        return "cybersecurity"
    if re.search(r"manager|director|vp|vice president|chief|president|head of", t):
        return "management"
    if re.search(r"consultant|advisor|analyst", t):
        return "consulting_analyst"

    return "other"


# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.log1p(np.maximum(arr, 0))


class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    """
    CV-safe target encoder.
    Pipeline içinde fit edildiği için validation fold target bilgisini görmez.
    y log salary olduğundan encoded değer de log scale'dedir.
    """
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

            temp = pd.DataFrame({
                col: X_df[col].fillna("Unknown").astype(str),
                "_target": y_arr,
            })

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
                X_df[col]
                .fillna("Unknown")
                .astype(str)
                .map(self.maps_[col])
                .fillna(self.global_)
                .astype(float)
            )

        return X_df


# =============================================================================
# DATA
# =============================================================================
def load_and_engineer():
    print("Loading data...")

    post = safe_read_csv(BASE / "postings.csv", usecols=[
        "job_id", "title", "description", "company_id", "location",
        "normalized_salary", "pay_period", "currency",
        "formatted_experience_level", "formatted_work_type",
        "remote_allowed", "views", "applies",
    ])

    salary_mask = post["normalized_salary"].notna()
    q_low = post.loc[salary_mask, "normalized_salary"].quantile(0.01)
    q_high = post.loc[salary_mask, "normalized_salary"].quantile(0.99)

    lower_bound = max(10_000, q_low)
    upper_bound = min(300_000, q_high)

    df = post[
        post["normalized_salary"].between(lower_bound, upper_bound) &
        (post["currency"].isin(["USD"]) | post["currency"].isna())
    ].copy()

    if RUN_YEARLY_ONLY:
        before = len(df)
        df = df[df["pay_period"].fillna("").str.upper().eq("YEARLY")].copy()
        print(f"  YEARLY-only enabled: {before:,} -> {len(df):,} rows")

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

            spec_count = (
                cspec.groupby("company_id")["speciality"]
                .nunique()
                .reset_index()
                .rename(columns={"speciality": "company_speciality_count"})
            )
            df = df.merge(spec_count, on="company_id", how="left")

            top_specs = cspec["speciality"].value_counts().head(20).index.tolist()
            spec_pivot = (
                cspec[cspec["speciality"].isin(top_specs)]
                .assign(val=1)
                .pivot_table(index="company_id", columns="speciality", values="val", fill_value=0, aggfunc="max")
            )
            spec_pivot.columns = [f"company_spec_{clean_col_name(c)}" for c in spec_pivot.columns]
            spec_pivot = spec_pivot.reset_index()
            speciality_cols = [c for c in spec_pivot.columns if c != "company_id"]
            df = df.merge(spec_pivot, on="company_id", how="left")

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
    primary_ind = (
        job_ind.groupby("job_id")["industry_name"]
        .first()
        .reset_index()
        .rename(columns={"industry_name": "primary_industry"})
    )
    df = df.merge(primary_ind, on="job_id", how="left")
    df["primary_industry"] = df["primary_industry"].fillna("Unknown")

    # Skills top 50
    jskills = safe_read_csv(BASE / "job_skills.csv")
    skl_ref = safe_read_csv(BASE / "skills.csv")
    jskills = jskills.merge(skl_ref, on="skill_abr", how="left")

    top_skills = (
        jskills[jskills["job_id"].isin(df["job_id"])]["skill_name"]
        .dropna()
        .value_counts()
        .head(50)
        .index
        .tolist()
    )

    skill_pivot = (
        jskills[jskills["job_id"].isin(df["job_id"]) & jskills["skill_name"].isin(top_skills)]
        .assign(val=1)
        .pivot_table(index="job_id", columns="skill_name", values="val", fill_value=0, aggfunc="max")
    )
    skill_pivot.columns = [f"skill_{clean_col_name(c)}" for c in skill_pivot.columns]
    skill_pivot = skill_pivot.reset_index()
    skill_cols = [c for c in skill_pivot.columns if c != "job_id"]
    df = df.merge(skill_pivot, on="job_id", how="left")

    for col in skill_cols:
        df[col] = df[col].fillna(0).astype("int8")

    skill_count = (
        jskills[jskills["job_id"].isin(df["job_id"])]
        .groupby("job_id")["skill_name"]
        .nunique()
        .reset_index()
        .rename(columns={"skill_name": "n_skills"})
    )
    df = df.merge(skill_count, on="job_id", how="left")
    df["n_skills"] = df["n_skills"].fillna(0)

    # Benefits
    benefit_cols = []
    benefits_path = BASE / "benefits.csv"
    if benefits_path.exists():
        benefits = safe_read_csv(benefits_path)
        if "job_id" in benefits.columns:
            benefit_count = benefits.groupby("job_id").size().reset_index(name="benefit_count")
            df = df.merge(benefit_count, on="job_id", how="left")
            df["benefit_count"] = df["benefit_count"].fillna(0)

            if "type" in benefits.columns:
                benefits["type"] = benefits["type"].fillna("Unknown").astype(str)
                top_benefits = benefits["type"].value_counts().head(10).index.tolist()
                benefit_pivot = (
                    benefits[benefits["type"].isin(top_benefits)]
                    .assign(val=1)
                    .pivot_table(index="job_id", columns="type", values="val", fill_value=0, aggfunc="max")
                )
                benefit_pivot.columns = [f"benefit_{clean_col_name(c)}" for c in benefit_pivot.columns]
                benefit_pivot = benefit_pivot.reset_index()
                benefit_cols = [c for c in benefit_pivot.columns if c != "job_id"]
                df = df.merge(benefit_pivot, on="job_id", how="left")

                for col in benefit_cols:
                    df[col] = df[col].fillna(0).astype("int8")
        else:
            df["benefit_count"] = 0
    else:
        df["benefit_count"] = 0

    # Title features
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
    df["title_nurse"]      = title_lower.str.contains(r"nurse|rn\b|lpn", regex=True).astype(int)
    df["title_product"]    = title_lower.str.contains(r"product", regex=True).astype(int)
    df["title_finance"]    = title_lower.str.contains(r"finance|accountant|controller|accounting|auditor", regex=True).astype(int)
    df["title_security"]   = title_lower.str.contains(r"security|cyber|infosec", regex=True).astype(int)
    df["title_marketing"]  = title_lower.str.contains(r"marketing|brand|seo|content", regex=True).astype(int)
    df["title_hr"]         = title_lower.str.contains(r"human resources|recruiter|talent|people partner", regex=True).astype(int)

    df["title_len"] = df["title"].fillna("").str.len()
    df["title_word_count"] = df["title"].fillna("").str.split().str.len().fillna(0)
    df["title_text"] = df["title"].fillna("").astype(str)
    df["title_family"] = title_lower.apply(make_title_family)

    # ── Description cleaning (leakage removal) ───────────────────────────────
    # Compiled patterns — bir kez derlenir, tüm satırlarda kullanılır
    _SALARY_KW = re.compile(
        r"salary\s*range|compensation\s*range|pay\s*range|base\s*salary|"
        r"annual\s*salary|hourly\s*rate|expected\s*salary|pay\s*rate|"
        r"bonus\s*eligible|compensation|salary",
        re.IGNORECASE,
    )
    _DOLLAR_RE = re.compile(
        r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|K)?"
        r"(?:\s*[-–—to]+\s*\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|K)?)?",
        re.IGNORECASE,
    )
    _HOURLY_RE = re.compile(
        r"\b\d{1,3}(?:\.\d+)?(?:\s*[-–—]+\s*\d{1,3}(?:\.\d+)?)?"
        r"\s*(?:per\s*hour|/hour|/hr|hourly)\b",
        re.IGNORECASE,
    )
    _CURR_RE = re.compile(
        r"\b(?:usd|dollars|per\s+year|yearly|annually)\b", re.IGNORECASE
    )
    _WS_RE = re.compile(r"\s+")

    _desc_bar = ProgressBar(total=6, title="Cleaning descriptions (leakage removal)", unit="steps")

    # Adım 1 — ham text al, whitespace normalize et
    desc_series = df["description"].fillna("").astype(str)
    desc_series = desc_series.str.replace(r"\s+", " ", regex=True).str.strip()
    _desc_bar.step("Whitespace normalized")

    # Adım 2 — cümle bazında salary keyword filtresi
    # (eski [^.]* yerine: önce cümlelere böl, keyword içerenleri at)
    def _filter_salary_sentences(text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(s for s in sentences if not _SALARY_KW.search(s))

    desc_series = desc_series.map(_filter_salary_sentences)
    _desc_bar.step("Salary sentences removed")

    # Adım 3 — dollar miktarları ($120,000 - $160,000, $55k vb.)
    desc_series = desc_series.str.replace(_DOLLAR_RE, " ", regex=True)
    _desc_bar.step("Dollar amounts removed")

    # Adım 4 — saatlik ücret ifadeleri (50/hr, 40-60 hourly vb.)
    desc_series = desc_series.str.replace(_HOURLY_RE, " ", regex=True)
    _desc_bar.step("Hourly rates removed")

    # Adım 5 — currency keywords (usd, dollars, per year vb.)
    desc_series = desc_series.str.replace(_CURR_RE, " ", regex=True)
    _desc_bar.step("Currency keywords removed")

    # Adım 6 — son whitespace temizliği & kaydet
    df["description_clean"] = desc_series.str.replace(r"\s+", " ", regex=True).str.strip()
    _desc_bar.finish(label=f"{len(df):,} rows cleaned")
    # ─────────────────────────────────────────────────────────────────────────
    # Title clustering
    if USE_TITLE_CLUSTER:
        print("  Building title clusters...")
        tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=3,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        mat = tfidf.fit_transform(df["title_text"].fillna(""))
        n_clusters = min(N_TITLE_CLUSTERS, max(5, len(df) // 300))
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            batch_size=2048,
            n_init=10,
        )
        df["title_cluster"] = km.fit_predict(mat).astype(str)
    else:
        df["title_cluster"] = "0"

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

    # Location
    df["city_loc"] = df["location"].apply(normalize_city)
    df["state_final"] = df.apply(lambda row: extract_state(row["location"], row.get("state", None)), axis=1)

    # Interactions
    df["city_industry"] = df["city_loc"].astype(str) + "__" + df["primary_industry"].astype(str)
    df["state_industry"] = df["state_final"].astype(str) + "__" + df["primary_industry"].astype(str)
    df["exp_industry"] = df["formatted_experience_level"].astype(str) + "__" + df["primary_industry"].astype(str)
    df["exp_title_cluster"] = df["formatted_experience_level"].astype(str) + "__" + df["title_cluster"].astype(str)
    df["remote_industry"] = df["remote_allowed"].fillna(0).astype(int).astype(str) + "__" + df["primary_industry"].astype(str)
    df["pay_work_type"] = df["pay_period"].fillna("Unknown").astype(str) + "__" + df["formatted_work_type"].fillna("Unknown").astype(str)

    # Flags
    df["is_hourly"] = (df["pay_period"] == "HOURLY").astype(int)
    df["remote_flag"] = df["remote_allowed"].fillna(0).astype(int)
    df["has_applies"] = df["applies"].notna().astype(int)
    df["has_views"] = df["views"].notna().astype(int)

    # Categorical cleanup
    df["formatted_work_type"] = df["formatted_work_type"].fillna("Unknown")
    df["pay_period"] = df["pay_period"].fillna("Unknown")
    df["company_size"] = df["company_size"].fillna("Unknown").astype(str)
    df["company_id"] = df["company_id"].fillna(-1).astype(str)
    df["title_cluster"] = df["title_cluster"].fillna("Unknown").astype(str)
    df["title_family"] = df["title_family"].fillna("Unknown").astype(str)

    print(f"  Top skills used: {len(skill_cols)}")
    print(f"  Benefit columns used: {len(benefit_cols)}")
    print(f"  Company speciality columns used: {len(speciality_cols)}")
    print(f"  Feature engineering done. Shape: {df.shape}")

    return df, skill_cols, benefit_cols, speciality_cols


# =============================================================================
# FEATURE LISTS
# =============================================================================
CAT_COLS = [
    "formatted_experience_level",
    "formatted_work_type",
    "primary_industry",
    "pay_period",
    "company_size",
    "state_final",
    "title_cluster",
    "title_family",
]

TARGET_ENCODE_COLS = [
    "state_final",
    "city_loc",
    "primary_industry",
    "company_id",
    "title_cluster",
    "title_family",
    "city_industry",
    "state_industry",
    "exp_industry",
    "exp_title_cluster",
    "remote_industry",
    "pay_work_type",
]

TE_NUM_COLS = [f"te_{c}" for c in TARGET_ENCODE_COLS]

LOG_NUM_COLS = [
    "views",
    "applies",
    "employee_count",
    "follower_count",
    "company_job_count",
    "description_len",
    "description_word_count",
]

NUM_COLS = [
    "n_skills",
    "benefit_count",
    "company_industry_count",
    "company_speciality_count",
    "follower_per_employee",
    "title_len",
    "title_word_count",
]

BINARY_COLS = [
    "is_hourly",
    "remote_flag",
    "has_applies",
    "has_views",
    "has_employee_count",
    "has_follower_count",
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
]


def build_feature_columns(skill_cols, benefit_cols, speciality_cols):
    cols = (
        CAT_COLS
        + TARGET_ENCODE_COLS
        + LOG_NUM_COLS
        + NUM_COLS
        + BINARY_COLS
        + skill_cols
        + benefit_cols
        + speciality_cols
        + ["title_text", "description_clean"]
    )

    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_X_y(df, skill_cols, benefit_cols, speciality_cols):
    feature_cols = build_feature_columns(skill_cols, benefit_cols, speciality_cols)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols].copy()
    y_log = np.log1p(df["normalized_salary"].values)

    return X, y_log, feature_cols


# =============================================================================
# PIPELINE
# =============================================================================
def build_preprocessor(skill_cols, benefit_cols, speciality_cols):
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])

    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", LogTransformer()),
        ("scale", StandardScaler()),
    ])

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    bin_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    transformers = [
        ("cat", cat_pipe, CAT_COLS),
        ("te", num_pipe, TE_NUM_COLS),
        ("log_num", log_pipe, LOG_NUM_COLS),
        ("num", num_pipe, NUM_COLS),
        ("bin", bin_pipe, BINARY_COLS + skill_cols + benefit_cols + speciality_cols),
    ]

    if USE_TITLE_TEXT_FEATURES:
        title_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=TITLE_TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
                min_df=3,
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=TITLE_SVD_COMPONENTS, random_state=RANDOM_STATE)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("title_text", title_pipe, "title_text"))

    if USE_DESCRIPTION_TEXT_FEATURES:
        desc_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=DESC_TFIDF_MAX_FEATURES,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.85,
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
                stop_words="english",
            )),
            ("svd", TruncatedSVD(n_components=DESC_SVD_COMPONENTS, random_state=RANDOM_STATE)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("description_text", desc_pipe, "description_clean"))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(model_name, params=None):
    params = params or {}

    if model_name == "LightGBM":
        defaults = dict(
            n_estimators=900,
            learning_rate=0.04,
            max_depth=8,
            num_leaves=120,
            subsample=0.90,
            colsample_bytree=0.85,
            min_child_samples=15,
            reg_alpha=0.03,
            reg_lambda=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
        defaults.update(params)
        return lgb.LGBMRegressor(**defaults)

    if model_name == "XGBoost":
        defaults = dict(
            n_estimators=800,
            learning_rate=0.045,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.05,
            reg_lambda=1.2,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
        defaults.update(params)
        return xgb.XGBRegressor(**defaults)

    if model_name == "CatBoost":
        defaults = dict(
            iterations=800,
            learning_rate=0.045,
            depth=7,
            l2_leaf_reg=3,
            subsample=0.85,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )
        defaults.update(params)
        return CatBoostRegressor(**defaults)

    raise ValueError(model_name)


def build_pipeline(model_name, skill_cols, benefit_cols, speciality_cols, params=None):
    return Pipeline([
        ("target_encoder", MedianTargetEncoder(
            cols=TARGET_ENCODE_COLS,
            smoothing=40.0,
            min_samples_leaf=8,
        )),
        ("preprocessor", build_preprocessor(skill_cols, benefit_cols, speciality_cols)),
        ("model", build_model(model_name, params=params)),
    ])


# =============================================================================
# EVALUATION
# =============================================================================
def calc_metrics(y_log, pred_log):
    true_raw = np.expm1(y_log)
    pred_raw = np.expm1(pred_log)

    return {
        "rmse": float(np.sqrt(mean_squared_error(true_raw, pred_raw))),
        "mae": float(mean_absolute_error(true_raw, pred_raw)),
        "r2": float(r2_score(true_raw, pred_raw)),
        "log_r2": float(r2_score(y_log, pred_log)),
    }


def evaluate_cv(model_name, X, y_log, skill_cols, benefit_cols, speciality_cols, params=None, n_splits=5, return_oof=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    metrics = []
    oof = np.zeros(len(y_log), dtype=float)

    split_iter = kf.split(X)
    if TQDM_AVAILABLE:
        split_iter = tqdm(
            split_iter,
            total=n_splits,
            desc=f"{model_name} CV",
            leave=False
        )

    for fold, (tr, val) in enumerate(split_iter, start=1):
        fold_t0 = time.time()
        pipe = build_pipeline(
            model_name=model_name,
            skill_cols=skill_cols,
            benefit_cols=benefit_cols,
            speciality_cols=speciality_cols,
            params=params,
        )

        pipe.fit(X.iloc[tr], y_log[tr])
        pred_log = pipe.predict(X.iloc[val])
        oof[val] = pred_log

        m = calc_metrics(y_log[val], pred_log)
        metrics.append(m)

        print(
            f"  {model_name} Fold {fold}: "
            f"RMSE=${m['rmse']:,.0f} "
            f"MAE=${m['mae']:,.0f} "
            f"Raw R²={m['r2']:.3f} "
            f"Log R²={m['log_r2']:.3f}"
        )

    out = {
        "rmse": float(np.mean([m["rmse"] for m in metrics])),
        "rmse_std": float(np.std([m["rmse"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
        "mae_std": float(np.std([m["mae"] for m in metrics])),
        "r2": float(np.mean([m["r2"] for m in metrics])),
        "r2_std": float(np.std([m["r2"] for m in metrics])),
        "log_r2": float(np.mean([m["log_r2"] for m in metrics])),
        "log_r2_std": float(np.std([m["log_r2"] for m in metrics])),
    }

    if return_oof:
        return out, oof

    return out


def baseline_cv(X, y_log, skill_cols, benefit_cols, speciality_cols):
    print("Running baseline CV (3-fold)...")
    results = {}

    for model_name in MODEL_NAMES:
        print(f"\nBaseline {model_name}:")
        res = evaluate_cv(
            model_name=model_name,
            X=X,
            y_log=y_log,
            skill_cols=skill_cols,
            benefit_cols=benefit_cols,
            speciality_cols=speciality_cols,
            params=None,
            n_splits=3,
            return_oof=False,
        )

        results[model_name] = res
        print(
            f"  ► {model_name}: "
            f"RMSE=${res['rmse']:,.0f} "
            f"MAE=${res['mae']:,.0f} "
            f"Raw R²={res['r2']:.3f} "
            f"Log R²={res['log_r2']:.3f}"
        )

    return results


# =============================================================================
# OPTUNA
# =============================================================================
def suggest_params(trial, model_name):
    if model_name == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 500, 1200),
            max_depth=trial.suggest_int("max_depth", 5, 11),
            num_leaves=trial.suggest_int("num_leaves", 50, 180),
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
            subsample=trial.suggest_float("subsample", 0.70, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 8, 70),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )

    if model_name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 500, 1200),
            max_depth=trial.suggest_int("max_depth", 5, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
            subsample=trial.suggest_float("subsample", 0.70, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )

    if model_name == "CatBoost":
        return dict(
            iterations=trial.suggest_int("iterations", 500, 1200),
            depth=trial.suggest_int("depth", 5, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
            subsample=trial.suggest_float("subsample", 0.70, 1.0),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.5, 5.0),
        )

    raise ValueError(model_name)

def tune_model(model_name, X, y_log, skill_cols, benefit_cols, speciality_cols):
    kf = KFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = suggest_params(trial, model_name)
        scores = []

        for tr, val in kf.split(X):
            pipe = build_pipeline(
                model_name=model_name,
                skill_cols=skill_cols,
                benefit_cols=benefit_cols,
                speciality_cols=speciality_cols,
                params=params,
            )

            pipe.fit(X.iloc[tr], y_log[tr])
            pred_log = pipe.predict(X.iloc[val])
            true_raw = np.expm1(y_log[val])
            pred_raw = np.expm1(pred_log)
            rmse = np.sqrt(mean_squared_error(true_raw, pred_raw))
            scores.append(rmse)

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    return study


# =============================================================================
# PLOTS AND SAVE
# =============================================================================
def plot_results(results):
    plt.rcParams.update(STYLE)

    model_names = list(results.keys())
    metrics = [
        ("rmse", "RMSE (USD)", "Lower is better"),
        ("mae", "MAE (USD)", "Lower is better"),
        ("r2", "Raw R²", "Higher is better"),
        ("log_r2", "Log R²", "Higher is better"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Model 03 Advanced — Final Results", fontsize=15, fontweight="bold", y=1.02)

    for ax, (key, title, subtitle) in zip(axes, metrics):
        vals = [results[m][key] for m in model_names]
        colors = [COLORS.get(m, "#10B981") for m in model_names]

        bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor="none", width=0.55)

        for bar, val in zip(bars, vals):
            label = f"${val:,.0f}" if key in ("rmse", "mae") else f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#E2E8F0",
                fontweight="bold",
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=9, color="#8B92A5")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)

        if key in ("rmse", "mae"):
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x / 1000:.0f}k"))

    plt.tight_layout()
    plt.savefig(OUT / "model_03_advanced_results.png")
    plt.close()
    print("  ✓ model_03_advanced_results.png")


def plot_tuning_curves(studies):
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, len(studies), figsize=(8 * len(studies), 5))
    if len(studies) == 1:
        axes = [axes]

    fig.suptitle("Model 03 — Optuna Tuning Curves", fontsize=15, fontweight="bold", y=1.02)

    for ax, (name, study) in zip(axes, studies.items()):
        trials = study.trials_dataframe()
        values = trials["value"].values
        best_so_far = np.minimum.accumulate(values)

        ax.scatter(range(len(values)), values, color=COLORS.get(name, "#3B82F6"), s=18, alpha=0.55, label="Trial RMSE")
        ax.plot(best_so_far, color="#F8FAFC", linewidth=2, label="Best so far")
        ax.axhline(study.best_value, linestyle="--", color=COLORS.get(name, "#3B82F6"), linewidth=1.2)

        ax.set_title(name, fontsize=12, fontweight="bold", color=COLORS.get(name, "#3B82F6"))
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV RMSE")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x / 1000:.0f}k"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "model_03_tuning_curves.png")
    plt.close()
    print("  ✓ model_03_tuning_curves.png")


def save_best_model(results, best_params_all, X, y_log, skill_cols, benefit_cols, speciality_cols, feature_cols):
    best_name = max(results, key=lambda k: results[k]["r2"])

    if best_name == "Ensemble":
        print("  Fitting full-data ensemble pipelines...")
        ensemble_weights = results["Ensemble"].get("weights", {m: 1/len(MODEL_NAMES) for m in MODEL_NAMES})
        pipelines = {}
        for mn in MODEL_NAMES:
            pipe = build_pipeline(mn, skill_cols, benefit_cols, speciality_cols, best_params_all[mn])
            pipe.fit(X, y_log)
            pipelines[mn] = pipe

        obj = {
            "type": "ensemble",
            "weights": ensemble_weights,
            "pipelines": pipelines,
            "feature_cols": feature_cols,
            "metrics": results[best_name],
            "log_target": True,
            "run_yearly_only": RUN_YEARLY_ONLY,
            "description_cleaned": True,
        }
    else:
        print(f"  Fitting full-data {best_name} pipeline...")
        pipe = build_pipeline(best_name, skill_cols, benefit_cols, speciality_cols, best_params_all[best_name])
        pipe.fit(X, y_log)

        obj = {
            "type": "single_model",
            "model_name": best_name,
            "pipeline": pipe,
            "best_params": best_params_all[best_name],
            "feature_cols": feature_cols,
            "metrics": results[best_name],
            "log_target": True,
            "run_yearly_only": RUN_YEARLY_ONLY,
            "description_cleaned": True,
        }

    path = MODELS / "best_salary_model_03_advanced_progress.joblib"
    joblib.dump(obj, path)

    metrics_path = OUT / "model_03_metrics_progress.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Saved advanced model → {path}")
    print(f"  ✓ Saved metrics → {metrics_path}")

    return path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 82)
    print("  model_03_salary_advanced.py — Advanced Salary Prediction")
    print("=" * 82)
    print(f"  RUN_YEARLY_ONLY = {RUN_YEARLY_ONLY}")
    print(f"  N_TRIALS = {N_TRIALS}")
    print(f"  Title TF-IDF/SVD = {USE_TITLE_TEXT_FEATURES}")
    print(f"  Description TF-IDF/SVD = {USE_DESCRIPTION_TEXT_FEATURES}")
    print(f"  Title clustering = {USE_TITLE_CLUSTER}")

    log_step("1/7 Loading and engineering data")
    _main_t0 = time.time()
    df, skill_cols, benefit_cols, speciality_cols = load_and_engineer()
    log_duration(_main_t0, "Data loading + feature engineering")
    log_step("2/7 Building feature matrix")
    _t0 = time.time()
    X, y_log, feature_cols = build_X_y(df, skill_cols, benefit_cols, speciality_cols)
    log_duration(_t0, "Feature matrix build")

    print("\nFeature matrix:")
    print(f"  Rows: {X.shape[0]:,}")
    print(f"  Input columns: {X.shape[1]:,}")
    print(f"  Skill cols: {len(skill_cols)}")
    print(f"  Benefit cols: {len(benefit_cols)}")
    print(f"  Company speciality cols: {len(speciality_cols)}")
    print(f"  Target range: ${df['normalized_salary'].min():,.0f} - ${df['normalized_salary'].max():,.0f}")
    print(f"  Target median: ${df['normalized_salary'].median():,.0f}")

    print("\n" + "-" * 82)
    print("BASELINE CV")
    print("-" * 82)
    log_step("3/7 Running baseline CV")
    _t0 = time.time()
    baseline_results = baseline_cv(X, y_log, skill_cols, benefit_cols, speciality_cols)
    log_duration(_t0, "Baseline CV")

    print("\n" + "-" * 82)
    print(f"OPTUNA TUNING ({N_TRIALS} trials × {len(MODEL_NAMES)} models × {N_TUNE_FOLDS}-fold CV)")
    print("-" * 82)

    studies = {}
    best_params_all = {}

    for model_name in MODEL_NAMES:
        print(f"\nTuning {model_name}...", flush=True)
        log_step(f"4/7 Optuna tuning started for {model_name}")
        _t0 = time.time()
        study = tune_model(model_name, X, y_log, skill_cols, benefit_cols, speciality_cols)
        log_duration(_t0, f"Optuna tuning {model_name}")
        studies[model_name] = study
        best_params_all[model_name] = study.best_params
        print(f"  Best tuning RMSE: ${study.best_value:,.0f}")
        print(f"  Best params: {study.best_params}")

    print("\n" + "-" * 82)
    print(f"FINAL {N_FINAL_FOLDS}-FOLD CV WITH BEST PARAMS")
    print("-" * 82)

    final_results = {}
    oof_preds = {}

    for model_name in MODEL_NAMES:
        print(f"\nFinal CV {model_name}:")
        res, oof = evaluate_cv(
            model_name=model_name,
            X=X,
            y_log=y_log,
            skill_cols=skill_cols,
            benefit_cols=benefit_cols,
            speciality_cols=speciality_cols,
            params=best_params_all[model_name],
            n_splits=N_FINAL_FOLDS,
            return_oof=True,
        )
        final_results[model_name] = res
        oof_preds[model_name] = oof

        print(
            f"  ► {model_name}: "
            f"RMSE=${res['rmse']:,.0f}±{res['rmse_std']:,.0f} "
            f"MAE=${res['mae']:,.0f} "
            f"Raw R²={res['r2']:.4f} "
            f"Log R²={res['log_r2']:.4f}"
        )

    available = [m for m in MODEL_NAMES if m in oof_preds]
    if len(available) >= 2:
        print("\nOOF Ensemble:")
        # Her modelin OOF RMSE'sine göre ters orantılı ağırlık
        rmse_scores = {m: final_results[m]["rmse"] for m in available}
        inv = {m: 1.0 / rmse_scores[m] for m in available}
        total = sum(inv.values())
        weights = {m: inv[m] / total for m in available}
        print(f"  Ensemble weights: { {m: f'{w:.3f}' for m, w in weights.items()} }")
        ensemble_oof = sum(weights[m] * oof_preds[m] for m in available)
        ens = calc_metrics(y_log, ensemble_oof)
        ens["rmse_std"] = 0.0
        ens["mae_std"] = 0.0
        ens["weights"] = weights
        final_results["Ensemble"] = ens
        print(
            f"  Ensemble: RMSE=${ens['rmse']:,.0f} "
            f"MAE=${ens['mae']:,.0f} "
            f"Raw R²={ens['r2']:.4f} "
            f"Log R²={ens['log_r2']:.4f}"
        )

    log_step("6/7 Generating plots")
    print("\nGenerating plots...")
    plot_tuning_curves(studies)
    plot_results(final_results)

    log_step("7/7 Saving best advanced model")
    print("\nSaving best advanced model...")
    model_path = save_best_model(
        results=final_results,
        best_params_all=best_params_all,
        X=X,
        y_log=y_log,
        skill_cols=skill_cols,
        benefit_cols=benefit_cols,
        speciality_cols=speciality_cols,
        feature_cols=feature_cols,
    )

    print("\n" + "=" * 82)
    print("  MODEL 03 FINAL SUMMARY")
    print("=" * 82)
    fmt = "{:<12} {:>18} {:>14} {:>10} {:>10}"
    print(fmt.format("Model", "RMSE (USD)", "MAE (USD)", "Raw R²", "Log R²"))
    print("-" * 82)

    for name, r in final_results.items():
        print(fmt.format(
            name,
            f"${r['rmse']:,.0f} ±{r.get('rmse_std', 0):,.0f}",
            f"${r['mae']:,.0f}",
            f"{r['r2']:.4f}",
            f"{r['log_r2']:.4f}",
        ))

    best_name = max(final_results, key=lambda k: final_results[k]["r2"])

    print("=" * 82)
    print(f"\n  Best model: {best_name}")
    print(f"  Best Raw R²: {final_results[best_name]['r2']:.4f}")
    print(f"  Best Log R²: {final_results[best_name]['log_r2']:.4f}")
    print(f"  Saved: {model_path}")
    print("\n  Outputs:")
    print("    - outputs/model_03_tuning_curves.png")
    print("    - outputs/model_03_advanced_results.png")
    print("    - outputs/model_03_metrics.json")

    print("\n  If Raw R² is still below 0.80:")
    print("    1. Increase N_TRIALS to 20/30/50")
    print("    2. Increase DESC_SVD_COMPONENTS to 80 or 100")
    print("    3. Compare RUN_YEARLY_ONLY=False")
    print("    4. Add separate models by pay_period")