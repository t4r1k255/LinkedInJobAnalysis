"""
prepare_shap_data.py
────────────────────
model_03'ün feature engineering'ini çalıştırır ve X, y ile feature_cols'u
models/ klasörüne kaydeder. shap_dependence.py bu dosyaları yükler.

Çalışma süresi: ~2-3 dakika (sadece feature engineering, training yok)
Çıktılar:
  models/shap_X.parquet        — feature matrix (35k × 122)
  models/shap_y.npy            — log salary array
  models/shap_feature_cols.json — feature column names
"""
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from utils_progress import ProgressBar, StepTracker

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG (model_03 ile aynı) ────────────────────────────────────────────────
BASE             = Path("data")
MODELS_PATH      = Path("models")
RANDOM_STATE     = 42
RUN_YEARLY_ONLY  = False
USE_TITLE_CLUSTER = True
N_TITLE_CLUSTERS = 35

MODELS_PATH.mkdir(exist_ok=True)

tracker = StepTracker(total_steps=4, script_name="prepare_shap_data.py — Feature Engineering for SHAP")

# =============================================================================
# HELPER FUNCTIONS (model_03 ile birebir aynı)
# =============================================================================
def safe_read_csv(path, usecols=None):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path, usecols=usecols)
    except ValueError as e:
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
            "san francisco bay area": "CA", "new york": "NY",
            "greater seattle": "WA", "greater boston": "MA",
            "greater chicago": "IL", "los angeles": "CA",
            "washington dc": "DC", "dallas": "TX",
            "houston": "TX", "phoenix": "AZ", "atlanta": "GA",
        }
        for key, val in metro_state_map.items():
            if key in low:
                return val
    if pd.notna(company_state):
        return str(company_state)
    return "Unknown"


def make_title_family(title_lower):
    t = title_lower or ""
    if re.search(r"nurse|rn\b|lpn|medical assistant|physician|therapist|clinical", t): return "healthcare"
    if re.search(r"machine learning|data scientist|data engineer|data analyst|analytics|bi\b|ai\b|ml\b", t): return "data_ai"
    if re.search(r"software|engineer|developer|swe|sde|backend|frontend|full stack|devops|cloud|architect", t): return "software_engineering"
    if re.search(r"product manager|product owner|product lead", t): return "product"
    if re.search(r"sales|account executive|business development|customer success", t): return "sales"
    if re.search(r"marketing|brand|seo|content|growth", t): return "marketing"
    if re.search(r"finance|accountant|accounting|controller|auditor|tax", t): return "finance"
    if re.search(r"human resources|recruiter|talent|people partner", t): return "hr_recruiting"
    if re.search(r"security|cyber|infosec", t): return "cybersecurity"
    if re.search(r"manager|director|vp|vice president|chief|president|head of", t): return "management"
    if re.search(r"consultant|advisor|analyst", t): return "consulting_analyst"
    return "other"


# =============================================================================
# FEATURE LISTS (model_03 ile aynı)
# =============================================================================
CAT_COLS = [
    "formatted_experience_level", "formatted_work_type",
    "primary_industry", "pay_period", "company_size",
    "state_final", "title_cluster", "title_family",
]

TARGET_ENCODE_COLS = [
    "state_final", "city_loc", "primary_industry", "company_id",
    "title_cluster", "title_family", "city_industry", "state_industry",
    "exp_industry", "exp_title_cluster", "remote_industry", "pay_work_type",
]

LOG_NUM_COLS = ["views", "applies", "employee_count", "follower_count",
                "company_job_count", "description_len", "description_word_count"]

NUM_COLS = ["n_skills", "benefit_count", "company_industry_count",
            "company_speciality_count", "follower_per_employee",
            "title_len", "title_word_count"]

BINARY_COLS = [
    "is_hourly", "remote_flag", "has_applies", "has_views",
    "has_employee_count", "has_follower_count",
    "title_senior", "title_principal", "title_director", "title_vp",
    "title_chief", "title_manager", "title_junior", "title_associate",
    "title_engineer", "title_architect", "title_data", "title_sales",
    "title_consultant", "title_nurse", "title_product", "title_finance",
    "title_security", "title_marketing", "title_hr",
]


def build_feature_columns(skill_cols, benefit_cols, speciality_cols):
    cols = (CAT_COLS + TARGET_ENCODE_COLS + LOG_NUM_COLS + NUM_COLS + BINARY_COLS
            + skill_cols + benefit_cols + speciality_cols + ["title_text", "description_clean"])
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# =============================================================================
# FEATURE ENGINEERING (model_03'ten kopyalandı)
# =============================================================================
tracker.start(1, "Loading raw data")
_bar = ProgressBar(total=8, title="Loading CSV files", unit="files")

post = safe_read_csv(BASE / "postings.csv", usecols=[
    "job_id", "title", "description", "company_id", "location",
    "normalized_salary", "pay_period", "currency",
    "formatted_experience_level", "formatted_work_type",
    "remote_allowed", "views", "applies",
])
_bar.step("postings.csv")

comp = safe_read_csv(BASE / "companies.csv", usecols=["company_id", "company_size", "state", "country"])
_bar.step("companies.csv")

emp = safe_read_csv(BASE / "employee_counts.csv", usecols=["company_id", "employee_count", "follower_count"])
_bar.step("employee_counts.csv")

job_ind = safe_read_csv(BASE / "job_industries.csv")
_bar.step("job_industries.csv")

ind_ref = safe_read_csv(BASE / "industries.csv")
_bar.step("industries.csv")

jskills = safe_read_csv(BASE / "job_skills.csv")
_bar.step("job_skills.csv")

skl_ref = safe_read_csv(BASE / "skills.csv")
_bar.step("skills.csv")

benefits = safe_read_csv(BASE / "benefits.csv")
_bar.step("benefits.csv")
_bar.finish()
tracker.done(1)

# ── Salary filter ─────────────────────────────────────────────────────────────
tracker.start(2, "Feature engineering")

salary_mask = post["normalized_salary"].notna()
q_low  = post.loc[salary_mask, "normalized_salary"].quantile(0.01)
q_high = post.loc[salary_mask, "normalized_salary"].quantile(0.99)
lower_bound = max(10_000, q_low)
upper_bound = min(300_000, q_high)

df = post[
    post["normalized_salary"].between(lower_bound, upper_bound) &
    (post["currency"].isin(["USD"]) | post["currency"].isna())
].copy()

if RUN_YEARLY_ONLY:
    df = df[df["pay_period"].fillna("").str.upper().eq("YEARLY")].copy()

df["job_id"]     = df["job_id"].astype("Int64")
df["company_id"] = df["company_id"].astype("Int64")
print(f"  Salary rows: {len(df):,}")

# ── Company features ──────────────────────────────────────────────────────────
emp = emp.sort_values("employee_count").drop_duplicates("company_id", keep="last")
comp = comp.merge(emp, on="company_id", how="left")
df = df.merge(comp, on="company_id", how="left")
df["company_job_count"]    = df.groupby("company_id")["job_id"].transform("count").fillna(0)
df["has_employee_count"]   = df["employee_count"].notna().astype(int)
df["has_follower_count"]   = df["follower_count"].notna().astype(int)
df["follower_per_employee"] = df["follower_count"].fillna(0) / (df["employee_count"].fillna(0) + 1)
df["company_industry_count"] = 0

# ── Company specialities ──────────────────────────────────────────────────────
speciality_cols = []
cspec_path = BASE / "company_specialities.csv"
if cspec_path.exists():
    cspec = safe_read_csv(cspec_path)
    company_col = find_column(cspec, ["company_id"])
    spec_col    = find_column(cspec, ["speciality", "specialty", "specialities", "specialties"])
    if company_col and spec_col:
        cspec = cspec.rename(columns={company_col: "company_id", spec_col: "speciality"})
        cspec["speciality"] = cspec["speciality"].fillna("Unknown").astype(str)
        spec_count = (cspec.groupby("company_id")["speciality"].nunique()
                      .reset_index().rename(columns={"speciality": "company_speciality_count"}))
        df = df.merge(spec_count, on="company_id", how="left")
        top_specs  = cspec["speciality"].value_counts().head(20).index.tolist()
        spec_pivot = (cspec[cspec["speciality"].isin(top_specs)].assign(val=1)
                      .pivot_table(index="company_id", columns="speciality", values="val",
                                   fill_value=0, aggfunc="max"))
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

# ── Primary industry ──────────────────────────────────────────────────────────
job_ind = job_ind.merge(ind_ref, on="industry_id", how="left")
primary_ind = (job_ind.groupby("job_id")["industry_name"].first()
               .reset_index().rename(columns={"industry_name": "primary_industry"}))
df = df.merge(primary_ind, on="job_id", how="left")
df["primary_industry"] = df["primary_industry"].fillna("Unknown")

# ── Skills ────────────────────────────────────────────────────────────────────
jskills = jskills.merge(skl_ref, on="skill_abr", how="left")
top_skills = (jskills[jskills["job_id"].isin(df["job_id"])]["skill_name"]
              .dropna().value_counts().head(50).index.tolist())
skill_pivot = (jskills[jskills["job_id"].isin(df["job_id"]) & jskills["skill_name"].isin(top_skills)]
               .assign(val=1)
               .pivot_table(index="job_id", columns="skill_name", values="val", fill_value=0, aggfunc="max"))
skill_pivot.columns = [f"skill_{clean_col_name(c)}" for c in skill_pivot.columns]
skill_pivot = skill_pivot.reset_index()
skill_cols  = [c for c in skill_pivot.columns if c != "job_id"]
df = df.merge(skill_pivot, on="job_id", how="left")
for col in skill_cols:
    df[col] = df[col].fillna(0).astype("int8")

skill_count = (jskills[jskills["job_id"].isin(df["job_id"])]
               .groupby("job_id")["skill_name"].nunique().reset_index()
               .rename(columns={"skill_name": "n_skills"}))
df = df.merge(skill_count, on="job_id", how="left")
df["n_skills"] = df["n_skills"].fillna(0)

# ── Benefits ──────────────────────────────────────────────────────────────────
benefit_cols = []
if "job_id" in benefits.columns:
    benefit_count = benefits.groupby("job_id").size().reset_index(name="benefit_count")
    df = df.merge(benefit_count, on="job_id", how="left")
    df["benefit_count"] = df["benefit_count"].fillna(0)
    if "type" in benefits.columns:
        benefits["type"] = benefits["type"].fillna("Unknown").astype(str)
        top_benefits = benefits["type"].value_counts().head(10).index.tolist()
        benefit_pivot = (benefits[benefits["type"].isin(top_benefits)].assign(val=1)
                         .pivot_table(index="job_id", columns="type", values="val",
                                      fill_value=0, aggfunc="max"))
        benefit_pivot.columns = [f"benefit_{clean_col_name(c)}" for c in benefit_pivot.columns]
        benefit_pivot = benefit_pivot.reset_index()
        benefit_cols = [c for c in benefit_pivot.columns if c != "job_id"]
        df = df.merge(benefit_pivot, on="job_id", how="left")
        for col in benefit_cols:
            df[col] = df[col].fillna(0).astype("int8")
else:
    df["benefit_count"] = 0

# ── Title features ────────────────────────────────────────────────────────────
title_lower = df["title"].fillna("").str.lower()
df["title_senior"]    = title_lower.str.contains(r"senior|sr\.?\b|lead|principal|staff\b", regex=True).astype(int)
df["title_principal"] = title_lower.str.contains(r"principal", regex=True).astype(int)
df["title_director"]  = title_lower.str.contains(r"director", regex=True).astype(int)
df["title_vp"]        = title_lower.str.contains(r"\bvp\b|vice pres", regex=True).astype(int)
df["title_chief"]     = title_lower.str.contains(r"\bchief\b|president|ceo|cto|cfo|coo", regex=True).astype(int)
df["title_manager"]   = title_lower.str.contains(r"manager|mgr", regex=True).astype(int)
df["title_junior"]    = title_lower.str.contains(r"junior|jr\.?\b|entry|intern", regex=True).astype(int)
df["title_associate"] = title_lower.str.contains(r"\bassociate\b", regex=True).astype(int)
df["title_engineer"]  = title_lower.str.contains(r"engineer|developer|software|swe|sde", regex=True).astype(int)
df["title_architect"] = title_lower.str.contains(r"architect", regex=True).astype(int)
df["title_data"]      = title_lower.str.contains(r"data|analyst|scientist|ml|ai|machine", regex=True).astype(int)
df["title_sales"]     = title_lower.str.contains(r"sales|account exec|business dev", regex=True).astype(int)
df["title_consultant"]= title_lower.str.contains(r"consultant|advisor", regex=True).astype(int)
df["title_nurse"]     = title_lower.str.contains(r"nurse|rn\b|lpn", regex=True).astype(int)
df["title_product"]   = title_lower.str.contains(r"product", regex=True).astype(int)
df["title_finance"]   = title_lower.str.contains(r"finance|accountant|controller|accounting|auditor", regex=True).astype(int)
df["title_security"]  = title_lower.str.contains(r"security|cyber|infosec", regex=True).astype(int)
df["title_marketing"] = title_lower.str.contains(r"marketing|brand|seo|content", regex=True).astype(int)
df["title_hr"]        = title_lower.str.contains(r"human resources|recruiter|talent|people partner", regex=True).astype(int)
df["title_len"]        = df["title"].fillna("").str.len()
df["title_word_count"] = df["title"].fillna("").str.split().str.len().fillna(0)
df["title_text"]       = df["title"].fillna("").astype(str)
df["title_family"]     = title_lower.apply(make_title_family)

# ── Description (basit temizlik, leakage removal) ─────────────────────────────
_SALARY_KW = re.compile(
    r"salary\s*range|compensation\s*range|pay\s*range|base\s*salary|"
    r"annual\s*salary|hourly\s*rate|expected\s*salary|pay\s*rate|"
    r"bonus\s*eligible|compensation|salary", re.IGNORECASE)
_DOLLAR_RE = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|K)?"
    r"(?:\s*[-–—to]+\s*\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:k|K)?)?", re.IGNORECASE)
_HOURLY_RE = re.compile(
    r"\b\d{1,3}(?:\.\d+)?(?:\s*[-–—]+\s*\d{1,3}(?:\.\d+)?)?"
    r"\s*(?:per\s*hour|/hour|/hr|hourly)\b", re.IGNORECASE)
_CURR_RE = re.compile(r"\b(?:usd|dollars|per\s+year|yearly|annually)\b", re.IGNORECASE)

_desc_bar = ProgressBar(total=4, title="Cleaning descriptions", unit="steps")
desc = df["description"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
_desc_bar.step("Whitespace normalized")
desc = desc.map(lambda t: " ".join(s for s in re.split(r"(?<=[.!?])\s+", t) if not _SALARY_KW.search(s)))
_desc_bar.step("Salary sentences removed")
desc = desc.str.replace(_DOLLAR_RE, " ", regex=True).str.replace(_HOURLY_RE, " ", regex=True)
_desc_bar.step("Dollar/hourly amounts removed")
df["description_clean"] = desc.str.replace(_CURR_RE, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
df["description_len"]        = df["description_clean"].str.len().fillna(0)
df["description_word_count"] = df["description_clean"].str.split().str.len().fillna(0)
_desc_bar.step("Description features built")
_desc_bar.finish()

# ── Title clustering ──────────────────────────────────────────────────────────
if USE_TITLE_CLUSTER:
    print("  Building title clusters...")
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=3,
                            lowercase=True, strip_accents="unicode", sublinear_tf=True)
    mat = tfidf.fit_transform(df["title_text"].fillna(""))
    n_clusters = min(N_TITLE_CLUSTERS, max(5, len(df) // 300))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, batch_size=2048, n_init=10)
    df["title_cluster"] = km.fit_predict(mat).astype(str)
else:
    df["title_cluster"] = "0"

# ── Experience imputation ─────────────────────────────────────────────────────
def impute_exp(row):
    if pd.notna(row["formatted_experience_level"]):
        return row["formatted_experience_level"]
    t = str(row["title"]).lower() if pd.notna(row["title"]) else ""
    if any(x in t for x in ["chief","president","ceo","cto","cfo","coo"]): return "Executive"
    if any(x in t for x in ["vp","vice pres","director","head of"]):       return "Director"
    if any(x in t for x in ["principal","senior","sr.","lead","staff","manager"]): return "Mid-Senior level"
    if any(x in t for x in ["junior","jr.","intern","entry"]):             return "Entry level"
    return "Mid-Senior level"

df["formatted_experience_level"] = df.apply(impute_exp, axis=1)

# ── Location & interaction features ──────────────────────────────────────────
df["city_loc"]    = df["location"].apply(normalize_city)
df["state_final"] = df.apply(lambda r: extract_state(r["location"], r.get("state", None)), axis=1)

df["city_industry"]     = df["city_loc"].astype(str) + "__" + df["primary_industry"].astype(str)
df["state_industry"]    = df["state_final"].astype(str) + "__" + df["primary_industry"].astype(str)
df["exp_industry"]      = df["formatted_experience_level"].astype(str) + "__" + df["primary_industry"].astype(str)
df["exp_title_cluster"] = df["formatted_experience_level"].astype(str) + "__" + df["title_cluster"].astype(str)
df["remote_industry"]   = df["remote_allowed"].fillna(0).astype(int).astype(str) + "__" + df["primary_industry"].astype(str)
df["pay_work_type"]     = df["pay_period"].fillna("Unknown").astype(str) + "__" + df["formatted_work_type"].fillna("Unknown").astype(str)

# ── Flags & cleanup ───────────────────────────────────────────────────────────
df["is_hourly"]  = (df["pay_period"] == "HOURLY").astype(int)
df["remote_flag"] = df["remote_allowed"].fillna(0).astype(int)
df["has_applies"] = df["applies"].notna().astype(int)
df["has_views"]   = df["views"].notna().astype(int)
df["formatted_work_type"] = df["formatted_work_type"].fillna("Unknown")
df["pay_period"]  = df["pay_period"].fillna("Unknown")
df["company_size"] = df["company_size"].fillna("Unknown").astype(str)
df["company_id"]  = df["company_id"].fillna(-1).astype(str)
df["title_cluster"] = df["title_cluster"].fillna("Unknown").astype(str)
df["title_family"]  = df["title_family"].fillna("Unknown").astype(str)

tracker.done(2, f"{len(df):,} rows engineered")

# =============================================================================
# BUILD X, y
# =============================================================================
tracker.start(3, "Building feature matrix")

feature_cols = build_feature_columns(skill_cols, benefit_cols, speciality_cols)
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0

X = df[feature_cols].copy()
y_log = np.log1p(df["normalized_salary"].values)
print(f"  X shape: {X.shape}")
print(f"  feature_cols: {len(feature_cols)}")
tracker.done(3)

# =============================================================================
# SAVE
# =============================================================================
tracker.start(4, "Saving to disk")

X.to_parquet(MODELS_PATH / "shap_X.parquet", index=False)
np.save(str(MODELS_PATH / "shap_y.npy"), y_log)
with open(MODELS_PATH / "shap_feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

print(f"  ✓ models/shap_X.parquet")
print(f"  ✓ models/shap_y.npy")
print(f"  ✓ models/shap_feature_cols.json")
tracker.done(4)
tracker.finish()
