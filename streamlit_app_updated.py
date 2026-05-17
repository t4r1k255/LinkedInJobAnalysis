
"""
streamlit_app.py — Local LinkedIn Job Analysis Dashboard v6

Run:
    streamlit run streamlit_app.py

Design goals:
- Local-first dashboard, no cloud/toml/bat requirement
- Basic / Advanced view modes
- Stakeholder-specific decision-support pages
- Salary prediction with role-context builder, similar postings, reliability signal, scenario comparison
- Model 04 metrics / interval / error-analysis integration when CSV/JSON outputs exist
"""

from __future__ import annotations

import json
import re
import sys
import time
import types
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
BASE = Path(__file__).parent
DATA = BASE / "data" if (BASE / "data").exists() else BASE
MODELS = BASE / "models" if (BASE / "models").exists() else BASE
OUT = BASE / "outputs" if (BASE / "outputs").exists() else BASE

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="LinkedIn Job Analysis Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: #0F1117; }
.section-title {
    font-size: 1.32rem;
    font-weight: 700;
    margin-top: 0.55rem;
    margin-bottom: 0.65rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}
.soft-box, .green-box, .amber-box, .red-box {
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0 14px 0;
}
.soft-box  { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.24); }
.green-box { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.24); }
.amber-box { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); }
.red-box   { background: rgba(244,63,94,0.08); border: 1px solid rgba(244,63,94,0.25); }
.persona-card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 14px;
    height: 100%;
    background: rgba(255,255,255,0.025);
}
.small-note { color: #9CA3AF; font-size: 0.88rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CUSTOM TRANSFORMERS REQUIRED FOR JOBLIB LOADING
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
            if col not in X_df.columns or col not in getattr(self, "maps_", {}):
                X_df[new_col] = getattr(self, "global_", 0.0)
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


for module_name in [
    "__main__",
    "model_03_salary_advanced_progress",
    "model_03_salary_advanced",
    "model_02_pipeline_v3_cv_safe",
    "model_04_stacking_intervals",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)
    setattr(sys.modules[module_name], "LogTransformer", LogTransformer)
    setattr(sys.modules[module_name], "MedianTargetEncoder", MedianTargetEncoder)

# =============================================================================
# CONSTANTS
# =============================================================================
TOP_SKILLS = [
    "Information Technology", "Sales", "Management", "Manufacturing",
    "Engineering", "Health Care Provider", "Business Development", "Finance",
    "Accounting/Auditing", "Administrative", "Marketing", "Project Management",
    "Analyst", "Customer Service", "Operations", "Legal", "Research", "Design",
    "Education", "Consulting",
]

EXP_LEVELS = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
WORK_TYPES = ["Full-time", "Contract", "Part-time", "Internship", "Temporary", "Other"]
PAY_PERIODS = ["YEARLY", "HOURLY", "MONTHLY", "ONCE"]
COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-500", "501-1K", "1K-5K", "5K-10K", "10K+"]
SIZE_MAP_INV = {"1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4, "501-1K": 5, "1K-5K": 6, "5K-10K": 7, "10K+": 8}

US_STATES = ["CA", "NY", "TX", "WA", "MA", "IL", "FL", "VA", "GA", "CO", "NC", "NJ", "OH", "PA", "AZ", "MN", "MI", "OR", "MD", "CT", "Unknown"]
CITY_OPTIONS = [
    "San Francisco", "New York", "Seattle", "Boston", "Chicago", "Los Angeles",
    "Austin", "Dallas", "Atlanta", "Washington", "Denver", "Phoenix", "Remote",
    "Other / custom city",
]

STATE_CITY_MAP = {
    "CA": ["San Francisco", "Los Angeles", "San Diego", "San Jose", "Sacramento"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany"],
    "TX": ["Austin", "Dallas", "Houston", "San Antonio"],
    "WA": ["Seattle", "Bellevue", "Redmond", "Spokane"],
    "MA": ["Boston", "Cambridge", "Worcester"],
    "IL": ["Chicago", "Springfield"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville"],
    "VA": ["Arlington", "Richmond", "Alexandria"],
    "GA": ["Atlanta", "Savannah"],
    "CO": ["Denver", "Boulder"],
    "NC": ["Charlotte", "Raleigh", "Durham"],
    "NJ": ["Jersey City", "Newark"],
    "OH": ["Columbus", "Cleveland", "Cincinnati"],
    "PA": ["Philadelphia", "Pittsburgh"],
    "AZ": ["Phoenix", "Scottsdale"],
    "MN": ["Minneapolis", "Saint Paul"],
    "MI": ["Detroit", "Ann Arbor"],
    "OR": ["Portland"],
    "MD": ["Baltimore", "Bethesda"],
    "CT": ["Hartford", "Stamford"],
}

def city_options_for_state(state: str) -> List[str]:
    base = STATE_CITY_MAP.get(state, [])
    # Remote and custom are always available because some postings are remote
    # or have uncommon city names.
    return base + ["Remote", "Other / custom city"]
INDUSTRIES = [
    "Software Development", "IT Services and IT Consulting", "Financial Services",
    "Hospitals and Health Care", "Staffing and Recruiting",
    "Technology, Information and Internet", "Business Consulting and Services",
    "Retail", "Manufacturing", "Accounting", "Marketing Services",
    "Higher Education", "Construction", "Telecommunications",
]
BENEFIT_OPTIONS = [
    "Medical insurance", "Dental insurance", "Vision insurance", "401(k)",
    "Paid time off", "Parental leave", "Life insurance", "Disability insurance",
    "Tuition assistance", "Employee discount", "Commuter benefits",
    "Remote or flexible work support",
]
RESPONSIBILITY_OPTIONS = [
    "Build scalable systems", "Analyze data", "Manage projects", "Lead a team",
    "Work with customers", "Create product strategy", "Support operations",
    "Develop software", "Deliver clinical care", "Handle financial reporting",
    "Drive business development", "Research and experimentation",
]
FOCUS_OPTIONS = [
    "Technical", "Management", "Client-facing", "Operations", "Research",
    "Creative", "Healthcare", "Finance", "Sales", "People / HR",
]

MODEL1_METRICS = {"r2": 0.653, "rmse": 30118, "mae": 20033}
MODEL2_METRICS = {"r2": 0.711, "rmse": 27494, "mae": 17839}
MODEL3_METRICS = {"r2": 0.757, "rmse": 25225, "mae": 16036}

# =============================================================================
# GENERAL HELPERS
# =============================================================================
def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def box(text: str, kind: str = "soft"):
    kind = kind if kind in {"soft", "green", "amber", "red"} else "soft"
    st.markdown(f'<div class="{kind}-box">{text}</div>', unsafe_allow_html=True)


def money(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def clean_col_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def input_signature(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def choose_existing(*paths: Path):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def safe_read_csv_existing(path: Path, wanted_cols=None, low_memory=False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        if wanted_cols is None:
            return pd.read_csv(path, low_memory=low_memory)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        keep = [c for c in wanted_cols if c in header]
        if not keep:
            return pd.read_csv(path, low_memory=low_memory)
        return pd.read_csv(path, usecols=keep, low_memory=low_memory)
    except Exception:
        return pd.DataFrame()


def show_df(df: pd.DataFrame, msg="No data available."):
    if df is None or df.empty:
        st.info(msg)
    else:
        st.dataframe(df, width="stretch", hide_index=True)


def show_bar(df: pd.DataFrame, label_col: str, value_col: str, sort_desc=True, top_n=None):
    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        st.info("Chart data is not available.")
        return
    temp = df[[label_col, value_col]].copy().dropna()
    if sort_desc:
        temp = temp.sort_values(value_col, ascending=False)
    if top_n:
        temp = temp.head(top_n)
    st.bar_chart(temp.set_index(label_col)[value_col])


def friendly_error(context: str, exc: Exception | None = None):
    """Show a clean error without exposing local Windows paths."""
    st.error(f"{context}. Please check the required local files and try again.")
    if exc is not None:
        with st.expander("Technical hint", expanded=False):
            msg = str(exc)
            msg = re.sub(r"[A-Za-z]:\\[^\n\r]+", "[local path hidden]", msg)
            st.code(msg[:1200])


def chart_help(what: str, how: str, warning: str | None = None):
    with st.expander("How to read this", expanded=False):
        st.markdown(f"**What this shows** — {what}")
        st.markdown(f"**How to read it** — {how}")
        if warning:
            st.markdown(f"**Interpretation warning** — {warning}")


def metric_glossary(items: Dict[str, str], expanded=False):
    with st.expander("Metric glossary", expanded=expanded):
        for key, val in items.items():
            st.markdown(f"**{key}** — {val}")


def page_explainer(page_purpose: str, questions: List[str], key_metrics: List[str], how_to_use: str, limitations: str):
    """Compact page guide.

    Earlier versions used top-level tabs. That made the page look like the
    explanation changed, while the actual content stayed almost the same.
    This compact guide keeps the explanation available without dominating
    the page. The actual page body below now changes more clearly by view mode.
    """
    with st.expander("Page guide / interpretation notes", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Purpose**")
            st.write(page_purpose)
            st.markdown("**Decision questions**")
            for q in questions:
                st.markdown(f"- {q}")
        with col2:
            st.markdown("**Key metrics**")
            for m in key_metrics:
                st.markdown(f"- {m}")
            st.markdown("**How to use**")
            st.write(how_to_use)
            st.markdown("**Limitations**")
            st.write(limitations)


def use_this_insight(points: List[str]):
    box("<b>Use this insight</b><br>" + "<br>".join([f"• {p}" for p in points]), "green")


def make_title_family(title: str) -> str:
    t = str(title or "").lower()
    rules = [
        (r"product", "product"),
        (r"data|machine learning|ai|analyst|scientist", "data_ai"),
        (r"sales|account executive|business development|sdr", "sales"),
        (r"manager|director|chief|vp|head of|leadership", "management"),
        (r"software|engineer|developer|sre|backend|frontend|full stack|platform", "software_engineering"),
        (r"nurse|doctor|clinical|medical|pharmacy|therap", "healthcare"),
        (r"marketing|brand|seo|content|growth", "marketing"),
        (r"consultant|advisor|strategy", "consulting_analyst"),
        (r"finance|account|audit|controller|treasury", "finance"),
        (r"security|cyber|infosec", "cybersecurity"),
        (r"human resources|recruiter|talent|people", "hr_recruiting"),
    ]
    for pattern, label in rules:
        if re.search(pattern, t):
            return label
    return "other"


def clean_salary_text_for_dashboard(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?[kK])?", " ", s)
    s = re.sub(r"\b\d+[\d,]*(?:\.\d+)?\s?(?:USD|usd|dollars?)\b", " ", s)
    s = re.sub(
        r"\b(?:salary range|compensation range|pay range|base salary|annual salary|hourly rate|expected salary|salary|compensation|pay rate|bonus eligible|usd|dollars|per year|yearly|annually)\b",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", s).strip()


# =============================================================================
# DATA
# =============================================================================
@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    wanted = [
        "job_id", "title", "location", "remote_allowed", "normalized_salary", "currency",
        "formatted_experience_level", "formatted_work_type", "pay_period",
        "views", "applies", "description", "company_id",
    ]
    post = safe_read_csv_existing(DATA / "postings.csv", wanted_cols=wanted, low_memory=False)
    if post.empty:
        return post

    defaults = {
        "title": "", "location": "Unknown", "remote_allowed": np.nan,
        "normalized_salary": np.nan, "currency": "USD",
        "formatted_experience_level": "Unknown", "formatted_work_type": "Unknown",
        "pay_period": "Unknown", "views": np.nan, "applies": np.nan,
        "description": "", "company_id": np.nan,
    }
    for col, default in defaults.items():
        if col not in post.columns:
            post[col] = default

    post["formatted_experience_level"] = post["formatted_experience_level"].fillna("Unknown")
    post["formatted_work_type"] = post["formatted_work_type"].fillna("Unknown")
    post["remote_flag"] = (post["remote_allowed"] == 1).astype(int)
    post["state"] = post["location"].astype(str).str.extract(r",\s*([A-Z]{2})(?:\s|$)")
    post["state"] = post["state"].fillna("Unknown")
    post["city"] = post["location"].astype(str).str.extract(r"^([^,]+)")
    post["city"] = post["city"].fillna("Unknown").str.strip()
    post["title_family"] = post["title"].fillna("").apply(make_title_family)

    post["primary_industry"] = "Unknown"
    job_ind = safe_read_csv_existing(DATA / "job_industries.csv", low_memory=False)
    ind = safe_read_csv_existing(DATA / "industries.csv", low_memory=False)
    if not job_ind.empty and not ind.empty and "job_id" in job_ind.columns:
        ind_id_col = "industry_id" if "industry_id" in job_ind.columns and "industry_id" in ind.columns else None
        name_col = None
        for c in ["industry_name", "name", "industry"]:
            if c in ind.columns:
                name_col = c
                break
        if ind_id_col and name_col:
            merged = job_ind[["job_id", ind_id_col]].merge(ind[[ind_id_col, name_col]], on=ind_id_col, how="left")
            primary = merged.dropna(subset=[name_col]).groupby("job_id")[name_col].first()
            post["primary_industry"] = post["job_id"].map(primary).fillna("Unknown")

    post["salary_usable"] = (
        post["normalized_salary"].between(10_000, 300_000)
        & (post["currency"].isin(["USD"]) | post["currency"].isna())
    )
    post["competition_score"] = np.where(
        post["views"].fillna(0) > 0,
        post["applies"].fillna(0) / post["views"].fillna(0),
        np.nan,
    )
    return post


@st.cache_data(show_spinner=False)
def load_postings_summary() -> Dict[str, float]:
    df = load_market_data()
    if df.empty:
        return {"total_postings": 123_849, "salary_count": 35_279, "median_salary": 81_734, "remote_pct": 12.3}
    sal = df[df["salary_usable"]]
    return {
        "total_postings": int(len(df)),
        "salary_count": int(len(sal)),
        "median_salary": int(sal["normalized_salary"].median()) if len(sal) else 0,
        "remote_pct": round((df["remote_flag"] == 1).mean() * 100, 1),
    }


@st.cache_data(show_spinner=False)
def data_quality_summary() -> pd.DataFrame:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame([{"Metric": "Dataset availability", "Count": 0, "Coverage": "Not found"}])
    total = len(df)
    rows = [
        {"Metric": "Total job postings", "Count": total, "Coverage": "100.0%"},
        {"Metric": "Usable salary rows", "Count": int(df["salary_usable"].sum()), "Coverage": f"{df['salary_usable'].mean()*100:.1f}%"},
        {"Metric": "Rows with description", "Count": int(df["description"].notna().sum()), "Coverage": f"{df['description'].notna().mean()*100:.1f}%"},
        {"Metric": "Rows with applications", "Count": int(df["applies"].notna().sum()), "Coverage": f"{df['applies'].notna().mean()*100:.1f}%"},
        {"Metric": "Rows with views", "Count": int(df["views"].notna().sum()), "Coverage": f"{df['views'].notna().mean()*100:.1f}%"},
        {"Metric": "Explicit remote postings", "Count": int((df["remote_flag"] == 1).sum()), "Coverage": f"{(df['remote_flag'] == 1).mean()*100:.1f}%"},
    ]
    ben = safe_read_csv_existing(DATA / "benefits.csv", low_memory=False)
    if not ben.empty and "job_id" in ben.columns:
        rows.append({
            "Metric": "Jobs with at least one benefit record",
            "Count": int(ben["job_id"].nunique()),
            "Coverage": f"{ben['job_id'].nunique()/max(total,1)*100:.1f}%",
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_job_skills_merged() -> pd.DataFrame:
    """Load job-skill mappings robustly across slightly different Kaggle schemas.

    Fixes the earlier KeyError: 'skill_name' caused by pandas suffixing
    skill_name_x / skill_name_y after merging job_skills with skills.
    """
    js = safe_read_csv_existing(DATA / "job_skills.csv", low_memory=False)
    sk = safe_read_csv_existing(DATA / "skills.csv", low_memory=False)
    if js.empty:
        return pd.DataFrame()

    js = js.copy()
    key_col = None
    if "skill_abr" in js.columns:
        key_col = "skill_abr"
    elif "skill_id" in js.columns:
        key_col = "skill_id"

    # Start with a fallback skill_name from the mapping table itself.
    if "skill_name" not in js.columns:
        if key_col:
            js["skill_name"] = js[key_col].astype(str)
        else:
            js["skill_name"] = "Unknown skill"

    if not sk.empty and key_col and key_col in sk.columns:
        name_col = next((c for c in ["skill_name", "name", "skill_abr"] if c in sk.columns), None)
        if name_col:
            base = js.drop(columns=["skill_name"], errors="ignore")
            lookup = sk[[key_col, name_col]].drop_duplicates(subset=[key_col]).copy()
            lookup = lookup.rename(columns={name_col: "_skill_label"})
            merged = base.merge(lookup, on=key_col, how="left")
            merged["skill_name"] = merged["_skill_label"].fillna(merged[key_col].astype(str))
            merged = merged.drop(columns=["_skill_label"], errors="ignore")
            return merged

    return js


@st.cache_data(show_spinner=False)
def compute_skill_demand(top_n=20) -> pd.DataFrame:
    merged = load_job_skills_merged()
    if merged.empty or "skill_name" not in merged.columns:
        return pd.DataFrame()
    return merged.groupby("skill_name").size().sort_values(ascending=False).head(top_n).reset_index(name="posting_count")


@st.cache_data(show_spinner=False)
def compute_skill_salary_premium(top_n=15) -> pd.DataFrame:
    market = load_market_data()
    skills = load_job_skills_merged()
    if market.empty or skills.empty or "job_id" not in market.columns or "job_id" not in skills.columns or "skill_name" not in skills.columns:
        return pd.DataFrame()
    sal = market.loc[market["salary_usable"], ["job_id", "normalized_salary"]].copy()
    merged = skills[["job_id", "skill_name"]].merge(sal, on="job_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    overall = sal["normalized_salary"].median()
    stats = merged.groupby("skill_name")["normalized_salary"].agg(["median", "count"]).reset_index()
    stats = stats[stats["count"] >= 100].copy()
    stats["premium_vs_overall"] = stats["median"] - overall
    stats = stats.sort_values("premium_vs_overall", ascending=False).head(top_n)
    return stats.rename(columns={"median": "median_salary"})


@st.cache_data(show_spinner=False)
def compute_benefit_summary() -> pd.DataFrame:
    market = load_market_data()
    ben = safe_read_csv_existing(DATA / "benefits.csv", low_memory=False)
    if market.empty or ben.empty or "job_id" not in ben.columns:
        return pd.DataFrame()
    counts = ben.groupby("job_id").size().rename("benefit_count")
    out = market[["job_id", "normalized_salary", "salary_usable"]].merge(counts, on="job_id", how="left")
    out["benefit_count"] = out["benefit_count"].fillna(0)
    out = out[out["salary_usable"]].copy()
    out["benefit_bucket"] = pd.cut(out["benefit_count"], bins=[-1, 0, 2, 4, 8, 100], labels=["0", "1-2", "3-4", "5-8", "9+"])
    return out.groupby("benefit_bucket")["normalized_salary"].agg(["median", "count"]).reset_index().rename(columns={"median": "median_salary"})


@st.cache_data(show_spinner=False)
def compute_industry_stats(min_count=50) -> pd.DataFrame:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame()
    base = df[df["salary_usable"]].copy()
    grp = base.groupby("primary_industry").agg(
        postings=("job_id", "count"),
        median_salary=("normalized_salary", "median"),
        remote_rate=("remote_flag", "mean"),
        mean_competition=("competition_score", "mean"),
    ).reset_index()
    grp = grp[grp["postings"] >= min_count].sort_values("postings", ascending=False)
    grp["remote_rate"] = (grp["remote_rate"] * 100).round(1)
    grp["mean_competition"] = grp["mean_competition"].round(3)
    return grp


@st.cache_data(show_spinner=False)
def compute_state_stats(min_count=50) -> pd.DataFrame:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame()
    base = df[df["salary_usable"]].copy()
    grp = base.groupby("state").agg(
        postings=("job_id", "count"),
        median_salary=("normalized_salary", "median"),
        remote_rate=("remote_flag", "mean"),
    ).reset_index()
    grp = grp[grp["postings"] >= min_count].sort_values("median_salary", ascending=False)
    grp["remote_rate"] = (grp["remote_rate"] * 100).round(1)
    return grp


@st.cache_data(show_spinner=False)
def compute_title_family_stats(min_count=50) -> pd.DataFrame:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame()
    base = df[df["salary_usable"]].copy()
    grp = base.groupby("title_family").agg(
        postings=("job_id", "count"),
        median_salary=("normalized_salary", "median"),
        avg_views=("views", "mean"),
        avg_applies=("applies", "mean"),
        competition=("competition_score", "mean"),
    ).reset_index()
    return grp[grp["postings"] >= min_count].sort_values("median_salary", ascending=False)


@st.cache_data(show_spinner=False)
def compute_career_ladder() -> pd.DataFrame:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame()
    base = df[df["salary_usable"]].copy()
    out = base.groupby("formatted_experience_level")["normalized_salary"].agg(["median", "count"]).reset_index()
    order = {v: i for i, v in enumerate(EXP_LEVELS)}
    out["order"] = out["formatted_experience_level"].map(order).fillna(999)
    return out.sort_values("order").drop(columns="order").rename(columns={"median": "median_salary"})


@st.cache_data(show_spinner=False)
def compute_remote_summary() -> pd.DataFrame:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame()
    base = df[df["salary_usable"]].copy()
    base["remote_label"] = np.where(base["remote_flag"] == 1, "Remote", "Not explicitly remote")
    return base.groupby("remote_label")["normalized_salary"].agg(["median", "count"]).reset_index().rename(columns={"median": "median_salary"})


# =============================================================================
# MODEL 03 + MODEL 04
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_salary_model():
    candidates = [
        MODELS / "best_salary_model_03_advanced_progress.joblib",
        MODELS / "best_salary_model_03_advanced.joblib",
        MODELS / "best_salary_pipeline.joblib",
        BASE / "best_salary_model_03_advanced_progress.joblib",
        BASE / "best_salary_model_03_advanced.joblib",
        BASE / "best_salary_pipeline.joblib",
    ]
    last_error = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            obj = joblib.load(path)
            feature_cols = obj.get("feature_cols", []) if isinstance(obj, dict) else []
            model_type = obj.get("type", "single_model") if isinstance(obj, dict) else "single_model"
            metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
            return obj, feature_cols, path.name, model_type, metrics, None
        except Exception as exc:
            last_error = str(exc)
    return None, [], None, None, {}, last_error


def predict_salary_from_model(obj, X_pred: pd.DataFrame) -> float:
    if isinstance(obj, dict) and obj.get("type") == "ensemble":
        weights = obj.get("weights", {})
        pipelines = obj.get("pipelines", {})
        pred_log = 0.0
        used_weight = 0.0
        for name, pipe in pipelines.items():
            w = float(weights.get(name, 0.0))
            if w == 0:
                continue
            pred_log += w * float(pipe.predict(X_pred)[0])
            used_weight += w
        if used_weight == 0:
            raise ValueError("Ensemble weights could not be read.")
        return pred_log / used_weight
    if isinstance(obj, dict) and obj.get("pipeline") is not None:
        return float(obj["pipeline"].predict(X_pred)[0])
    return float(obj.predict(X_pred)[0])


@st.cache_data(show_spinner=False)
def load_model4_bundle() -> Dict[str, object]:
    bundle = {}
    json_path = choose_existing(OUT / "model_04_metrics.json", BASE / "model_04_metrics.json")
    try:
        bundle["metrics_json"] = json.loads(json_path.read_text(encoding="utf-8")) if json_path else {}
    except Exception:
        bundle["metrics_json"] = {}

    files = {
        "comparison": "model_04_comparison_metrics.csv",
        "intervals": "model_04_interval_method_scores.csv",
        "coefficients": "model_04_ridge_coefficients.csv",
        "error_experience": "model_04_error_by_experience.csv",
        "error_industry": "model_04_error_by_industry.csv",
        "error_state": "model_04_error_by_state.csv",
        "error_salary_band": "model_04_error_by_salary_band.csv",
        "error_title_family": "model_04_error_by_title_family.csv",
        "top_error_titles": "model_04_top_error_titles.csv",
        "top_overpredicted": "model_04_top_overpredicted.csv",
        "top_underpredicted": "model_04_top_underpredicted.csv",
        "error_remote": "model_04_error_by_remote.csv",
        "error_pay_period": "model_04_error_by_pay_period.csv",
    }
    for key, filename in files.items():
        path = choose_existing(OUT / filename, BASE / filename)
        try:
            bundle[key] = pd.read_csv(path) if path else pd.DataFrame()
        except Exception:
            bundle[key] = pd.DataFrame()
    return bundle


# =============================================================================
# PREDICTION HELPERS
# =============================================================================
def build_role_context(responsibilities: List[str], focuses: List[str], custom_text: str) -> str:
    if custom_text and custom_text.strip():
        return custom_text.strip()
    parts = []
    if responsibilities:
        parts.append("Responsible for " + ", ".join([r.lower() for r in responsibilities[:4]]) + ".")
    if focuses:
        parts.append("Primary focus: " + ", ".join([f.lower() for f in focuses[:3]]) + ".")
    if not parts:
        parts.append("Collaborate with cross-functional teams and support role-specific responsibilities.")
    return " ".join(parts)


def build_prediction_row(
    feature_cols, title, description, exp_level, work_type, industry, pay_period, state, city,
    company_size, is_remote, skills_sel, selected_benefits, views, applies, employee_count,
    follower_count, company_job_count
) -> pd.DataFrame:
    row = {c: 0 for c in feature_cols}
    title_text = str(title or "").strip()
    desc_text = clean_salary_text_for_dashboard(description)
    title_lower = title_text.lower()
    title_family = make_title_family(title_lower)
    remote_flag = int(is_remote)
    title_cluster = "Unknown"
    city_value = str(city or "Unknown").strip() or "Unknown"
    state_value = str(state or "Unknown").strip() or "Unknown"
    pay_value = str(pay_period or "YEARLY").strip() or "YEARLY"
    company_size_code = str(SIZE_MAP_INV.get(company_size, 5))

    updates = {
        "formatted_experience_level": exp_level,
        "formatted_work_type": work_type,
        "primary_industry": industry,
        "pay_period": pay_value,
        "company_size": company_size_code,
        "state_final": state_value,
        "city_loc": city_value,
        "company_id": "dashboard_demo_company",
        "title_cluster": title_cluster,
        "title_family": title_family,
        "city_industry": f"{city_value}__{industry}",
        "state_industry": f"{state_value}__{industry}",
        "exp_industry": f"{exp_level}__{industry}",
        "exp_title_cluster": f"{exp_level}__{title_cluster}",
        "remote_industry": f"{remote_flag}__{industry}",
        "pay_work_type": f"{pay_value}__{work_type}",
        "title_text": title_text,
        "description_clean": desc_text,
        "views": float(views),
        "applies": float(applies),
        "employee_count": float(employee_count),
        "follower_count": float(follower_count),
        "company_job_count": float(company_job_count),
        "description_len": len(desc_text),
        "description_word_count": len(desc_text.split()),
        "n_skills": len(skills_sel),
        "benefit_count": len(selected_benefits),
        "company_industry_count": 1,
        "company_speciality_count": 0,
        "follower_per_employee": float(follower_count) / (float(employee_count) + 1.0),
        "title_len": len(title_text),
        "title_word_count": len(title_text.split()),
        "is_hourly": int(pay_value.upper() == "HOURLY"),
        "remote_flag": remote_flag,
        "has_applies": int(float(applies) > 0),
        "has_views": int(float(views) > 0),
        "has_employee_count": int(float(employee_count) > 0),
        "has_follower_count": int(float(follower_count) > 0),
        "title_senior": int(bool(re.search(r"senior|sr\.?\b|lead|staff", title_lower))),
        "title_principal": int("principal" in title_lower),
        "title_director": int(bool(re.search(r"director|head of", title_lower))),
        "title_vp": int(bool(re.search(r"\bvp\b|vice pres", title_lower))),
        "title_chief": int(bool(re.search(r"\bchief\b|president|ceo|cto|cfo|coo", title_lower))),
        "title_manager": int(bool(re.search(r"manager|mgr", title_lower))),
        "title_junior": int(bool(re.search(r"junior|jr\.?\b|entry|intern", title_lower))),
        "title_associate": int("associate" in title_lower),
        "title_engineer": int(bool(re.search(r"engineer|developer|software|swe|sde", title_lower))),
        "title_architect": int("architect" in title_lower),
        "title_data": int(bool(re.search(r"data|analyst|scientist|ml|ai|machine", title_lower))),
        "title_sales": int(bool(re.search(r"sales|account exec|business dev", title_lower))),
        "title_consultant": int(bool(re.search(r"consultant|advisor", title_lower))),
        "title_nurse": int(bool(re.search(r"nurse|rn\b|lpn", title_lower))),
        "title_product": int("product" in title_lower),
        "title_finance": int(bool(re.search(r"finance|accountant|controller|accounting|auditor", title_lower))),
        "title_security": int(bool(re.search(r"security|cyber|infosec", title_lower))),
        "title_marketing": int(bool(re.search(r"marketing|brand|seo|content", title_lower))),
        "title_hr": int(bool(re.search(r"human resources|recruiter|talent|people partner", title_lower))),
    }

    for col, val in updates.items():
        if col in row:
            row[col] = val

    for skill in skills_sel:
        col_name = f"skill_{clean_col_name(skill)}"
        if col_name in row:
            row[col_name] = 1

    for benefit in selected_benefits:
        clean = clean_col_name(benefit)
        candidates = [f"benefit_{clean}", f"benefit_type_{clean}", f"ben_{clean}", clean]
        for col_name in candidates:
            if col_name in row:
                row[col_name] = 1
        for existing_col in row.keys():
            if existing_col.startswith(("benefit_", "ben_")) and clean in existing_col:
                row[existing_col] = 1

    return pd.DataFrame([row], columns=feature_cols)


def benefits_package_strength(selected_benefits: List[str]) -> Tuple[str, str]:
    high_value = [
        "Medical insurance", "Dental insurance", "Vision insurance", "401(k)",
        "Paid time off", "Parental leave", "Tuition assistance",
    ]
    high_count = sum(1 for b in (selected_benefits or []) if b in high_value)
    if len(selected_benefits or []) >= 6 or high_count >= 4:
        return "Strong package", "green"
    if len(selected_benefits or []) >= 3 or high_count >= 2:
        return "Standard package", "amber"
    return "Basic package", "red"


def prediction_explanation(payload: Dict[str, object]) -> Tuple[List[str], List[str]]:
    positives, cautions = [], []
    title = str(payload.get("job_title", "")).lower()
    exp = payload.get("exp_level", "")
    state = payload.get("state", "")
    city = payload.get("city", "")
    remote = payload.get("is_remote", False)
    skills = payload.get("skills_sel", [])
    benefits = payload.get("selected_benefits", [])
    company_size = payload.get("company_size", "")

    if exp in ["Mid-Senior level", "Director", "Executive"]:
        positives.append("Higher seniority usually increases salary expectations.")
    elif exp in ["Entry level", "Associate"]:
        cautions.append("Lower seniority usually limits the expected salary range.")
    if re.search(r"senior|principal|staff|lead|director|vp|chief", title):
        positives.append("The title contains seniority or leadership language.")
    if re.search(r"software|engineer|data|machine learning|ai|security|cloud", title):
        positives.append("The title belongs to a technical job family that often pays above the overall median.")
    if state in ["CA", "NY", "WA", "MA"]:
        positives.append(f"{state} is associated with higher salary postings in this dataset.")
    elif state not in ["Unknown", "CA", "NY", "WA", "MA"]:
        cautions.append("Location can move the estimate up or down depending on local salary patterns.")
    if remote:
        positives.append("Remote roles are sometimes linked to higher compensation in this dataset, but this is correlational.")
    if len(skills) >= 3:
        positives.append("Multiple selected skills create a stronger skill-profile signal.")
    elif len(skills) == 0:
        cautions.append("No selected skills means the model receives less skill information.")
    if company_size in ["1K-5K", "5K-10K", "10K+"]:
        positives.append("Larger company size often signals a more structured compensation package.")
    if len(benefits) >= 5:
        positives.append("A stronger benefit package can indicate a higher-quality employer package signal.")
    elif len(benefits) <= 1:
        cautions.append("Few listed benefits may indicate a weaker employer package signal.")
    if city == "Other / custom city":
        cautions.append("Custom cities may map to less familiar categories and reduce certainty.")
    return positives[:5], cautions[:5]


def similar_postings_analysis(payload: Dict[str, object]) -> Tuple[pd.DataFrame, str]:
    df = load_market_data()
    if df.empty:
        return pd.DataFrame(), "No data loaded"
    salary_df = df[df["salary_usable"]].copy()
    if salary_df.empty:
        return pd.DataFrame(), "No usable salary rows"

    target_family = make_title_family(payload.get("job_title", ""))
    exp = payload.get("exp_level", "")
    state = payload.get("state", "")
    industry = str(payload.get("industry", "")).lower()
    remote_flag = 1 if payload.get("is_remote", False) else 0

    rules = [
        ("same experience + state + industry + title family + remote", lambda d: d[(d["formatted_experience_level"] == exp) & (d["state"] == state) & (d["primary_industry"].astype(str).str.lower().str.contains(re.escape(industry), na=False)) & (d["title_family"] == target_family) & (d["remote_flag"] == remote_flag)]),
        ("same experience + state + title family", lambda d: d[(d["formatted_experience_level"] == exp) & (d["state"] == state) & (d["title_family"] == target_family)]),
        ("same experience + industry + title family", lambda d: d[(d["formatted_experience_level"] == exp) & (d["primary_industry"].astype(str).str.lower().str.contains(re.escape(industry), na=False)) & (d["title_family"] == target_family)]),
        ("same experience + title family", lambda d: d[(d["formatted_experience_level"] == exp) & (d["title_family"] == target_family)]),
        ("same experience + state", lambda d: d[(d["formatted_experience_level"] == exp) & (d["state"] == state)]),
        ("same experience only", lambda d: d[d["formatted_experience_level"] == exp]),
        ("all salary postings", lambda d: d),
    ]

    for label, fn in rules:
        subset = fn(salary_df)
        if len(subset) >= 30:
            return subset, label
    return salary_df, "all salary postings"


def reliability_signal(pred_salary: float, similar_df: pd.DataFrame, payload: Dict[str, object]) -> Tuple[str, str, List[str]]:
    reasons = []
    score = 0

    if len(similar_df) >= 150:
        score += 2
        reasons.append("large similar-posting reference set")
    elif len(similar_df) >= 50:
        score += 1
        reasons.append("reasonable similar-posting reference set")
    else:
        reasons.append("small similar-posting reference set")

    if not similar_df.empty:
        q25 = float(similar_df["normalized_salary"].quantile(0.25))
        q75 = float(similar_df["normalized_salary"].quantile(0.75))
        width = q75 - q25
        if width <= 35_000:
            score += 2
            reasons.append("similar postings have a fairly tight middle range")
        elif width <= 60_000:
            score += 1
            reasons.append("similar postings have a moderate middle range")
        else:
            reasons.append("similar postings show a wide market range")

        med = float(similar_df["normalized_salary"].median())
        if abs(pred_salary - med) <= max(10_000, med * 0.12):
            score += 1
            reasons.append("prediction is close to the similar-postings median")
        else:
            reasons.append("prediction is relatively far from the similar-postings median")

    if payload.get("city") != "Other / custom city":
        score += 1
    if payload.get("exp_level") not in ["Entry level", "Associate"]:
        score += 1

    if score >= 6:
        return "High", "green", reasons
    if score >= 4:
        return "Medium", "amber", reasons
    return "Low", "red", reasons


def scenario_variants(payload: Dict[str, object]) -> List[Tuple[str, Dict[str, object]]]:
    base = dict(payload)
    scenarios = [("Current scenario", base)]

    senior_map = {
        "Entry level": "Associate",
        "Associate": "Mid-Senior level",
        "Mid-Senior level": "Director",
        "Director": "Executive",
        "Executive": "Executive",
    }
    s2 = dict(base)
    s2["exp_level"] = senior_map.get(base.get("exp_level"), base.get("exp_level"))
    scenarios.append(("One-step higher seniority", s2))

    s3 = dict(base)
    skills = list(s3.get("skills_sel", []))
    for item in ["Information Technology", "Project Management", "Analyst"]:
        if item not in skills and len(skills) < 8:
            skills.append(item)
    s3["skills_sel"] = skills
    benefits = list(s3.get("selected_benefits", []))
    for item in ["Medical insurance", "401(k)", "Paid time off"]:
        if item not in benefits:
            benefits.append(item)
    s3["selected_benefits"] = benefits[:8]
    scenarios.append(("Stronger profile / package", s3))

    s4 = dict(base)
    s4["is_remote"] = True
    scenarios.append(("Remote version", s4))
    return scenarios


def download_report_button(result: Dict[str, object], similar_df: pd.DataFrame, reliability: str):
    payload = result["payload"]
    q25 = q50 = q75 = None
    if similar_df is not None and not similar_df.empty:
        q25 = float(similar_df["normalized_salary"].quantile(0.25))
        q50 = float(similar_df["normalized_salary"].quantile(0.50))
        q75 = float(similar_df["normalized_salary"].quantile(0.75))

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": result.get("model_name"),
        "input_signature": result.get("signature"),
        "prediction": round(result.get("salary", 0), 2),
        "practical_range": [round(result.get("practical_low", 0), 2), round(result.get("practical_high", 0), 2)],
        "expected_zone_mae": [round(result.get("mae_low", 0), 2), round(result.get("mae_high", 0), 2)],
        "broad_zone_rmse": [round(result.get("rmse_low", 0), 2), round(result.get("rmse_high", 0), 2)],
        "similar_postings_p25_p50_p75": [q25, q50, q75],
        "reliability": reliability,
        "payload": payload,
    }
    st.download_button(
        "Download salary estimate report (.json)",
        data=json.dumps(report, indent=2),
        file_name="salary_estimate_report.json",
        mime="application/json",
    )


# =============================================================================
# SIDEBAR
# =============================================================================
summary = load_postings_summary()

with st.sidebar:
    st.markdown("## 💼 LinkedIn Jobs")
    st.markdown("*Kaggle 2024 — job postings analysis*")
    st.caption(f"{summary['total_postings']:,} postings · {summary['salary_count']:,} salary-usable rows")
    st.markdown("---")

    page = st.radio(
        "Choose a perspective",
        [
            "🏠 Home",
            "👤 Job Seeker",
            "🏢 HR / Recruiting",
            "🎓 Education / Curriculum Planner",
            "📈 Investor / Market Analyst",
            "🏛️ Policy Maker / Labor Market Analyst",
            "🔬 Researcher / ML Evaluation",
        ],
    )

    view_mode = st.radio("View mode", ["Basic", "Advanced"], index=0)
    BASIC = view_mode == "Basic"
    ADVANCED = view_mode == "Advanced"
    # Research is now a separate page, not a third global view mode.
    RESEARCH = page == "🔬 Researcher / ML Evaluation"

    st.markdown("---")
    st.caption("Models: XGBoost · LightGBM · CatBoost · Model 04 Ridge Stacking")
    st.caption("Best OOF R² ≈ 0.759 (Model 04)")
    if view_mode == "Basic":
        st.caption("Basic mode hides most raw tables and technical detail.")
    else:
        st.caption("Advanced mode shows tables, error analysis, and interpretation detail.")


# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("LinkedIn Job Analysis Dashboard")
    st.caption("A multi-perspective dashboard for labor-market analysis, salary prediction, and ML evaluation.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total postings", f"{summary['total_postings']:,}")
    c2.metric("Salary-usable rows", f"{summary['salary_count']:,}")
    c3.metric("Median annual salary", money(summary["median_salary"]))
    c4.metric("Explicit remote rate", f"{summary['remote_pct']:.1f}%")

    box(
        "<b>What this dashboard is</b><br>"
        "This is a decision-support dashboard, not only a chart gallery. "
        "Each page is organized around a user perspective and the selected view mode controls technical depth.",
        "soft",
    )

    box(
        "<b>Local-first usage</b><br>"
        "This version is designed to run locally after downloading the project files. "
        "It does not require Streamlit Cloud, .toml configuration, or .bat launcher files.",
        "green",
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="persona-card"><b>Basic mode</b><br>Short explanations, fewer technical details, and decision-oriented reading.</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="persona-card"><b>Advanced mode</b><br>Market ranges, comparison tables, and more detailed analysis.</div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="persona-card"><b>Researcher page</b><br>OOF metrics, intervals, diagnostics, and segment error analysis are collected in the dedicated Researcher / ML Evaluation page.</div>', unsafe_allow_html=True)

    section("Dataset quality snapshot")
    show_df(data_quality_summary())

    if ADVANCED:
        section("Quick market snapshots")
        col1, col2 = st.columns(2)
        with col1:
            industry_stats = compute_industry_stats(min_count=100).head(10)
            st.markdown("**Top industries by posting volume**")
            show_bar(industry_stats, "primary_industry", "postings", top_n=10)
        with col2:
            state_stats = compute_state_stats(min_count=100).head(10)
            st.markdown("**Highest-median-salary states**")
            show_bar(state_stats, "state", "median_salary", top_n=10)

    box(
        "<b>Important limitation</b><br>"
        "The dataset contains job postings, not accepted offers and not records of who was hired. "
        "The dashboard estimates salary patterns and market signals; it cannot guarantee salary offers or hiring probability.",
        "amber",
    )


# =============================================================================
# JOB SEEKER
# =============================================================================
elif page == "👤 Job Seeker":
    st.title("Job Seeker")
    st.caption("Salary estimation, scenario comparison, similar postings, and decision-support context.")

    page_explainer(
        page_purpose="This page helps individual job seekers estimate salary and compare the result against similar postings.",
        questions=[
            "What annual salary does the model estimate for this role?",
            "What market range do similar postings suggest?",
            "How reliable is the estimate?",
            "How does the estimate change under alternative scenarios?",
        ],
        key_metrics=[
            "Predicted salary", "Practical range", "Expected zone (±MAE)",
            "Similar postings P25/P50/P75", "Reliability signal",
        ],
        how_to_use="Use the model estimate as a starting point. For decisions, rely more on similar-postings P25–P75 and the reliability signal.",
        limitations="The prediction is based on posting patterns. It is not a guaranteed offer and does not estimate hiring probability.",
    )

    metric_glossary(
        {
            "Predicted salary": "The model's point estimate for annual salary.",
            "Practical range": "A simple ±15% reading band to avoid over-reading a single number.",
            "Expected zone": "Prediction ± average absolute model error.",
            "Broad zone": "Prediction ± RMSE. This is wider and more conservative.",
            "Similar postings": "Real salary-usable postings selected with progressive matching rules.",
            "Reliability signal": "A simple support signal based on similar-posting count and range stability.",
        },
        expanded=BASIC,
    )

    obj, feature_cols, model_name, model_type, model_metrics, load_error = load_salary_model()
    model4 = load_model4_bundle()
    m4json = model4.get("metrics_json", {}) or {}
    mae_model4 = float(m4json.get("model04_ridge_oof", {}).get("mae", MODEL3_METRICS["mae"]))
    rmse_model4 = float(m4json.get("model04_ridge_oof", {}).get("rmse", MODEL3_METRICS["rmse"]))

    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        section("Input form")
        with st.form("salary_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                job_title = st.text_input("Job title", value="Software Engineer")
                exp_level = st.selectbox("Experience level", EXP_LEVELS, index=2)
                work_type = st.selectbox("Work type", WORK_TYPES, index=0)
                pay_period = st.selectbox("Pay period", PAY_PERIODS, index=0)
                industry = st.selectbox("Primary industry", INDUSTRIES, index=0)

            with col_b:
                state = st.selectbox(
                    "State / region",
                    US_STATES,
                    index=0,
                    help="State is the broader regional salary signal. It is usually more stable than city.",
                )
                city_choices = city_options_for_state(state)
                city_choice = st.selectbox(
                    "City / metro area",
                    city_choices,
                    index=0,
                    help="City is a more local signal inside the selected state. Choose custom only if the city is not listed.",
                )
                if city_choice == "Other / custom city":
                    custom_city = st.text_input("Custom city", value="", placeholder="Type the city name here")
                    city = custom_city.strip() if custom_city.strip() else "Unknown"
                    if city == "Unknown":
                        st.caption("Custom city is empty, so the model will use Unknown as the city signal.")
                else:
                    st.text_input("Custom city", value="", disabled=True, help="Enabled only when City / metro area is set to Other / custom city.")
                    city = city_choice
                company_size = st.selectbox("Company size", COMPANY_SIZES, index=5)
                is_remote = st.toggle("Remote role", value=False)

            st.markdown("**Skills and benefits**")
            s1, s2 = st.columns(2)
            with s1:
                skills_sel = st.multiselect("Relevant skills", TOP_SKILLS, default=["Information Technology", "Engineering"])
            with s2:
                selected_benefits = st.multiselect("Benefits offered by employer", BENEFIT_OPTIONS, default=["Medical insurance", "Paid time off"])
                st.caption(f"Selected benefit count: {len(selected_benefits)}")

            st.markdown("**Role context builder**")
            rb1, rb2 = st.columns(2)
            with rb1:
                responsibilities = st.multiselect("Main responsibilities", RESPONSIBILITY_OPTIONS, default=["Build scalable systems"])
            with rb2:
                focuses = st.multiselect("Role focus", FOCUS_OPTIONS, default=["Technical"])

            with st.expander("Optional: paste a real job description instead"):
                custom_description = st.text_area(
                    "Actual job description (optional)",
                    height=120,
                    placeholder="Paste a real job description if you have one. Otherwise the app builds a short role context automatically.",
                )

            with st.expander("Optional model signal inputs"):
                v1, v2, v3 = st.columns(3)
                with v1:
                    views = st.number_input("Estimated views", min_value=0, value=500, step=50)
                    applies = st.number_input("Estimated applications", min_value=0, value=30, step=5)
                with v2:
                    employee_count = st.number_input("Estimated employee count", min_value=0, value=500, step=50)
                    follower_count = st.number_input("Estimated company followers", min_value=0, value=5000, step=100)
                with v3:
                    company_job_count = st.number_input("Estimated active jobs from this company", min_value=0, value=5, step=1)

            generated_description = build_role_context(responsibilities, focuses, custom_description)
            st.caption("Generated role context used by the model")
            st.code(generated_description, language=None)

            submit = st.form_submit_button("🔮 Predict Salary")

        if submit:
            payload = {
                "job_title": job_title,
                "exp_level": exp_level,
                "work_type": work_type,
                "pay_period": pay_period,
                "industry": industry,
                "city": city,
                "state": state,
                "company_size": company_size,
                "is_remote": is_remote,
                "selected_benefits": selected_benefits,
                "benefit_count": len(selected_benefits),
                "skills_sel": skills_sel,
                "description": generated_description,
                "views": views,
                "applies": applies,
                "employee_count": employee_count,
                "follower_count": follower_count,
                "company_job_count": company_job_count,
            }

            if obj is None:
                st.session_state["salary_prediction_error"] = "Model file could not be loaded. " + (f"Details: {load_error}" if load_error else "")
            elif not feature_cols:
                st.session_state["salary_prediction_error"] = "The loaded model does not contain feature_cols."
            else:
                try:
                    X_pred = build_prediction_row(
                        feature_cols=feature_cols,
                        title=job_title,
                        description=generated_description,
                        exp_level=exp_level,
                        work_type=work_type,
                        industry=industry,
                        pay_period=pay_period,
                        state=state,
                        city=city,
                        company_size=company_size,
                        is_remote=is_remote,
                        skills_sel=skills_sel,
                        selected_benefits=selected_benefits,
                        views=views,
                        applies=applies,
                        employee_count=employee_count,
                        follower_count=follower_count,
                        company_job_count=company_job_count,
                    )
                    pred_log = predict_salary_from_model(obj, X_pred)
                    pred_salary = float(np.expm1(pred_log))
                    st.session_state["salary_prediction_result"] = {
                        "salary": pred_salary,
                        "practical_low": pred_salary * 0.85,
                        "practical_high": pred_salary * 1.15,
                        "mae_low": max(0, pred_salary - mae_model4),
                        "mae_high": pred_salary + mae_model4,
                        "rmse_low": max(0, pred_salary - rmse_model4),
                        "rmse_high": pred_salary + rmse_model4,
                        "model_name": model_name,
                        "model_type": model_type,
                        "signature": input_signature(payload),
                        "timestamp": time.strftime("%H:%M:%S"),
                        "payload": payload,
                    }
                    st.session_state.pop("salary_prediction_error", None)
                except Exception as exc:
                    msg = str(exc)
                    msg = re.sub(r"[A-Za-z]:\\[^\n\r]+", "[local path hidden]", msg)
                    st.session_state["salary_prediction_error"] = (
                        "Salary prediction could not be completed. "
                        "Please check model files, selected inputs, and local data files. "
                        f"Technical hint: {msg[:500]}"
                    )

    with right:
        section("Prediction result")

        if "salary_prediction_error" in st.session_state:
            st.error(st.session_state["salary_prediction_error"])

        result = st.session_state.get("salary_prediction_result")
        if not result:
            st.info("Fill the form and click **Predict Salary**.")
            box(
                "<b>How this works</b><br>"
                "The app builds a model-ready input row, runs the saved salary model, then compares the estimate with similar postings.",
                "soft",
            )
        else:
            salary = result["salary"]
            payload = result["payload"]
            similar_df, similar_rule = similar_postings_analysis(payload)
            reliability, rel_color, rel_reasons = reliability_signal(salary, similar_df, payload)

            q25 = q50 = q75 = None
            if not similar_df.empty:
                q25 = float(similar_df["normalized_salary"].quantile(0.25))
                q50 = float(similar_df["normalized_salary"].quantile(0.50))
                q75 = float(similar_df["normalized_salary"].quantile(0.75))

            st.success("### Estimated Annual Salary")
            st.caption(f"Model: {result['model_name']} · Recomputed at {result['timestamp']} · Input signature: {result['signature']}")

            a, b = st.columns([1.2, 0.8])
            with a:
                st.metric("Prediction", money(salary))
                st.metric("Practical range", f"{money(result['practical_low'])} — {money(result['practical_high'])}")
            with b:
                st.metric("Reliability signal", reliability)
                st.metric("Benefit package", benefits_package_strength(payload["selected_benefits"])[0])

            if q25 is not None:
                box(
                    f"<b>Recommended market reading</b><br>"
                    f"Use <b>{money(q25)} — {money(q75)}</b> as the main market range because it comes from similar postings. "
                    f"The model estimate is <b>{money(salary)}</b>, and the similar-postings median is <b>{money(q50)}</b>.",
                    "green",
                )

            st.markdown("**Salary bands**")
            bands = pd.DataFrame(
                [
                    ["Practical range", "Simple reading band (±15%)", f"{money(result['practical_low'])} — {money(result['practical_high'])}"],
                    ["Expected zone", "Prediction ± MAE", f"{money(result['mae_low'])} — {money(result['mae_high'])}"],
                    ["Broad zone", "Prediction ± RMSE", f"{money(result['rmse_low'])} — {money(result['rmse_high'])}"],
                ],
                columns=["Band", "Interpretation", "Range"],
            )
            show_df(bands)

            if ADVANCED:
                with st.expander("Stress-test range (very wide; not the main recommended decision range)"):
                    stress_low = max(0, salary - 2 * rmse_model4)
                    stress_high = salary + 2 * rmse_model4
                    st.write(f"Stress-test range: **{money(stress_low)} — {money(stress_high)}**")
                    st.caption("This is a broad model-error band, not a likely salary-offer range.")

            positives, cautions = prediction_explanation(payload)
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Higher-salary signals**")
                for item in positives or ["No strong positive rule-based signal detected."]:
                    st.markdown(f"- {item}")
            with p2:
                st.markdown("**Caution / uncertainty signals**")
                for item in cautions or ["No major caution signal detected."]:
                    st.markdown(f"- {item}")

            box("<b>Reliability details</b><br>" + "<br>".join([f"• {r}" for r in rel_reasons]), rel_color)

            section("Similar postings reference")
            if similar_df.empty:
                st.info("A similar-postings set could not be built.")
            else:
                st.caption(f"Matched set: {similar_rule} · Similar postings found: {len(similar_df):,}")
                show_df(pd.DataFrame({"Statistic": ["P25", "Median", "P75"], "Salary": [money(q25), money(q50), money(q75)]}))
                bins = [0, 50_000, 75_000, 100_000, 130_000, 160_000, 200_000, 300_000]
                labels = ["<$50k", "$50k–75k", "$75k–100k", "$100k–130k", "$130k–160k", "$160k–200k", "$200k+"]
                band_counts = (
                    pd.cut(similar_df["normalized_salary"], bins=bins, labels=labels, include_lowest=True)
                    .value_counts(normalize=True)
                    .sort_index()
                    .mul(100)
                    .round(1)
                )
                st.bar_chart(band_counts)
                chart_help(
                    "This chart shows how similar postings are distributed across salary bands.",
                    "Higher bars mean more similar postings fall into that salary band.",
                    "These are posting distributions, not hiring probabilities.",
                )

            section("Scenario comparison")
            if obj is not None and feature_cols:
                rows = []
                for label, sc in scenario_variants(payload):
                    try:
                        X_tmp = build_prediction_row(
                            feature_cols=feature_cols, title=sc["job_title"], description=sc["description"],
                            exp_level=sc["exp_level"], work_type=sc["work_type"], industry=sc["industry"],
                            pay_period=sc["pay_period"], state=sc["state"], city=sc["city"],
                            company_size=sc["company_size"], is_remote=sc["is_remote"],
                            skills_sel=sc["skills_sel"], selected_benefits=sc["selected_benefits"],
                            views=sc["views"], applies=sc["applies"], employee_count=sc["employee_count"],
                            follower_count=sc["follower_count"], company_job_count=sc["company_job_count"],
                        )
                        pred = float(np.expm1(predict_salary_from_model(obj, X_tmp)))
                        rows.append({"Scenario": label, "Predicted salary": pred, "Difference vs current": pred - salary})
                    except Exception:
                        pass
                if rows:
                    df_scn = pd.DataFrame(rows)
                    df_scn["Predicted salary"] = df_scn["Predicted salary"].map(money)
                    df_scn["Difference vs current"] = df_scn["Difference vs current"].map(lambda x: f"{x:+,.0f} USD")
                    show_df(df_scn)

            use_this_insight([
                "Use similar-postings P25–P75 as your main negotiation or expectation band.",
                "Use scenario comparison to see how seniority, benefits, or remote status can change the estimate.",
                "If reliability is low, rely less on the single number and more on broader market comparison.",
            ])
            download_report_button(result, similar_df, reliability)


# =============================================================================
# HR / RECRUITING
# =============================================================================
elif page == "🏢 HR / Recruiting":
    st.title("HR / Recruiting")
    st.caption("Benchmark salary, benefits, and competition to support role design and offer strategy.")

    page_explainer(
        page_purpose="This page helps HR and recruiting teams benchmark salary ranges and competition signals.",
        questions=[
            "What salary level is typical for a role or segment?",
            "Where is hiring competition stronger?",
            "How does benefit-package strength relate to salary level?",
            "Should an offer target median, P75, or above-market salary?",
        ],
        key_metrics=["Median salary", "Competition score", "Benefit count", "Title-family benchmark"],
        how_to_use="Use salary medians as baseline anchors, then combine them with benefit and competition signals.",
        limitations="Competition uses applications/views. It does not show who was hired or candidate quality.",
    )

    df = load_market_data()
    if df.empty:
        st.warning("Market data could not be loaded.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            exp = st.selectbox("Experience filter", ["All"] + EXP_LEVELS, key="hr_exp")
        with f2:
            ind_values = sorted([x for x in df["primary_industry"].dropna().astype(str).unique().tolist() if x])
            ind_sel = st.selectbox("Industry filter", ["All"] + ind_values[:200], key="hr_ind")
        with f3:
            remote_sel = st.selectbox("Remote filter", ["All", "Remote", "Non-remote"], key="hr_remote")

        temp = df.copy()
        if exp != "All":
            temp = temp[temp["formatted_experience_level"] == exp]
        if ind_sel != "All":
            temp = temp[temp["primary_industry"] == ind_sel]
        if remote_sel != "All":
            temp = temp[temp["remote_flag"] == (1 if remote_sel == "Remote" else 0)]
        sal = temp[temp["salary_usable"]].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filtered postings", f"{len(temp):,}")
        c2.metric("Salary-usable rows", f"{len(sal):,}")
        c3.metric("Median salary", money(sal["normalized_salary"].median()) if len(sal) else "—")
        c4.metric("Competition score", f"{sal['competition_score'].mean():.3f}" if len(sal) else "—")

        section("Role-family benchmark")
        if sal.empty:
            st.info("No salary rows after filtering.")
        else:
            tf = sal.groupby("title_family").agg(
                postings=("job_id", "count"),
                median_salary=("normalized_salary", "median"),
                avg_competition=("competition_score", "mean"),
            ).reset_index()
            tf = tf[tf["postings"] >= 20].sort_values("median_salary", ascending=False)
            if ADVANCED:
                show_df(tf.head(15))
            else:
                box("Basic reading: compare the salary bars first. Open Advanced mode if you want the table behind the chart.", "soft")
            show_bar(tf, "title_family", "median_salary", top_n=12)
            chart_help(
                "This compares median salary across title families under the current filters.",
                "Higher bars mean higher median posted salary.",
            )

        section("Benefit package benchmark")
        ben = compute_benefit_summary()
        if ADVANCED:
            show_df(ben, "Benefit summary could not be built.")
        if not ben.empty:
            show_bar(ben, "benefit_bucket", "median_salary", sort_desc=False)
            chart_help(
                "This shows the relationship between benefit-count buckets and median salary.",
                "Moving right usually means more listed benefits.",
                "A higher benefit count can reflect richer packages, but not every benefit has equal value.",
            )

        section("Competition snapshot")
        if not sal.empty:
            comp = sal.groupby("title_family").agg(
                postings=("job_id", "count"),
                avg_views=("views", "mean"),
                avg_applies=("applies", "mean"),
                competition=("competition_score", "mean"),
            ).reset_index()
            comp = comp[comp["postings"] >= 20].sort_values("competition", ascending=False)
            if ADVANCED:
                show_df(comp.head(12))
                show_bar(comp, "title_family", "competition", top_n=12)
            else:
                box("Basic reading: competition is summarized as applications per view. Use Advanced mode to see the detailed table and chart.", "soft")
                chart_help(
                    "This uses applications / views as a competition-intensity score.",
                    "Higher bars mean more applications per view on average.",
                    "This is not a hiring-success metric and does not show candidate quality.",
                )

        use_this_insight([
            "Use median salary as a baseline anchor and similar market ranges for offer calibration.",
            "High-competition role families may require stronger compensation or benefits.",
            "Benefit package strength can support employer attractiveness but does not replace base salary.",
        ])


# =============================================================================
# EDUCATION
# =============================================================================
elif page == "🎓 Education / Curriculum Planner":
    st.title("Education / Curriculum Planner")
    st.caption("Inspect skill demand and salary-associated skills for curriculum and career-readiness planning.")

    page_explainer(
        page_purpose="This page helps education users inspect skill demand and broad skill-to-salary associations.",
        questions=[
            "Which skills appear most often in postings?",
            "Which skills are associated with higher posted salaries?",
            "How does salary change across the career ladder?",
        ],
        key_metrics=["Posting count by skill", "Skill salary premium", "Career-ladder median salary"],
        how_to_use="Prioritize skills that are both common in postings and associated with higher salaries.",
        limitations="Skill premium is association, not causation. It may overlap with seniority, industry, and location.",
    )

    section("Most frequent skills")
    demand = compute_skill_demand(20)
    if ADVANCED:
        show_df(demand)
    if not demand.empty:
        show_bar(demand, "skill_name", "posting_count", top_n=15)

    section("Skills associated with higher salaries")
    premium = compute_skill_salary_premium(15)
    if ADVANCED:
        show_df(premium)
    if not premium.empty:
        show_bar(premium.rename(columns={"premium_vs_overall": "premium"}), "skill_name", "premium", top_n=12)
        chart_help(
            "This compares a skill's median salary against the overall median.",
            "Positive values mean postings mentioning the skill tend to pay above the dataset median.",
            "This is not proof that learning the skill alone causes the salary increase.",
        )

    section("Career ladder")
    ladder = compute_career_ladder()
    if ADVANCED:
        show_df(ladder)
    if not ladder.empty:
        show_bar(ladder, "formatted_experience_level", "median_salary", sort_desc=False)

    use_this_insight([
        "Prioritize skills that are both frequent and salary-associated.",
        "Use career-ladder charts to explain how experience affects salary expectations.",
        "Avoid presenting skill premium as a guaranteed salary increase.",
    ])


# =============================================================================
# INVESTOR / MARKET ANALYST
# =============================================================================
elif page == "📈 Investor / Market Analyst":
    st.title("Investor / Market Analyst")
    st.caption("Explore hiring volume, salary concentration, and labor-market structure.")

    page_explainer(
        page_purpose="This page helps market-oriented users inspect industry hiring volume and salary concentration.",
        questions=[
            "Which industries are hiring the most?",
            "Which industries combine high salary and high volume?",
            "Which states show stronger salary levels?",
        ],
        key_metrics=["Posting volume", "Median salary", "Remote rate", "Competition score"],
        how_to_use="Use this page to identify labor-market signals, not to make direct investment decisions in isolation.",
        limitations="This is a LinkedIn posting sample, not the entire economy. It is not financial advice.",
    )

    industry = compute_industry_stats(min_count=100)
    state = compute_state_stats(min_count=100)
    title_stats = compute_title_family_stats(min_count=100)

    section("Industry hiring volume")
    if ADVANCED:
        show_df(industry.head(15))
    if not industry.empty:
        show_bar(industry, "primary_industry", "postings", top_n=12)

    section("Industry salary concentration")
    if not industry.empty:
        show_bar(industry.sort_values("median_salary", ascending=False), "primary_industry", "median_salary", top_n=12)

    section("Geographic salary snapshot")
    if ADVANCED:
        show_df(state.head(15))
    if not state.empty:
        show_bar(state.head(12), "state", "median_salary", top_n=12)

    if ADVANCED:
        section("Title-family market structure")
        show_df(title_stats.head(12))
        if not title_stats.empty:
            show_bar(title_stats, "title_family", "median_salary", top_n=10)

    use_this_insight([
        "Look for segments that combine hiring volume and strong salary levels.",
        "Use state and industry views together because geography and industry interact.",
        "Treat this as a labor-market signal, not investment advice.",
    ])


# =============================================================================
# POLICY / LABOR MARKET ANALYST
# =============================================================================
elif page == "🏛️ Policy Maker / Labor Market Analyst":
    st.title("Policy Maker / Labor Market Analyst")
    st.caption("Regional inequality, remote access, and salary-pattern interpretation.")

    page_explainer(
        page_purpose="This page helps policy-oriented users inspect regional salary differences and remote-access patterns.",
        questions=[
            "Which regions show higher or lower salary levels?",
            "How does remote access vary across states?",
            "Where are market differences wider?",
        ],
        key_metrics=["Median salary by state", "Remote rate by state", "Remote vs non-remote salary"],
        how_to_use="Use this page to inspect broad opportunity patterns and possible regional inequality signals.",
        limitations="This is a job-posting sample, not a full labor-force dataset. Wider salary differences do not prove policy failure on their own.",
    )

    state = compute_state_stats(min_count=100)
    remote = compute_remote_summary()

    section("State salary comparison")
    if ADVANCED:
        show_df(state.head(20))
    if not state.empty:
        show_bar(state.head(15), "state", "median_salary", top_n=15)

    section("Remote access snapshot")
    if ADVANCED:
        show_df(remote)
    if not remote.empty:
        show_bar(remote, "remote_label", "median_salary", sort_desc=False)
        chart_help(
            "This compares median salary for postings explicitly marked remote versus not explicitly remote.",
            "Remote labels can be incomplete, so read this as a posting-level signal.",
            "A non-remote label may include postings where remote status was missing.",
        )

    if ADVANCED and not state.empty:
        section("States with higher remote share")
        state_rate = state[["state", "remote_rate"]].sort_values("remote_rate", ascending=False).head(15)
        show_bar(state_rate, "state", "remote_rate", top_n=15)

    # Salary transparency / negotiation-gap outputs from analysis_13, if available.
    gap_files = [
        ("68_salary_gap_distribution.png", "Salary gap distribution"),
        ("69_salary_gap_by_industry.png", "Negotiation gap by industry"),
        ("70_salary_gap_by_exp.png", "Negotiation gap by experience level"),
        ("71_salary_band_vs_gap.png", "Salary band vs negotiation gap"),
        ("72_salary_gap_negotiation_map.png", "Industry × experience negotiation map"),
    ]
    available_gap = [(f, title) for f, title in gap_files if (OUT / f).exists()]
    if available_gap:
        section("Salary transparency / negotiation-gap signals")
        box(
            "<b>Policy interpretation</b><br>"
            "Salary gap means max_salary - min_salary in postings with salary ranges. "
            "Wider gaps can suggest more negotiation room or less precise pay transparency.",
            "soft",
        )
        for filename, title in available_gap:
            st.markdown(f"**{title}**")
            st.image(str(OUT / filename), width="stretch")
            chart_help(
                "This chart comes from the salary-gap analysis module.",
                "Use it to inspect where posted salary ranges are wider.",
                "Wider posted salary ranges do not automatically prove unfairness; they are transparency / negotiation signals.",
            )

    use_this_insight([
        "Use state salary levels together with remote-rate differences to discuss opportunity distribution.",
        "Treat platform-posting patterns as directional signals, not the entire labor market.",
        "Where market ranges are wide, salary transparency may be weaker and workers may face more uncertainty.",
    ])


# =============================================================================
# RESEARCHER / ML EVALUATION
# =============================================================================
elif page == "🔬 Researcher / ML Evaluation":
    st.title("Researcher / ML Evaluation")
    st.caption("Cross-validation, uncertainty, diagnostics, and segment error analysis.")

    page_explainer(
        page_purpose="This page documents how salary models performed and where larger errors remain.",
        questions=[
            "How did Models 01–04 perform?",
            "How much did Model 04 improve over Model 03?",
            "Which interval method is most useful?",
            "Which segments have larger error or bias?",
        ],
        key_metrics=["Raw R²", "RMSE", "MAE", "OOF / cross-validation", "Coverage", "Interval width"],
        how_to_use="Use validation metrics rather than training metrics. Segment-error tables show where the model is less stable.",
        limitations="Some segment-level tables may reflect sparse categories or naturally harder-to-model compensation structures.",
    )

    metric_glossary(
        {
            "R²": "Explained variance score. Higher is better.",
            "RMSE": "Root Mean Squared Error. Penalizes larger mistakes more.",
            "MAE": "Mean Absolute Error. Average absolute prediction error.",
            "OOF": "Out-of-fold predictions from cross-validation.",
            "Coverage": "How often the true value falls inside a prediction interval.",
            "Interval width": "How wide the interval is on average.",
            "Calibration": "How closely predicted levels align with actual levels.",
        },
        expanded=True,
    )

    model4 = load_model4_bundle()
    m4json = model4.get("metrics_json", {}) or {}
    m3 = m4json.get("model03_weighted_oof", {})
    m4 = m4json.get("model04_ridge_oof", {})
    intervals = model4.get("intervals", pd.DataFrame())
    coeffs = model4.get("coefficients", pd.DataFrame())

    section("Model comparison")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model 01 Raw R²", f"{MODEL1_METRICS['r2']:.3f}")
    c2.metric("Model 02 Raw R²", f"{MODEL2_METRICS['r2']:.3f}")
    c3.metric("Model 03 Raw R²", f"{MODEL3_METRICS['r2']:.3f}")
    c4.metric("Model 04 Raw R²", f"{float(m4.get('r2', np.nan)):.3f}" if m4 else "—")

    comp = pd.DataFrame([
        {"Model": "Model 01", "Raw R²": MODEL1_METRICS["r2"], "RMSE": MODEL1_METRICS["rmse"], "MAE": MODEL1_METRICS["mae"]},
        {"Model": "Model 02", "Raw R²": MODEL2_METRICS["r2"], "RMSE": MODEL2_METRICS["rmse"], "MAE": MODEL2_METRICS["mae"]},
        {"Model": "Model 03", "Raw R²": float(m3.get("r2", MODEL3_METRICS["r2"])), "RMSE": float(m3.get("rmse", MODEL3_METRICS["rmse"])), "MAE": float(m3.get("mae", MODEL3_METRICS["mae"]))},
        {"Model": "Model 04", "Raw R²": float(m4.get("r2", np.nan)), "RMSE": float(m4.get("rmse", np.nan)), "MAE": float(m4.get("mae", np.nan))},
    ])
    show_df(comp)

    if m3 and m4:
        delta_r2 = float(m4.get("r2", 0)) - float(m3.get("r2", 0))
        delta_rmse = float(m4.get("rmse", 0)) - float(m3.get("rmse", 0))
        delta_mae = float(m4.get("mae", 0)) - float(m3.get("mae", 0))
        box(
            f"<b>Model 04 vs Model 03</b><br>"
            f"R² improved by <b>{delta_r2:+.4f}</b>. RMSE changed by <b>{delta_rmse:,.0f}</b> USD. "
            f"MAE changed by <b>{delta_mae:,.0f}</b> USD. "
            "The gain is modest but healthy: Model 04 mainly adds stronger uncertainty and error-analysis layers.",
            "green",
        )

    section("Prediction-interval methods")
    show_df(intervals, "Interval-method scores are not available.")
    if not intervals.empty:
        width_col = "avg_width" if "avg_width" in intervals.columns else intervals.columns[-2]
        cov_col = "coverage_pct" if "coverage_pct" in intervals.columns else intervals.columns[1]
        show_bar(intervals.rename(columns={width_col: "Average interval width"}), "method", "Average interval width", top_n=len(intervals))
        chart_help(
            "This compares interval width across uncertainty methods.",
            "Narrower intervals are useful only if coverage remains near the target.",
            "A narrow interval with poor coverage can be misleading.",
        )
        rec = intervals.sort_values([cov_col, width_col], ascending=[False, True]).iloc[0]
        box(
            f"<b>Recommended interval method</b><br>"
            f"Current best practical choice: <b>{rec['method']}</b>. It balances coverage and interval width better than the alternatives in this run.",
            "green",
        )

    section("How Model 04 is built")
    show_df(coeffs, "Ridge-stacking coefficient table is not available.")
    if not coeffs.empty:
        label_col = coeffs.columns[0]
        val_col = coeffs.columns[1]
        show_bar(coeffs, label_col, val_col, top_n=len(coeffs))

    section("Segment error analysis")
    tabs = st.tabs(["Experience", "Industry", "State", "Salary band", "Title family", "Error titles"])

    with tabs[0]:
        df0 = model4.get("error_experience", pd.DataFrame())
        show_df(df0, "Experience-level error table is not available.")
        if not df0.empty:
            label = df0.columns[0]
            value = next((c for c in df0.columns if c.lower() in {"rmse", "mae"}), df0.columns[-1])
            show_bar(df0, label, value, top_n=len(df0))

    with tabs[1]:
        df1 = model4.get("error_industry", pd.DataFrame())
        show_df(df1.head(20), "Industry error table is not available.")
        if not df1.empty:
            label = df1.columns[0]
            value = next((c for c in df1.columns if c.lower() in {"rmse", "mae"}), df1.columns[-1])
            show_bar(df1.head(15), label, value, top_n=15)
            chart_help(
                "This shows the highest-error industries.",
                "Use it to identify sectors where salary structure is harder to predict.",
                "Some industries may have more variable compensation structures or lower sample quality.",
            )

    with tabs[2]:
        df2 = model4.get("error_state", pd.DataFrame())
        show_df(df2.head(20), "State error table is not available.")
        if not df2.empty:
            label = df2.columns[0]
            value = next((c for c in df2.columns if c.lower() in {"rmse", "mae"}), df2.columns[-1])
            show_bar(df2.head(15), label, value, top_n=15)

    with tabs[3]:
        df3 = model4.get("error_salary_band", pd.DataFrame())
        show_df(df3, "Salary-band error table is not available.")
        if not df3.empty:
            label = df3.columns[0]
            value = next((c for c in df3.columns if c.lower() in {"rmse", "mae"}), df3.columns[-1])
            show_bar(df3, label, value, top_n=len(df3))
            chart_help(
                "This shows how error changes across salary bands.",
                "A rising pattern at higher salary bands is common because high-end compensation is more variable.",
                "Do not read high-band error as total model failure; salary dispersion is wider there.",
            )

    with tabs[4]:
        df4 = model4.get("error_title_family", pd.DataFrame())
        show_df(df4, "Title-family error table is not available.")
        if not df4.empty:
            label = df4.columns[0]
            value = next((c for c in df4.columns if c.lower() in {"rmse", "mae"}), df4.columns[-1])
            show_bar(df4, label, value, top_n=len(df4))

    with tabs[5]:
        show_df(model4.get("top_error_titles", pd.DataFrame()).head(20), "Top-error-title table is not available.")
        if ADVANCED:
            st.markdown("**Top overpredicted examples**")
            show_df(model4.get("top_overpredicted", pd.DataFrame()).head(15), "Top-overpredicted table is not available.")
            st.markdown("**Top underpredicted examples**")
            show_df(model4.get("top_underpredicted", pd.DataFrame()).head(15), "Top-underpredicted table is not available.")
            chart_help(
                "These are diagnostic examples where errors were especially large.",
                "Use them to understand edge cases, not as global truth.",
                "Some groups may be sparse or internally heterogeneous.",
            )

    section("Data quality and methodology")
    show_df(data_quality_summary())
    box(
        "<b>Methodology note</b><br>"
        "Model evaluation should be read from cross-validation / out-of-fold performance. "
        "Model 04 adds a Ridge stacking layer and interval methods, so its value is not only a slightly higher R², "
        "but also better uncertainty and segment-level error interpretation.",
        "soft",
    )
    use_this_insight([
        "Prioritize OOF / cross-validation metrics over training metrics.",
        "Use segment-error tables to identify where the model is less stable.",
        "Treat interval methods as part of the decision output, not only a technical appendix.",
    ])
