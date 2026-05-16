"""
streamlit_app.py
LinkedIn Job Analysis Dashboard

Run:
    streamlit run streamlit_app.py
"""

import os
import re
import sys
import json
import time
import types
import hashlib
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "outputs"
MODELS = BASE / "models"

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="LinkedIn Job Analysis",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(
    """
<style>
[data-testid="stSidebar"] { background: #0F1117; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 4px 0; }
.metric-card {
    background: #1A1D27;
    border: 1px solid #2E3347;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-label { color: #8B92A5; font-size: 13px; margin-bottom: 4px; }
.metric-value { color: #E2E8F0; font-size: 26px; font-weight: 600; }
.metric-delta { font-size: 12px; margin-top: 4px; }
.metric-delta.pos { color: #10B981; }
.metric-delta.neg { color: #F43F5E; }
.section-title {
    color: #C8CDD8;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2E3347;
}
.insight-box {
    background: #1A1D27;
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0 14px 0;
    color: #C8CDD8;
    font-size: 14px;
}
.insight-box.green  { border-left-color: #10B981; }
.insight-box.amber  { border-left-color: #F59E0B; }
.insight-box.red    { border-left-color: #F43F5E; }
.small-note {
    color: #8B92A5;
    font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CUSTOM TRANSFORMERS FOR JOBLIB LOADING
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


# The model was saved from a script, so joblib may look for classes under __main__
# or under the model script module name. Register the classes in all likely places.
for module_name in ["__main__", "model_03_salary_advanced_progress", "model_03_salary_advanced", "model_02_pipeline_v3_cv_safe"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)
    setattr(sys.modules[module_name], "LogTransformer", LogTransformer)
    setattr(sys.modules[module_name], "MedianTargetEncoder", MedianTargetEncoder)


# =============================================================================
# HELPERS
# =============================================================================
def img(filename, caption=None):
    path = OUT / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Chart not found: {filename}")


def metric_card(label, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta:
        cls = "pos" if delta_pos else "neg"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'

    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def insight(text, color="purple"):
    st.markdown(f'<div class="insight-box {color}">{text}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def explain(title, bullets):
    """Small explanatory text block for charts and tables."""
    items = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(
        f"""
        <div class="insight-box">
            <b>{title}</b>
            <ul style="margin-top:6px;margin-bottom:0;padding-left:20px">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_col_name(value):
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value if value else "unknown"


def make_title_family(title):
    t = str(title or "").lower()

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


def clean_salary_text_for_dashboard(text):
    """A lightweight leakage-removal version for live user input."""
    s = str(text or "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(
        r"\$\s?\d+(?:[,\d]*)(?:\.\d+)?\s*(?:k|K)?(?:\s*(?:-|to|–|—)\s*\$?\s?\d+(?:[,\d]*)(?:\.\d+)?\s*(?:k|K)?)?",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(?:salary range|compensation range|pay range|base salary|annual salary|hourly rate|expected salary|salary|compensation|pay rate|bonus eligible|usd|dollars|per year|yearly|annually)\b",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", s).strip()


@st.cache_data(show_spinner=False)
def load_postings_summary():
    try:
        post = pd.read_csv(
            DATA / "postings.csv",
            usecols=["job_id", "normalized_salary", "currency", "remote_allowed"],
            low_memory=False,
        )
        sal = post[
            post["normalized_salary"].between(10_000, 300_000)
            & (post["currency"].isin(["USD"]) | post["currency"].isna())
        ]
        return {
            "total_postings": len(post),
            "salary_count": len(sal),
            "median_salary": int(sal["normalized_salary"].median()),
            "remote_pct": round((post["remote_allowed"] == 1.0).mean() * 100, 1),
        }
    except Exception:
        return {
            "total_postings": 123_849,
            "salary_count": 35_279,
            "median_salary": 81_734,
            "remote_pct": 12.3,
        }


@st.cache_resource(show_spinner=False)
def load_salary_model():
    """Load final Model 03 ensemble first, then fall back to Model 02."""
    candidates = [
        MODELS / "best_salary_model_03_advanced_progress.joblib",
        MODELS / "best_salary_model_03_advanced.joblib",
        MODELS / "best_salary_pipeline.joblib",
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


def predict_salary_from_model(obj, X_pred):
    """Return a log-salary prediction from either a single pipeline or an ensemble object."""
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


def build_prediction_row(
    feature_cols,
    title,
    description,
    exp_level,
    work_type,
    industry,
    pay_period,
    state,
    city,
    company_size,
    is_remote,
    skills_sel,
    selected_benefits,
    views,
    applies,
    employee_count,
    follower_count,
    company_job_count,
):
    """Build a single prediction row that matches Model 03 feature_cols."""
    row = {c: 0 for c in feature_cols}

    title_text = str(title or "").strip()
    desc_text = clean_salary_text_for_dashboard(description)
    title_lower = title_text.lower()
    title_family = make_title_family(title_lower)

    remote_flag = int(is_remote)
    title_cluster = "Unknown"  # Live KMeans assignment is not stored; unknown is safely handled.
    city_value = str(city or "Unknown").strip() or "Unknown"
    state_value = str(state or "Unknown").strip() or "Unknown"
    pay_value = str(pay_period or "YEARLY").strip() or "YEARLY"
    company_size_code = str(SIZE_MAP_INV.get(company_size, 5))

    updates = {
        # Categorical and target-encoded columns
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

        # Text
        "title_text": title_text,
        "description_clean": desc_text,

        # Numeric
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

        # Binary / title flags
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

    # Benefit multi-hot columns. The exact feature names can differ depending on
    # the training script, so we try several safe candidates.
    for benefit in selected_benefits:
        clean = clean_col_name(benefit)
        candidates = [
            f"benefit_{clean}",
            f"benefit_type_{clean}",
            f"ben_{clean}",
            clean,
        ]
        for col_name in candidates:
            if col_name in row:
                row[col_name] = 1

        # Fallback for columns that contain the clean benefit token.
        for existing_col in row.keys():
            if existing_col.startswith(("benefit_", "ben_")) and clean in existing_col:
                row[existing_col] = 1

    return pd.DataFrame([row], columns=feature_cols)


def input_signature(payload):
    """Create a short stable signature so users can see the prediction was recomputed."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def safe_read_csv_existing(path, wanted_cols=None, low_memory=False):
    """Read only columns that exist. This keeps the dashboard robust across dataset versions."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if wanted_cols is None:
            return pd.read_csv(path, low_memory=low_memory)
        usecols = [c for c in wanted_cols if c in header]
        if not usecols:
            return pd.DataFrame()
        return pd.read_csv(path, usecols=usecols, low_memory=low_memory)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_market_data():
    """
    Lightweight, filterable dataset for interactive dashboard components.
    This does not train a model; it only supports summaries and similar-posting analysis.
    """
    wanted = [
        "job_id", "title", "location", "remote_allowed", "normalized_salary", "currency",
        "formatted_experience_level", "formatted_work_type", "pay_period",
        "views", "applies", "description", "company_id",
    ]

    post = safe_read_csv_existing(DATA / "postings.csv", wanted_cols=wanted, low_memory=False)
    if post.empty:
        return post

    # Add missing columns safely.
    defaults = {
        "title": "",
        "location": "Unknown",
        "remote_allowed": np.nan,
        "normalized_salary": np.nan,
        "currency": "USD",
        "formatted_experience_level": "Unknown",
        "formatted_work_type": "Unknown",
        "pay_period": "Unknown",
        "views": np.nan,
        "applies": np.nan,
        "description": "",
        "company_id": np.nan,
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

    # Primary industry join.
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
            merged = job_ind[["job_id", ind_id_col]].merge(
                ind[[ind_id_col, name_col]], on=ind_id_col, how="left"
            )
            primary = merged.dropna(subset=[name_col]).groupby("job_id")[name_col].first()
            post["primary_industry"] = post["job_id"].map(primary).fillna("Unknown")

    salary_ok = (
        post["normalized_salary"].between(10_000, 300_000)
        & (post["currency"].isin(["USD"]) | post["currency"].isna())
    )
    post["salary_usable"] = salary_ok

    return post


def filter_market_data(df, exp=None, industry=None, state=None, remote=None, pay_period=None, company_size=None):
    out = df.copy()
    if out.empty:
        return out

    if exp and exp != "All":
        out = out[out["formatted_experience_level"] == exp]
    if industry and industry != "All":
        ind_norm = industry.lower()
        out = out[out["primary_industry"].astype(str).str.lower().str.contains(re.escape(ind_norm), na=False)]
        if out.empty:
            out = df[df["primary_industry"].astype(str).str.lower() == ind_norm]
    if state and state != "All":
        out = out[out["state"] == state]
    if remote and remote != "All":
        out = out[out["remote_flag"] == (1 if remote == "Remote" else 0)]
    if pay_period and pay_period != "All" and "pay_period" in out.columns:
        out = out[out["pay_period"].astype(str).str.upper() == pay_period.upper()]

    return out


def interactive_market_explorer():
    df = load_market_data()

    if df.empty:
        st.warning("Interactive market data could not be loaded.")
        return

    st.markdown(
        "Use these filters to inspect the dataset directly. This complements the static PNG charts "
        "with a lightweight interactive summary."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        exp = st.selectbox("Filter: experience", ["All"] + EXP_LEVELS, key="global_exp")
    with c2:
        industry = st.selectbox("Filter: industry", ["All"] + INDUSTRIES, key="global_industry")
    with c3:
        state = st.selectbox("Filter: state", ["All"] + US_STATES, key="global_state")
    with c4:
        remote = st.selectbox("Filter: remote", ["All", "Remote", "Non-remote / unknown"], key="global_remote")

    c5, c6 = st.columns(2)
    with c5:
        pay_period = st.selectbox("Filter: pay period", ["All"] + PAY_PERIODS, key="global_pay_period")
    with c6:
        show_rows = st.slider("Rows to preview", 5, 50, 10, key="global_preview_rows")

    filtered = filter_market_data(df, exp, industry, state, remote, pay_period)
    sal = filtered[filtered["salary_usable"]].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Filtered postings", f"{len(filtered):,}")
    with c2:
        metric_card("With usable salary", f"{len(sal):,}")
    with c3:
        val = f"${int(sal['normalized_salary'].median()):,}" if len(sal) else "N/A"
        metric_card("Filtered median salary", val)
    with c4:
        val = f"{round(filtered['remote_flag'].mean()*100, 1)}%" if len(filtered) else "N/A"
        metric_card("Filtered remote rate", val)

    if len(sal) > 0:
        st.markdown("**Filtered salary percentiles**")
        qs = sal["normalized_salary"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        st.dataframe(
            pd.DataFrame({
                "Percentile": ["P10", "P25", "Median", "P75", "P90"],
                "Salary": [f"${v:,.0f}" for v in qs.values],
            }),
            width="stretch",
            hide_index=True,
        )

        band_bins = [0, 50_000, 75_000, 100_000, 130_000, 160_000, 200_000, 300_000]
        band_labels = ["<$50k", "$50k–75k", "$75k–100k", "$100k–130k", "$130k–160k", "$160k–200k", "$200k+"]
        band_counts = pd.cut(sal["normalized_salary"], bins=band_bins, labels=band_labels, include_lowest=True).value_counts().sort_index()
        st.bar_chart(band_counts)

    st.markdown("**Preview of filtered postings**")
    preview_cols = ["title", "location", "formatted_experience_level", "primary_industry", "normalized_salary", "remote_flag"]
    preview_cols = [c for c in preview_cols if c in filtered.columns]
    st.dataframe(filtered[preview_cols].head(show_rows), width="stretch", hide_index=True)

    st.download_button(
        "Download filtered summary as CSV",
        data=filtered[preview_cols].head(5000).to_csv(index=False).encode("utf-8"),
        file_name="filtered_linkedin_postings_preview.csv",
        mime="text/csv",
    )


def benefit_package_strength(selected_benefits):
    high_value = [
        "Medical insurance", "401(k)", "Paid time off", "Parental leave",
        "Tuition assistance", "Disability insurance", "Life insurance",
    ]
    selected = selected_benefits or []
    high_count = sum(1 for b in selected if b in high_value)

    if len(selected) >= 6 or high_count >= 4:
        return "Strong package", "green"
    if len(selected) >= 3 or high_count >= 2:
        return "Standard package", "amber"
    return "Basic package", "red"


def prediction_explanation(payload):
    """Rule-based explanation. It does not replace SHAP, but it is fast and understandable."""
    positives = []
    cautions = []

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
        positives.append("The job title contains seniority/leadership language.")
    if re.search(r"software|engineer|data|machine learning|ai|security|cloud", title):
        positives.append("The title belongs to a high-value technical job family.")

    if state in ["CA", "NY", "WA", "MA"]:
        positives.append(f"{state} is often associated with higher salary postings in this dataset.")
    elif state not in ["Unknown", "CA", "NY", "WA", "MA"]:
        cautions.append("Location can reduce or raise the prediction depending on local salary patterns.")

    if remote:
        positives.append("Remote roles are often associated with higher-compensation postings, but this is correlational.")

    if len(skills) >= 3:
        positives.append("Multiple selected skills create a stronger skill profile signal.")
    elif len(skills) == 0:
        cautions.append("No selected skills reduces the available skill signal for the model.")

    if company_size in ["1K-5K", "5K-10K", "10K+"]:
        positives.append("Larger company size often signals a more structured compensation package.")

    if len(benefits) >= 5:
        positives.append("A stronger benefit package can indicate a higher-quality employer compensation package.")
    elif len(benefits) <= 1:
        cautions.append("Few listed benefits may indicate a weaker employer package signal.")

    if city == "Other / custom city":
        cautions.append("Custom cities may map to less familiar categories and can reduce model certainty.")

    return positives[:5], cautions[:5]


def similar_postings_analysis(payload):
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

    # Progressive relaxation. The first rule with enough rows wins.
    rules = [
        ("same experience + state + industry + title family + remote",
         lambda d: d[
             (d["formatted_experience_level"] == exp)
             & (d["state"] == state)
             & (d["primary_industry"].astype(str).str.lower().str.contains(re.escape(industry), na=False))
             & (d["title_family"] == target_family)
             & (d["remote_flag"] == remote_flag)
         ]),
        ("same experience + state + title family",
         lambda d: d[(d["formatted_experience_level"] == exp) & (d["state"] == state) & (d["title_family"] == target_family)]),
        ("same experience + industry + title family",
         lambda d: d[
             (d["formatted_experience_level"] == exp)
             & (d["primary_industry"].astype(str).str.lower().str.contains(re.escape(industry), na=False))
             & (d["title_family"] == target_family)
         ]),
        ("same experience + title family",
         lambda d: d[(d["formatted_experience_level"] == exp) & (d["title_family"] == target_family)]),
        ("same experience + state",
         lambda d: d[(d["formatted_experience_level"] == exp) & (d["state"] == state)]),
        ("same experience only",
         lambda d: d[d["formatted_experience_level"] == exp]),
        ("all salary postings",
         lambda d: d),
    ]

    for label, fn in rules:
        subset = fn(salary_df)
        if len(subset) >= 30:
            return subset, label

    return salary_df, "all salary postings"


@st.cache_data(show_spinner=False)
def data_quality_summary():
    df = load_market_data()
    rows = []

    if not df.empty:
        total = len(df)
        rows.extend([
            {"Metric": "Total job postings", "Count": total, "Coverage": "100.0%"},
            {"Metric": "Usable salary rows", "Count": int(df["salary_usable"].sum()), "Coverage": f"{df['salary_usable'].mean()*100:.1f}%"},
            {"Metric": "Rows with description", "Count": int(df["description"].notna().sum()), "Coverage": f"{df['description'].notna().mean()*100:.1f}%"},
            {"Metric": "Rows with applications", "Count": int(df["applies"].notna().sum()), "Coverage": f"{df['applies'].notna().mean()*100:.1f}%"},
            {"Metric": "Rows with views", "Count": int(df["views"].notna().sum()), "Coverage": f"{df['views'].notna().mean()*100:.1f}%"},
            {"Metric": "Explicit remote postings", "Count": int((df["remote_flag"] == 1).sum()), "Coverage": f"{(df['remote_flag'] == 1).mean()*100:.1f}%"},
        ])

    benefits = safe_read_csv_existing(DATA / "benefits.csv", low_memory=False)
    if not benefits.empty and "job_id" in benefits.columns and not df.empty:
        total = len(df)
        unique_benefit_jobs = benefits["job_id"].nunique()
        rows.append({
            "Metric": "Postings with listed benefits",
            "Count": int(unique_benefit_jobs),
            "Coverage": f"{unique_benefit_jobs / max(total, 1) * 100:.1f}%",
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def chart_inventory():
    try:
        source_text = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=["Chart", "Status"])

    charts = sorted(set(re.findall(r'img\("([^"]+)"', source_text)))
    rows = []
    for chart in charts:
        exists = (OUT / chart).exists()
        rows.append({"Chart": chart, "Status": "Found" if exists else "Missing"})
    return pd.DataFrame(rows)


# =============================================================================
# SELECT OPTIONS
# =============================================================================
TOP_SKILLS = [
    "Information Technology", "Sales", "Management", "Manufacturing",
    "Engineering", "Health Care Provider", "Business Development",
    "Finance", "Accounting/Auditing", "Administrative",
    "Marketing", "Project Management", "Analyst", "Customer Service",
    "Operations", "Legal", "Research", "Design", "Education", "Consulting",
]

EXP_LEVELS = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]

INDUSTRIES = [
    "Software Development",
    "IT Services and IT Consulting",
    "Financial Services",
    "Hospitals and Health Care",
    "Staffing and Recruiting",
    "Technology, Information and Internet",
    "Business Consulting and Services",
    "Retail",
    "Manufacturing",
    "Accounting",
    "Marketing Services",
    "Higher Education",
    "Construction",
    "Telecommunications",
]

WORK_TYPES = ["Full-time", "Contract", "Part-time", "Internship", "Temporary", "Other"]
PAY_PERIODS = ["YEARLY", "HOURLY", "MONTHLY", "ONCE"]
COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-500", "501-1K", "1K-5K", "5K-10K", "10K+"]
SIZE_MAP_INV = {"1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4, "501-1K": 5, "1K-5K": 6, "5K-10K": 7, "10K+": 8}
US_STATES = [
    "CA", "NY", "TX", "WA", "MA", "IL", "FL", "VA", "GA", "CO",
    "NC", "NJ", "OH", "PA", "AZ", "MN", "MI", "OR", "MD", "CT",
]

CITY_OPTIONS = [
    "San Francisco", "New York", "Seattle", "Boston", "Chicago", "Los Angeles",
    "Austin", "Dallas", "Atlanta", "Washington", "Denver", "Phoenix", "Remote",
    "Other / custom city",
]

BENEFIT_OPTIONS = [
    "Medical insurance",
    "Dental insurance",
    "Vision insurance",
    "401(k)",
    "Paid time off",
    "Parental leave",
    "Life insurance",
    "Disability insurance",
    "Tuition assistance",
    "Employee discount",
    "Commuter benefits",
    "Remote or flexible work support",
]


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 💼 LinkedIn Jobs")
    st.markdown("*123k job postings · Kaggle 2024*")
    st.markdown("---")

    page = st.radio(
        "Choose a perspective:",
        options=[
            "🏠 Home",
            "👤 Job Seeker",
            "🏢 HR / Recruiting",
            "🎓 Education",
            "📈 Investor",
            "🏛️ Policy Maker",
            "🔬 Researcher",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#4B5563'>Dataset: arshkon/linkedin-job-postings<br>"
        "Models: XGBoost · LightGBM · CatBoost<br>"
        "Best Raw R² = 0.757 (Model 03 ensemble)</small>",
        unsafe_allow_html=True,
    )


# =============================================================================
# HOME
# =============================================================================
if "Home" in page:
    st.title("LinkedIn Job Postings — Comprehensive Analysis")
    st.markdown(
        "A multi-perspective analysis of LinkedIn job postings using the Kaggle "
        "`arshkon/linkedin-job-postings` dataset. The project covers labor market trends, "
        "salary prediction, skill demand, competition metrics, remote work, benefits, "
        "and model explainability."
    )

    summary = load_postings_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Postings", f"{summary['total_postings']:,}")
    with c2:
        metric_card("Postings with Salary", f"{summary['salary_count']:,}")
    with c3:
        metric_card("Median Salary (USD)", f"${summary['median_salary']:,}")
    with c4:
        metric_card("Remote Posting Rate", f"{summary['remote_pct']}%")

    st.markdown("<br>", unsafe_allow_html=True)

    section("Interactive Market Explorer")
    explain(
        "Why this matters",
        [
            "Static charts show the final exported analysis, while this panel lets users filter the underlying postings.",
            "It helps answer practical questions such as salary percentiles by experience, industry, state, and remote status.",
        ],
    )
    with st.expander("Open filters", expanded=False):
        interactive_market_explorer()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        section("Market Overview")
        explain("How to read this", ["This chart shows where hiring demand is concentrated by experience level.", "It helps establish the market baseline before salary modeling."])
        img("01_experience_distribution.png")
    with col2:
        section("Salary Distribution")
        explain("How to read this", ["Salary differs strongly by seniority.", "This is one reason experience level is a key model feature."])
        img("05_salary_by_experience.png")

    st.markdown("---")
    section("Project Scope")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            **📊 Analysis Scripts**
            - Market overview
            - Skills and salary premium
            - Cross-sectional insights
            - Geographic and company analysis
            - Remote work, benefits, competition
            - Salary gap and negotiation range
            """
        )
    with c2:
        st.markdown(
            """
            **🤖 Modeling**
            - Model 01: feature-rich baseline, Raw R²≈0.653
            - Model 02: CV-safe pipeline + Optuna, Raw R²=0.711
            - Model 03: advanced OOF ensemble, Raw R²=0.757
            - XGBoost · LightGBM · CatBoost
            - SHAP and diagnostics
            """
        )
    with c3:
        st.markdown(
            """
            **🔍 Six Perspectives**
            - Job seeker
            - HR / recruiting
            - Education
            - Investor
            - Policy maker
            - Researcher
            """
        )


# =============================================================================
# JOB SEEKER
# =============================================================================
elif "Job Seeker" in page:
    st.title("👤 Job Seeker Perspective")
    st.markdown("*Salary prediction, skill guidance, career ladder, and competition analysis.*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Salary Prediction",
        "🎯 Skill Guide",
        "📊 Career Ladder",
        "⚔️ Competition",
    ])

    with tab1:
        section("Live Salary Prediction")
        insight(
            "Final model: Model 03 Advanced OOF Ensemble · Raw R²=0.7570 · Log R²=0.8065 · "
            "RMSE=$25,225 · MAE=$16,004. The model file is cached, but every button click recomputes the prediction.",
            "green",
        )
        insight(
            "Important: the dataset contains job postings, salary fields, views, and applications. "
            "It does not contain who was hired or which offer was accepted, so the app cannot estimate hiring probability.",
            "amber",
        )

        obj, feature_cols, model_name, model_type, metrics, load_error = load_salary_model()

        col_form, col_result = st.columns([1.05, 1])

        with col_form:
            with st.form("salary_form", clear_on_submit=False):
                job_title = st.text_input(
                    "Job title",
                    value="Senior Software Engineer",
                    help=(
                        "Free text is intentional. The model uses title text features, "
                        "so 'Software Engineer', 'Senior Software Engineer', and "
                        "'Principal Backend Engineer' can produce different signals."
                    ),
                )
                st.caption("Examples: Data Analyst, Product Manager, Registered Nurse, Finance Manager, Principal Backend Engineer")

                exp_level = st.selectbox("Experience level", EXP_LEVELS, index=2)
                work_type = st.selectbox("Work type", WORK_TYPES, index=0)
                pay_period = st.selectbox("Pay period", PAY_PERIODS, index=0)
                industry = st.selectbox("Industry", INDUSTRIES, index=0)

                city_choice = st.selectbox(
                    "City",
                    CITY_OPTIONS,
                    index=0,
                    help="Top cities are listed to reduce input noise. Use custom city only if needed.",
                )
                if city_choice == "Other / custom city":
                    city = st.text_input("Custom city", value="San Francisco")
                else:
                    city = city_choice

                state = st.selectbox("US state", US_STATES, index=0)
                company_size = st.selectbox("Company size", COMPANY_SIZES, index=5)
                is_remote = st.checkbox("Remote position", value=False)

                skills_sel = st.multiselect(
                    "Skills / job functions",
                    TOP_SKILLS,
                    default=["Information Technology", "Engineering"],
                )

                selected_benefits = st.multiselect(
                    "Benefits offered by employer",
                    BENEFIT_OPTIONS,
                    default=["Medical insurance", "401(k)", "Paid time off"],
                    help=(
                        "These are employer-side package features, not personal qualifications. "
                        "The app automatically converts your selections into benefit_count and benefit indicators."
                    ),
                )
                st.caption(f"Selected benefit count: {len(selected_benefits)}")

                description = st.text_area(
                    "Short job description / responsibilities",
                    value=(
                        "Responsible for building scalable products, collaborating with cross-functional teams, "
                        "and improving system performance."
                    ),
                    height=90,
                )

                with st.expander("Optional model signal inputs"):
                    views = st.number_input("Estimated views", min_value=0, value=500, step=50)
                    applies = st.number_input("Estimated applications", min_value=0, value=30, step=5)
                    employee_count = st.number_input("Estimated employee count", min_value=0, value=1000, step=100)
                    follower_count = st.number_input("Estimated company followers", min_value=0, value=5000, step=500)
                    company_job_count = st.number_input("Estimated active job count from this company", min_value=0, value=5, step=1)

                submitted = st.form_submit_button("🔮 Predict Salary", width="stretch")

        if submitted:
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
                "description": description,
                "views": views,
                "applies": applies,
                "employee_count": employee_count,
                "follower_count": follower_count,
                "company_job_count": company_job_count,
            }

            signature = input_signature(payload)

            with st.spinner("Recomputing salary prediction..."):
                if obj is None:
                    st.session_state["salary_prediction_error"] = (
                        "Model file could not be loaded. Expected: models/best_salary_model_03_advanced_progress.joblib"
                    )
                    if load_error:
                        st.session_state["salary_prediction_error"] += f"\nDetails: {load_error}"
                elif not feature_cols:
                    st.session_state["salary_prediction_error"] = (
                        "The model does not contain feature_cols. Re-save the joblib from the model_03 script."
                    )
                else:
                    try:
                        X_pred = build_prediction_row(
                            feature_cols=feature_cols,
                            title=job_title,
                            description=description,
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
                            "low": pred_salary * 0.85,
                            "high": pred_salary * 1.15,
                            "model_name": model_name,
                            "model_type": model_type,
                            "signature": signature,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "payload": payload,
                        }
                        st.session_state.pop("salary_prediction_error", None)
                    except Exception as exc:
                        st.session_state["salary_prediction_error"] = str(exc)

        with col_result:
            if "salary_prediction_error" in st.session_state:
                st.error(st.session_state["salary_prediction_error"])

            result = st.session_state.get("salary_prediction_result")

            if result:
                st.success("### Estimated Annual Salary")
                st.caption(
                    f"Model: {result['model_name']} · Recalculated at {result['timestamp']} · "
                    f"Input signature: {result['signature']}"
                )

                st.markdown(
                    f"""
                    <div class="metric-card" style="margin:8px 0">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value" style="font-size:36px;color:#10B981">
                            ${result['salary']:,.0f}
                        </div>
                        <div class="metric-delta pos">
                            Practical range: ${result['low']:,.0f} — ${result['high']:,.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                mae = 16004
                rmse = 25225
                salary = result["salary"]
                st.markdown("**Estimated salary bands**")
                st.caption(
                    "These are model-error bands, not hiring probabilities. The dataset contains job postings, "
                    "not records of who was hired or which offer was accepted."
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Band": "Expected zone",
                                "Interpretation": "Prediction ± MAE",
                                "Range": f"${max(0, salary - mae):,.0f} — ${salary + mae:,.0f}",
                            },
                            {
                                "Band": "Broad zone",
                                "Interpretation": "Prediction ± RMSE",
                                "Range": f"${max(0, salary - rmse):,.0f} — ${salary + rmse:,.0f}",
                            },
                            {
                                "Band": "Conservative zone",
                                "Interpretation": "Prediction ± 2×RMSE",
                                "Range": f"${max(0, salary - 2*rmse):,.0f} — ${salary + 2*rmse:,.0f}",
                            },
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

                positives, cautions = prediction_explanation(result["payload"])
                strength, strength_color = benefit_package_strength(result["payload"].get("selected_benefits", []))

                st.markdown("**Why this prediction may be high or low**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Higher-salary signals**")
                    if positives:
                        for item in positives:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("- No strong positive rule-based signal detected.")
                with col_b:
                    st.markdown("**Caution / uncertainty signals**")
                    if cautions:
                        for item in cautions:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("- No major caution signal detected.")

                insight(f"Benefit package strength: **{strength}**. Benefits are employer-provided package features, not personal qualifications.", strength_color)

                st.markdown("**Similar postings salary distribution**")
                similar_df, similar_rule = similar_postings_analysis(result["payload"])
                if len(similar_df) > 0:
                    q = similar_df["normalized_salary"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                    st.caption(f"Matched set: {similar_rule} · Similar postings found: {len(similar_df):,}")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Statistic": ["P10", "P25", "Median", "P75", "P90"],
                                "Salary": [f"${v:,.0f}" for v in q.values],
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    bins = [0, 50_000, 75_000, 100_000, 130_000, 160_000, 200_000, 300_000]
                    labels = ["<$50k", "$50k–75k", "$75k–100k", "$100k–130k", "$130k–160k", "$160k–200k", "$200k+"]
                    band_counts = (
                        pd.cut(similar_df["normalized_salary"], bins=bins, labels=labels, include_lowest=True)
                        .value_counts(normalize=True)
                        .sort_index()
                        .mul(100)
                        .round(1)
                    )
                    st.markdown("**Share of similar postings by salary band**")
                    st.dataframe(
                        band_counts.rename("Share of postings (%)").reset_index().rename(columns={"index": "Salary band"}),
                        width="stretch",
                        hide_index=True,
                    )
                    st.bar_chart(band_counts)

                p = result["payload"]
                st.markdown(
                    f"""
                    | Input | Value |
                    |---|---|
                    | Job title | {p['job_title']} |
                    | Experience | {p['exp_level']} |
                    | Industry | {p['industry']} |
                    | Location | {p['city']}, {p['state']} |
                    | Pay period | {p['pay_period']} |
                    | Work type | {p['work_type']} |
                    | Company size | {p['company_size']} |
                    | Remote | {'Yes' if p['is_remote'] else 'No'} |
                    | Selected skills | {len(p['skills_sel'])} |
                    | Benefits offered | {p['benefit_count']} selected |
                    | Benefit types | {", ".join(p.get('selected_benefits', [])) if p.get('selected_benefits') else "None"} |
                    """
                )
            else:
                st.info("Fill the form and click **Predict Salary**.")
                st.markdown(
                    """
                    **How this works**
                    - The model itself is cached for speed.
                    - The prediction is not cached.
                    - Every click builds a new input row and recomputes the result.
                    - Job title is free text because the model uses title text features.
                    - City is mostly dropdown-based to reduce noisy inputs.
                    - Benefits are employer-side package features, not personal qualifications.
                    - The dataset does not include who was hired, so the app estimates salary bands, not hiring probability.
                    """
                )

    with tab2:
        section("Skill Salary Premium")
        insight("Skill premium is analyzed using salary differences and statistical testing.", "green")
        explain("What this section tells you", ["It compares salaries for postings that mention a skill versus those that do not.", "Treat the result as association, not guaranteed causal impact."])
        col1, col2 = st.columns(2)
        with col1:
            img("42_skill_salary_premium.png", "Skill salary premium")
        with col2:
            img("43_skill_salary_boxplot.png", "Top skill salary distribution")
        col1, col2 = st.columns(2)
        with col1:
            img("44_exp_skill_salary_heatmap.png", "Experience × skill salary matrix")
        with col2:
            img("46_skill_count_salary.png", "Skill count vs salary")

    with tab3:
        section("Career Ladder")
        explain("What this section tells you", ["The charts show how salary changes across experience levels.", "Use it to understand expected salary progression rather than exact individual outcomes."])
        col1, col2 = st.columns(2)
        with col1:
            img("52_career_ladder_violin.png", "Salary distribution by experience level")
        with col2:
            img("55_salary_growth_rate.png", "Salary growth rate by segment")
        img("53_career_ladder_by_industry.png", "Career ladder by industry")

    with tab4:
        section("Competition Analysis")
        insight("Competition score = applications / views. Higher scores imply more competitive postings.", "red")
        explain("What this section tells you", ["High competition means many applications relative to views.", "The dataset has applications/views, but it does not contain actual hiring outcomes."])
        col1, col2 = st.columns(2)
        with col1:
            img("47_most_competitive_titles.png", "Most competitive job titles")
        with col2:
            img("49_competition_by_experience.png", "Competition by experience level")
        col1, col2 = st.columns(2)
        with col1:
            img("51_low_competition_opportunities.png", "Low-competition high-salary opportunities")
        with col2:
            img("50_competition_remote_vs_office.png", "Remote vs non-remote competition")


# =============================================================================
# HR / RECRUITING
# =============================================================================
elif "HR" in page:
    st.title("🏢 HR / Recruiting Perspective")
    st.markdown("*Market benchmarks, competition analysis, company profiles, and benefit strategy.*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Salary Benchmark",
        "🏆 Competition",
        "🏗️ Company Analysis",
        "🎁 Benefits",
    ])

    with tab1:
        section("Market Salary Benchmark")
        explain("What this section tells you", ["These charts help compare compensation expectations by experience, industry, and company size.", "They are useful for salary benchmarking and offer calibration."])
        col1, col2 = st.columns(2)
        with col1:
            img("05_salary_by_experience.png", "Median salary by experience level")
        with col2:
            img("17_salary_band_experience.png", "Experience × salary band")
        img("18_salary_band_industry.png", "Industry × salary band")
        col1, col2 = st.columns(2)
        with col1:
            img("28_company_size_salary.png", "Salary by company size")
        with col2:
            img("69_salary_gap_by_industry.png", "Negotiation gap by industry")

    with tab2:
        section("Competition and Job Position Analysis")
        explain("What this section tells you", ["This section shows which titles and industries attract more applications.", "It helps HR understand where hiring may require stronger compensation or sourcing effort."])
        col1, col2 = st.columns(2)
        with col1:
            img("48_competition_by_industry.png", "Competition score by industry")
        with col2:
            img("47_most_competitive_titles.png", "Most competitive job titles")
        col1, col2 = st.columns(2)
        with col1:
            img("01_experience_distribution.png", "Experience-level demand")
        with col2:
            img("20_worktype_experience.png", "Work type × experience")
        img("45_industry_skill_salary_heatmap.png", "Industry × skill salary matrix")

    with tab3:
        section("Company Profile Analysis")
        explain("What this section tells you", ["Company size and hiring volume are used to compare employer segments.", "These are market signals, not judgments about individual company quality."])
        col1, col2 = st.columns(2)
        with col1:
            img("27_top_companies.png", "Companies with the most postings")
        with col2:
            img("30_top_paying_companies.png", "Top-paying companies")
        col1, col2 = st.columns(2)
        with col1:
            img("29_company_size_industry.png", "Company size × industry")
        with col2:
            img("31_company_size_experience.png", "Company size × experience")

    with tab4:
        section("Benefits and Compensation Package")
        explain("What this section tells you", ["Benefits are employer-provided package features.", "More benefits may indicate a more structured compensation package, but benefit count alone does not prove higher salary."])
        col1, col2 = st.columns(2)
        with col1:
            img("37_top_benefits.png", "Most common benefits")
        with col2:
            img("38_benefits_by_company_size.png", "Benefits by company size")
        col1, col2 = st.columns(2)
        with col1:
            img("39_benefits_by_industry.png", "Benefits by industry")
        with col2:
            img("40_benefit_count_salary.png", "Benefit count vs salary")
        img("41_benefits_by_experience.png", "Benefits by experience level")


# =============================================================================
# EDUCATION
# =============================================================================
elif "Education" in page:
    st.title("🎓 Education Perspective")
    st.markdown("*Market demand signals for curriculum planning and career preparation.*")

    tab1, tab2, tab3 = st.tabs(["📚 Skill Demand", "🎓 Requirements", "🚀 Early Career"])

    with tab1:
        section("Most Demanded Skills")
        explain("What this section tells you", ["These charts identify skills and job functions that appear most often in postings.", "They can guide curriculum design and career preparation."])
        col1, col2 = st.columns(2)
        with col1:
            img("07_top_20_skills.png", "Top 20 demanded skills")
        with col2:
            img("09_skill_by_experience_heatmap.png", "Experience × skill demand")
        img("11_skill_by_industry_stacked.png", "Skill composition by industry")
        col1, col2 = st.columns(2)
        with col1:
            img("08_top_industries.png", "Top hiring industries")
        with col2:
            img("10_top_paying_skills.png", "Top-paying skills")

    with tab2:
        section("Degree and Job Requirement Signals")
        insight("A 500-posting sample was analyzed with Gemini for degree requirements, soft skills, urgency, and tech-role signals.", "amber")
        explain("What this section tells you", ["LLM analysis extracts qualitative requirements from descriptions.", "This is a sampled analysis and should be interpreted as directional evidence."])
        col1, col2 = st.columns(2)
        with col1:
            img("12_degree_requirement.png", "Degree requirement")
        with col2:
            img("16_degree_by_experience.png", "Degree requirement by experience")
        col1, col2 = st.columns(2)
        with col1:
            img("13_soft_skills.png", "Soft skill demand")
        with col2:
            img("15_tech_vs_nontech_salary.png", "Tech vs non-tech salary")

    with tab3:
        section("Early Career Signals")
        col1, col2 = st.columns(2)
        with col1:
            img("52_career_ladder_violin.png", "Salary distribution by career level")
        with col2:
            img("70_salary_gap_by_exp.png", "Salary range gap by experience")
        col1, col2 = st.columns(2)
        with col1:
            img("55_salary_growth_rate.png", "Salary growth by segment")
        with col2:
            img("46_skill_count_salary.png", "Skill count vs salary")


# =============================================================================
# INVESTOR
# =============================================================================
elif "Investor" in page:
    st.title("📈 Investor Perspective")
    st.markdown("*Sector growth signals, talent concentration, and company segmentation.*")

    tab1, tab2, tab3 = st.tabs(["🏭 Sector Signals", "🗺️ Geography", "💼 Company Segmentation"])

    with tab1:
        section("Sector Hiring Volume and Salary Profile")
        explain("What this section tells you", ["High posting volume suggests active hiring demand.", "Salary bands and competition scores help compare sector attractiveness."])
        col1, col2 = st.columns(2)
        with col1:
            img("08_top_industries.png", "Top hiring industries")
        with col2:
            img("18_salary_band_industry.png", "Industry × salary band")
        img("29_company_size_industry.png", "Industry × company size")
        col1, col2 = st.columns(2)
        with col1:
            img("15_tech_vs_nontech_salary.png", "Tech vs non-tech salary")
        with col2:
            img("48_competition_by_industry.png", "Talent competition by industry")

    with tab2:
        section("Geographic Talent Concentration")
        explain("What this section tells you", ["The maps/charts compare job concentration and salary by location.", "Use this to identify regional labor-market clusters."])
        col1, col2 = st.columns(2)
        with col1:
            img("22_state_median_salary.png", "Median salary by state")
        with col2:
            img("04_top_states.png", "Top states by postings")
        img("25_state_industry_heatmap.png", "State × industry heatmap")
        col1, col2 = st.columns(2)
        with col1:
            img("24_top_cities.png", "Top cities by postings")
        with col2:
            img("19_state_remote_salary_bubble.png", "Remote rate vs salary by state")

    with tab3:
        section("Company Segmentation")
        col1, col2 = st.columns(2)
        with col1:
            img("27_top_companies.png", "Most active hiring companies")
        with col2:
            img("30_top_paying_companies.png", "Top-paying companies")
        col1, col2 = st.columns(2)
        with col1:
            img("28_company_size_salary.png", "Company size vs salary")
        with col2:
            img("31_company_size_experience.png", "Company size × experience")


# =============================================================================
# POLICY MAKER
# =============================================================================
elif "Policy" in page:
    st.title("🏛️ Policy Maker Perspective")
    st.markdown("*Employment geography, salary transparency, regional inequality, and remote work patterns.*")

    tab1, tab2, tab3 = st.tabs(["🗺️ Regional Inequality", "⚖️ Salary Transparency", "🏠 Work Models"])

    with tab1:
        section("Regional Employment and Salary Inequality")
        explain("What this section tells you", ["The charts compare salary and job patterns across states.", "Differences may reflect industry mix, cost of labor, and concentration of high-paying roles."])
        img("22_state_median_salary.png", "State-level median salary")
        col1, col2 = st.columns(2)
        with col1:
            img("26_state_experience.png", "Experience profile by state")
        with col2:
            img("23_state_remote_rate.png", "Remote rate by state")
        img("25_state_industry_heatmap.png", "State × industry distribution")

    with tab2:
        section("Salary Transparency and Negotiation Gap")
        explain("What this section tells you", ["Salary gap means max_salary minus min_salary in listed salary ranges.", "A wider gap may indicate more negotiation room or less precise compensation transparency."])
        col1, col2 = st.columns(2)
        with col1:
            img("68_salary_gap_distribution.png", "Salary range gap distribution")
        with col2:
            img("71_salary_band_vs_gap.png", "Salary level vs range gap")
        img("72_salary_gap_negotiation_map.png", "Industry × experience negotiation map")
        col1, col2 = st.columns(2)
        with col1:
            img("69_salary_gap_by_industry.png", "Salary range gap by industry")
        with col2:
            img("70_salary_gap_by_exp.png", "Salary range gap by experience")

    with tab3:
        section("Remote Work and Employment Models")
        col1, col2 = st.columns(2)
        with col1:
            img("03_remote_vs_office.png", "Remote vs non-remote distribution")
        with col2:
            img("06_remote_by_company_size.png", "Remote rate by company size")
        col1, col2 = st.columns(2)
        with col1:
            img("59_remote_by_size_salary.png", "Company size: remote rate and salary")
        with col2:
            img("61_remote_exp_salary_grouped.png", "Experience × work model salary")
        img("60_remote_state_scatter.png", "Remote rate vs median salary by state")


# =============================================================================
# RESEARCHER
# =============================================================================
elif "Researcher" in page:
    st.title("🔬 Researcher Perspective")
    st.markdown("*Model transparency, SHAP analysis, diagnostics, and methodology.*")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Model Comparison",
        "🧠 SHAP",
        "🔍 Feature Engineering",
        "🧪 Diagnostics",
        "🧾 Data Quality",
        "📖 Methodology",
    ])

    with tab1:
        section("Model Performance Comparison")
        explain("What this section tells you", ["Model performance is evaluated using cross-validation / out-of-fold results.", "Final selection is based on validation performance, not training performance."])
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Model 01 Raw R²", "0.653", "feature-rich baseline")
        with c2:
            metric_card("Model 02 Raw R²", "0.711", "CV-safe pipeline")
        with c3:
            metric_card("Model 03 Raw R²", "0.757", "advanced ensemble")
        with c4:
            metric_card("Best RMSE", "$25,225", "Model 03 OOF ensemble")

        col1, col2 = st.columns(2)
        with col1:
            img("model_01_comparison.png", "Model 01 comparison")
        with col2:
            img("model_02_baseline_vs_tuned.png", "Model 02 baseline vs tuned")
        col1, col2 = st.columns(2)
        with col1:
            img("model_03_advanced_results.png", "Model 03 final results")
        with col2:
            img("model_03_tuning_curves.png", "Model 03 Optuna tuning curves")

        insight(
            "Model 03 improved performance through cleaned description text features, title clustering, interaction features, "
            "CV-safe target encoding, and an OOF weighted ensemble.",
            "green",
        )

    with tab2:
        section("SHAP Explainability")
        insight("SHAP/dependence analysis is used to explain which features most influence salary predictions.", "green")

        col1, col2 = st.columns(2)
        with col1:
            img("62_shap_beeswarm.png", "SHAP beeswarm")
        with col2:
            img("63_shap_bar.png", "SHAP bar importance")

        st.markdown("**SHAP dependence plots**")
        col1, col2 = st.columns(2)
        with col1:
            img("65_shap_dependence_follower_count.png", "follower_count effect")
            img("67_shap_dependence_desc_svd_6.png", "description SVD effect")
        with col2:
            img("66_shap_dependence_applies.png", "applies effect")
            img("68_shap_dependence_exp_title_cluster.png", "experience × title cluster effect")

        insight(
            "Engagement features such as follower_count and applies are correlational, not causal. "
            "High-paying postings may naturally receive more views and applications.",
            "red",
        )

    with tab3:
        section("Feature Engineering Layers")
        col1, col2 = st.columns(2)
        with col1:
            img("model_01_shap_summary.png", "Model 01 SHAP summary")
        with col2:
            img("model_01_shap_importance.png", "Model 01 SHAP importance")
        img("model_02_param_importance.png", "Model 02 Optuna parameter importance")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                **Categorical**
                - experience level
                - work type
                - primary industry
                - pay period
                - company size
                - state and title cluster
                """
            )
        with col2:
            st.markdown(
                """
                **Text**
                - title TF-IDF + SVD
                - cleaned description TF-IDF + SVD
                - salary/compensation leakage removal
                """
            )
        with col3:
            st.markdown(
                """
                **CV-safe target encoding**
                - city × industry
                - state × industry
                - experience × industry
                - experience × title cluster
                - remote × industry
                """
            )

    with tab4:
        section("Model Diagnostics")
        explain("What this section tells you", ["Diagnostics check overfitting, residual patterns, and fold stability.", "The model overfits on training data, but fold validation is stable, so OOF performance is the reliable number."])
        insight(
            "Diagnostics showed Train R²=0.985 and OOF R²=0.757, indicating overfitting. "
            "However, fold-level validation was stable with Fold R² std=0.0081. Final selection is based on OOF/CV performance.",
            "amber",
        )

        col1, col2 = st.columns(2)
        with col1:
            img("73_diag_train_vs_oof.png", "Train vs OOF comparison")
        with col2:
            img("74_diag_learning_curve.png", "Learning curve")

        col1, col2 = st.columns(2)
        with col1:
            img("75_diag_residual_by_range.png", "Residuals by salary range")
        with col2:
            img("76_diag_error_distribution.png", "Error distribution")
        img("77_diag_fold_stability.png", "Fold stability")

    with tab5:
        section("Data Quality and Dashboard Coverage")
        explain(
            "What this section tells you",
            [
                "This panel shows how much of the dataset is available for salary modeling and dashboard analysis.",
                "It also checks whether the PNG charts referenced by the dashboard exist in the outputs folder.",
            ],
        )

        st.markdown("**Dataset coverage**")
        dq = data_quality_summary()
        if dq.empty:
            st.warning("Data quality summary could not be loaded.")
        else:
            st.dataframe(dq, width="stretch", hide_index=True)

        st.markdown("**Chart file checker**")
        inv = chart_inventory()
        if inv.empty:
            st.warning("Chart inventory could not be generated.")
        else:
            c1, c2, c3 = st.columns(3)
            found = int((inv["Status"] == "Found").sum())
            missing = int((inv["Status"] == "Missing").sum())
            with c1:
                metric_card("Referenced charts", f"{len(inv):,}")
            with c2:
                metric_card("Found", f"{found:,}")
            with c3:
                metric_card("Missing", f"{missing:,}", delta_pos=False)

            if missing:
                st.dataframe(inv[inv["Status"] == "Missing"], width="stretch", hide_index=True)
            else:
                st.success("All referenced dashboard chart files were found.")

            st.download_button(
                "Download chart inventory",
                data=inv.to_csv(index=False).encode("utf-8"),
                file_name="dashboard_chart_inventory.csv",
                mime="text/csv",
            )

    with tab6:
        section("Methodology and Dataset")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Dataset**
                - Source: Kaggle — arshkon/linkedin-job-postings
                - Size: 123,849 job postings
                - Tables: postings, companies, job_skills, skills, job_industries, industries, benefits, employee_counts, salaries
                - Snapshot: April 2024

                **Analysis**
                - 13 analysis scripts
                - 77+ generated charts
                - Mann-Whitney U test for skill premium
                - TF-IDF + K-Means title clustering
                - Gemini API for LLM-assisted description analysis
                """
            )
        with col2:
            st.markdown(
                """
                **Modeling**
                - Target: normalized_salary (USD)
                - Usable salary rows: 35,279
                - Final model: Model 03 OOF ensemble
                - Algorithms: LightGBM, XGBoost, CatBoost
                - Leakage prevention: CV-safe target encoding and salary-text cleanup

                **Limitations**
                - Salary values are missing for many postings
                - The dataset does not show who was hired or which salary offer was accepted
                - remote_allowed=0 can mean non-remote or missing remote information
                - The data is mostly US-centered
                - Engagement features are correlational, not causal
                """
            )

        st.markdown("---")
        st.markdown("**Project structure**")
        st.code(
            """
LinkedInJobAnalysis/
├── data/
├── models/
├── outputs/
├── analysis_01-13.py
├── model_01-03.py
├── shap_dependence.py
├── model_03_diagnostics.py
├── utils_progress.py
└── streamlit_app.py
            """,
            language="text",
        )
