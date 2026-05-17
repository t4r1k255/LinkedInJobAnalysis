# LinkedIn Job Postings — Advanced Analysis & Salary Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Dataset-123%2C849%20postings-informational?style=flat-square" />
  <img src="https://img.shields.io/badge/Best%20R²-0.757-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Charts-77%2B-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

A comprehensive data analysis and machine learning project on **123,849 LinkedIn job postings** from the United States. The project covers market exploration, skill demand analysis, salary prediction, SHAP-based model explainability, and an interactive Streamlit dashboard with six distinct stakeholder perspectives.

---

## Contents

- [Project Highlights](#project-highlights)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Analysis Modules](#analysis-modules)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [SHAP Explainability](#shap-explainability)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Setup](#setup)
- [Running Order](#running-order)
- [Key Findings](#key-findings)
- [Limitations](#limitations)

---

## Project Highlights

| Category | Detail |
|---|---|
| Dataset | 123,849 LinkedIn job postings · 9 relational tables · Kaggle 2024 snapshot |
| Analysis | 13 scripts · 77+ charts · 6 analysis dimensions |
| Modeling | XGBoost · LightGBM · CatBoost · OOF Ensemble |
| Best model | R² = **0.757** · RMSE = **$25,225** · Log R² = **0.807** |
| Text features | TF-IDF + SVD on titles and descriptions · leakage-safe cleaning |
| Explainability | SHAP TreeExplainer · beeswarm · bar · dependence plots |
| Dashboard | Streamlit · 6 perspectives · live salary prediction |

---

## Dataset

**Source:** [Kaggle — arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

Download the dataset from Kaggle and place all CSV files inside a `data/` folder at the project root.

| File | Rows | Description |
|---|---|---|
| `postings.csv` | 123,849 | Job title, salary, location, experience, work type, views, applies |
| `companies.csv` | 24,473 | Company name, size, state, country |
| `job_skills.csv` | 993,285 | Job-to-skill mappings |
| `skills.csv` | 35,645 | Skill abbreviation → name lookup |
| `job_industries.csv` | 139,699 | Job-to-industry mappings |
| `industries.csv` | 149 | Industry ID → name lookup |
| `salaries.csv` | 50,854 | Min/max/med salary, pay period, currency |
| `benefits.csv` | 254,898 | Benefit types by job posting |
| `employee_counts.csv` | 186,027 | Company headcount and follower history |

> The dataset is not included in this repository. Create a local `data/` directory and place the CSV files there.

---

## Project Structure

```
LinkedInJobAnalysis/
│
├── data/                                   # Kaggle CSV files (not tracked)
├── outputs/                                # Generated PNG charts (not tracked)
├── models/                                 # Saved .joblib model files (not tracked)
│
├── data_loader.py                          # Schema inspection and data profiling
├── data_cleaning.py                        # Cleaning pipeline with summary stats
│
├── analysis_01_market.py                   # Market overview (6 charts)
├── analysis_02_skills.py                   # Skill demand analysis (5 charts)
├── analysis_03_llm.py                      # Gemini API job description analysis (5 charts)
├── analysis_04_crosssection.py             # Cross-sectional salary/skill analysis (5 charts)
├── analysis_05_geo.py                      # Geographic distribution (5 charts)
├── analysis_06_company.py                  # Company-level analysis (5 charts)
├── analysis_07_title_clustering.py         # TF-IDF + K-Means title clustering (5 charts)
├── analysis_08_benefits.py                 # Benefits and perks analysis (5 charts)
├── analysis_09_skill_salary.py             # Skill salary premium — Mann-Whitney U (5 charts)
├── analysis_10_competition.py              # Competition score: applies/views (5 charts)
├── analysis_11_career_ladder.py            # Career progression and salary growth (5 charts)
├── analysis_12_remote.py                   # Remote work patterns (5 charts)
├── analysis_13_salary_gap.py               # Salary negotiation range analysis (5 charts)
│
├── model_01_salary_v3_final.py             # Feature-rich baseline — XGB/LGB/Cat + SHAP
├── model_02_pipeline_v3_cv_safe.py         # CV-safe sklearn Pipeline + Optuna tuning
├── model_03_salary_advanced_progress.py    # Advanced ensemble: TF-IDF/SVD + interactions
├── model_04_stacking_intervals.py          # Ridge meta-learner stacking + prediction intervals
├── model_03_diagnostics.py                 # Overfitting, residual, learning-curve diagnostics
│
├── prepare_shap_data.py                    # Feature engineering for SHAP (run before shap_dependence)
├── shap_dependence.py                      # SHAP beeswarm, bar, and dependence plots
│
├── streamlit_app.py                        # Interactive dashboard (6 perspectives)
├── utils_progress.py                       # Shared progress bar utility (ProgressBar, StepTracker)
│
├── requirements.txt
└── README.md
```

---

## Analysis Modules

Each script loads the full 123k dataset (no sampling), produces charts to `outputs/`, and is instrumented with `utils_progress.py` for step-by-step progress tracking.

### 01 · Market Overview
`analysis_01_market.py`

High-level picture of the dataset: experience-level distribution, work type breakdown, remote vs on-site split, top states by volume, salary by experience level, and remote rate by company size.

### 02 · Skill Demand
`analysis_02_skills.py`

Frequency analysis of the 35k+ distinct skills in the dataset. Top-20 skills overall, top-paying skills (median salary), skill demand by experience level, and skill composition by industry.

### 03 · LLM-Assisted Analysis
`analysis_03_llm.py`

Uses the **Gemini API** on a stratified sample of 500 job descriptions to extract structured signals not available in the structured columns: degree requirement, soft skill type, hiring urgency, and tech vs non-tech classification. Results cached to `outputs/llm_results.csv`.

> Requires `GEMINI_API_KEY` in a `.env` file.

### 04 · Cross-Section
`analysis_04_crosssection.py`

Multi-dimensional salary analysis: salary band × experience, salary band × industry, remote ratio vs salary by state (bubble chart), work type × experience, and skill × salary band heatmap.

### 05 · Geographic Analysis
`analysis_05_geo.py`

State-level salary ranking, remote rate by state, top cities by volume, state × industry heatmap, and experience distribution by state.

### 06 · Company Analysis
`analysis_06_company.py`

Top hiring companies, salary by company size (1-10 to 10K+), company size × industry heatmap, top-paying companies (min 20 postings), and company size × experience distribution.

### 07 · Title Clustering
`analysis_07_title_clustering.py`

TF-IDF vectorization of job titles (300 features, bigrams) followed by K-Means clustering (k=8). PCA 2D visualization on 10k sample, salary and experience profile by cluster, remote rate by cluster.

### 08 · Benefits Analysis
`analysis_08_benefits.py`

Most common benefits across 254k records, average benefit count by company size, benefit composition by industry and experience level, and benefit count vs median salary.

### 09 · Skill Salary Premium
`analysis_09_skill_salary.py`

For every skill with ≥200 postings, computes the median salary difference between postings with and without that skill. Uses **Mann-Whitney U** test (p<0.05) to filter statistically significant premiums. Includes experience × skill and industry × skill salary matrices.

### 10 · Competition Analysis
`analysis_10_competition.py`

Derives a competition score as `applies / views`. Identifies the most competitive job titles and industries, competition by experience level, remote vs on-site comparison, and a low-competition / high-salary opportunity scatter.

### 11 · Career Ladder
`analysis_11_career_ladder.py`

Salary distribution by experience level (violin), industry-specific career trajectories (line chart), company size × experience salary matrix (heatmap), entry→mid-senior growth rate by industry, and remote vs on-site career ladder comparison.

### 12 · Remote Work Analysis
`analysis_12_remote.py`

Remote salary premium by industry, remote vs on-site skill profiles, remote rate by company size (dual-axis), state-level remote rate vs salary scatter with trend line, and experience × work type salary grouped bars.

### 13 · Salary Gap / Negotiation Range
`analysis_13_salary_gap.py`

Uses `salaries.csv` (21k yearly USD records) to analyze the width of posted salary ranges as a negotiation signal. Gap distribution, gap by industry and experience, gap by salary band, and a sector × experience negotiation heatmap.

---

## Machine Learning Pipeline

Salary prediction is developed in three stages, each building on the previous.

### Stage 1 — Feature-Rich Baseline
**File:** `model_01_salary_v3_final.py`

Establishes a strong baseline with hand-crafted features: skills multi-hot (top 20), company size, title keyword flags, experience level, industry, location. Evaluates three models with 5-fold CV.

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| XGBoost | $30,118 | $20,033 | 0.653 | 0.717 |
| LightGBM | $30,237 | $20,090 | 0.651 | 0.715 |
| CatBoost | $30,942 | $20,720 | 0.634 | 0.700 |

---

### Stage 2 — CV-Safe Pipeline + Optuna
**File:** `model_02_pipeline_v3_cv_safe.py`

Wraps preprocessing in a `sklearn.Pipeline` to prevent target leakage. Adds CV-safe median target encoding, title TF-IDF + SVD (25 components), and Optuna hyperparameter search (40 trials per model). Best pipeline saved to `models/best_salary_pipeline.joblib`.

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| XGBoost | $27,562 | $17,726 | 0.7098 | 0.7630 |
| **LightGBM** | **$27,494** | **$17,839** | **0.7112** | **0.7636** |
| CatBoost | $28,434 | $18,707 | 0.6912 | 0.7492 |

---

### Stage 3 — Advanced Ensemble (Final)
**File:** `model_03_salary_advanced_progress.py`

The production model. Key additions over Stage 2:

- **Salary leakage removal** from job descriptions (sentence-level filter, vectorized)
- **Description TF-IDF + SVD** (100 components, after leakage cleaning)
- **10 interaction features** (city × industry, exp × title cluster, etc.)
- **CV-safe target encoding** with smoothing
- **OOF inverse-RMSE weighted ensemble** across all three models
- Saved to `models/best_salary_model_03_advanced_progress.joblib`

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| LightGBM | $25,338 | $15,983 | 0.7547 | 0.8048 |
| XGBoost | $25,616 | $16,386 | 0.7493 | 0.7995 |
| CatBoost | $25,663 | $16,487 | 0.7484 | 0.7980 |
| **OOF Ensemble** | **$25,225** | **$16,004** | **0.7570** | **0.8065** |

---

### Stage 4 — Stacking + Prediction Intervals
**File:** `model_04_stacking_intervals.py`

Ridge meta-learner trained on OOF predictions from Stage 3 models. Adds quantile regression for prediction intervals and a fairness/error analysis by salary range.

---

### Diagnostics
**File:** `model_03_diagnostics.py`

| Metric | Value |
|---|---:|
| Train R² | 0.9850 |
| OOF R² | 0.7570 |
| Overfit gap | 0.2280 |
| Fold R² std | 0.0081 |
| OOF RMSE | $25,225 |

The train/OOF gap (0.228) indicates overfitting in the training fit, but fold stability (std = 0.008) confirms reliable generalization. All reported metrics are OOF/CV-based.

---

## SHAP Explainability

**Files:** `prepare_shap_data.py` → `shap_dependence.py`

Run `prepare_shap_data.py` first to build the feature matrix and save it to `models/shap_X.parquet`. Then run `shap_dependence.py` to generate:

- **SHAP beeswarm plot** — feature impact distribution across 2k samples
- **SHAP bar plot** — mean |SHAP| importance ranking
- **SHAP dependence plots** — top 4 features: value vs SHAP value with trend line

```bash
python prepare_shap_data.py    # ~30 seconds
python shap_dependence.py      # ~2 minutes
```

> **Note:** `follower_count` and `applies` appear in the top features by SHAP value. These are correlational signals — higher-paying or more prominent postings attract more engagement — not causal salary drivers.

---

## Streamlit Dashboard

**File:** `streamlit_app.py`

```bash
streamlit run streamlit_app.py
```

Six stakeholder perspectives, each with dedicated tabs:

| Perspective | Key content |
|---|---|
| **Job Seeker** | Live salary prediction form · skill premium guide · career ladder · competition score |
| **HR / Recruiting** | Market salary benchmark · competition by role · company size profiles · benefits |
| **Education** | Top skills by demand · diploma requirement rates · skill × curriculum mapping |
| **Investor** | Sector hiring volume · geographic talent density · company size segmentation |
| **Policy Maker** | Regional salary inequality · salary gap / transparency · remote work policy signals |
| **Researcher** | Model comparison (01→03) · SHAP plots · feature importance · methodology · limitations |

The **live salary prediction** (Job Seeker tab) uses `models/best_salary_pipeline.joblib` and accepts: experience level, work type, industry, state, company size, remote flag, and up to 20 skills.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/t4r1k255/LinkedInJobAnalysis.git
cd LinkedInJobAnalysis
```

### 2. Create virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

Download from Kaggle and place all CSV files in:
```
data/
```

### 5. Gemini API key (optional)

Required only for `analysis_03_llm.py`. Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

---

## Running Order

```bash
# 1. Inspect and clean data
python data_loader.py
python data_cleaning.py

# 2. Run analysis scripts (independent, any order)
python analysis_01_market.py
python analysis_02_skills.py
python analysis_03_llm.py          # requires GEMINI_API_KEY
python analysis_04_crosssection.py
python analysis_05_geo.py
python analysis_06_company.py
python analysis_07_title_clustering.py
python analysis_08_benefits.py
python analysis_09_skill_salary.py
python analysis_10_competition.py
python analysis_11_career_ladder.py
python analysis_12_remote.py
python analysis_13_salary_gap.py

# 3. Train models (sequential — each builds on the previous)
python model_01_salary_v3_final.py
python model_02_pipeline_v3_cv_safe.py
python model_03_salary_advanced_progress.py
python model_04_stacking_intervals.py

# 4. Diagnostics and explainability
python model_03_diagnostics.py
python prepare_shap_data.py        # must run before shap_dependence
python shap_dependence.py

# 5. Dashboard
streamlit run streamlit_app.py
```

---

## Key Findings

A selection of the most actionable results across all analyses.

**Salary and experience**
- Median salary jumps from **$72k** (Entry level) to **$130k** (Director) — an 80% increase.
- The largest single step is Entry → Mid-Senior: approximately **$35–50k** depending on industry.

**Skill premium**
- Skills with the highest statistically significant salary premium include Engineering, Finance, and Information Technology (all p<0.05, Mann-Whitney U).
- Jobs listing 3–5 skills have higher median salaries than those listing 1–2 or 8+.

**Remote work**
- Remote postings represent ~12% of all listings but show a salary premium of **$15–20k** vs non-remote postings.
- Remote rate peaks at mid-size companies (1K–5K employees).

**Competition**
- Healthcare and nursing roles have the lowest competition scores (applies/views), while entry-level sales and administrative roles have the highest.
- Low-competition + high-salary opportunities cluster in niche engineering and specialized finance roles.

**Salary negotiation gap**
- Median posted salary range: **$30,000** (~32% of minimum salary).
- Financial services and staffing sectors show the widest negotiation gaps; construction and retail the narrowest.

**Modeling**
- Adding leakage-cleaned description text (100 SVD components) and interaction features improved R² from 0.711 to 0.757 — the single largest jump across all three stages.
- `follower_count` and `applies` are the top SHAP features but should be treated as correlational proxies, not causal salary drivers.

---

## Limitations

- **Salary coverage:** salary data is available for ~29% of postings (35,604 of 123,849). Results may not generalize to postings without salary disclosure.
- **US-centric:** nearly all postings are from the United States. Results do not apply to other labor markets.
- **Single snapshot:** the dataset reflects April 2024. Trends may not represent current conditions.
- **`remote_allowed` sparsity:** 87% of postings have no remote flag. Null cannot be reliably interpreted as "on-site."
- **Engagement features:** `views`, `applies`, and `follower_count` correlate with salary but the direction of causality is unclear. They should not be used in isolation to explain salary levels.
- **No hiring outcomes:** the dataset records job postings, not hires. Salary predictions reflect posted ranges, not offer or negotiated amounts.
- **Overfitting gap:** train R² (0.985) vs OOF R² (0.757) shows the model overfits in full-data fit. All reported metrics are CV-based.

---

## License

MIT — see [LICENSE](LICENSE) for details.
