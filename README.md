# LinkedIn Job Postings — Advanced Analysis & Salary Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Dataset-123%2C849%20postings-informational?style=flat-square" />
  <img src="https://img.shields.io/badge/Best%20Raw%20R²-0.759-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Charts-80%2B-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

A local-first data analysis and machine learning project on **123,849 LinkedIn job postings** from the Kaggle 2024 LinkedIn Job Postings dataset.

The project covers labor-market exploration, skill demand, salary patterns, company and geographic differences, benefits, competition, remote work, salary-gap analysis, salary prediction, SHAP explainability, prediction intervals, model diagnostics, and a multi-perspective Streamlit dashboard.

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
- [License](#license)

---

## Project Highlights

| Category | Detail |
|---|---|
| Dataset | 123,849 LinkedIn job postings · 9 relational tables · Kaggle 2024 snapshot |
| Analysis | 13 scripts · 80+ generated outputs · labor-market, skill, company, geo, remote, benefits, competition, and salary-gap analysis |
| Modeling | XGBoost · LightGBM · CatBoost · OOF ensemble · Ridge stacking |
| Best model | Model 04 · Raw R² ≈ **0.7593** · RMSE ≈ **$25,106** · MAE ≈ **$15,929** |
| Text features | TF-IDF + SVD on job titles and descriptions · salary-text leakage removal |
| Explainability | SHAP TreeExplainer · beeswarm · bar · dependence plots |
| Uncertainty | Residual quantile intervals · quantile regression intervals · conformal intervals |
| Dashboard | Streamlit · 6 stakeholder perspectives · Basic / Advanced modes · local-first usage |

---

## Dataset

**Source:** [Kaggle — arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

The dataset is **not included in this repository** because the raw files are large. Download the dataset from Kaggle and place the CSV files inside a `data/` folder at the project root.

| File | Rows | Description |
|---|---:|---|
| `postings.csv` | 123,849 | Main job-posting table: title, salary, location, experience, work type, views, applies |
| `companies.csv` | 24,473 | Company metadata such as company name, size, state, and country |
| `job_skills.csv` | 993,285 | Job-to-skill mapping table |
| `skills.csv` | 35,645 | Skill abbreviation / skill name lookup |
| `job_industries.csv` | 139,699 | Job-to-industry mapping table |
| `industries.csv` | 149 | Industry ID / industry name lookup |
| `salaries.csv` | 50,854 | Salary range table with min / max / median salary, pay period, and currency |
| `benefits.csv` | 254,898 | Benefit types by job posting |
| `employee_counts.csv` | 186,027 | Company headcount and follower history |

Expected local structure:

```text
LinkedInJobAnalysis/
└── data/
    ├── postings.csv
    ├── companies.csv
    ├── job_skills.csv
    ├── skills.csv
    ├── job_industries.csv
    ├── industries.csv
    ├── salaries.csv
    ├── benefits.csv
    └── employee_counts.csv
```

Main salary target:

```text
postings.csv -> normalized_salary
```

Final model salary filter:

```text
Salary range: $10,000 - $300,000
Currency: USD or missing
```

Known salary-usable row count:

```text
35,279 rows
```

---

## Project Structure

```text
LinkedInJobAnalysis/
│
├── data/                                   # Kaggle CSV files, not tracked
├── outputs/                                # Generated charts and lightweight output CSV/JSON files
├── models/                                 # Model registry tracked; large .joblib artifacts not tracked
├── docs/                                   # Project documentation
├── reports/                                # Optional report drafts
│
├── config.py                               # Shared paths and constants
├── data_loader.py                          # Schema inspection and data profiling
├── data_cleaning.py                        # Cleaning pipeline and summary stats
│
├── analysis_01_market.py                   # Market overview
├── analysis_02_skills.py                   # Skill demand analysis
├── analysis_03_llm.py                      # Gemini API job-description analysis
├── analysis_04_crosssection.py             # Cross-sectional salary / skill analysis
├── analysis_05_geo.py                      # Geographic distribution
├── analysis_06_company.py                  # Company-level analysis
├── analysis_07_title_clustering.py         # TF-IDF + K-Means title clustering
├── analysis_08_benefits.py                 # Benefits and perks analysis
├── analysis_09_skill_salary.py             # Skill salary premium analysis
├── analysis_10_competition.py              # Competition score: applies / views
├── analysis_11_career_ladder.py            # Career ladder and salary progression
├── analysis_12_remote.py                   # Remote work patterns
├── analysis_13_salary_gap.py               # Salary range / negotiation-gap analysis
│
├── model_01_salary_v3_final.py             # Feature-rich baseline model comparison
├── model_02_pipeline_v3_cv_safe.py         # CV-safe sklearn pipeline + Optuna tuning
├── model_03_salary_advanced_progress.py    # Advanced ensemble with text, interactions, and OOF weighting
├── model_04_stacking_intervals.py          # Ridge stacking + prediction intervals + segment error analysis
├── model_03_diagnostics.py                 # Overfitting, residual, and learning-curve diagnostics
│
├── prepare_shap_data.py                    # Feature matrix export for SHAP
├── shap_dependence.py                      # SHAP beeswarm, bar, and dependence plots
├── streamlit_app.py                        # Local Streamlit dashboard
├── utils_progress.py                       # Shared terminal progress utilities
│
├── requirements.txt
├── README.md
└── LICENSE
```

Additional documentation:

```text
docs/PROJECT_STRUCTURE.md
docs/RUN_ORDER.md
docs/DATASET.md
docs/MODEL_CARD.md
docs/DASHBOARD_GUIDE.md
docs/LIMITATIONS.md
```

---

## Analysis Modules

Each analysis script reads the local Kaggle dataset and saves outputs under `outputs/`.

### 01 · Market Overview
`analysis_01_market.py`

Experience-level distribution, work-type breakdown, remote vs on-site split, top states by volume, salary by experience level, and remote rate by company size.

### 02 · Skill Demand
`analysis_02_skills.py`

Top skills by demand, top industries, skill demand by experience level, highest-paying skills, and skill composition by industry.

### 03 · LLM-Assisted Analysis
`analysis_03_llm.py`

Uses the Gemini API on a stratified sample of job descriptions to extract structured signals such as degree requirement, soft-skill type, urgency, and tech vs non-tech classification.

> Requires `GEMINI_API_KEY` in a `.env` file.

### 04 · Cross-Section
`analysis_04_crosssection.py`

Multi-dimensional salary analysis: salary band × experience, salary band × industry, remote ratio vs salary by state, work type × experience, and skill × salary band heatmap.

### 05 · Geography
`analysis_05_geo.py`

State and city-level job distribution, salary differences, remote access, and geographic salary patterns.

### 06 · Company
`analysis_06_company.py`

Company size, company salary differences, top hiring companies, company-size × industry views, and company-level signals.

### 07 · Title Clustering
`analysis_07_title_clustering.py`

TF-IDF vectorization and K-Means clustering of job titles, with cluster-level salary and market interpretation.

### 08 · Benefits
`analysis_08_benefits.py`

Benefit count, benefit type distribution, benefit package strength, and relationship between benefits and salary.

### 09 · Skill Salary
`analysis_09_skill_salary.py`

Skill salary premium analysis with statistical testing and skill-level salary comparison.

### 10 · Competition
`analysis_10_competition.py`

Competition score based on applications / views. Includes high-competition roles, low-competition high-salary opportunities, and segment-level competition patterns.

### 11 · Career Ladder
`analysis_11_career_ladder.py`

Salary progression from entry-level roles to senior and director-level roles.

### 12 · Remote Work
`analysis_12_remote.py`

Remote vs non-remote salary patterns, remote access by state, company size, and experience level.

### 13 · Salary Gap
`analysis_13_salary_gap.py`

Salary range width analysis using:

```text
salary_gap = max_salary - min_salary
```

This is interpreted as a salary transparency / negotiation-range signal.

---

## Machine Learning Pipeline

Salary prediction is developed in **four stages**, each building on the previous one.

### Stage 1 — Baseline Modeling
**File:** `model_01_salary_v3_final.py`

Compares XGBoost, LightGBM, and CatBoost with feature-rich baseline engineering.

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| XGBoost | ~$40k | ~$23k | ~0.56 | — |
| LightGBM | ~$41k | ~$23k | ~0.54 | — |
| CatBoost | ~$42k | ~$24k | ~0.52 | — |

Stage 1 is mainly an initial benchmark.

### Stage 2 — CV-Safe Pipeline
**File:** `model_02_pipeline_v3_cv_safe.py`

Introduces a safer scikit-learn pipeline with cross-validation-safe target encoding and Optuna tuning.

Known result:

```text
Raw R² ≈ 0.711
RMSE ≈ $27,494
MAE ≈ $17,839
```

### Stage 3 — Advanced OOF Ensemble
**File:** `model_03_salary_advanced_progress.py`

Adds:

- salary-text leakage removal from descriptions
- title TF-IDF + SVD
- description TF-IDF + SVD
- title clustering
- interaction features
- CV-safe target encoding
- LightGBM, XGBoost, and CatBoost
- inverse-RMSE weighted OOF ensemble

Known final Model 03 metrics:

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| LightGBM | $25,338 | $15,983 | 0.7547 | 0.8048 |
| XGBoost | $25,616 | $16,386 | 0.7493 | 0.7995 |
| CatBoost | $25,663 | $16,487 | 0.7484 | 0.7980 |
| OOF Ensemble | $25,225 | $16,004 | 0.7570 | 0.8065 |

Saved model artifact:

```text
models/best_salary_model_03_advanced_progress.joblib
```

### Stage 4 — Ridge Stacking + Prediction Intervals
**File:** `model_04_stacking_intervals.py`

Model 04 uses Model 03 base-model predictions and builds a Ridge meta-learner on top of them. It also adds prediction intervals and segment-level error analysis.

Adds:

- Ridge meta-learner over Model 03 base-model predictions
- residual quantile interval
- quantile regression interval
- conformal interval
- error analysis by experience, industry, state, remote flag, salary band, and title family
- Model 03 vs Model 04 comparison

Known final Model 04 metrics:

```text
Raw R² ≈ 0.7593
RMSE ≈ $25,106
MAE ≈ $15,929
```

Model 04 gives only a modest accuracy improvement over Model 03, but it adds stronger uncertainty and diagnostic interpretation.

---

## Diagnostics

**File:** `model_03_diagnostics.py`

Known diagnostic summary:

| Metric | Value |
|---|---:|
| Train R² | 0.9850 |
| OOF R² | 0.7570 |
| Overfit gap | 0.2280 |
| Fold R² std | 0.0081 |
| OOF RMSE | $25,225 |

The train/OOF gap indicates overfitting in full training fit. Final evaluation should be read from OOF / cross-validation results, not from training score.

Generated diagnostic charts include:

```text
73_diag_train_vs_oof.png
74_diag_learning_curve.png
75_diag_residual_by_range.png
76_diag_error_distribution.png
77_diag_fold_stability.png
```

---

## SHAP Explainability

**Files:** `prepare_shap_data.py` → `shap_dependence.py`

Run `prepare_shap_data.py` first to build and save the feature matrix:

```text
models/shap_X.parquet
models/shap_y.npy
models/shap_feature_cols.json
```

Then run:

```bash
python shap_dependence.py
```

SHAP outputs include:

- beeswarm plot
- mean absolute SHAP bar plot
- dependence plots for the strongest features

> Important: `follower_count`, `applies`, and similar engagement features are correlational signals, not causal salary drivers.

---

## Streamlit Dashboard

**File:** `streamlit_app.py`

Run:

```bash
streamlit run streamlit_app.py
```

The current dashboard is local-first and does **not** require Streamlit Cloud, `.toml` configuration, or `.bat` launcher files.

It includes six stakeholder perspectives:

| Perspective | Key content |
|---|---|
| **Job Seeker** | Salary prediction · role context builder · similar postings P25/P50/P75 · reliability signal · scenario comparison |
| **HR / Recruiting** | Role-family salary benchmark · benefit package benchmark · competition snapshot |
| **Education / Curriculum Planner** | Skill demand · salary-associated skills · career ladder |
| **Investor / Market Analyst** | Industry hiring volume · industry salary concentration · geographic salary snapshot |
| **Policy Maker / Labor Market Analyst** | State salary differences · remote access · salary transparency / negotiation-gap signals |
| **Researcher / ML Evaluation** | Model comparison 01→04 · prediction intervals · Ridge coefficients · segment error analysis · methodology |

The dashboard has two global view modes:

```text
Basic
Advanced
```

Research-level content is collected in the dedicated `Researcher / ML Evaluation` page.

The live salary prediction uses the advanced Model 03 artifact when available:

```text
models/best_salary_model_03_advanced_progress.joblib
```

Model 04 outputs are used for model comparison, interval interpretation, and segment-level error analysis when the relevant CSV / JSON files exist under `outputs/`.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/t4r1k255/LinkedInJobAnalysis.git
cd LinkedInJobAnalysis
```

### 2. Create virtual environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

Download the Kaggle dataset and place all CSV files inside:

```text
data/
```

### 5. Gemini API key, optional

Only required for `analysis_03_llm.py`.

Create `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

---

## Running Order

### Data inspection and cleaning

```bash
python data_loader.py
python data_cleaning.py
```

### Analysis scripts

```bash
python analysis_01_market.py
python analysis_02_skills.py
python analysis_03_llm.py
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
```

### Model scripts

```bash
python model_01_salary_v3_final.py
python model_02_pipeline_v3_cv_safe.py
python model_03_salary_advanced_progress.py
```

### SHAP preparation, diagnostics, and Model 04

```bash
python prepare_shap_data.py
python model_03_diagnostics.py
python shap_dependence.py
python model_04_stacking_intervals.py
```

### Dashboard

```bash
streamlit run streamlit_app.py
```

---

## Outputs

Generated files are saved under:

```text
outputs/
```

Main output categories:

- EDA charts
- skill and salary charts
- geographic charts
- company charts
- remote work charts
- benefit charts
- competition charts
- salary-gap charts
- SHAP charts
- diagnostic charts
- Model 04 comparison, interval, calibration, and error-analysis files

Saved large model artifacts are stored locally under:

```text
models/
```

Large `.joblib` artifacts are not intended to be committed to GitHub.

Lightweight registry / metadata files can be tracked, for example:

```text
models/model_registry.json
outputs/outputs_manifest.csv
```

---

## Key Findings

A selection of high-level findings:

- Salary increases clearly with experience level, especially from entry / associate levels toward mid-senior and director levels.
- Technical and data-oriented title families generally show stronger salary levels than the overall median.
- Skill premiums are useful market signals but should not be read as causal salary guarantees.
- Remote labels are useful but require caution because `remote_allowed = 0` can also mean missing or unclear remote information.
- Benefit count can support employer package interpretation, but benefit quality matters more than raw count alone.
- Salary range width can be used as a salary transparency / negotiation-range signal, not as direct proof of unfairness.
- Model 04 improves Model 03 only slightly in R², but adds uncertainty, intervals, calibration, and segment error analysis.

---

## Limitations

- The dataset contains job postings, not accepted job offers.
- The project cannot determine who was hired.
- The project cannot estimate hiring probability.
- Salary information is missing for many postings.
- `remote_allowed = 0` may indicate non-remote status or missing remote metadata.
- Views, applications, follower counts, and company-job counts are correlational signals, not causal salary drivers.
- High-salary roles are naturally harder to predict due to bonuses, equity, leadership scope, and company-specific compensation policy.
- The dataset is mostly US-centered and reflects a specific historical snapshot.
- Model evaluation is based on cross-validation / out-of-fold predictions, not on an external holdout dataset.

---

## License

MIT
