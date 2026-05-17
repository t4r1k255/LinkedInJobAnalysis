# Project Structure

This document explains the main files and folders in the project.

## Root files

| File | Purpose |
|---|---|
| `README.md` | Main project overview and local setup instructions |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Prevents large data/model/cache files from being committed |
| `config.py` | Shared path and constant definitions |
| `streamlit_app.py` | Local dashboard application |
| `LICENSE` | Project license |

## Data files

The full dataset is expected under:

```text
data/
```

The raw Kaggle data is not committed to GitHub due to file size.

Expected files:

```text
postings.csv
companies.csv
job_skills.csv
skills.csv
job_industries.csv
industries.csv
salaries.csv
benefits.csv
employee_counts.csv
```

## Analysis scripts

| File | Purpose |
|---|---|
| `analysis_01_market.py` | Market overview |
| `analysis_02_skills.py` | Skill demand and skill distribution |
| `analysis_03_llm.py` | Gemini-based job-description analysis |
| `analysis_04_crosssection.py` | Cross-sectional salary, industry, skill, and work-type analysis |
| `analysis_05_geo.py` | Geographic salary and job-market analysis |
| `analysis_06_company.py` | Company-level analysis |
| `analysis_07_title_clustering.py` | TF-IDF and K-Means title clustering |
| `analysis_08_benefits.py` | Benefits analysis |
| `analysis_09_skill_salary.py` | Skill salary premium analysis |
| `analysis_10_competition.py` | Competition score analysis |
| `analysis_11_career_ladder.py` | Salary progression across experience levels |
| `analysis_12_remote.py` | Remote work analysis |
| `analysis_13_salary_gap.py` | Salary range width / negotiation-gap analysis |

## Model scripts

| File | Purpose |
|---|---|
| `model_01_salary_v3_final.py` | Baseline model comparison |
| `model_02_pipeline_v3_cv_safe.py` | CV-safe pipeline and Optuna tuning |
| `model_03_salary_advanced_progress.py` | Final advanced GBM ensemble |
| `prepare_shap_data.py` | Prepares feature matrix for SHAP |
| `shap_dependence.py` | SHAP explainability plots |
| `model_03_diagnostics.py` | Overfitting, residual, learning-curve diagnostics |
| `model_04_stacking_intervals.py` | Ridge stacking, intervals, segment error analysis |

## Output folders

| Folder | Purpose |
|---|---|
| `outputs/` | Generated charts, metrics, interval scores, error tables |
| `models/` | Saved model artifacts and model registry |
| `docs/` | Project documentation |
| `reports/` | Optional final report files |
