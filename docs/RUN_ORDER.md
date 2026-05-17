# Run Order

Recommended execution order for reproducing the project locally.

## 0. Prepare dataset

Download the Kaggle LinkedIn Job Postings dataset and place CSV files under:

```text
data/
```

## 1. Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data inspection and cleaning

```powershell
python data_loader.py
python data_cleaning.py
```

## 3. Analysis scripts

```powershell
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

`analysis_03_llm.py` requires `GEMINI_API_KEY` in `.env`. If no key is available, it can be skipped.

## 4. Model training

```powershell
python model_01_salary_v3_final.py
python model_02_pipeline_v3_cv_safe.py
python model_03_salary_advanced_progress.py
```

## 5. SHAP and diagnostics

```powershell
python prepare_shap_data.py
python shap_dependence.py
python model_03_diagnostics.py
```

## 6. Model 04

```powershell
python model_04_stacking_intervals.py
```

Model 04 adds Ridge stacking, prediction intervals, and segment-level error analysis.

## 7. Run dashboard

```powershell
streamlit run streamlit_app.py
```

Model training can be slow, especially Model 03 and Model 04.
