# LinkedIn Job Postings — Advanced Analysis & Salary Prediction

A comprehensive data analysis and machine learning project built on 123,000+ LinkedIn job postings. The project produces deep analytical outputs across six stakeholder perspectives and a multi-model salary prediction pipeline with full interpretability.

---

## Dataset

**Source:** [Kaggle — LinkedIn Job Postings (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

| File | Description |
|---|---|
| `postings.csv` | 123k job listings with salary, location, experience level |
| `companies.csv` | Company metadata, size, industry |
| `job_skills.csv` / `skills.csv` | Skill tags per posting |
| `job_industries.csv` / `industries.csv` | Industry classifications |
| `salaries.csv` | Min/max/median salary ranges |
| `benefits.csv` | Benefits offered per posting |
| `employee_counts.csv` | Company headcount and follower data |

> Download the dataset and place all CSV files in a `data/` folder at the project root.

---

## Project Structure

```
LinkedInJobAnalysis/
├── data/                           # CSV files (not tracked, download from Kaggle)
├── outputs/                        # Generated plots (not tracked)
├── models/                         # Saved model files (not tracked)
│
├── Data_loader.py                  # Initial data loading & schema inspection
├── data_cleaning.py                # Cleaning pipeline
│
├── analysis_01_market.py           # Market overview
├── analysis_02_skills.py           # Skill demand analysis
├── analysis_03_llm.py              # LLM-powered analysis (Gemini API)
├── analysis_04_crosssection.py     # Cross-sectional analysis
├── analysis_05_geo.py              # Geographic analysis
├── Analysis_06_company.py          # Company-level analysis
├── analysis_07_title_clustering.py # TF-IDF + K-Means title clustering
├── analysis_08_benefits.py         # Benefits analysis
├── analysis_09_skill_salary.py     # Skill salary premium (Mann-Whitney U)
├── analysis_10_competition.py      # Job competition analysis (applies/views)
├── analysis_11_career_ladder.py    # Career progression & salary growth
├── analysis_12_remote.py           # Remote work analysis
├── analysis_13_salary_gap.py       # Salary negotiation gap (planned)
│
├── model_01_salary_v3_final.py     # Baseline: XGBoost + LightGBM + CatBoost + SHAP
├── model_02_pipeline_v3_cv_safe.py # CV-safe pipeline (leakage prevention)
├── model_03_salary_advanced_progress.py  # Tuned GBM + TF-IDF/SVD + Optuna
├── model_04_stacking.py            # Stacking + prediction intervals (planned)
├── model_05_tabnet.py              # TabNet + attention visualization (planned)
│
├── utils_progress.py               # Progress bar utility (shared across scripts)
├── requirements.txt
└── README.md
```

---

## Analysis Modules — 61 Plots, 12 Scripts

### Market & Structure
- **analysis_01** — Experience level distribution, work type breakdown, remote vs office, top states by volume, salary by experience, remote rate by company size
- **analysis_04** — Salary band × experience heatmap, sector × salary heatmap, state remote/salary bubble chart, work type × experience stacked bar, skill × salary band heatmap

### Skills & Compensation
- **analysis_02** — Top 20 skills by demand, top industries, skill by experience heatmap, highest-paying skills, skill by industry stacked bar
- **analysis_09** — Skill salary premium with Mann-Whitney U significance testing, salary boxplot by skill, experience × skill salary matrix, industry × skill salary matrix, skill count vs salary

### Geographic
- **analysis_05** — State median salary ranking, state remote rates, top cities by volume, state × industry heatmap, state × experience distribution
- **analysis_12** — Remote salary premium by sector, remote vs office skill profiles, company size × remote/salary dual axis, state remote/salary scatter with trend, experience × work type salary comparison

### Company & Career
- **analysis_06** — Top hiring companies, salary by company size, company size × industry heatmap, top-paying companies (min 20 postings), company size × experience distribution
- **analysis_11** — Career ladder violin plots, sector career trajectories, company size × experience salary matrix, entry→mid-senior growth rates, remote vs office career ladder

### Job Market Dynamics
- **analysis_10** — Most competitive titles (applies/views), competition by industry, competition by experience level, remote vs office competition, low-competition high-salary opportunities
- **analysis_08** — Top 20 benefits, benefit count by company size, benefits by industry heatmap, benefit count vs salary, benefits by experience heatmap

### LLM-Powered (Gemini API)
- **analysis_03** — Degree requirement analysis, soft skill type distribution, urgency classification, tech vs non-tech salary comparison, degree requirement by experience (stratified 500-posting sample)

### Title Intelligence
- **analysis_07** — TF-IDF vectorization + K-Means clustering (k=8), PCA 2D projection, salary by cluster, experience by cluster, remote rate by cluster

---

## Six Stakeholder Perspectives

The same analytical foundation is framed across six distinct audiences. Each perspective draws from the same outputs but prioritizes different questions and metrics.

| Perspective | Core Questions |
|---|---|
| **Job Seeker** | Which skills, locations, and title combinations maximize salary? |
| **HR / Recruiter** | What does a competitive offer look like for this role in this market? |
| **Educator / Bootcamp** | Which skills are rising or declining in demand? |
| **Investor / Market Analyst** | Which sectors are hiring most aggressively? |
| **Policy Maker** | Where are the regional skill and employment gaps? |
| **Researcher** | How do NLP methods and labor economics intersect here? |

---

## Salary Prediction Models

### Model 01 — Baseline Comparison
Three gradient boosting models, 5-fold CV, SHAP interpretability.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | $40,441 | $23,118 | 0.560 |
| LightGBM | $41,252 | $23,406 | 0.542 |
| CatBoost | $42,378 | $24,031 | 0.517 |

### Model 02 — CV-Safe Pipeline
Introduces target encoding fitted inside cross-validation folds to prevent data leakage. Establishes the correct methodological baseline for all subsequent models.

### Model 03 — Advanced Pipeline *(current best)*
- YEARLY-only salary filter for clean target distribution
- Description text cleaning to remove salary leakage via vectorized regex
- TF-IDF + SVD on job titles and descriptions (NLP features)
- Title clustering features from K-Means
- Optuna hyperparameter tuning (30 trials per model)
- LightGBM + XGBoost ensemble with OOF inverse-RMSE weighting

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LightGBM | $29,378 | $20,117 | 0.667 |
| XGBoost | $29,003 | $19,665 | 0.675 |
| **Ensemble** | **$28,937** | **$19,681** | **0.677** |

### Model 04 — Stacking Ensemble *(planned)*
- Ridge meta-learner trained on OOF predictions from Model 03 base models
- Prediction intervals: point estimate + uncertainty range per prediction
- Fairness analysis: RMSE breakdown by state and industry
- Error analysis: systematic bias detection by job profile type

### Model 05 — TabNet *(planned)*
- Attention-based neural architecture designed specifically for tabular data
- Built-in feature importance via sparse attention masks (no SHAP needed)
- Full gradient boosting vs deep learning comparison across all five models
- Training curve and attention visualization

---

## Key Technical Decisions

**Why log-transform the target?**
Salary is right-skewed. Log transform stabilizes variance, improves model fit, and makes residuals more interpretable.

**Why YEARLY-only?**
Mixed pay periods add noise even after annualization. Restricting to yearly-reported salaries produces a cleaner target distribution with less structural noise.

**Why clean description text?**
Job descriptions sometimes contain salary ranges explicitly. Without cleaning, models learn to extract salary from text — this is data leakage that artificially inflates R² without improving real predictive power.

**Why CV-safe target encoding?**
Fitting encoders on the full training set before CV allows target information to leak into validation folds. Model 02 establishes the pattern of fitting all transformations inside each fold only.

**Why OOF ensemble weights?**
Fixed weights are arbitrary. OOF-based inverse-RMSE weighting lets validation performance determine each model's contribution to the ensemble.

---

## Setup

```bash
git clone https://github.com/t4r1k255/LinkedInJobAnalysis.git
cd LinkedInJobAnalysis

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Download the dataset from Kaggle and place all CSV files in `data/`.

For `analysis_03` (LLM features), create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

---

## Running Order

```bash
# Data
python Data_loader.py
python data_cleaning.py

# Analyses (independent, any order)
python analysis_01_market.py
python analysis_02_skills.py
python analysis_03_llm.py
python analysis_04_crosssection.py
python analysis_05_geo.py
python Analysis_06_company.py
python analysis_07_title_clustering.py
python analysis_08_benefits.py
python analysis_09_skill_salary.py
python analysis_10_competition.py
python analysis_11_career_ladder.py
python analysis_12_remote.py

# Models (sequential)
python model_01_salary_v3_final.py
python model_02_pipeline_v3_cv_safe.py
python model_03_salary_advanced_progress.py
```

---

## Dependencies

```
numpy · pandas · matplotlib · seaborn · scikit-learn · scipy
xgboost · lightgbm · catboost · optuna · shap · tqdm
python-dotenv · google-genai
```

---

## License

MIT
