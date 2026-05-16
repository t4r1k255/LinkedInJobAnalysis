# LinkedIn Job Postings — Advanced Analysis & Salary Prediction

A comprehensive data analysis and machine learning project based on the Kaggle **LinkedIn Job Postings** dataset.  
The project analyzes labor-market trends, skill demand, salary patterns, company and geographic differences, competition metrics, benefits, remote work, and salary prediction using machine learning.

The final deliverable includes:

- 13 analysis scripts
- 77+ generated visual outputs
- 3 machine learning model stages
- Final advanced ensemble salary prediction model
- SHAP explainability
- Overfitting / diagnostics analysis
- Streamlit dashboard with six stakeholder perspectives

---

## Dataset

**Source:** Kaggle — `arshkon/linkedin-job-postings`

Download the dataset from Kaggle and place all CSV files inside a `data/` folder at the project root.

Expected files:

| File | Description |
|---|---|
| `postings.csv` | Main job posting table with title, salary, location, experience, work type, views/applications |
| `companies.csv` | Company metadata such as company size, location, and company profile information |
| `job_skills.csv` | Job-to-skill mappings |
| `skills.csv` | Skill labels |
| `job_industries.csv` | Job-to-industry mappings |
| `industries.csv` | Industry labels |
| `salaries.csv` | Salary ranges, pay period, and currency |
| `benefits.csv` | Benefits offered by postings |
| `employee_counts.csv` | Company employee and follower count history |

> The dataset is not included in this repository because of size and licensing.  
> Create a local `data/` directory and place the CSV files there.

---

## Project Structure

```text
LinkedInJobAnalysis/
├── data/                              # Kaggle CSV files, not tracked
├── outputs/                           # Generated charts and model outputs
├── models/                            # Saved .joblib models
│
├── data_loader.py                     # Initial schema and file inspection
├── data_cleaning.py                   # Data cleaning workflow
│
├── analysis_01_market.py              # Market overview
├── analysis_02_skills.py              # Skill demand analysis
├── analysis_03_llm.py                 # Gemini-based LLM analysis
├── analysis_04_crosssection.py        # Cross-sectional salary/work/skill analysis
├── analysis_05_geo.py                 # Geographic analysis
├── analysis_06_company.py             # Company-level analysis
├── analysis_07_title_clustering.py    # TF-IDF + K-Means title clustering
├── analysis_08_benefits.py            # Benefits analysis
├── analysis_09_skill_salary.py        # Skill salary premium analysis
├── analysis_10_competition.py         # Competition score analysis
├── analysis_11_career_ladder.py       # Career ladder and salary growth
├── analysis_12_remote.py              # Remote work analysis
├── analysis_13_salary_gap.py          # Salary negotiation gap analysis
│
├── model_01_salary_v3_final.py        # Baseline model comparison
├── model_02_pipeline_v3_cv_safe.py    # CV-safe pipeline and Optuna tuning
├── model_03_salary_advanced_progress.py # Final advanced ensemble model
├── model_03_diagnostics.py            # Overfitting, residual, learning-curve diagnostics
├── shap_dependence.py                 # SHAP explainability plots
│
├── streamlit_app.py                   # Final dashboard
├── utils_progress.py                  # Shared progress bar utilities
├── requirements.txt
└── README.md
```

---

## Analysis Modules

### 1. Market Overview
`analysis_01_market.py`

Explores the overall job market structure:

- Experience-level distribution
- Work type breakdown
- Remote vs non-remote postings
- Top states by job volume
- Salary by experience level
- Remote rate by company size

### 2. Skill Demand
`analysis_02_skills.py`

Analyzes the skills and job functions most frequently requested:

- Top 20 skills
- Top industries
- Skill demand by experience level
- Highest-paying skills
- Skill distribution by industry

### 3. LLM-Assisted Job Description Analysis
`analysis_03_llm.py`

Uses Gemini API on a stratified sample of job descriptions:

- Degree requirement extraction
- Soft skill identification
- Urgency classification
- Tech vs non-tech role classification
- Degree requirement by experience level

> Requires a `.env` file containing `GEMINI_API_KEY`.

### 4. Cross-Sectional Analysis
`analysis_04_crosssection.py`

Studies relationships between salary bands and other dimensions:

- Salary band × experience level
- Salary band × industry
- State-level remote ratio and salary bubble chart
- Work type × experience level
- Skill × salary band heatmap

### 5. Geographic Analysis
`analysis_05_geo.py`

Explores location-based differences:

- State median salary ranking
- State remote rate
- Top cities by job volume
- State × industry heatmap
- State × experience distribution

### 6. Company-Level Analysis
`analysis_06_company.py`

Analyzes company-related patterns:

- Top hiring companies
- Salary by company size
- Company size × industry heatmap
- Top-paying companies
- Company size × experience distribution

### 7. Title Clustering
`analysis_07_title_clustering.py`

Groups job titles using text mining:

- TF-IDF vectorization
- K-Means clustering
- PCA visualization
- Salary by title cluster
- Experience and remote rate by cluster

### 8. Benefits Analysis
`analysis_08_benefits.py`

Examines employer-provided benefits:

- Most common benefits
- Benefits by company size
- Benefits by industry
- Benefit count vs salary
- Benefits by experience level

### 9. Skill Salary Premium
`analysis_09_skill_salary.py`

Measures association between skills and salary:

- Skill salary premium
- Mann-Whitney U significance testing
- Salary boxplots by skill
- Experience × skill salary matrix
- Industry × skill salary matrix
- Skill count vs salary

### 10. Competition Analysis
`analysis_10_competition.py`

Uses `applies / views` as a job competition score:

- Most competitive titles
- Competition by industry
- Competition by experience level
- Remote vs non-remote competition
- Low-competition high-salary opportunities

### 11. Career Ladder
`analysis_11_career_ladder.py`

Analyzes salary progression across experience levels:

- Career ladder salary distribution
- Industry-specific career trajectories
- Company size × experience salary matrix
- Entry-to-mid-senior salary growth
- Remote vs non-remote career ladder

### 12. Remote Work Analysis
`analysis_12_remote.py`

Explores remote work patterns:

- Remote vs non-remote salary comparison
- Remote salary premium by sector
- Remote skill profiles
- Remote rate by company size
- State-level remote and salary patterns

### 13. Salary Gap / Negotiation Range
`analysis_13_salary_gap.py`

Uses salary range width as a negotiation-gap signal:

- Salary gap distribution
- Salary gap by industry
- Salary gap by experience level
- Salary level vs salary gap
- Industry × experience negotiation map

---

## Machine Learning Models

The project uses a staged modeling process.

### Model 01 — Feature-Rich Baseline

File: `model_01_salary_v3_final.py`

Models tested:

- XGBoost
- LightGBM
- CatBoost

Main purpose:

- Establish a strong initial salary prediction baseline
- Add engineered features such as skills, benefits, company size, title keywords, and location
- Generate initial SHAP explanations

Best result:

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| XGBoost | $30,118 | $20,033 | 0.653 | 0.717 |
| LightGBM | $30,237 | $20,090 | 0.651 | 0.715 |
| CatBoost | $30,942 | $20,720 | 0.634 | 0.700 |

---

### Model 02 — CV-Safe Pipeline

File: `model_02_pipeline_v3_cv_safe.py`

Main improvements:

- Scikit-learn Pipeline
- CV-safe target encoding
- Optuna hyperparameter tuning
- Title TF-IDF + SVD features
- Saved production-ready pipeline

Best result:

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| XGBoost | $27,562 | $17,726 | 0.7098 | 0.7630 |
| **LightGBM** | **$27,494** | **$17,839** | **0.7112** | **0.7636** |
| CatBoost | $28,434 | $18,707 | 0.6912 | 0.7492 |

Saved model:

```text
models/best_salary_pipeline.joblib
```

---

### Model 03 — Final Advanced Ensemble

File: `model_03_salary_advanced_progress.py`

Final selected model.

Main improvements:

- Cleaned job description text to remove salary leakage
- Title TF-IDF + SVD
- Description TF-IDF + SVD
- Title clustering
- Interaction features
- CV-safe target encoding
- Optuna tuning
- Out-of-fold weighted ensemble
- LightGBM + XGBoost + CatBoost

Final result:

| Model | RMSE | MAE | Raw R² | Log R² |
|---|---:|---:|---:|---:|
| LightGBM | $25,338 | $15,983 | 0.7547 | 0.8048 |
| XGBoost | $25,616 | $16,386 | 0.7493 | 0.7995 |
| CatBoost | $25,663 | $16,487 | 0.7484 | 0.7980 |
| **OOF Ensemble** | **$25,225** | **$16,004** | **0.7570** | **0.8065** |

Saved model:

```text
models/best_salary_model_03_advanced_progress.joblib
```

---

## Diagnostics

File: `model_03_diagnostics.py`

Diagnostic outputs:

- Train vs OOF comparison
- Learning curve
- Residual analysis by salary range
- Error distribution
- Fold stability

Diagnostic summary:

| Metric | Value |
|---|---:|
| Train R² | 0.9850 |
| OOF R² | 0.7570 |
| Overfit gap | 0.2280 |
| Fold R² std | 0.0081 |
| OOF RMSE | $25,225 |

Interpretation:

- The model has overfitting tendency because train R² is much higher than OOF R².
- Final performance is based on OOF/CV results, not training score.
- Fold-level validation is stable, so the model performance is consistent across folds.

---

## SHAP Explainability

File: `shap_dependence.py`

The SHAP analysis explains which features most influence salary prediction.

Expected outputs include:

- SHAP beeswarm plot
- SHAP bar importance plot
- SHAP dependence plots for selected high-impact features

Important interpretation:

- Engagement features such as views, applies, and follower count are correlational.
- They should not be interpreted as causal drivers of salary.
- Higher-paying or more attractive jobs may naturally receive more views and applications.

---

## Streamlit Dashboard

File:

```text
streamlit_app.py
```

Run:

```bash
streamlit run streamlit_app.py
```

Dashboard perspectives:

| Perspective | Purpose |
|---|---|
| Job Seeker | Salary prediction, similar posting salary distribution, skills, career ladder, competition |
| HR / Recruiting | Salary benchmark, role competition, company profile, benefits |
| Education | Skill demand and curriculum planning |
| Investor | Sector and geographic market signals |
| Policy Maker | Regional salary differences, transparency, remote work |
| Researcher | Model comparison, SHAP, diagnostics, methodology, data quality |

Key dashboard features:

- Final Model 03 salary prediction
- Similar postings salary distribution
- Salary uncertainty bands based on MAE/RMSE
- Benefit type selection and benefit package strength
- Interactive market explorer
- Data quality panel
- Chart file checker
- Downloadable filtered summary

Important limitation:

> The dataset contains job postings, salaries, views, and applications.  
> It does **not** contain who was hired or which salary offer was accepted.  
> Therefore, the dashboard estimates salary ranges and posting-level salary distributions, not hiring probability.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/t4r1k255/LinkedInJobAnalysis.git
cd LinkedInJobAnalysis
```

### 2. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
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

### 5. Add Gemini API key, optional

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

### Diagnostics and SHAP

```bash
python model_03_diagnostics.py
python shap_dependence.py
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
- Skill and salary charts
- Geographic charts
- Company charts
- Remote work charts
- Benefit charts
- Competition charts
- Salary gap charts
- Model comparison charts
- SHAP charts
- Diagnostic charts

Saved models are stored under:

```text
models/
```

---

## Notes on Reproducibility

- Random seed is fixed where possible.
- Model performance is based on cross-validation and out-of-fold predictions.
- Target encoding is implemented in a CV-safe way to reduce leakage.
- Description salary text is cleaned before text feature extraction.
- Large generated folders such as `data/`, `outputs/`, and `models/` may be excluded from GitHub depending on repository size.

---

## Limitations

- Salary data is missing for many postings.
- `remote_allowed = 0` may mean non-remote or missing remote information.
- The dataset is mostly US-centered.
- The data is a snapshot of the labor market and may not represent current conditions.
- Views/applications/follower counts are correlational features, not causal salary drivers.
- The dataset does not include actual hiring outcomes or accepted salary offers.
- Salary prediction is an estimate and should not be treated as a guaranteed compensation value.

---

## License

MIT
