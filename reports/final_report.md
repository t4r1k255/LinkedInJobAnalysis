# Final Report Draft

## 1. Introduction

This project analyzes LinkedIn job postings to understand labor-market patterns and build a salary prediction system. The project combines exploratory data analysis, machine learning, diagnostics, explainability, and an interactive Streamlit dashboard.

## 2. Dataset

The project uses the Kaggle LinkedIn Job Postings dataset. The full dataset is not included in the repository because the raw files are large. Users should download the dataset from Kaggle and place the CSV files inside the `data/` folder.

The salary prediction target is `normalized_salary` from `postings.csv`. Salary rows were filtered to the range $10,000 to $300,000.

## 3. Analysis Modules

The project includes 13 analysis modules covering market overview, skill demand, LLM-assisted description analysis, cross-sectional patterns, geography, companies, title clustering, benefits, skill salary premiums, competition, career ladder, remote work, and salary gaps.

## 4. Modeling

Model 01 establishes baseline gradient boosting performance. Model 02 improves validation safety using a CV-safe pipeline. Model 03 builds an advanced weighted ensemble using LightGBM, XGBoost, and CatBoost. Model 04 adds Ridge stacking, prediction intervals, and segment-level error analysis.

## 5. Final Model Results

Known final Model 04 metrics:

```text
Raw R² ≈ 0.7593
RMSE ≈ $25,106
MAE ≈ $15,929
```

## 6. Dashboard

The Streamlit dashboard is organized by user perspective: Job Seeker, HR / Recruiting, Education / Curriculum Planner, Investor / Market Analyst, Policy Maker / Labor Market Analyst, and Researcher / ML Evaluation.

## 7. Limitations

The dataset includes job postings, not actual hiring outcomes. It does not include who was hired or the final accepted salary. Therefore, model predictions are market estimates and should not be treated as guaranteed salary offers.

## 8. Conclusion

The project demonstrates a complete workflow from labor-market data analysis to machine learning salary prediction and dashboard-based interpretation.
