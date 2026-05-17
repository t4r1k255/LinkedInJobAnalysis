# Model Card

## Model purpose

The salary prediction models estimate annual salary from LinkedIn job posting data.

The prediction target is:

```text
normalized_salary
```

The task is regression.

## Training data

The model is trained on salary-usable postings after filtering:

```text
Salary bounds: $10,000 - $300,000
Currency: USD or missing
```

Known final model dataset size:

```text
35,279 salary-usable rows
```

## Feature groups

The models use job title text, cleaned description text, experience level, work type, pay period, industry, state/city, remote flag, skills, benefits, company size, views, applications, employee count, follower count, company job count, title-derived features, interaction features, and target-encoded categorical features.

## Model stages

| Model | Type | Main contribution |
|---|---|---|
| Model 01 | XGBoost / LightGBM / CatBoost baseline | Initial benchmark |
| Model 02 | CV-safe pipeline | Safer preprocessing and tuning |
| Model 03 | Advanced GBM ensemble | Final weighted ensemble for live prediction |
| Model 04 | Ridge stacking + intervals | Slightly improved OOF score plus uncertainty and error analysis |

## Final known metrics

### Model 03

```text
Raw R² ≈ 0.7570
RMSE ≈ $25,225
MAE ≈ $16,004
Log R² ≈ 0.8065
```

### Model 04

```text
Raw R² ≈ 0.7593
RMSE ≈ $25,106
MAE ≈ $15,929
```

## Validation method

The dataset does not provide a separate external test split.

Evaluation uses cross-validation, out-of-fold predictions, and fold stability analysis.

## Known overfitting signal

Model 03 diagnostics showed:

```text
Train R² ≈ 0.9850
OOF R² ≈ 0.7570
Overfit gap ≈ 0.2280
```

OOF/CV metrics should be trusted more than training metrics.

## Interpretation warnings

Features such as views, applies, follower_count, employee_count, and company_job_count are correlational market signals. They should not be interpreted as direct causal salary drivers.

## Intended use

Appropriate use: salary range estimation, market benchmarking, model comparison, and dashboard-based exploratory analysis.

Not appropriate use: guaranteeing a salary offer, predicting hiring probability, making causal claims, or making employment/policy decisions without external validation.
