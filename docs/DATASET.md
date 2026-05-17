# Dataset Guide

## Source

The project uses the Kaggle dataset:

```text
arshkon/linkedin-job-postings
```

The full dataset is not included in the repository because the raw files are large.

## Required folder

Create a folder named:

```text
data/
```

Place the Kaggle CSV files inside it.

## Expected files

| File | Used for |
|---|---|
| `postings.csv` | Main job posting table and salary prediction target |
| `companies.csv` | Company metadata |
| `job_skills.csv` | Job-to-skill mappings |
| `skills.csv` | Skill labels |
| `job_industries.csv` | Job-to-industry mappings |
| `industries.csv` | Industry labels |
| `salaries.csv` | Salary range / negotiation-gap analysis |
| `benefits.csv` | Benefits analysis |
| `employee_counts.csv` | Company scale and follower signals |

## Salary target

The salary prediction target is:

```text
postings.csv -> normalized_salary
```

Filtering used in the final model:

```text
Salary bounds: $10,000 - $300,000
Currency: USD or missing
```

Known project count:

```text
Total postings: about 123,849
Final salary-usable rows: about 35,279
```

## Salary gap analysis

`analysis_13_salary_gap.py` uses:

```text
salaries.csv -> min_salary
salaries.csv -> max_salary
```

Salary gap is calculated as:

```text
salary_gap = max_salary - min_salary
```

This is interpreted as a posting-level salary transparency / negotiation-range signal.

## Important limitations

The dataset contains job postings, not actual hiring outcomes.

It does not contain who was hired, accepted final offers, candidate profiles, interview outcomes, or negotiated salary.
