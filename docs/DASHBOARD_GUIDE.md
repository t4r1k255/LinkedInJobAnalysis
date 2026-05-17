# Dashboard Guide

Run:

```powershell
streamlit run streamlit_app.py
```

## View modes

The dashboard has two global modes:

```text
Basic
Advanced
```

Basic mode hides most raw tables and technical detail.

Advanced mode shows tables, extra chart interpretation, and more detailed metrics.

The Researcher / ML Evaluation page is the dedicated technical model evaluation page.

## Pages

### Home

Dataset summary, salary-usable rows, remote rate, data quality snapshot, quick market snapshots.

### Job Seeker

Salary prediction, role context builder, similar postings P25/P50/P75, reliability signal, scenario comparison, downloadable salary report.

### HR / Recruiting

Role-family salary benchmark, benefit package benchmark, competition snapshot.

### Education / Curriculum Planner

Most frequent skills, salary-associated skills, career ladder.

### Investor / Market Analyst

Industry hiring volume, industry salary concentration, geographic salary snapshot, title-family market structure.

### Policy Maker / Labor Market Analyst

State salary comparison, remote access snapshot, salary transparency / negotiation-gap signals.

### Researcher / ML Evaluation

Model 01-04 comparison, Model 03 vs Model 04 improvement, prediction interval methods, Ridge coefficients, segment error analysis, data quality, methodology.

## Common terms

| Term | Meaning |
|---|---|
| Prediction | Model salary estimate |
| MAE | Mean absolute error |
| RMSE | Root mean squared error |
| R² | Explained variance score |
| OOF | Out-of-fold validation prediction |
| P25 / P75 | 25th / 75th percentiles |
| Similar postings | Real postings matched using a progressive matching rule |
| Reliability signal | Support signal based on similar-posting count and range width |
