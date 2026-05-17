# Limitations

## 1. Job postings are not accepted offers

The dataset contains job postings. It does not contain accepted salary offers, candidate profiles, who was hired, interview results, or negotiation outcomes.

## 2. No hiring probability

The dashboard cannot estimate the probability of getting hired.

Applications and views are used only as posting engagement signals.

## 3. Correlation is not causation

Features such as views, applies, follower_count, company size, benefit count, and remote flag may be associated with salary, but they should not be interpreted as direct causes.

## 4. Missing salary data

Many postings do not include salary information. The model uses only salary-usable rows after filtering.

## 5. Remote flag ambiguity

`remote_allowed = 0` may mean non-remote, missing remote information, or unclear posting metadata.

## 6. Geographic and industry imbalance

Some states, industries, and title groups have more observations than others. Predictions are generally more reliable for common segments than rare segments.

## 7. High salary roles are harder to predict

Higher compensation roles vary more because of company scope, equity, bonus structure, seniority, leadership responsibility, and industry differences.

## 8. External validation is limited

The dataset does not provide a separate official test set. Performance is evaluated using cross-validation and out-of-fold predictions.

## 9. LLM analysis limitations

If `analysis_03_llm.py` is used, results depend on sample selection, prompt design, Gemini API availability, and output consistency.
