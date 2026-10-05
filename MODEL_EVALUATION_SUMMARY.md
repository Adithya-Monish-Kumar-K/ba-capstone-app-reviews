# Model Evaluation Summary

## Overview

Two classification models were evaluated for detecting problematic app reviews:

- Logistic Regression
- Random Forest

The evaluation uses accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC on a held-out test set of 227,598 reviews (20% of 1,137,987 reviews from 11 apps). 19.5% of reviews are problematic (rated 1–2★).

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.8053 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1947 |
| Logistic Regression | 0.9216 | 0.7672 | 0.8579 | 0.8100 | 0.9400 | 0.8710 |
| Random Forest | 0.9143 | 0.7372 | 0.8701 | 0.7982 | 0.9411 | 0.8735 |

Out-of-time check (train April–August 2026, test 1–20 September): Random Forest PR-AUC 0.8660, Logistic Regression 0.8617. Per-domain and per-app results are in `data/model_results_by_group.csv`; see `MODEL_EVALUATION.md` §7.

Robustness checks (`MODEL_EVALUATION.md` §8): 95% bootstrap intervals are about ±0.003 for PR-AUC and ±0.001 for accuracy. In a feature-group ablation the 200 text components alone reach PR-AUC 0.852 (Logistic Regression) and 0.854 (Random Forest), against 0.739 and 0.802 for the 4 sentiment scores alone. On a validation split of the training data, `C=1` is the best Logistic Regression setting of four tried, and the Random Forest's 100-review leaves are within 0.005 PR-AUC of the best setting tried.

## Evaluation Artifacts

The following evaluation outputs are included in the project:

- `figures/model_performance_comparison.png`
- `figures/logistic_regression_confusion_matrix.png`
- `figures/random_forest_confusion_matrix.png`
- `figures/roc_curves.png`
- `figures/precision_recall_curves.png`
- `figures/normalized_confusion_matrix_comparison.png`
- `figures/random_forest_feature_importance.png`, `figures/logistic_regression_feature_importance.png`, `figures/top_tfidf_terms.png`
- `data/auc_summary.csv`
- `data/model_results.csv`, `data/baseline_results.csv`
- `data/model_results_by_group.csv`, `data/model_results_out_of_time.csv`
- `data/model_bootstrap_ci.csv`, `data/model_ablation.csv`, `data/model_tuning.csv`

## Interpretation

The evaluation provides multiple views of classifier performance. Accuracy measures overall classification correctness, while precision and recall describe performance for the problematic-review class. F1-score provides a combined precision-recall measure.

ROC-AUC and PR-AUC summarize ranking performance across classification thresholds. The confusion matrices provide the corresponding counts of correct and incorrect predictions.

These metrics should be considered together rather than relying on a single evaluation measure. Because only 19.5% of reviews are problematic, a model that never flags anything already scores 80.5% accuracy, so PR-AUC, precision and recall are the meaningful measures here. The two models are close. Logistic Regression is better at the default threshold (higher accuracy, precision and F1, 16% fewer false alarms); Random Forest ranks reviews slightly better (higher PR-AUC and ROC-AUC) and has the higher recall.
