# Model Evaluation Summary

## Overview

Two classification models were evaluated for detecting problematic app reviews:

- Logistic Regression
- Random Forest

The evaluation uses accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8926 | 0.9485 | 0.9041 | 0.9258 | 0.9513 | 0.9798 |
| Random Forest | 0.9059 | 0.9045 | 0.9761 | 0.9390 | 0.9439 | 0.9738 |

## Evaluation Artifacts

The following evaluation outputs are included in the project:

- `figures/model_performance_comparison.png`
- `figures/logistic_regression_confusion_matrix.png`
- `figures/random_forest_confusion_matrix.png`
- `figures/roc_curves.png`
- `figures/precision_recall_curves.png`
- `data/auc_summary.csv`
- `data/model_results.csv`

## Interpretation

The evaluation provides multiple views of classifier performance. Accuracy measures overall classification correctness, while precision and recall describe performance for the problematic-review class. F1-score provides a combined precision-recall measure.

ROC-AUC and PR-AUC summarize ranking performance across classification thresholds. The confusion matrices provide the corresponding counts of correct and incorrect predictions.

These metrics should be considered together rather than relying on a single evaluation measure.
