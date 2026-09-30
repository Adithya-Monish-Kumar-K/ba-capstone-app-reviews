# Model Evaluation & Interpretation — Findings & Method

**Capstone Project — Stage 5: Predictive Modeling**

*Owner: Person 4 · Input: `data/engineered_features.npz` and `data/target.npy` · Code: `scripts/06_model_training.py`*

Outputs: trained Logistic Regression and Random Forest models, evaluation metrics, prediction data, ROC/PR and confusion-matrix charts, and model interpretation charts.

---

## 1. Headline findings

1. **The predictive task is binary classification of problematic reviews.**  
   A review is labelled `1` (`problematic`) when its rating score is ≤2, and `0` (`non-problematic`) otherwise.

2. **The dataset contains 14,988 reviews and 115 engineered features.**  
   The feature matrix combines 100 SVD components derived from TF-IDF text features with 15 structural and sentiment features.

3. **The target is imbalanced.**  
   There are 11,108 problematic reviews and 3,880 non-problematic reviews. Therefore, accuracy is reported together with precision, recall, F1-score, ROC-AUC and PR-AUC.

4. **Logistic Regression provides a strong baseline.**  
   It achieved 89.26% accuracy, 92.58% F1-score, 95.13% ROC-AUC and 97.98% PR-AUC on the held-out test set.

5. **Random Forest produced a different error profile.**  
   It achieved 90.59% accuracy, 93.90% F1-score, 94.39% ROC-AUC and 97.38% PR-AUC.

6. **The two models show a precision-recall trade-off.**  
   Logistic Regression produced higher precision, while Random Forest produced higher recall. This difference is visible in their confusion matrices.

7. **Model evaluation uses a fixed stratified 80/20 train-test split.**  
   The training set contains 11,990 reviews and the test set contains 2,998 reviews.

---

## 2. Method

| Choice | Details |
|---|---|
| Target | `is_problematic = 1` when `score <= 2`, otherwise `0` |
| Input features | 115 engineered features |
| Text representation | TF-IDF followed by Truncated SVD |
| SVD components | 100 |
| Structural features | 15 |
| Train/test split | 80% / 20% |
| Random state | 42 |
| Split strategy | Stratified |
| Models | Logistic Regression and Random Forest |
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Random Forest | 200 trees, `class_weight="balanced"` |
| Evaluation metrics | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |

The evaluation was performed on the held-out test set rather than on the training data.

---

## 3. Dataset and feature representation

The engineered feature matrix contains:

- **5,000 TF-IDF text features** before dimensionality reduction.
- **100 SVD components** representing the reduced text feature space.
- **15 structural/sentiment features**.

The resulting model input therefore contains:

**14,988 reviews × 115 features**

The target distribution is:

| Target | Count | Meaning |
|---|---:|---|
| 0 | 3,880 | Non-problematic |
| 1 | 11,108 | Problematic |

The target is therefore not evenly distributed, which is why class balancing and multiple evaluation metrics are used.

---

## 4. Model performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8926 | 0.9485 | 0.9041 | 0.9258 | 0.9513 | 0.9798 |
| Random Forest | 0.9059 | 0.9045 | 0.9761 | 0.9390 | 0.9439 | 0.9738 |

The metrics were calculated using the 2,998-review held-out test set.

---

## 5. Confusion matrices

### Logistic Regression

```text
                 Predicted
               Non-Prob  Prob
Actual Non-Prob    667    109
       Prob        213   2009
The corresponding chart is:

[`figures/random_forest_confusion_matrix.png`](figures/random_forest_confusion_matrix.png)

---

## 6. ROC and Precision-Recall evaluation

Two threshold-independent evaluation views were generated.

### ROC curve

The ROC curve compares the true-positive rate with the false-positive rate across classification thresholds.

[`figures/roc_curves.png`](figures/roc_curves.png)

The ROC-AUC values are:

- Logistic Regression: **0.9513**
- Random Forest: **0.9439**

### Precision-Recall curve

The Precision-Recall curve is particularly informative when the target classes are imbalanced because it focuses on the relationship between precision and recall.

[`figures/precision_recall_curves.png`](figures/precision_recall_curves.png)

The PR-AUC values are:

- Logistic Regression: **0.9798**
- Random Forest: **0.9738**

---

## 7. Model interpretation

Model interpretation outputs were generated to inspect which engineered dimensions and text terms contributed most strongly to the fitted models.

### Logistic Regression feature importance

The absolute coefficients were used to identify the most influential model features.

[`figures/logistic_regression_feature_importance.png`](figures/logistic_regression_feature_importance.png)

The detailed values are stored in:

`data/logistic_feature_importance.csv`

### Random Forest feature importance

Random Forest impurity-based feature importance was extracted and the highest-importance features were visualized.

[`figures/random_forest_feature_importance.png`](figures/random_forest_feature_importance.png)

The detailed values are stored in:

`data/random_forest_feature_importance.csv`

### TF-IDF term interpretation

The TF-IDF vocabulary and SVD components were used to calculate mean absolute SVD loadings for the original text terms.

[`figures/top_tfidf_terms.png`](figures/top_tfidf_terms.png)

The resulting terms and scores are stored in:

`data/top_tfidf_terms.csv`

Because the classifier operates on the reduced SVD representation, the model feature-importance charts describe SVD dimensions rather than individual raw words. The TF-IDF chart provides a separate view of which original text terms have the strongest average SVD loadings.

---

## 7A. Detailed Findings and Model Comparison

### Overall classification performance

The two classifiers show strong performance across the reported evaluation metrics.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8926 | 0.9059 |
| Precision | 0.9485 | 0.9045 |
| Recall | 0.9041 | 0.9761 |
| F1 Score | 0.9258 | 0.9390 |
| ROC-AUC | 0.9513 | 0.9439 |
| PR-AUC | 0.9798 | 0.9738 |

The results show that the models have different precision-recall characteristics. Logistic Regression has higher precision, meaning that a larger proportion of its reviews predicted as problematic are actually problematic. Random Forest has higher recall, meaning that it identifies a larger proportion of the actual problematic reviews.

The F1 scores summarize this precision-recall balance and are 0.9258 for Logistic Regression and 0.9390 for Random Forest.

### Confusion-matrix findings

The confusion-matrix counts provide a more detailed view of classification errors.

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|
| Logistic Regression | 667 | 109 | 213 | 2009 |
| Random Forest | 547 | 229 | 53 | 2169 |

For Logistic Regression, 2,009 problematic reviews were correctly identified, while 213 problematic reviews were classified as non-problematic. The model produced 109 false positives.

For Random Forest, 2,169 problematic reviews were correctly identified and only 53 problematic reviews were classified as non-problematic. It produced 229 false positives.

These results illustrate the trade-off between false positives and false negatives. In this dataset, the models therefore provide different error profiles despite both achieving strong overall performance.

### Threshold-based evaluation

The ROC and Precision-Recall curves evaluate model behaviour across different classification thresholds rather than only the default prediction threshold.

The ROC-AUC values of 0.9513 for Logistic Regression and 0.9439 for Random Forest indicate strong separation between the two target classes across thresholds.

The PR-AUC values of 0.9798 for Logistic Regression and 0.9738 for Random Forest provide an additional view of precision-recall behaviour for the problematic-review classification task.

### Interpretation of model features

The Logistic Regression interpretation is based on model coefficients, while the Random Forest interpretation uses impurity-based feature importance.

Because the predictive pipeline applies TF-IDF followed by SVD, the classifier operates on the resulting reduced feature representation rather than directly on individual raw words.

The TF-IDF interpretation therefore provides complementary information about the original text vocabulary. It should not be interpreted as a direct list of classifier coefficients for individual words.

### Practical interpretation

The evaluation indicates that problematic-review detection can be approached using a combination of text-derived and structural review features.

The evaluation outputs provide three complementary levels of analysis:

1. **Classification performance** — accuracy, precision, recall, F1, ROC-AUC and PR-AUC.
2. **Prediction errors** — true positives, true negatives, false positives and false negatives.
3. **Feature interpretation** — model feature importance and TF-IDF/SVD term analysis.

Together, these outputs provide a reproducible basis for understanding classifier behaviour and the engineered feature representation.

## 8. Reproducibility outputs

The modeling stage produces the following files:

| File | Purpose |
|---|---|
| `models/logistic_regression.pkl` | Trained Logistic Regression model |
| `models/random_forest.pkl` | Trained Random Forest model |
| `data/model_results.csv` | Model-level evaluation metrics |
| `data/model_predictions.csv` | Test-set predictions and probabilities |
| `data/model_curve_data.json` | ROC and Precision-Recall curve data |
| `data/logistic_feature_importance.csv` | Logistic Regression feature coefficients |
| `data/random_forest_feature_importance.csv` | Random Forest feature importance |
| `data/top_tfidf_terms.csv` | Top TF-IDF terms by SVD loading |
| `figures/` | Model evaluation and interpretation charts |

---

## 9. Limitations

1. **The target is derived directly from the review rating.**  
   Therefore, the model predicts the project's defined `problematic` label rather than an independently observed business failure outcome.

2. **The dataset is imbalanced.**  
   Problematic reviews substantially outnumber non-problematic reviews, so accuracy alone should not be used to describe model performance.

3. **The models operate on a reduced feature representation.**  
   The text features are compressed from 5,000 TF-IDF dimensions into 100 SVD components, so individual classifier features do not directly correspond to individual words.

4. **The evaluation uses one fixed train-test split.**  
   The reported metrics describe this held-out test set and may vary with a different split or validation procedure.

5. **Model interpretation should be treated as feature-level evidence rather than causal explanation.**  
   Feature importance indicates association with model predictions and does not establish that a feature causes a review to be problematic.

---

## 10. Files and scripts

**Feature engineering**

`scripts/05_feature_engineering.py`

**Model training and evaluation**

`scripts/06_model_training.py`

**Evaluation visualization**

`scripts/07_model_evaluation_charts.py`

**Model interpretation**

`scripts/08_model_interpretation.py`

---

## 11. Summary

The predictive modeling stage converts the engineered review representation into a binary problematic-review classifier using Logistic Regression and Random Forest. Both models achieve strong test-set discrimination, while their confusion matrices demonstrate different precision-recall trade-offs. Evaluation charts and interpretation outputs provide reproducible evidence for comparing model behaviour and understanding the engineered feature space.

---
