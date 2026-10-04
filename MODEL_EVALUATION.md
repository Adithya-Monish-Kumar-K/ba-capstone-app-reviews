# Model Evaluation & Interpretation — Findings & Method

**Capstone Project — Stage 5: Predictive Modeling**

*Owners: Person 4 and Person 5 · Input: `data/engineered_features.npz`, `data/target.npy`, `data/model_row_index.csv.gz` (all built by `scripts/05_feature_engineering.py`) · Code: `scripts/06_model_training.py` to `scripts/12_normalized_confusion_matrices.py`*

Outputs: trained Logistic Regression and Random Forest models (`models/`), evaluation metrics, test-set predictions, per-app and out-of-time results, and the charts in `figures/`.

This is the second version of the evaluation, on 1,137,987 reviews from 11 apps (`Sort.NEWEST`, 1 Apr – 20 Sep 2026). The first version (14,988 `MOST_RELEVANT` reviews, 5 apps) is in git history.

---

## 1. Headline findings

1. **The task is binary classification of problematic reviews.** A review is `1` (problematic) when its rating is ≤ 2, otherwise `0`.
2. **The classes have flipped since v1.** 19.5% of reviews are problematic (v1: 74.1%). A model that never flags anything scores **80.5% accuracy with 0% recall**, so accuracy alone is not a useful headline. We report PR-AUC, recall and precision as well.
3. **Both models score above 91% accuracy and about 0.94 ROC-AUC on 227,598 held-out reviews.** They are close, and each is better on different measures.
4. **Logistic Regression is better at the default threshold.** Accuracy 92.2%, precision 76.7%, F1 0.810. It raises 2,205 fewer false alarms than Random Forest.
5. **Random Forest ranks reviews slightly better.** PR-AUC 0.873 (vs a 0.195 base rate), ROC-AUC 0.941 and recall 87.0%, so it misses the fewest complaints.
6. **Performance holds on future data.** Trained on April–August and tested on 1–20 September, PR-AUC drops by less than 0.01 (Random Forest 0.866, Logistic Regression 0.862).
7. **Payments is the hardest domain.** Random Forest PR-AUC is 0.895 for Food & Grocery and 0.869 for Shopping, but 0.750 for Payments. PhonePe is the hardest app (0.678): its complaints are rare (9.7%) and short (median 6 words, against 12 for all apps).
8. **Sentiment carries the Random Forest.** The positive, compound and negative VADER scores are its top three features; with the neutral score (7th) the four make up 42% of total importance. Text components and review length follow.

---

## 2. Method

| Choice | Details |
|---|---|
| Target | `is_problematic = 1` when `score <= 2`, otherwise `0` |
| Input features | 215: 200 SVD text components + 15 structural features |
| Text representation | TF-IDF (20,000 terms, 1–2 grams, min_df 3, max_df 0.95, sublinear term frequency, no stop-word list) → Truncated SVD (200 components) |
| Train/test split | 80% / 20%, stratified, random state 42 (910,389 / 227,598 reviews) |
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Random Forest | 150 trees, each on a 30% bootstrap sample, `min_samples_leaf=100`, `class_weight="balanced"` |
| Baseline | Always predict the majority class (`data/baseline_results.csv`) |
| Extra checks | Per-domain and per-app metrics (`data/model_results_by_group.csv`); out-of-time test, train < 1 Sep 2026 ≤ test (`data/model_results_out_of_time.csv`) |
| Evaluation metrics | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |

**Why no stop-word list.** The standard English stop-word list removes "not", "no" and "never", so "not good" and "good" would look the same. Without it, negated phrases such as "not good" and "very bad" become features.

**Why the Random Forest settings changed from v1 (200 fully grown trees).** The v1 forest was 26 MB for about 12,000 training rows. Fully grown trees on 910,389 rows would be roughly 75 times larger (about 2 GB by that estimate), which GitHub cannot store and which is slow to load. Subsampling each tree to 30% of rows and requiring at least 100 reviews per leaf keeps the model at 9.8 MB. The whole training script (both models, baseline, out-of-time re-fit) runs in about 3 minutes on 20 cores.

---

## 3. Dataset and feature representation

| Item | Value |
|---|---|
| Reviews | 1,137,987 |
| TF-IDF terms | 20,000 |
| SVD components | 200 (62.7% of TF-IDF variance kept; v1: 21.8%) |
| Structural features | 15: 9 issue flags, 4 VADER scores, `review_length`, `thumbs_up` |
| Total features | 215 |

| Target | Count | Share | Meaning |
|---|---:|---:|---|
| 0 | 916,406 | 80.5% | Non-problematic |
| 1 | 221,581 | 19.5% | Problematic |

The SVD keeps far more variance than in v1 because most reviews now use a small vocabulary ("good", "nice app", "worst service").

---

## 4. Model performance (test set, 227,598 reviews)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.805 | 0.000 | 0.000 | 0.000 | 0.500 | 0.195 |
| Logistic Regression | **0.922** | **0.767** | 0.858 | **0.810** | 0.940 | 0.871 |
| Random Forest | 0.914 | 0.737 | **0.870** | 0.798 | **0.941** | **0.873** |

Bold = better of the two models.

![Model comparison](figures/model_performance_comparison.png)

---

## 5. Confusion matrices

| Model | True negative | False positive | False negative | True positive |
|---|---:|---:|---:|---:|
| Logistic Regression | 171,744 | 11,538 | 6,297 | 38,019 |
| Random Forest | 169,539 | 13,743 | 5,756 | 38,560 |

Random Forest finds 541 more of the 44,316 problem reviews (38,560 vs 38,019). Logistic Regression raises 2,205 (16%) fewer false alarms.

![Normalized confusion matrices](figures/normalized_confusion_matrix_comparison.png)

---

## 6. ROC and Precision-Recall evaluation

![ROC curves](figures/roc_curves.png)

![Precision-Recall curves](figures/precision_recall_curves.png)

The dashed line on the PR chart is the 19.5% base rate, the precision of random guessing. At 80% recall, Random Forest's precision is 0.85 (Logistic Regression 0.84). At 90% recall both are still at 0.58, about three times the base rate. The two models' curves almost overlap: the difference between them is mainly where the default 0.5 threshold falls on each curve, not how well they rank reviews.

---

## 7. Per-domain, per-app and out-of-time results (Random Forest)

| Group | Test reviews | % problematic | Precision | Recall | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Food & Grocery | 99,491 | 21.7 | 0.738 | 0.905 | 0.895 |
| Shopping | 97,360 | 19.2 | 0.764 | 0.843 | 0.869 |
| Payments | 30,747 | 12.9 | 0.626 | 0.805 | 0.750 |
| Amazon | 6,705 | 54.8 | 0.908 | 0.961 | 0.969 |
| Swiggy | 15,001 | 35.2 | 0.873 | 0.935 | 0.948 |
| Myntra | 16,954 | 10.4 | 0.654 | 0.949 | 0.926 |
| Meesho | 20,509 | 17.8 | 0.693 | 0.942 | 0.909 |
| Domino's | 8,568 | 23.9 | 0.784 | 0.889 | 0.898 |
| Zomato | 30,215 | 18.7 | 0.739 | 0.881 | 0.879 |
| Blinkit | 45,707 | 19.0 | 0.664 | 0.906 | 0.865 |
| Google Pay | 4,049 | 23.4 | 0.685 | 0.855 | 0.813 |
| Flipkart | 53,192 | 18.1 | 0.771 | 0.742 | 0.800 |
| Paytm | 9,026 | 14.4 | 0.661 | 0.840 | 0.798 |
| PhonePe | 17,672 | 9.7 | 0.569 | 0.751 | 0.678 |

PR-AUC rises with the share of problem reviews, so compare each app against its own base rate. Even so, the payment apps are the weakest: their complaints are rarer and shorter (median 8 words in Payments, against 11 in Food & Grocery and 16 in Shopping). Flipkart and PhonePe have the lowest recall (74% and 75%). For Flipkart this is consistent with EDA §5: the most distinctive words in untagged 1–2★ reviews are praise words ("nice product", "mast", "super"), mostly from Flipkart, i.e. users who mis-rate.

| Out-of-time test (train < 1 Sep ≤ test) | Train | Test | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 1,008,932 | 129,055 | 0.748 | 0.849 | 0.938 | 0.862 |
| Random Forest | 1,008,932 | 129,055 | 0.721 | 0.862 | 0.939 | 0.866 |

PR-AUC drops by less than 0.01 compared with the random split. The models do not depend on mixing past and future reviews.

---

## 8. Model interpretation

![Random Forest feature importance](figures/random_forest_feature_importance.png)

| Rank | Random Forest feature | Importance |
|---:|---|---:|
| 1 | `sentiment_pos` | 0.155 |
| 2 | `sentiment_compound` | 0.136 |
| 3 | `sentiment_neg` | 0.094 |
| 4 | SVD_12 | 0.059 |
| 5 | SVD_11 | 0.042 |
| 6 | SVD_7 | 0.039 |
| 7 | `sentiment_neu` | 0.039 |
| 8 | `review_length` | 0.035 |

v1 plotted every feature as `SVD_<n>`, including the 15 structural ones. The charts now use the real names (saved in `data/feature_names.json`). Logistic Regression's largest coefficients are all text components (SVD_27 −10.2, SVD_23 −9.7). The SVD components are not standardised while the 15 structural features are, so coefficient sizes cannot be compared across the two groups. Among the structural features, the largest are `sentiment_compound` (−0.92: more positive text, less likely a problem) and `review_length` (+0.24: longer reviews are more likely complaints).

![Top TF-IDF terms](figures/top_tfidf_terms.png)

The terms with the largest SVD loadings are a mix of praise ("very fast", "the best", "fast service", "fast delivery", "good delivery") and complaint phrases ("very bad", "not good", "worst app"). Because stop words are kept, negated phrases such as "not good" now appear among them, along with very common words ("to", "my", "the"). The text components separate these vocabularies.

---

## 9. Practical interpretation

- **Triage:** at the default threshold the Random Forest flags about 52,000 of 228,000 test reviews and catches 87.0% of real complaints; about 3 in 4 flagged reviews are real complaints. Logistic Regression flags about 50,000, catches 85.8% and is right about 77% of the time. Use Random Forest when missing a complaint is costly and Logistic Regression when reviewer time is the limit.
- **Per domain:** the model can be used as-is for Food & Grocery and Shopping. For payment apps, expect about 4 in 10 flags to be false alarms.
- **Short reviews:** most reviews are 1–3 words. For these the model mostly reads sentiment ("worst" vs "good"). The text components matter for the longer, more informative reviews.

---

## 10. Limitations

- The target comes from the star rating, so the model predicts "low rating", not a confirmed failure. Mis-ratings (1★ with "nice product") are counted as problems.
- TF-IDF, SVD and the scaler are fitted on all rows before the split. They do not use the target, but a stricter setup would fit them on training rows only.
- No hyper-parameter search; one random split plus one out-of-time split, no cross-validation.
- During the 21 Apr – 5 May feed gap, positive reviews are under-represented for five apps. This changes the label mix for those two weeks only.

---

## 11. Files and scripts

| File | Contents |
|---|---|
| `scripts/05_feature_engineering.py` | TF-IDF, SVD, structural features; writes the feature matrix (not tracked in git, ~570 MB, rebuilt in about 2.5 minutes) |
| `scripts/06_model_training.py` | Training, test metrics, baseline, per-group and out-of-time results |
| `scripts/07`–`12` | Charts, feature importance, AUC summary, error summary, normalized confusion matrices |
| `data/model_results.csv`, `data/baseline_results.csv`, `data/model_results_by_group.csv`, `data/model_results_out_of_time.csv` | Metric tables |
| `data/model_predictions.csv.gz` | Test-set predictions with app and domain |
| `data/feature_engineering_summary.json`, `data/feature_names.json` | Feature matrix summary and names |
| `models/*.pkl` | TF-IDF, SVD, scaler, Logistic Regression, Random Forest |

```bash
python scripts/05_feature_engineering.py   # ~2.5 min
python scripts/06_model_training.py        # ~3 min on 20 cores
for s in 07_model_evaluation_charts 08_model_interpretation 09_model_performance_comparison \
         10_auc_summary 11_prediction_error_summary 12_normalized_confusion_matrices; do python scripts/$s.py; done
```
