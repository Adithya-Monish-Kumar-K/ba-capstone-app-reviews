"""Check the Review 1 classifiers with preprocessing fitted on the training rows only.

scripts/05_feature_engineering.py fits TF-IDF, SVD and the scaler on all 1,137,987 reviews before
the 80/20 split. None of them uses the target, but the test reviews still shape the vocabulary,
the SVD components and the scaling. This script repeats the exact Review 1 setup (same split,
same feature settings, same model settings) with all three fitted on the 910,389 training reviews
only, and applied unchanged to the 227,598 test reviews.

Output: data/model_train_only_preprocessing.csv (both setups side by side)
Run:    python scripts/25_train_only_preprocessing_check.py   (about 15 minutes)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import read_stage  # noqa: E402

ISSUE_COLS = ["issue_crash_bugs_stability", "issue_payment_refund", "issue_delivery_delay",
              "issue_order_quality_fulfillment", "issue_cancellation_return", "issue_customer_support",
              "issue_account_login_otp", "issue_pricing_charges_fraud", "issue_ui_ux_update"]
STRUCTURAL = ISSUE_COLS + ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound",
                           "review_length", "thumbs_up"]

df = read_stage("tagged", columns=["score", "content"] + STRUCTURAL)
y = (df["score"] <= 2).astype(int).to_numpy()
text = df["content"].fillna("").astype(str).to_numpy()
struct = df[STRUCTURAL].fillna(0).astype(float).to_numpy()

# identical split to scripts/06_model_training.py
idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.20, random_state=42, stratify=y)
print(f"train {len(idx_train):,}  test {len(idx_test):,}")

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3, max_df=0.95,
                        sublinear_tf=True, dtype=np.float32)
svd = TruncatedSVD(n_components=200, random_state=42)
scaler = StandardScaler()

X_train = np.hstack([svd.fit_transform(tfidf.fit_transform(text[idx_train])),
                     scaler.fit_transform(struct[idx_train])]).astype(np.float32)
X_test = np.hstack([svd.transform(tfidf.transform(text[idx_test])),
                    scaler.transform(struct[idx_test])]).astype(np.float32)
y_train, y_test = y[idx_train], y[idx_test]
print("features built:", X_train.shape, X_test.shape)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=150, max_samples=0.3, min_samples_leaf=100,
                                            random_state=42, class_weight="balanced", n_jobs=-1),
}
original = pd.read_csv("data/model_results.csv").set_index("model")
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    new = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
           "recall": recall_score(y_test, pred), "f1_score": f1_score(y_test, pred),
           "roc_auc": roc_auc_score(y_test, prob), "pr_auc": average_precision_score(y_test, prob)}
    for setup, vals in [("fitted on all rows (Review 1)", original.loc[name]), ("fitted on training rows only", new)]:
        rows.append({"model": name, "preprocessing": setup, **{k: round(float(vals[k]), 4) for k in new}})
    print(rows[-2]); print(rows[-1])

pd.DataFrame(rows).to_csv("data/model_train_only_preprocessing.csv", index=False)
print("saved data/model_train_only_preprocessing.csv")
