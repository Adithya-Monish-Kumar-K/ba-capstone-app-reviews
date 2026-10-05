"""
Robustness checks for the Review 1 classifiers (run after scripts/06_model_training.py).

1. Bootstrap confidence intervals: 95% intervals for every test-set metric of the saved models,
   and for the difference between the two models.
2. Feature-group ablation: retrain both models on subsets of the 215 features (text only,
   sentiment only, structural only, all except sentiment) and compare on the same test set.
3. Settings check: compare a small range of settings on a validation split carved out of the
   training data (the test set is never used to choose settings).

Outputs: data/model_bootstrap_ci.csv, data/model_ablation.csv, data/model_tuning.csv
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

FEATURE_PATH = "data/engineered_features.npz"
TARGET_PATH = "data/target.npy"
FEATURE_NAMES_PATH = "data/feature_names.json"
MODEL_DIR = "models"
RESULT_DIR = "data"

N_BOOT = 1000
MODELS = ["logistic_regression", "random_forest"]


def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def tree_nodes(rf):
    return int(sum(t.tree_.node_count for t in rf.estimators_))


# --------------------------------------------------
# Load features and rebuild the same split as scripts/06
# --------------------------------------------------

print("Loading engineered features...")
X = np.load(FEATURE_PATH)["X"]
y = np.load(TARGET_PATH)
names = json.load(open(FEATURE_NAMES_PATH))

idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.20, random_state=42, stratify=y)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
print(f"Train {X_train.shape}, test {X_test.shape}")

final = {m: joblib.load(os.path.join(MODEL_DIR, f"{m}.pkl")) for m in MODELS}
probs = {m: final[m].predict_proba(X_test)[:, 1] for m in MODELS}
preds = {m: (probs[m] >= 0.5).astype(int) for m in MODELS}

# The saved models must reproduce the reported results exactly
reported = pd.read_csv(os.path.join(RESULT_DIR, "model_results.csv")).set_index("model")
for m in MODELS:
    got = metrics(y_test, preds[m], probs[m])
    for k, v in got.items():
        assert abs(v - reported.loc[m, k]) < 1e-9, (m, k, v, reported.loc[m, k])
print("Saved models reproduce data/model_results.csv")

# --------------------------------------------------
# 1. Bootstrap 95% confidence intervals on the test set
# --------------------------------------------------

print(f"\nBootstrap: {N_BOOT} resamples of the {len(y_test):,} test reviews...")


def one_boot(seed):
    i = np.random.default_rng(seed).integers(0, len(y_test), len(y_test))
    out = {m: metrics(y_test[i], preds[m][i], probs[m][i]) for m in MODELS}
    out["diff"] = {k: out["random_forest"][k] - out["logistic_regression"][k] for k in out["random_forest"]}
    return out


boots = Parallel(n_jobs=-1)(delayed(one_boot)(s) for s in range(N_BOOT))

ci_rows = []
for m in MODELS + ["diff"]:
    point = (metrics(y_test, preds[m], probs[m]) if m != "diff" else
             {k: metrics(y_test, preds["random_forest"], probs["random_forest"])[k]
              - metrics(y_test, preds["logistic_regression"], probs["logistic_regression"])[k]
              for k in reported.columns})
    for k in reported.columns:
        vals = np.array([b[m][k] for b in boots])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci_rows.append({"model": "random_forest_minus_logistic_regression" if m == "diff" else m,
                        "metric": k, "estimate": point[k], "ci_low": lo, "ci_high": hi,
                        "ci_half_width": (hi - lo) / 2})
ci = pd.DataFrame(ci_rows)
ci.round(5).to_csv(os.path.join(RESULT_DIR, "model_bootstrap_ci.csv"), index=False)
print(ci.round(4).to_string(index=False))

# --------------------------------------------------
# 2. Feature-group ablation (same split, same settings, test set used for reporting only)
# --------------------------------------------------

groups = {
    "text": [i for i, n in enumerate(names) if n.startswith("SVD_")],
    "issue": [i for i, n in enumerate(names) if n.startswith("issue_")],
    "sentiment": [i for i, n in enumerate(names) if n.startswith("sentiment_")],
    "length_upvotes": [names.index("review_length"), names.index("thumbs_up")],
}
variants = {
    "text_only": groups["text"],
    "sentiment_only": groups["sentiment"],
    "structural_only": groups["issue"] + groups["sentiment"] + groups["length_upvotes"],
    "all_except_sentiment": groups["text"] + groups["issue"] + groups["length_upvotes"],
    "all_features": list(range(len(names))),
}

ab_rows = []
for v, cols in variants.items():
    for m in MODELS:
        if v == "all_features":
            p = probs[m]                                  # the final model, already trained on all 215
        else:
            print(f"Ablation: {v} ({len(cols)} features), {m}...")
            model = clone(final[m]).fit(X_train[:, cols], y_train)
            p = model.predict_proba(X_test[:, cols])[:, 1]
        ab_rows.append({"features": v, "n_features": len(cols), "model": m,
                        **metrics(y_test, (p >= 0.5).astype(int), p)})
ablation = pd.DataFrame(ab_rows)
ablation.round(4).to_csv(os.path.join(RESULT_DIR, "model_ablation.csv"), index=False)
print("\nFeature-group ablation (test PR-AUC):")
print(ablation.pivot(index="features", columns="model", values="pr_auc").round(4))

# --------------------------------------------------
# 3. Settings check on a validation split carved from the training data
# --------------------------------------------------

i_fit, i_val = train_test_split(np.arange(len(y_train)), test_size=0.20, random_state=42, stratify=y_train)
X_fit, X_val, y_fit, y_val = X_train[i_fit], X_train[i_val], y_train[i_fit], y_train[i_val]
print(f"\nSettings check: fit on {len(y_fit):,} training rows, validate on {len(y_val):,} (test set untouched)")

grid = ([("logistic_regression", "C", c, c == 1.0) for c in [0.01, 0.1, 1.0, 10.0]] +
        [("random_forest", "min_samples_leaf", leaf, leaf == 100) for leaf in [25, 50, 100, 200]])

tune_rows = []
for m, param, value, current in grid:
    model = clone(final[m]).set_params(**{param: value}).fit(X_fit, y_fit)
    p = model.predict_proba(X_val)[:, 1]
    row = {"model": m, "parameter": param, "value": value, "current_setting": current,
           **metrics(y_val, (p >= 0.5).astype(int), p)}
    if m == "random_forest":
        row["tree_nodes"] = tree_nodes(model)
    tune_rows.append(row)
    print(f"{m} {param}={value}: val PR-AUC {row['pr_auc']:.4f}, F1 {row['f1_score']:.4f}")
tuning = pd.DataFrame(tune_rows)
cur_nodes = tuning.loc[(tuning["model"] == "random_forest") & tuning["current_setting"], "tree_nodes"].iloc[0]
tuning["size_vs_current"] = (tuning["tree_nodes"] / cur_nodes).round(2)
tuning.round(4).to_csv(os.path.join(RESULT_DIR, "model_tuning.csv"), index=False)

print("\nRobustness checks completed.")
