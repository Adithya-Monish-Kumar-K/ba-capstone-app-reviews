import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

FEATURE_PATH = "data/engineered_features.npz"
TARGET_PATH = "data/target.npy"
ROW_INDEX_PATH = "data/model_row_index.csv.gz"

MODEL_DIR = "models"
RESULT_DIR = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_OF_TIME_CUTOFF = pd.Timestamp("2026-09-01")   # train on Apr-Aug, test on 1-20 Sep
CURVE_POINTS = 500


def make_models():
    # Random Forest is size-capped (subsampled trees, leaves >= 100 reviews) so the saved model
    # stays well under GitHub's 100 MB file limit on ~0.9M training rows.
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_samples=0.3,
            min_samples_leaf=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }


def metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan
    }


def downsample(*arrays, n=CURVE_POINTS):
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n).astype(int))
    return [np.asarray(a)[idx].round(5).tolist() for a in arrays]


# --------------------------------------------------
# Load engineered features
# --------------------------------------------------

print("Loading engineered features...")

X = np.load(FEATURE_PATH)["X"]
y = np.load(TARGET_PATH)
rows = pd.read_csv(ROW_INDEX_PATH, parse_dates=["review_date"])

print(f"Feature matrix: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nTarget distribution:")
print(pd.Series(y).value_counts().sort_index())

# --------------------------------------------------
# Train-test split (stratified, random 80/20)
# --------------------------------------------------

idx_train, idx_test = train_test_split(
    np.arange(len(y)),
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

print("\nTrain/Test split:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")

# --------------------------------------------------
# Baseline: always predict the majority class
# --------------------------------------------------

majority = int(pd.Series(y_train).mode()[0])
baseline = metrics(y_test, np.full_like(y_test, majority), np.full(len(y_test), float(majority)))
baseline = {"model": f"majority_class_baseline (always {majority})", **baseline}
pd.DataFrame([baseline]).to_csv(os.path.join(RESULT_DIR, "baseline_results.csv"), index=False)
print(f"\nMajority-class baseline accuracy: {baseline['accuracy']:.4f} (recall {baseline['recall']:.4f})")

# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

models = make_models()
results = []
predictions = {}

for model_name, model in models.items():

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    m = metrics(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    for k, v in m.items():
        print(f"{k:9s}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append({"model": model_name, **m})

    predictions[model_name] = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm.tolist()
    }

    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path, compress=3)

    print(f"\nSaved model: {model_path} ({os.path.getsize(model_path) / 1e6:.1f} MB)")

# --------------------------------------------------
# Save model results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(RESULT_DIR, "model_results.csv")

results_df.to_csv(results_path, index=False)

print("\nModel Results:")
print(results_df.to_string(index=False))

print(f"\nSaved results: {results_path}")

# --------------------------------------------------
# Save detailed predictions
# --------------------------------------------------

test_rows = rows.iloc[idx_test].reset_index(drop=True)

prediction_data = pd.DataFrame({
    "app_name": test_rows["app_name"],
    "domain": test_rows["domain"],
    "actual": y_test
})

for model_name in models:
    prediction_data[f"{model_name}_prediction"] = predictions[model_name]["y_pred"]
    prediction_data[f"{model_name}_probability"] = predictions[model_name]["y_prob"].round(4)

prediction_path = os.path.join(RESULT_DIR, "model_predictions.csv.gz")

prediction_data.to_csv(prediction_path, index=False, compression="gzip")

print(f"Saved predictions: {prediction_path}")

# --------------------------------------------------
# Per-app and per-domain results on the same test set
# --------------------------------------------------

group_rows = []

for level in ["domain", "app_name"]:
    for group, g in prediction_data.groupby(level):
        for model_name in models:
            m = metrics(g["actual"], g[f"{model_name}_prediction"], g[f"{model_name}_probability"])
            group_rows.append({
                "level": level,
                "group": group,
                "model": model_name,
                "n_test": len(g),
                "positive_rate": g["actual"].mean(),
                **m
            })

by_group = pd.DataFrame(group_rows)
by_group.round(4).to_csv(os.path.join(RESULT_DIR, "model_results_by_group.csv"), index=False)

print("\nPR-AUC by domain:")
print(by_group[by_group["level"] == "domain"].pivot(index="group", columns="model", values="pr_auc").round(3))

# --------------------------------------------------
# Out-of-time check: train on Apr-Aug, test on 1-20 Sep
# --------------------------------------------------

print(f"\n{'=' * 60}")
print(f"Out-of-time check: train < {OUT_OF_TIME_CUTOFF:%Y-%m-%d} <= test")
print(f"{'=' * 60}")

past = (rows["review_date"] < OUT_OF_TIME_CUTOFF).to_numpy()
oot_rows = []

for model_name, model in make_models().items():
    model.fit(X[past], y[past])
    y_prob = model.predict_proba(X[~past])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    m = metrics(y[~past], y_pred, y_prob)
    oot_rows.append({
        "model": model_name,
        "n_train": int(past.sum()),
        "n_test": int((~past).sum()),
        "test_positive_rate": float(y[~past].mean()),
        **m
    })
    print(f"{model_name}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()))

pd.DataFrame(oot_rows).round(4).to_csv(os.path.join(RESULT_DIR, "model_results_out_of_time.csv"), index=False)

# --------------------------------------------------
# Save evaluation curves (downsampled)
# --------------------------------------------------

curve_data = {}

for model_name in models:

    y_prob = predictions[model_name]["y_prob"]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fpr, tpr = downsample(fpr, tpr)
    precision, recall = downsample(precision, recall)

    curve_data[model_name] = {
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "pr_precision": precision,
        "pr_recall": recall
    }

curve_data["positive_rate"] = float(y_test.mean())

curve_path = os.path.join(RESULT_DIR, "model_curve_data.json")

with open(curve_path, "w") as f:
    json.dump(curve_data, f)

print(f"Saved curve data: {curve_path}")

print("\nClassifier training and evaluation completed.")
