import os
import json
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

MODEL_DIR = "models"
RESULT_DIR = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# --------------------------------------------------
# Load engineered features
# --------------------------------------------------

print("Loading engineered features...")

X = np.load(FEATURE_PATH)["X"]
y = np.load(TARGET_PATH)

print(f"Feature matrix: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nTarget distribution:")
print(pd.Series(y).value_counts().sort_index())

# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")

# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = []

predictions = {}

# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

for model_name, model in models.items():

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

    predictions[model_name] = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm.tolist()
    }

    # Save trained model
    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    import joblib
    joblib.dump(model, model_path)

    print(f"\nSaved model: {model_path}")

# --------------------------------------------------
# Save model results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    RESULT_DIR,
    "model_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\nModel Results:")
print(results_df.to_string(index=False))

print(f"\nSaved results: {results_path}")

# --------------------------------------------------
# Save detailed predictions
# --------------------------------------------------

prediction_data = pd.DataFrame({
    "actual": y_test
})

for model_name in models:
    prediction_data[f"{model_name}_prediction"] = predictions[model_name]["y_pred"]
    prediction_data[f"{model_name}_probability"] = predictions[model_name]["y_prob"]

prediction_path = os.path.join(
    RESULT_DIR,
    "model_predictions.csv"
)

prediction_data.to_csv(prediction_path, index=False)

print(f"Saved predictions: {prediction_path}")

# --------------------------------------------------
# Save evaluation curves
# --------------------------------------------------

curve_data = {}

for model_name in models:

    y_prob = predictions[model_name]["y_prob"]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    curve_data[model_name] = {
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "pr_precision": precision.tolist(),
        "pr_recall": recall.tolist()
    }

curve_path = os.path.join(
    RESULT_DIR,
    "model_curve_data.json"
)

with open(curve_path, "w") as f:
    json.dump(curve_data, f)

print(f"Saved curve data: {curve_path}")

print("\nClassifier training and evaluation completed.")
