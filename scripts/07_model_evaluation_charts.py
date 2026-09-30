import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PREDICTION_PATH = "data/model_predictions.csv"
CURVE_PATH = "data/model_curve_data.json"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# Load prediction and curve data
# --------------------------------------------------

predictions = pd.read_csv(PREDICTION_PATH)

with open(CURVE_PATH, "r") as f:
    curve_data = json.load(f)

print("Loaded prediction data:", predictions.shape)

# --------------------------------------------------
# Model names
# --------------------------------------------------

models = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest"
}

# --------------------------------------------------
# 1. Confusion Matrix
# --------------------------------------------------

for model_key, model_name in models.items():

    y_true = predictions["actual"]
    y_pred = predictions[f"{model_key}_prediction"]

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Problematic", "Problematic"]
    )

    disp.plot()

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        f"{model_key}_confusion_matrix.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

# --------------------------------------------------
# 2. ROC Curves
# --------------------------------------------------

plt.figure()

for model_key, model_name in models.items():

    fpr = curve_data[model_key]["roc_fpr"]
    tpr = curve_data[model_key]["roc_tpr"]

    plt.plot(
        fpr,
        tpr,
        label=model_name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(
    FIGURE_DIR,
    "roc_curves.png"
)

plt.savefig(roc_path, dpi=300)
plt.close()

print(f"Saved: {roc_path}")

# --------------------------------------------------
# 3. Precision-Recall Curves
# --------------------------------------------------

plt.figure()

for model_key, model_name in models.items():

    precision = curve_data[model_key]["pr_precision"]
    recall = curve_data[model_key]["pr_recall"]

    plt.plot(
        recall,
        precision,
        label=model_name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()

pr_path = os.path.join(
    FIGURE_DIR,
    "precision_recall_curves.png"
)

plt.savefig(pr_path, dpi=300)
plt.close()

print(f"Saved: {pr_path}")

print("\nModel evaluation charts generated successfully.")
