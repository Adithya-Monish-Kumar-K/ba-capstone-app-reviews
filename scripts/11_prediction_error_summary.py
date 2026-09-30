import pandas as pd
from sklearn.metrics import confusion_matrix

PREDICTION_PATH = "data/model_predictions.csv"
OUTPUT_PATH = "data/prediction_error_summary.csv"

predictions = pd.read_csv(PREDICTION_PATH)

models = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest"
}

rows = []

for model_key, model_name in models.items():
    y_true = predictions["actual"]
    y_pred = predictions[f"{model_key}_prediction"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    rows.append({
        "model": model_name,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    })

summary = pd.DataFrame(rows)

summary.to_csv(OUTPUT_PATH, index=False)

print("Prediction error summary:")
print(summary)

print(f"\nSaved: {OUTPUT_PATH}")
