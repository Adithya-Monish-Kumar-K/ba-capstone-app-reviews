import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

RESULTS_PATH = "data/model_results.csv"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# Load model results
# --------------------------------------------------

results = pd.read_csv(RESULTS_PATH)

print("Loaded model results:")
print(results)

# --------------------------------------------------
# Prepare metrics
# --------------------------------------------------

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc"
]

model_labels = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest"
}

results["model"] = results["model"].map(model_labels)

# --------------------------------------------------
# Create comparison chart
# --------------------------------------------------

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    results.iloc[0][metrics],
    width=width,
    label=results.iloc[0]["model"]
)

plt.bar(
    [i + width / 2 for i in x],
    results.iloc[1][metrics],
    width=width,
    label=results.iloc[1]["model"]
)

plt.xticks(
    list(x),
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
)

plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()

output_path = os.path.join(
    FIGURE_DIR,
    "model_performance_comparison.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")
print("\nModel performance comparison generated successfully.")
