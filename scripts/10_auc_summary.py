import pandas as pd

RESULTS_PATH = "data/model_results.csv"
OUTPUT_PATH = "data/auc_summary.csv"

# Load model evaluation results
results = pd.read_csv(RESULTS_PATH)

# Select the AUC metrics
auc_summary = results[
    ["model", "roc_auc", "pr_auc"]
].copy()

# Convert model names for readability
auc_summary["model"] = auc_summary["model"].replace({
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest"
})

# Save summary
auc_summary.to_csv(OUTPUT_PATH, index=False)

print("AUC summary:")
print(auc_summary)
print(f"\nSaved: {OUTPUT_PATH}")
