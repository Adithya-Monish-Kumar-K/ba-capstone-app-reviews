import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_DIR = "models"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# Load trained models and feature-engineering artifacts
# --------------------------------------------------

logistic_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

tfidf = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

svd = joblib.load(
    os.path.join(MODEL_DIR, "svd_model.pkl")
)

print("Loaded trained models and feature-engineering artifacts.")

# --------------------------------------------------
# 1. Logistic Regression feature importance
# --------------------------------------------------

logistic_coefficients = logistic_model.coef_[0]

feature_importance = pd.DataFrame({
    "feature_index": np.arange(len(logistic_coefficients)),
    "coefficient": logistic_coefficients,
    "absolute_coefficient": np.abs(logistic_coefficients)
})

top_logistic = feature_importance.sort_values(
    "absolute_coefficient",
    ascending=False
).head(20)

plt.figure(figsize=(10, 7))

plt.barh(
    range(len(top_logistic)),
    top_logistic["coefficient"].values
)

plt.yticks(
    range(len(top_logistic)),
    [f"SVD_{i}" for i in top_logistic["feature_index"]]
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 20 Logistic Regression Features")
plt.gca().invert_yaxis()
plt.tight_layout()

output_path = os.path.join(
    FIGURE_DIR,
    "logistic_regression_feature_importance.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")

# --------------------------------------------------
# 2. Random Forest feature importance
# --------------------------------------------------

rf_importance = pd.DataFrame({
    "feature_index": np.arange(
        len(random_forest_model.feature_importances_)
    ),
    "importance": random_forest_model.feature_importances_
})

top_rf = rf_importance.sort_values(
    "importance",
    ascending=False
).head(20)

plt.figure(figsize=(10, 7))

plt.barh(
    range(len(top_rf)),
    top_rf["importance"].values
)

plt.yticks(
    range(len(top_rf)),
    [f"SVD_{i}" for i in top_rf["feature_index"]]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Random Forest Features")
plt.gca().invert_yaxis()
plt.tight_layout()

output_path = os.path.join(
    FIGURE_DIR,
    "random_forest_feature_importance.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")

# --------------------------------------------------
# 3. Top TF-IDF terms
# --------------------------------------------------

terms = np.array(tfidf.get_feature_names_out())

components = svd.components_

term_scores = np.mean(
    np.abs(components),
    axis=0
)

top_term_indices = np.argsort(
    term_scores
)[-20:][::-1]

top_terms = pd.DataFrame({
    "term": terms[top_term_indices],
    "importance": term_scores[top_term_indices]
})

plt.figure(figsize=(10, 7))

plt.barh(
    range(len(top_terms)),
    top_terms["importance"].values
)

plt.yticks(
    range(len(top_terms)),
    top_terms["term"]
)

plt.xlabel("Mean Absolute SVD Loading")
plt.ylabel("TF-IDF Term")
plt.title("Top 20 TF-IDF Terms by SVD Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

output_path = os.path.join(
    FIGURE_DIR,
    "top_tfidf_terms.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")

# --------------------------------------------------
# 4. Save interpretation data
# --------------------------------------------------

top_logistic.to_csv(
    "data/logistic_feature_importance.csv",
    index=False
)

top_rf.to_csv(
    "data/random_forest_feature_importance.csv",
    index=False
)

top_terms.to_csv(
    "data/top_tfidf_terms.csv",
    index=False
)

print("Saved interpretation data files.")

print("\nModel interpretation completed.")
