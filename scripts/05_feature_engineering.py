import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import read_stage  # noqa: E402


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

OUTPUT_PATH = "data/engineered_features.npz"
TARGET_PATH = "data/target.npy"
ROW_INDEX_PATH = "data/model_row_index.csv.gz"      # app / domain / date per row, for per-app and out-of-time evaluation
FEATURE_NAMES_PATH = "data/feature_names.json"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Structural feature definitions
# ---------------------------------------------------------

issue_cols = [
    "issue_crash_bugs_stability",
    "issue_payment_refund",
    "issue_delivery_delay",
    "issue_order_quality_fulfillment",
    "issue_cancellation_return",
    "issue_customer_support",
    "issue_account_login_otp",
    "issue_pricing_charges_fraud",
    "issue_ui_ux_update"
]

sentiment_cols = [
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_pos",
    "sentiment_compound"
]

numeric_cols = [
    "review_length",
    "thumbs_up"
]

structural_cols = issue_cols + sentiment_cols + numeric_cols


# ---------------------------------------------------------
# 3. Load tagged dataset
# ---------------------------------------------------------

print("Loading tagged review dataset (data/tagged/*.csv.gz)...")

df = read_stage(
    "tagged",
    columns=["app_name", "domain", "review_date", "score", "content"] + structural_cols
)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# 4. Create target variable
# ---------------------------------------------------------

# Problematic review = rating <= 2
df["is_problematic"] = (df["score"] <= 2).astype(int)

y = df["is_problematic"].to_numpy()

print("\nTarget distribution:")
print(df["is_problematic"].value_counts())


# ---------------------------------------------------------
# 5. Text feature engineering: TF-IDF
# ---------------------------------------------------------

print("\nCreating TF-IDF features...")

text = df["content"].fillna("").astype(str)

# No stop-word list: the standard English list removes "not", "no" and "never", so "not good" would look like "good".
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32
)

X_tfidf = tfidf.fit_transform(text)

print("TF-IDF shape:", X_tfidf.shape)


# ---------------------------------------------------------
# 6. Dimensionality reduction: Truncated SVD
# ---------------------------------------------------------

print("\nApplying Truncated SVD...")

svd = TruncatedSVD(
    n_components=200,
    random_state=42
)

X_svd = svd.fit_transform(X_tfidf)

print("SVD shape:", X_svd.shape)

print(
    "Explained variance ratio:",
    svd.explained_variance_ratio_.sum()
)


# ---------------------------------------------------------
# 7. Structural features
# ---------------------------------------------------------

X_structural = (
    df[structural_cols]
    .fillna(0)
    .astype(float)
)

# Scale structural numerical features
scaler = StandardScaler()

X_structural_scaled = scaler.fit_transform(X_structural)


# ---------------------------------------------------------
# 8. Combine SVD + structural features
# ---------------------------------------------------------

print("\nCombining features...")

X = np.hstack([
    X_svd,
    X_structural_scaled
]).astype(np.float32)

feature_names = [f"SVD_{i}" for i in range(X_svd.shape[1])] + structural_cols

print("Final feature matrix shape:", X.shape)


# ---------------------------------------------------------
# 9. Save engineered features, target and row index
# ---------------------------------------------------------

print("\nSaving engineered features...")

np.savez_compressed(
    OUTPUT_PATH,
    X=X
)

np.save(
    TARGET_PATH,
    y
)

df[["app_name", "domain", "review_date"]].to_csv(ROW_INDEX_PATH, index=False, compression="gzip")

with open(FEATURE_NAMES_PATH, "w") as f:
    json.dump(feature_names, f)

with open("data/feature_engineering_summary.json", "w") as f:
    json.dump({
        "rows": int(X.shape[0]),
        "tfidf_features": int(X_tfidf.shape[1]),
        "svd_components": int(X_svd.shape[1]),
        "svd_explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
        "structural_features": len(structural_cols),
        "total_features": int(X.shape[1]),
        "positive_class_rate": round(float(y.mean()), 4)
    }, f, indent=2)


# ---------------------------------------------------------
# 10. Save preprocessing objects
# ---------------------------------------------------------

joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    compress=3
)

joblib.dump(
    svd,
    os.path.join(MODEL_DIR, "svd_model.pkl"),
    compress=3
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "structural_scaler.pkl")
)


# ---------------------------------------------------------
# 11. Final summary
# ---------------------------------------------------------

print("\nFeature engineering completed successfully.")

print("Final X shape:", X.shape)
print("Target shape:", y.shape)
print("Feature dataset:", OUTPUT_PATH)
print("Target dataset:", TARGET_PATH)
