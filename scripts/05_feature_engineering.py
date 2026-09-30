import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

INPUT_PATH = "data/app_reviews_tagged.csv"
OUTPUT_PATH = "data/engineered_features.npz"
TARGET_PATH = "data/target.npy"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Load tagged dataset
# ---------------------------------------------------------

print("Loading tagged review dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# 3. Create target variable
# ---------------------------------------------------------

# Problematic review = rating <= 2
df["is_problematic"] = (df["score"] <= 2).astype(int)

y = df["is_problematic"].to_numpy()

print("\nTarget distribution:")
print(df["is_problematic"].value_counts())


# ---------------------------------------------------------
# 4. Text feature engineering: TF-IDF
# ---------------------------------------------------------

print("\nCreating TF-IDF features...")

text = df["content"].fillna("").astype(str)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(text)

print("TF-IDF shape:", X_tfidf.shape)


# ---------------------------------------------------------
# 5. Dimensionality reduction: Truncated SVD
# ---------------------------------------------------------

print("\nApplying Truncated SVD...")

svd = TruncatedSVD(
    n_components=100,
    random_state=42
)

X_svd = svd.fit_transform(X_tfidf)

print("SVD shape:", X_svd.shape)

print(
    "Explained variance ratio:",
    svd.explained_variance_ratio_.sum()
)


# ---------------------------------------------------------
# 6. Structural features
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

X_structural = (
    df[structural_cols]
    .fillna(0)
    .astype(float)
)

# Scale structural numerical features
scaler = StandardScaler()

X_structural_scaled = scaler.fit_transform(X_structural)


# ---------------------------------------------------------
# 7. Combine SVD + structural features
# ---------------------------------------------------------

print("\nCombining features...")

X = np.hstack([
    X_svd,
    X_structural_scaled
])

print("Final feature matrix shape:", X.shape)


# ---------------------------------------------------------
# 8. Save engineered features and target
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


# ---------------------------------------------------------
# 9. Save preprocessing objects
# ---------------------------------------------------------

joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

joblib.dump(
    svd,
    os.path.join(MODEL_DIR, "svd_model.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "structural_scaler.pkl")
)


# ---------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------

print("\nFeature engineering completed successfully.")

print("Final X shape:", X.shape)
print("Target shape:", y.shape)
print("Feature dataset:", OUTPUT_PATH)
print("Target dataset:", TARGET_PATH)
