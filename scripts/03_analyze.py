import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score,
                              roc_curve, precision_recall_curve, auc)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

BASE = "/home/Adithya/Desktop/Sem 7/BA/BAProject/topic1_app_review_dataset"
FIG = f"{BASE}/figures"
import os
os.makedirs(FIG, exist_ok=True)

RESULTS = {}

# ============================================================
# 1. LOAD + CLEAN
# ============================================================
df = pd.read_csv(f"{BASE}/data/app_reviews_tagged.csv")
df["review_date"] = pd.to_datetime(df["review_date"])
n_raw = len(df)

df["content"] = df["content"].fillna("").astype(str)
df = df.drop_duplicates(subset="review_id")
n_after_dedupe = len(df)

df = df[df["content"].str.strip().str.len() >= 3]
n_after_empty = len(df)

def ascii_ratio(s):
    if len(s) == 0:
        return 0
    return sum(c.isascii() for c in s) / len(s)

df["ascii_ratio"] = df["content"].apply(ascii_ratio)
df = df[df["ascii_ratio"] >= 0.85]
n_after_lang = len(df)

df["month"] = df["review_date"].dt.to_period("M").astype(str)
df["review_length"] = df["content"].str.split().str.len()

RESULTS["cleaning"] = {
    "raw_rows": n_raw,
    "after_dedupe": n_after_dedupe,
    "after_empty_removed": n_after_empty,
    "after_non_english_removed": n_after_lang,
    "final_rows": len(df),
}

# month-level sample size filter (used only for the monthly trend / time-series view)
month_counts = df.groupby(["app_name", "month"]).size().rename("n").reset_index()
reliable_months = set(zip(month_counts[month_counts["n"] >= 30]["app_name"],
                           month_counts[month_counts["n"] >= 30]["month"]))
df["month_reliable"] = df.apply(lambda r: (r["app_name"], r["month"]) in reliable_months, axis=1)

df.to_csv(f"{BASE}/data/app_reviews_clean.csv", index=False)
print("Cleaning:", RESULTS["cleaning"])

# ============================================================
# 2. EDA
# ============================================================
ISSUE_COLS = [c for c in df.columns if c.startswith("issue_")]

# 2a. rating distribution per app
fig, ax = plt.subplots(figsize=(8, 4.5))
order = df.groupby("app_name")["score"].mean().sort_values().index
sns.countplot(data=df, x="score", hue="app_name", hue_order=order, ax=ax)
ax.set_title("Rating Distribution by App")
ax.set_xlabel("Star rating"); ax.set_ylabel("Number of reviews")
plt.tight_layout(); plt.savefig(f"{FIG}/01_rating_distribution.png"); plt.close()

# 2b. review volume over time
monthly_vol = df.groupby(["app_name", "month"]).size().reset_index(name="review_count")
fig, ax = plt.subplots(figsize=(10, 4.5))
for name, g in monthly_vol.groupby("app_name"):
    g = g.sort_values("month")
    ax.plot(g["month"], g["review_count"], marker="o", label=name, alpha=0.85)
ax.set_title("Monthly Review Volume by App")
ax.set_ylabel("Reviews per month"); ax.tick_params(axis="x", rotation=75)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/02_review_volume.png"); plt.close()

# 2c. issue category frequency
issue_rates = df[ISSUE_COLS].mean().sort_values(ascending=False)
issue_rates.index = [c.replace("issue_", "") for c in issue_rates.index]
fig, ax = plt.subplots(figsize=(8, 4.5))
issue_rates.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[0])
ax.set_title("Share of Reviews Mentioning Each Issue Category")
ax.set_xlabel("Fraction of all reviews")
plt.tight_layout(); plt.savefig(f"{FIG}/03_issue_category_frequency.png"); plt.close()

# 2d. monthly avg rating vs issue-mention rate (reliable months only) - the KEY chart
df["any_issue"] = df[ISSUE_COLS].any(axis=1)
rel = df[df["month_reliable"]]
monthly_rel = rel.groupby(["app_name", "month"]).agg(
    avg_score=("score", "mean"), issue_rate=("any_issue", "mean"), n=("score", "size")
).reset_index().sort_values(["app_name", "month"])
monthly_rel.to_csv(f"{BASE}/data/monthly_trend_reliable.csv", index=False)

apps = monthly_rel["app_name"].unique()
fig, axes = plt.subplots(len(apps), 1, figsize=(10, 2.6 * len(apps)), sharex=False)
for ax, name in zip(axes, apps):
    g = monthly_rel[monthly_rel["app_name"] == name]
    ax2 = ax.twinx()
    ax.plot(g["month"], g["avg_score"], color="#1f6f6f", marker="o", label="avg rating")
    ax2.plot(g["month"], g["issue_rate"], color="#b23a34", marker="s", linestyle="--", label="issue mention rate")
    ax.set_ylabel("avg rating", color="#1f6f6f"); ax2.set_ylabel("issue rate", color="#b23a34")
    ax.set_title(name, fontsize=10, loc="left")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/04_rating_vs_issue_trend.png"); plt.close()

# correlation per app (reliable months only)
corr_results = {}
for name, g in monthly_rel.groupby("app_name"):
    if len(g) >= 4:
        corr_results[name] = round(float(g["avg_score"].corr(g["issue_rate"])), 3)
RESULTS["monthly_corr_avg_score_vs_issue_rate"] = corr_results
print("Correlation (reliable months only):", corr_results)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
df["is_problematic"] = (df["score"] <= 2).astype(int)

X_text = df["content"]
tfidf = TfidfVectorizer(max_features=800, stop_words="english", ngram_range=(1, 2), min_df=5)
X_tfidf = tfidf.fit_transform(X_text)

svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_tfidf)
RESULTS["dimensionality_reduction"] = {
    "tfidf_features": X_tfidf.shape[1],
    "svd_components": 50,
    "explained_variance_ratio_sum": round(float(svd.explained_variance_ratio_.sum()), 3),
}

extra_numeric = df[ISSUE_COLS + ["review_length", "thumbs_up"]].fillna(0).values
X_full = np.hstack([X_svd, extra_numeric])
feature_names = [f"svd_{i}" for i in range(50)] + ISSUE_COLS + ["review_length", "thumbs_up"]

y = df["is_problematic"].values

# ============================================================
# 4. PREDICTIVE MODEL
# ============================================================
X_train, X_test, y_train, y_test, svd_train, svd_test = train_test_split(
    X_full, y, X_svd, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)
pred_lr = log_reg.predict(X_test)
proba_lr = log_reg.predict_proba(X_test)[:, 1]

rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

def eval_model(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "f1_problematic_class": round(f1_score(y_true, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }

RESULTS["model_performance"] = [
    eval_model("Logistic Regression", y_test, pred_lr, proba_lr),
    eval_model("Random Forest", y_test, pred_rf, proba_rf),
]
print("Model performance:", RESULTS["model_performance"])

RESULTS["confusion_matrix_logreg"] = confusion_matrix(y_test, pred_lr).tolist()
RESULTS["classification_report_logreg"] = classification_report(y_test, pred_lr, target_names=["OK (3-5*)", "Problematic (1-2*)"], output_dict=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, cm) in zip(axes, [("Logistic Regression", confusion_matrix(y_test, pred_lr)),
                                   ("Random Forest", confusion_matrix(y_test, pred_rf))]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["OK", "Problematic"], yticklabels=["OK", "Problematic"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(f"{FIG}/05_confusion_matrices.png"); plt.close()

# feature importance (RF) - map svd components back approx via top issue flags + top tfidf terms via LR coefficients on original tfidf space
lr_full_text = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_full_text.fit(X_tfidf, y)
coefs = lr_full_text.coef_[0]
terms = np.array(tfidf.get_feature_names_out())
top_pos_idx = np.argsort(coefs)[-15:][::-1]
top_neg_idx = np.argsort(coefs)[:15]
RESULTS["top_terms_driving_problematic_reviews"] = list(zip(terms[top_pos_idx].tolist(), coefs[top_pos_idx].round(3).tolist()))
RESULTS["top_terms_driving_positive_reviews"] = list(zip(terms[top_neg_idx].tolist(), coefs[top_neg_idx].round(3).tolist()))

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(15)
ax.barh(y_pos, coefs[top_pos_idx][::-1], color="#b23a34")
ax.set_yticks(y_pos); ax.set_yticklabels(terms[top_pos_idx][::-1])
ax.set_title("Top Words/Phrases Driving 'Problematic Review' Prediction")
ax.set_xlabel("Logistic regression coefficient (higher = more predictive of 1-2 stars)")
plt.tight_layout(); plt.savefig(f"{FIG}/06_top_negative_terms.png"); plt.close()

RF_importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 5))
RF_importance[::-1].plot(kind="barh", ax=ax, color="#0f6e6e")
ax.set_title("Random Forest — Top 15 Feature Importances")
plt.tight_layout(); plt.savefig(f"{FIG}/07_rf_feature_importance.png"); plt.close()

# ============================================================
# 5. ADDITIONAL VISUALIZATIONS (ROC/PR, dimensionality-reduction
#    projection, per-app issue breakdown, review length vs rating)
# ============================================================

# 5a. ROC curves
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, proba, color in [("Logistic Regression", proba_lr, "#1C7293"), ("Random Forest", proba_rf, "#E4572E")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color=color, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="#B0B8BA", linewidth=1, label="Random baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Problematic Review Classifier")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/08_roc_curve.png"); plt.close()

# 5b. Precision-Recall curve (matters more given the 74/26 class split)
fig, ax = plt.subplots(figsize=(6.5, 6))
base_rate = y_test.mean()
for name, proba, color in [("Logistic Regression", proba_lr, "#1C7293"), ("Random Forest", proba_rf, "#E4572E")]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, label=f"{name} (AUC={auc(rec, prec):.3f})", color=color, linewidth=2)
ax.axhline(base_rate, linestyle="--", color="#B0B8BA", linewidth=1, label=f"Baseline (class rate={base_rate:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Problematic Review Classifier")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/09_precision_recall_curve.png"); plt.close()

# 5c. SVD (dimensionality reduction) 2D projection, colored by class
fig, ax = plt.subplots(figsize=(8, 6.5))
idx_ok, idx_prob = y_test == 0, y_test == 1
ax.scatter(svd_test[idx_ok, 0], svd_test[idx_ok, 1], s=8, alpha=0.35, color="#1C7293", label="OK (3-5 stars)")
ax.scatter(svd_test[idx_prob, 0], svd_test[idx_prob, 1], s=8, alpha=0.35, color="#E4572E", label="Problematic (1-2 stars)")
ax.set_xlabel("SVD component 1"); ax.set_ylabel("SVD component 2")
ax.set_title("Review Text in Reduced Space (first 2 of 50 SVD components)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/10_svd_projection.png"); plt.close()

# 5d. Issue-category composition, per app (pooled view in 2c hides this)
per_app_issue = df.groupby("app_name")[ISSUE_COLS].mean()
per_app_issue.columns = [c.replace("issue_", "") for c in per_app_issue.columns]
top_issues = per_app_issue.mean().sort_values(ascending=False).head(6).index
fig, ax = plt.subplots(figsize=(10, 5.5))
per_app_issue[top_issues].plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(top_issues)))
ax.set_ylabel("Fraction of that app's reviews")
ax.set_title("Which Issues Dominate, Per App (top 6 categories)")
ax.legend(title=None, fontsize=9, ncol=3)
plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{FIG}/11_issue_breakdown_by_app.png"); plt.close()

# 5e. Review length vs. rating
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(data=df, x="score", y="review_length", ax=ax, color="#1C7293", showfliers=False)
ax.set_xlabel("Star rating"); ax.set_ylabel("Review length (words)")
ax.set_title("Do Angrier Reviews Run Longer?")
plt.tight_layout(); plt.savefig(f"{FIG}/12_review_length_vs_rating.png"); plt.close()

RESULTS["pr_auc"] = {
    "logistic_regression": round(float(auc(*precision_recall_curve(y_test, proba_lr)[1::-1])), 3),
    "random_forest": round(float(auc(*precision_recall_curve(y_test, proba_rf)[1::-1])), 3),
}
RESULTS["test_set_problematic_rate"] = round(float(base_rate), 3)
RESULTS["review_length_median_by_score"] = df.groupby("score")["review_length"].median().to_dict()
RESULTS["issue_breakdown_by_app"] = per_app_issue[top_issues].round(4).to_dict(orient="index")

with open(f"{BASE}/data/analysis_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("\nAll figures saved to:", FIG)
print("Results JSON saved to:", f"{BASE}/data/analysis_results.json")
print("\nFINAL feature/row counts: X_full shape", X_full.shape, "positive class rate:", round(y.mean(), 3))
print("PR-AUC:", RESULTS["pr_auc"])
print("Review length median by score:", RESULTS["review_length_median_by_score"])
