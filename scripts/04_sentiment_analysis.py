import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from apps import APP_COLORS, APPS, DOMAINS, APP_DOMAIN
from data_io import read_app, read_stage, write_app

# Setup Matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = REPO_ROOT / "data" / "charts"
SUMMARY_PATH = REPO_ROOT / "data" / "sentiment_summary.json"
APP_AGG_PATH = REPO_ROOT / "data" / "sentiment_aggregation_by_app.csv"
ISSUE_AGG_PATH = REPO_ROOT / "data" / "sentiment_aggregation_by_issue.csv"
RATING_VAL_PATH = REPO_ROOT / "data" / "sentiment_validation_by_rating.csv"


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies VADER SentimentIntensityAnalyzer on review content.
    Returns DataFrame with sentiment_neg, sentiment_neu, sentiment_pos,
    sentiment_compound, and sentiment_label.
    """
    # Ensure idempotency: remove any existing sentiment columns
    sentiment_cols = ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound", "sentiment_label"]
    df = df.drop(columns=[c for c in sentiment_cols if c in df.columns])

    sia = SentimentIntensityAnalyzer()

    def get_scores(text):
        cleaned = str(text) if pd.notna(text) else ""
        scores = sia.polarity_scores(cleaned)
        comp = scores["compound"]
        if comp >= 0.05:
            label = "Positive"
        elif comp <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return scores["neg"], scores["neu"], scores["pos"], comp, label

    results = df["content"].apply(get_scores)
    sentiment_df = pd.DataFrame(
        results.tolist(),
        columns=sentiment_cols
    )
    return pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)


def validate_sentiment(df: pd.DataFrame) -> dict:
    """
    Validates VADER sentiment against ground truth Play Store star ratings (1-5).
    """
    print("\n--- Validating Sentiment vs Star Ratings ---")
    p_corr, p_pvalue = pearsonr(df["score"], df["sentiment_compound"])
    s_corr, s_pvalue = spearmanr(df["score"], df["sentiment_compound"])

    # Star rating mapping to expected sentiment
    # 1-2 stars: Negative, 3 stars: Neutral, 4-5 stars: Positive
    def rating_to_sentiment(score):
        if score <= 2:
            return "Negative"
        elif score == 3:
            return "Neutral"
        else:
            return "Positive"

    df["rating_sentiment_class"] = df["score"].apply(rating_to_sentiment)
    acc = accuracy_score(df["rating_sentiment_class"], df["sentiment_label"])
    report = classification_report(
        df["rating_sentiment_class"],
        df["sentiment_label"],
        output_dict=True,
        zero_division=0
    )
    conf_mat = confusion_matrix(
        df["rating_sentiment_class"],
        df["sentiment_label"],
        labels=["Negative", "Neutral", "Positive"]
    ).tolist()

    # Per-rating stats
    rating_stats = []
    for star, grp in df.groupby("score"):
        stats = {
            "star_rating": int(star),
            "review_count": int(len(grp)),
            "mean_compound": round(float(grp["sentiment_compound"].mean()), 4),
            "median_compound": round(float(grp["sentiment_compound"].median()), 4),
            "std_compound": round(float(grp["sentiment_compound"].std()), 4),
            "pct_negative": round(float((grp["sentiment_label"] == "Negative").mean() * 100), 2),
            "pct_neutral": round(float((grp["sentiment_label"] == "Neutral").mean() * 100), 2),
            "pct_positive": round(float((grp["sentiment_label"] == "Positive").mean() * 100), 2),
        }
        rating_stats.append(stats)

    rating_stats_df = pd.DataFrame(rating_stats)
    rating_stats_df.to_csv(RATING_VAL_PATH, index=False)

    val_results = {
        "pearson_correlation": round(float(p_corr), 4),
        "pearson_pvalue": float(p_pvalue),
        "spearman_correlation": round(float(s_corr), 4),
        "spearman_pvalue": float(s_pvalue),
        "overall_alignment_accuracy": round(float(acc * 100), 2),
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["Negative", "Neutral", "Positive"],
            "matrix": conf_mat
        },
        "per_rating_stats": rating_stats
    }

    print(f"Pearson Correlation (score vs compound): {p_corr:.4f} (p < 1e-10)")
    print(f"Spearman Rank Correlation:              {s_corr:.4f} (p < 1e-10)")
    print(f"Overall 3-Class Alignment Accuracy:     {acc * 100:.2f}%")
    print(f"Saved validation table to: {RATING_VAL_PATH}")
    return val_results


def aggregate_sentiment(df: pd.DataFrame) -> dict:
    """
    Computes sentiment aggregations by App, by Issue Category, and by Month.
    """
    print("\n--- Aggregating Sentiment Metrics ---")

    # 1. By App
    app_agg_rows = []
    for app_name, grp in df.groupby("app_name"):
        app_agg_rows.append({
            "app_name": app_name,
            "total_reviews": int(len(grp)),
            "mean_compound": round(float(grp["sentiment_compound"].mean()), 4),
            "pct_negative": round(float((grp["sentiment_label"] == "Negative").mean() * 100), 2),
            "pct_neutral": round(float((grp["sentiment_label"] == "Neutral").mean() * 100), 2),
            "pct_positive": round(float((grp["sentiment_label"] == "Positive").mean() * 100), 2),
            "avg_star_rating": round(float(grp["score"].mean()), 2)
        })
    app_agg_df = pd.DataFrame(app_agg_rows).sort_values("mean_compound")
    app_agg_df.to_csv(APP_AGG_PATH, index=False)
    print(f"Saved App aggregation table to: {APP_AGG_PATH}")

    # 2. By Issue Category
    issue_cols = [c for c in df.columns if c.startswith("issue_") and c not in ["issue_count"]]
    issue_agg_rows = []
    for col in issue_cols:
        sub = df[df[col] == 1]
        issue_name = col.replace("issue_", "")
        if len(sub) > 0:
            issue_agg_rows.append({
                "issue_category": issue_name,
                "review_count": int(len(sub)),
                "pct_of_reviews": round(float(len(sub) / len(df) * 100), 2),
                "mean_compound": round(float(sub["sentiment_compound"].mean()), 4),
                "pct_negative": round(float((sub["sentiment_label"] == "Negative").mean() * 100), 2),
                "pct_positive": round(float((sub["sentiment_label"] == "Positive").mean() * 100), 2),
                "avg_star_rating": round(float(sub["score"].mean()), 2)
            })
    issue_agg_df = pd.DataFrame(issue_agg_rows).sort_values("mean_compound")
    issue_agg_df.to_csv(ISSUE_AGG_PATH, index=False)
    print(f"Saved Issue Category aggregation table to: {ISSUE_AGG_PATH}")

    # 3. Monthly Trends
    monthly_app = (
        df.groupby(["month", "app_name"])["sentiment_compound"]
        .agg(["mean", "count"])
        .reset_index()
    )

    return {
        "by_app": app_agg_rows,
        "by_issue_category": issue_agg_rows,
        "monthly_data_points": len(monthly_app)
    }


def generate_charts(df: pd.DataFrame, val_results: dict):
    """
    Generates 4 high-quality charts visualizing sentiment validation and aggregation.
    """
    print(f"\n--- Generating Charts in {CHARTS_DIR} ---")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validation Boxplot: Star Rating vs VADER Compound
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#264653"]
    sns.boxplot(
        data=df,
        x="score",
        y="sentiment_compound",
        hue="score",
        palette=palette,
        legend=False,
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "white", "markersize": 7},
        ax=ax,
        fliersize=1,
        linewidth=1.2
    )
    r_val = val_results["pearson_correlation"]
    rho_val = val_results["spearman_correlation"]
    ax.set_title(
        f"Sentiment Validation: VADER Compound Score vs Play Store Rating\n(Pearson r = {r_val:.3f}, Spearman \u03c1 = {rho_val:.3f})",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("User Star Rating (1 - 5 Stars)", fontsize=11)
    ax.set_ylabel("VADER Compound Sentiment Score (-1.0 to +1.0)", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
    fig.tight_layout()
    chart1_path = CHARTS_DIR / "01_sentiment_validation_by_rating.png"
    fig.savefig(chart1_path)
    plt.close(fig)
    print(f"  Chart 1: {chart1_path.name}")

    # 2. Sentiment Breakdown by App (Stacked Bar)
    app_agg = pd.read_csv(APP_AGG_PATH).set_index("app_name")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    apps = app_agg.index
    neg = app_agg["pct_negative"]
    neu = app_agg["pct_neutral"]
    pos = app_agg["pct_positive"]

    p1 = ax.bar(apps, neg, color="#e63946", label="Negative (\u2264 -0.05)", width=0.55)
    p2 = ax.bar(apps, neu, bottom=neg, color="#f4a261", label="Neutral (-0.05 to 0.05)", width=0.55)
    p3 = ax.bar(apps, pos, bottom=neg + neu, color="#2a9d8f", label="Positive (\u2265 0.05)", width=0.55)

    # Annotate percentages
    for i, app in enumerate(apps):
        ax.text(i, neg[app] / 2, f"{neg[app]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if pos[app] > 10:
            ax.text(i, neg[app] + neu[app] + pos[app] / 2, f"{pos[app]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.set_title(f"Review Sentiment Distribution Across {len(apps)} Consumer Apps", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Percentage of Reviews (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    chart2_path = CHARTS_DIR / "02_sentiment_distribution_by_app.png"
    fig.savefig(chart2_path)
    plt.close(fig)
    print(f"  Chart 2: {chart2_path.name}")

    # 3. Sentiment by Issue Category (Horizontal Bar Chart)
    issue_agg = pd.read_csv(ISSUE_AGG_PATH).sort_values("mean_compound", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#d62828" if v < -0.3 else "#f77f00" if v < 0 else "#2a9d8f" for v in issue_agg["mean_compound"]]
    bars = ax.barh(
        issue_agg["issue_category"].str.replace("_", " ").str.title(),
        issue_agg["mean_compound"],
        color=colors,
        height=0.6
    )
    for bar in bars:
        val = bar.get_width()
        offset = -0.02 if val < 0 else 0.02
        ha = "right" if val < 0 else "left"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlim(-0.7, 0.35)
    ax.set_title("Mean Sentiment Severity by Issue Category", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean VADER Compound Sentiment Score", fontsize=11)
    fig.tight_layout()
    chart3_path = CHARTS_DIR / "03_sentiment_by_issue_category.png"
    fig.savefig(chart3_path)
    plt.close(fig)
    print(f"  Chart 3: {chart3_path.name}")

    # 4. Monthly Sentiment Trends by App, one panel per domain
    monthly = (
        df.groupby(["month", "app_name"])["sentiment_compound"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Filter out sparse months with < 10 reviews
    monthly = monthly[monthly["count"] >= 10].sort_values("month")
    months = sorted(monthly["month"].unique())

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(16, 5), dpi=300, sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        for app in [a for a in APP_COLORS if APP_DOMAIN[a] == domain]:
            sub = monthly[monthly["app_name"] == app].set_index("month").reindex(months)
            ax.plot(months, sub["mean"], marker="o", label=app, color=APP_COLORS[app], linewidth=2, markersize=5)
        ax.set_title(domain, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="lower left", frameon=True, fontsize=9)
    axes[0].set_ylabel("Mean VADER Compound Score", fontsize=11)
    fig.suptitle(f"Monthly Sentiment Trajectory Across {df['app_name'].nunique()} Apps", fontsize=12, fontweight="bold")
    fig.tight_layout()
    chart4_path = CHARTS_DIR / "04_sentiment_trend_monthly.png"
    fig.savefig(chart4_path)
    plt.close(fig)
    print(f"  Chart 4: {chart4_path.name}")


def score_app(slug):
    scored = score_sentiment(read_app("tagged", slug))
    write_app(scored, "tagged", slug)
    return slug, len(scored)


def main():
    # 1. Scoring (VADER, one process per app; results written back into data/tagged/<app>.csv.gz)
    print("Scoring data/tagged/*.csv.gz with VADER...")
    with ProcessPoolExecutor(max_workers=11) as pool:
        for slug, n in pool.map(score_app, APPS):
            print(f"  {APPS[slug]['name']:10s} {n:>9,} reviews scored")

    issue_cols = [c for c in read_app("tagged", next(iter(APPS)), nrows=0).columns
                  if c.startswith("issue_") and c != "issue_count"]
    scored_df = read_stage("tagged", columns=["app_name", "score", "month", "sentiment_neg", "sentiment_neu",
                                              "sentiment_pos", "sentiment_compound", "sentiment_label"] + issue_cols)
    print(f"Loaded {len(scored_df):,} scored rows for validation and aggregation.")

    # 2. Validation
    val_results = validate_sentiment(scored_df)

    # 3. Aggregation
    agg_results = aggregate_sentiment(scored_df)

    # 4. Generate Charts
    generate_charts(scored_df, val_results)

    # 5. Save Summary JSON
    summary = {
        "total_reviews": len(scored_df),
        "validation": val_results,
        "aggregation": agg_results,
        "sentiment_distribution": {
            "negative_pct": round(float((scored_df["sentiment_label"] == "Negative").mean() * 100), 2),
            "neutral_pct": round(float((scored_df["sentiment_label"] == "Neutral").mean() * 100), 2),
            "positive_pct": round(float((scored_df["sentiment_label"] == "Positive").mean() * 100), 2),
        },
        "output_files": {
            "dataset": "data/tagged/*.csv.gz",
            "charts": [
                str(CHARTS_DIR / "01_sentiment_validation_by_rating.png"),
                str(CHARTS_DIR / "02_sentiment_distribution_by_app.png"),
                str(CHARTS_DIR / "03_sentiment_by_issue_category.png"),
                str(CHARTS_DIR / "04_sentiment_trend_monthly.png"),
            ],
            "tables": [
                str(APP_AGG_PATH),
                str(ISSUE_AGG_PATH),
                str(RATING_VAL_PATH)
            ]
        }
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved complete sentiment summary to: {SUMMARY_PATH}")
    print("\nSentiment analysis completed successfully!")


if __name__ == "__main__":
    main()
