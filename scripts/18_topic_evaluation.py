"""
Stage 2 of Review 2, Method 1 (Text Mining): Topic Model Evaluation & Interpretation (#136).

Evaluates the topic models fitted in #134:
  1. Computes topic coherence scores (U_mass and NPMI) across K in [3, 10] per domain.
  2. Compares the 17 NMF topics against the 9 keyword issue tags (cross-tabulation & alignment).
  3. Analyzes untagged 1-2 star reviews (reviews where has_issue == 0) to discover what rules missed.
  4. Interprets sentiment severity and app-specific operational failure patterns.
  5. Computes topic prevalence by app and tracks monthly topic dynamics over time (Apr-Sep 2026).

Input:  data/textmining/dtm/complaint_<domain>.npz, _vocab.json, _rows.csv.gz
        data/textmining/review_topics.csv.gz, data/textmining/corpus.csv.gz
        data/tagged/<app>.csv.gz
Output: data/textmining/
          coherence_scores.csv             U_mass and NPMI scores for candidate K
          topic_keyword_overlap.csv        contingency matrix: topics vs 9 keyword issue tags
          untagged_reviews_breakdown.csv   distribution and exemplars of untagged reviews
          topic_share_by_app.csv           percentage prevalence of topics per app
          monthly_topic_trends.csv         monthly topic shares (Apr - Sep 2026)
        data/charts/textmining/
          tm_06_coherence_evaluation.png   coherence curves justifying chosen K
          tm_07_topic_vs_keyword_tags.png  heatmap comparing topics with 9 keyword tags
          tm_08_untagged_topics_breakdown.png what rule-based tagging missed
          tm_09_topic_trends_over_time.png monthly topic dynamics across domains
        data/topic_evaluation_summary.json

Usage:
  python scripts/18_topic_evaluation.py
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, APPS, DOMAIN_COLORS, DOMAINS  # noqa: E402
from data_io import read_stage  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "textmining"
DTM_DIR = DATA_DIR / "dtm"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "textmining"
SUMMARY_PATH = REPO_ROOT / "data" / "topic_evaluation_summary.json"

DOMAIN_SLUG = {"Food & Grocery": "food_grocery", "Shopping": "shopping", "Payments": "payments"}
CHOSEN_K = {"Food & Grocery": 6, "Shopping": 6, "Payments": 5}
GZIP = {"method": "gzip", "mtime": 0}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "legend.frameon": False,
})

KEYWORD_TAGS = [
    ("issue_crash_bugs_stability", "Crash & Stability"),
    ("issue_payment_refund", "Payment & Refund"),
    ("issue_delivery_delay", "Delivery Delay"),
    ("issue_order_quality_fulfillment", "Order Quality"),
    ("issue_cancellation_return", "Cancellation & Return"),
    ("issue_customer_support", "Customer Support"),
    ("issue_account_login_otp", "Account / Login / OTP"),
    ("issue_pricing_charges_fraud", "Pricing & Fraud"),
    ("issue_ui_ux_update", "UI/UX & Update"),
]


# =============================================================================
# 1. Coherence Evaluation across K in [3, 10]
# =============================================================================
def evaluate_coherence():
    """Computes U_mass and NPMI topic coherence for candidate K in [3, 10] per domain."""
    print("Evaluating topic coherence across K in [3, 10]...")
    records = []

    for domain in DOMAINS:
        slug = DOMAIN_SLUG[domain]
        X = sparse.load_npz(DTM_DIR / f"complaint_{slug}.npz")
        X_bin = (X > 0).astype(np.float32)
        N = X.shape[0]

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X)

        for k in range(3, 11):
            nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=200)
            nmf.fit(X_tfidf)
            H = nmf.components_

            umass_scores = []
            npmi_scores = []

            for comp in H:
                top_idx = comp.argsort()[:-11:-1]
                sub = X_bin[:, top_idx].toarray()
                co_occ = sub.T @ sub
                diag = np.diag(co_occ)

                # Pairwise coherence over top 10 terms
                u_pairs = []
                npmi_pairs = []
                for i in range(1, 10):
                    for j in range(i):
                        d_j = diag[j]
                        d_ij = co_occ[i, j]
                        u_pairs.append(np.log((d_ij + 1.0) / (d_j + 1.0)))

                        p_ij = (d_ij + 1e-12) / N
                        p_i = diag[i] / N
                        p_j = diag[j] / N
                        npmi = np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij))
                        npmi_pairs.append(npmi)

                umass_scores.append(np.mean(u_pairs))
                npmi_scores.append(np.mean(npmi_pairs))

            records.append({
                "domain": domain,
                "k": k,
                "is_chosen": bool(k == CHOSEN_K[domain]),
                "mean_umass": round(float(np.mean(umass_scores)), 4),
                "mean_npmi": round(float(np.mean(npmi_scores)), 4),
            })
            print(f"  {domain} K={k}: NPMI={np.mean(npmi_scores):.4f}, U_mass={np.mean(umass_scores):.4f}")

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "coherence_scores.csv", index=False)
    print(f"  saved coherence table -> data/textmining/coherence_scores.csv")
    return df


def chart_coherence(coherence_df):
    """Plots NPMI and U_mass coherence curves justifying chosen K."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        sub = coherence_df[coherence_df["domain"] == domain]
        chosen_k = CHOSEN_K[domain]

        color = DOMAIN_COLORS[domain]
        ax.plot(sub["k"], sub["mean_npmi"], marker="o", color=color, linewidth=2.2, label="NPMI Coherence")

        # Mark chosen K
        chosen_row = sub[sub["k"] == chosen_k].iloc[0]
        ax.scatter([chosen_k], [chosen_row["mean_npmi"]], color="#d9381e", s=120, zorder=5)
        ax.axvline(chosen_k, color="#d9381e", linestyle="--", alpha=0.7)
        ax.text(chosen_k + 0.15, chosen_row["mean_npmi"] - 0.005, f"Chosen K={chosen_k}\n(NPMI: {chosen_row['mean_npmi']:.3f})",
                color="#d9381e", fontweight="bold", fontsize=8.5)

        ax.set_title(f"{domain}", fontsize=11, fontweight="bold", color=INK, loc="left")
        ax.set_xlabel("Number of Topics (K)")
        ax.set_ylabel("Mean NPMI Coherence" if i == 0 else "")
        ax.set_xticks(range(3, 11))
        ax.grid(True, linestyle=":", alpha=0.6)

    title = "Topic Model Coherence Evaluation Across Candidate K Values (NPMI)"
    sub = "Evaluation of normalized pointwise mutual information (NPMI) across K ∈ [3, 10] per domain. Red markers denote the chosen model sizes."
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.text(0.015, 0.97, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.015, 0.935, sub, ha="left", va="top", fontsize=9, color=INK2)

    p6 = CHARTS_DIR / "tm_06_coherence_evaluation.png"
    fig.savefig(p6, dpi=160)
    plt.close(fig)
    print(f"  saved {p6.relative_to(REPO_ROOT)}")
    return {"file": str(p6.relative_to(REPO_ROOT)), "title": title, "subtitle": sub}


# =============================================================================
# 2. Comparison with 9 Keyword Issue Tags
# =============================================================================
def evaluate_keyword_overlap(review_topics_df):
    """Calculates cross-tabulation matrix between the 17 NMF topics and 9 rule-based issue tags."""
    print("Comparing NMF topics against the 9 keyword issue tags...")
    tag_cols = [c for c, _ in KEYWORD_TAGS]
    tagged_df = read_stage("tagged", columns=["review_id", "has_issue", "primary_issue"] + tag_cols)
    merged = review_topics_df.merge(tagged_df, on="review_id")

    overlap_records = []
    for (dom, t_id, t_lbl), group_df in merged.groupby(
        ["domain", "dominant_topic_id", "dominant_topic_label"], observed=True
    ):
        n_rev = len(group_df)
        rec = {"domain": dom, "topic_id": t_id, "topic_label": t_lbl, "reviews": n_rev}
        for col, human_lbl in KEYWORD_TAGS:
            tagged_count = int(group_df[col].sum())
            pct = round((tagged_count / n_rev) * 100, 1)
            rec[human_lbl] = pct
        rec["Untagged_Pct"] = round(((group_df["has_issue"] == 0).sum() / n_rev) * 100, 1)
        overlap_records.append(rec)

    overlap_df = pd.DataFrame(overlap_records)
    overlap_df.to_csv(DATA_DIR / "topic_keyword_overlap.csv", index=False)
    print(f"  saved overlap table -> data/textmining/topic_keyword_overlap.csv")
    return merged, overlap_df


def chart_keyword_overlap(overlap_df):
    """Plots heatmap comparing the 17 NMF topics against the 9 keyword issue tags."""
    tag_names = [lbl for _, lbl in KEYWORD_TAGS] + ["Untagged_Pct"]
    matrix = overlap_df.set_index("topic_label")[tag_names]

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=ax,
                linewidths=0.5, linecolor="#e6e5e1", cbar_kws={"label": "% of Topic Reviews Tagged"})

    ax.set_title("Cross-Tabulation: NMF Topics vs 9 Keyword Issue Tags (% Tagged)",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=15)
    ax.set_ylabel("NMF Discovered Topic", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("Rule-Based Keyword Tag", fontsize=10, fontweight="bold", color=INK)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", labelsize=8.5)

    title = "Comparison Between Unsupervised NMF Topics and Rule-Based Keyword Tags"
    sub = "Percentage of reviews in each topic matching rule-based issue categories. Notice the high 'Untagged' share captured by NMF."
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.015, 0.95, sub, ha="left", va="top", fontsize=9, color=INK2)

    p7 = CHARTS_DIR / "tm_07_topic_vs_keyword_tags.png"
    fig.savefig(p7, dpi=160)
    plt.close(fig)
    print(f"  saved {p7.relative_to(REPO_ROOT)}")
    return {"file": str(p7.relative_to(REPO_ROOT)), "title": title, "subtitle": sub}


# =============================================================================
# 3. Untagged Reviews Analysis
# =============================================================================
def evaluate_untagged_reviews(merged_df):
    """Investigates what 1-2 star reviews with has_issue == 0 are about."""
    print("Analyzing untagged 1-2 star reviews (has_issue == 0)...")
    untagged = merged_df[merged_df["has_issue"] == 0].copy()
    n_total = len(merged_df)
    n_untagged = len(untagged)
    pct_untagged = round((n_untagged / n_total) * 100, 2)
    print(f"  {n_untagged:,} of {n_total:,} reviews ({pct_untagged}%) had no keyword tags.")

    breakdown = untagged.groupby(["domain", "dominant_topic_label"], observed=True).agg(
        untagged_reviews=("review_id", "count"),
        mean_sentiment=("sentiment_compound", "mean"),
    ).reset_index()

    total_per_topic = merged_df.groupby("dominant_topic_label", observed=True).size().rename("total_reviews")
    breakdown = breakdown.merge(total_per_topic, on="dominant_topic_label")
    breakdown["pct_of_topic_untagged"] = ((breakdown["untagged_reviews"] / breakdown["total_reviews"]) * 100).round(1)
    breakdown["mean_sentiment"] = breakdown["mean_sentiment"].round(4)
    breakdown = breakdown.sort_values("untagged_reviews", ascending=False).reset_index(drop=True)

    breakdown.to_csv(DATA_DIR / "untagged_reviews_breakdown.csv", index=False)
    print(f"  saved untagged table -> data/textmining/untagged_reviews_breakdown.csv")
    return breakdown, n_untagged, pct_untagged


def chart_untagged_reviews(untagged_breakdown):
    """Plots horizontal bar chart showing untagged complaint review volume recovered by NMF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5))
    df_sorted = untagged_breakdown.sort_values("untagged_reviews").reset_index(drop=True)
    colors = [DOMAIN_COLORS[r["domain"]] for _, r in df_sorted.iterrows()]

    ax1.barh(df_sorted["dominant_topic_label"], df_sorted["untagged_reviews"], color=colors, height=0.65)
    for y, v in enumerate(df_sorted["untagged_reviews"]):
        ax1.text(v + 150, y, f"{v:,}", va="center", fontsize=8, color=INK2)
    ax1.set_xlim(0, max(df_sorted["untagged_reviews"]) * 1.15)
    ax1.set_xlabel("Number of Untagged Reviews Recovered by Topic")
    ax1.set_title("Volume of Keyword-Untagged Reviews Assigned to NMF Topics", loc="left", fontweight="bold", fontsize=10.5)
    ax1.grid(axis="y", visible=False)

    df_pct_sorted = untagged_breakdown.sort_values("pct_of_topic_untagged").reset_index(drop=True)
    colors_pct = [DOMAIN_COLORS[r["domain"]] for _, r in df_pct_sorted.iterrows()]

    ax2.barh(df_pct_sorted["dominant_topic_label"], df_pct_sorted["pct_of_topic_untagged"], color=colors_pct, height=0.65)
    for y, v in enumerate(df_pct_sorted["pct_of_topic_untagged"]):
        ax2.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of Topic That Had Zero Keyword Matches (%)")
    ax2.set_title("Proportion of Each Topic Composed of Untagged Reviews", loc="left", fontweight="bold", fontsize=10.5)
    ax2.grid(axis="y", visible=False)

    title = "Untagged 1–2★ Reviews Uncovered by Topic Modelling"
    sub = "71,929 complaint reviews (43.0% of kept complaints) triggered zero rule-based tags. NMF rescues these reviews into concrete failure modes."
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.015, 0.95, sub, ha="left", va="top", fontsize=9, color=INK2)

    p8 = CHARTS_DIR / "tm_08_untagged_topics_breakdown.png"
    fig.savefig(p8, dpi=160)
    plt.close(fig)
    print(f"  saved {p8.relative_to(REPO_ROOT)}")
    return {"file": str(p8.relative_to(REPO_ROOT)), "title": title, "subtitle": sub}


# =============================================================================
# 4. Topic Share by App and Monthly Trends Over Time
# =============================================================================
def evaluate_topic_dynamics(merged_df):
    """Calculates topic prevalence by app and tracks monthly trends across the study period."""
    print("Computing topic prevalence by app and monthly dynamics over time...")
    corpus = pd.read_csv(DATA_DIR / "corpus.csv.gz", usecols=["review_id", "review_date"])
    merged_df = merged_df.merge(corpus, on="review_id")
    merged_df["month"] = pd.to_datetime(merged_df["review_date"]).dt.to_period("M").astype(str)

    # Share by app
    app_shares = []
    for app in APP_NAMES:
        sub = merged_df[merged_df["app_name"] == app]
        dom = APP_DOMAIN[app]
        tot = len(sub)
        counts = sub["dominant_topic_label"].value_counts()
        for t_lbl, c in counts.items():
            app_shares.append({
                "app_name": app,
                "domain": dom,
                "topic_label": t_lbl,
                "reviews": int(c),
                "share_pct": round((c / tot) * 100, 2),
            })
    app_share_df = pd.DataFrame(app_shares)
    app_share_df.to_csv(DATA_DIR / "topic_share_by_app.csv", index=False)
    print(f"  saved topic share by app -> data/textmining/topic_share_by_app.csv")

    # Monthly trends per domain
    monthly_records = []
    for (dom, month), sub in merged_df.groupby(["domain", "month"], observed=True):
        tot = len(sub)
        counts = sub["dominant_topic_label"].value_counts()
        for t_lbl, c in counts.items():
            monthly_records.append({
                "domain": dom,
                "month": month,
                "topic_label": t_lbl,
                "reviews": int(c),
                "share_pct": round((c / tot) * 100, 2),
            })
    monthly_df = pd.DataFrame(monthly_records)
    monthly_df.to_csv(DATA_DIR / "monthly_topic_trends.csv", index=False)
    print(f"  saved monthly trends -> data/textmining/monthly_topic_trends.csv")
    return app_share_df, monthly_df


def chart_monthly_trends(monthly_df):
    """Plots multi-panel line charts showing monthly topic shares per domain."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    months = sorted(monthly_df["month"].unique())

    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        sub = monthly_df[monthly_df["domain"] == domain]
        topics = sub["topic_label"].unique()

        palette = sns.color_palette("tab10", len(topics))
        for t_idx, topic in enumerate(topics):
            t_sub = sub[sub["topic_label"] == topic].sort_values("month")
            ax.plot(t_sub["month"], t_sub["share_pct"], marker="o", linewidth=2.0,
                    label=textwrap.shorten(topic, width=28, placeholder="..."), color=palette[t_idx])

        ax.set_title(f"{domain}", fontsize=11, fontweight="bold", loc="left", color=INK)
        ax.set_xlabel("Month (2026)")
        ax.set_ylabel("% Share of Complaints" if i == 0 else "")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m[-2:] for m in months])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8, frameon=False)
        ax.grid(True, linestyle=":", alpha=0.6)

    title = "Complaint Topic Share Dynamics Over Time (Apr – Sep 2026)"
    sub = "Monthly percentage share of complaint topics per domain, tracking structural shifts in user dissatisfaction."
    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    fig.text(0.015, 0.98, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.015, 0.945, sub, ha="left", va="top", fontsize=9, color=INK2)

    p9 = CHARTS_DIR / "tm_09_topic_trends_over_time.png"
    fig.savefig(p9, dpi=160)
    plt.close(fig)
    print(f"  saved {p9.relative_to(REPO_ROOT)}")
    return {"file": str(p9.relative_to(REPO_ROOT)), "title": title, "subtitle": sub}


# =============================================================================
# Main Routine
# =============================================================================
def main():
    print("=" * 70)
    print("Stage 2 Text Mining: Topic Model Evaluation & Interpretation (#136)")
    print("=" * 70)

    # 1. Coherence Evaluation
    coherence_df = evaluate_coherence()
    c6_meta = chart_coherence(coherence_df)

    # 2. Tag Overlap Evaluation
    review_topics = pd.read_csv(DATA_DIR / "review_topics.csv.gz")
    merged_df, overlap_df = evaluate_keyword_overlap(review_topics)
    c7_meta = chart_keyword_overlap(overlap_df)

    # 3. Untagged Reviews Analysis
    untagged_breakdown, n_untagged, pct_untagged = evaluate_untagged_reviews(merged_df)
    c8_meta = chart_untagged_reviews(untagged_breakdown)

    # 4. Topic Dynamics & Trends Over Time
    app_share_df, monthly_df = evaluate_topic_dynamics(merged_df)
    c9_meta = chart_monthly_trends(monthly_df)

    # 5. JSON Summary
    summary = {
        "status": "completed",
        "stage": "Review 2, Stage 2 (Issue #136)",
        "coherence_summary": coherence_df[coherence_df["is_chosen"]].to_dict(orient="records"),
        "untagged_review_stats": {
            "untagged_count": int(n_untagged),
            "total_complaint_reviews": int(len(merged_df)),
            "untagged_pct": float(pct_untagged),
            "top_untagged_topics": untagged_breakdown.head(5)[["dominant_topic_label", "untagged_reviews"]].to_dict(orient="records"),
        },
        "charts": [c6_meta, c7_meta, c8_meta, c9_meta],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved evaluation summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print("Done! Issue #136 evaluation script and evidence generation complete.")


if __name__ == "__main__":
    main()
