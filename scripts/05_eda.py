"""
Stage 4: Exploratory Data Analysis (Person 3)

Reads data/app_reviews_tagged.csv and produces
  * 13 charts              -> data/charts/eda/eda_XX_*.png
  * data/eda_summary.json  -> every number quoted in EDA.md, plus the statistical tests
  * data/eda/monthly_trend.csv   -> monthly per-app metrics over all months, with window/regime flags (Person 5)
  * data/eda/version_metrics.csv -> rating per app version with z-scores (Person 5)

Charts
  01 coverage timeline           07 issue co-occurrence
  02 ratings + sample bias       08 issues by star rating (tagger false positives)
  03 length + upvotes            09 taxonomy coverage gap
  04 correlation matrix          10 monthly trends (common window)
  05 issue size vs severity      11 July-2026 sampling regime shift
  06 issues by app               12 rating by app version
                                 13 which issues are elevated in the worse-rated versions

Key methodological choice: Swiggy/Zomato/Myntra reviews are almost all from 2026,
while Paytm/PhonePe reach back to 2018 (see DATA_SOURCES.md, MOST_RELEVANT sampling).
Cross-app *trend* comparisons are therefore restricted to a COMMON WINDOW of months in
which every app has at least MIN_MONTH_N reviews.
"""
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "app_reviews_tagged.csv"
META_PATH = REPO_ROOT / "data" / "app_metadata.csv"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "eda"
TABLES_DIR = REPO_ROOT / "data" / "eda"
SUMMARY_PATH = REPO_ROOT / "data" / "eda_summary.json"

APPS = ["Swiggy", "Zomato", "Myntra", "Paytm", "PhonePe"]
ISSUES = [
    "crash_bugs_stability",
    "payment_refund",
    "delivery_delay",
    "order_quality_fulfillment",
    "cancellation_return",
    "customer_support",
    "account_login_otp",
    "pricing_charges_fraud",
    "ui_ux_update",
]
ISSUE_LABELS = {
    "crash_bugs_stability": "Crash & Stability",
    "payment_refund": "Payment & Refund",
    "delivery_delay": "Delivery Delay",
    "order_quality_fulfillment": "Order Quality",
    "cancellation_return": "Cancellation & Return",
    "customer_support": "Customer Support",
    "account_login_otp": "Account / Login / OTP",
    "pricing_charges_fraud": "Pricing & Fraud",
    "ui_ux_update": "UI/UX & Update",
}
ISSUE_COLS = [f"issue_{i}" for i in ISSUES]

MIN_MONTH_N = 50        # min reviews per app-month to count as "covered"
MIN_CELL_N = 30         # min reviews before a rate/mean is plotted or flagged
PARTIAL_MONTH = "2026-09"   # scrape ran on 20 Sep 2026 -> last month is incomplete

# ----------------------------------------------------------------------------
# Visual style (palette validated with dataviz validate_palette.js)
# ----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
APP_COLORS = dict(zip(APPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
STAR_COLORS = {1: "#c93a39", 2: "#ee8f8a", 3: "#c9c8c3", 4: "#86b6ef", 5: "#2a78d6"}
BLUE = "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fd", "#9ec5f4", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_rb", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

CHART_INDEX = []   # (filename, title) for the summary json / EDA.md


def save(fig, name, title, sub=None):
    """Left-aligned action title + muted subtitle (fixed inch offsets), then write the PNG."""
    h = fig.get_figheight()
    lines = textwrap.wrap(sub, int(fig.get_figwidth() * 13.5)) if sub else []
    fig.tight_layout(rect=[0, 0, 1, 1 - ((0.95 + 0.17 * (len(lines) - 1)) if sub else 0.6) / h])
    t = fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.canvas.draw()
    while t.get_window_extent().width > fig.get_figwidth() * fig.dpi * 0.97 and t.get_fontsize() > 9:
        t.set_fontsize(t.get_fontsize() - 0.5)
        fig.canvas.draw()
    if sub:
        fig.text(0.012, 1 - 0.46 / h, "\n".join(lines), ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.4)
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    CHART_INDEX.append({"file": f"data/charts/eda/{name}.png", "title": title})
    print(f"  saved {path.name}")


def text_on(rgb_or_hex):
    r, g, b = to_rgb(rgb_or_hex)
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def heatmap(ax, data, xlabels, ylabels, cmap, fmt="{:.0f}", vmin=None, vmax=None, center=None):
    vals = np.asarray(data, dtype=float)
    if center is not None:
        span = max(abs(np.nanmin(vals) - center), abs(np.nanmax(vals) - center))
        vmin, vmax = center - span, center + span
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isnan(vals[i, j]):
                continue
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=SURFACE, lw=2))
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=8.5,
                    color=text_on(im.cmap(im.norm(vals[i, j]))[:3]))
    return im


def cramers_v(table):
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(chi2), float(p), int(dof), float(np.sqrt(chi2 / (n * k)))


def month_label(m):
    return pd.Period(m).strftime("%b '%y")


# ----------------------------------------------------------------------------
# Load & prepare
# ----------------------------------------------------------------------------
def load():
    df = pd.read_csv(INPUT_PATH, parse_dates=["review_date"])
    df.attrs["n_source_columns"] = df.shape[1]
    df["app_name"] = pd.Categorical(df["app_name"], APPS, ordered=True)
    df["hour"] = df["review_date"].dt.hour
    df["dow"] = df["review_date"].dt.dayofweek
    df["week"] = df["review_date"].dt.to_period("W-SUN").dt.start_time
    df["is_low"] = (df["score"] <= 2).astype(int)         # Person 4's `is_problematic`
    df["log_thumbs"] = np.log1p(df["thumbs_up"])
    return df


def common_window(df):
    counts = df.groupby(["month", "app_name"], observed=True).size().unstack(fill_value=0)
    ok = counts[(counts.reindex(columns=APPS, fill_value=0) >= MIN_MONTH_N).all(axis=1)]
    return list(ok.index), counts


# ============================================================================
# A. DATA QUALITY & COVERAGE
# ============================================================================
def iqr_outliers(x):
    """Tukey fences (1.5 x IQR): how many values fall outside, and how extreme the maximum is."""
    q1, q3 = x.quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    out = (x < lo) | (x > hi)
    return {"fence_low": round(float(lo), 2), "fence_high": round(float(hi), 2), "n_outside": int(out.sum()),
            "pct_outside": round(float(out.mean() * 100), 2), "max": float(x.max()), "p99": float(x.quantile(.99))}


def a_quality_audit(df, summary):
    print("\n[A] Data quality & coverage audit")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].to_dict()
    summary["data_quality"] = {
        "rows": int(len(df)),
        "columns": int(df.attrs["n_source_columns"]),
        "null_counts": {k: int(v) for k, v in nulls.items()},
        "app_version_missing_pct": round(float(df["app_version"].isna().mean() * 100), 2),
        "duplicate_review_ids": int(df["review_id"].duplicated().sum()),
        "date_min": str(df["review_date"].min()),
        "date_max": str(df["review_date"].max()),
        "zero_thumbs_pct": round(float((df["thumbs_up"] == 0).mean() * 100), 2),
        "thumbs_up_max": int(df["thumbs_up"].max()),
        "review_length_words": {k: float(v) for k, v in df["review_length"].describe().round(2).items()},
        "very_short_reviews_le3_words_pct": round(float((df["review_length"] <= 3).mean() * 100), 2),
        "untagged_reviews_pct": round(float((df["has_issue"] == 0).mean() * 100), 2),
        "outliers_iqr": {c: iqr_outliers(df[c]) for c in ["thumbs_up", "review_length", "sentiment_compound"]},
        "outlier_policy": "No rows removed. thumbs_up outliers are genuine viral reviews (kept, handled with medians / log1p / rank statistics); "
                          "review_length and sentiment_compound outliers are few and plausible.",
    }
    per_app = df.groupby("app_name", observed=True).agg(
        reviews=("review_id", "size"), first_review=("review_date", "min"),
        last_review=("review_date", "max"), months_covered=("month", "nunique"),
        distinct_versions=("app_version", "nunique"),
        version_missing_pct=("app_version", lambda s: s.isna().mean() * 100),
        mean_rating=("score", "mean"), median_thumbs_up=("thumbs_up", "median"),
        median_length_words=("review_length", "median"),
    )
    per_app = per_app.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c)
    summary["coverage_by_app"] = json.loads(per_app.reset_index().astype(str).to_json(orient="records"))


def chart_02_ratings_and_bias(df, meta, summary):
    """Rating mix per app (left) and how far the sample sits below the public rating (right)."""
    ct = pd.crosstab(df["app_name"], df["score"], normalize="index") * 100
    ct.loc["All apps"] = df["score"].value_counts(normalize=True).sort_index() * 100
    order = ["All apps"] + APPS
    ct = ct.loc[order]
    summary["rating_distribution_pct"] = {k: {int(s_): round(float(v), 2) for s_, v in r.items()} for k, r in ct.iterrows()}
    m = meta.copy()
    m["app_name"] = m["app_name"].str.extract(r"^(Swiggy|Zomato|Myntra|Paytm|PhonePe)")[0]
    tbl = pd.DataFrame({"public_rating": m.set_index("app_name")["current_score"],
                        "sample_mean_rating": df.groupby("app_name", observed=True)["score"].mean()}).loc[APPS]
    tbl["gap"] = tbl["public_rating"] - tbl["sample_mean_rating"]
    summary["sample_vs_public_rating"] = json.loads(tbl.round(3).to_json(orient="index"))
    chi2, p, dof, v = cramers_v(pd.crosstab(df["app_name"], df["score"]))
    summary["rating_vs_app_chi2"] = {"chi2": round(chi2, 1), "dof": dof, "p": p, "cramers_v": round(v, 3)}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [6, 5]})
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for s_ in range(1, 6):
        w = ct[s_].values
        ax.barh(y, w, left=left, color=STAR_COLORS[s_], edgecolor=SURFACE, linewidth=2, height=0.62, label=f"{s_}★")
        for yi, li, wi in zip(y, left, w):
            if wi >= 4:
                ax.text(li + wi / 2, yi, f"{wi:.0f}%", ha="center", va="center", fontsize=9, color=text_on(STAR_COLORS[s_]))
        left += w
    ax.set_yticks(y, order)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of sampled reviews")
    ax.set_title("Rating mix", loc="left", fontsize=10.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)

    y = np.arange(len(APPS))[::-1]
    for yi, app in zip(y, APPS):
        r = tbl.loc[app]
        bx.plot([r["sample_mean_rating"], r["public_rating"]], [yi, yi], color=GRID, lw=3, zorder=1)
        bx.scatter(r["public_rating"], yi, s=90, color=INK2, zorder=3, edgecolor=SURFACE, linewidth=2)
        bx.scatter(r["sample_mean_rating"], yi, s=90, color=APP_COLORS[app], zorder=3, edgecolor=SURFACE, linewidth=2)
        bx.text(r["sample_mean_rating"] - 0.06, yi, f"{r['sample_mean_rating']:.2f}", ha="right", va="center", fontsize=9)
        bx.text(r["public_rating"] + 0.06, yi, f"{r['public_rating']:.2f}", ha="left", va="center", fontsize=9, color=INK2)
    bx.set_yticks(y, APPS)
    bx.set_xlim(0.7, 5.2)
    bx.set_xlabel("Mean star rating (grey = public Play Store rating)")
    bx.set_title("Sample vs public rating", loc="left", fontsize=10.5, color=INK2)
    bx.grid(axis="y", visible=False)
    save(fig, "eda_02_ratings_and_sample_bias",
         f"{ct.loc['All apps', 1]:.0f}% of sampled reviews are 1★, and every app sits {tbl['gap'].min():.1f}–{tbl['gap'].max():.1f} stars below its public rating",
         f"Swiggy ({ct.loc['Swiggy', 1]:.0f}% 1★) and Zomato ({ct.loc['Zomato', 1]:.0f}%) are far harsher than Myntra ({ct.loc['Myntra', 1]:.0f}%). "
         f"App effect on rating: Cramér's V = {v:.2f}. Compare apps and months; never quote absolute negativity.")


def chart_03_engagement(df, summary):
    """Review length by rating, and how concentrated upvotes are."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    data = [df.loc[df["score"] == s_, "review_length"] for s_ in range(1, 6)]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color=INK, lw=2),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for patch, s_ in zip(bp["boxes"], range(1, 6)):
        patch.set(facecolor=STAR_COLORS[s_], edgecolor=SURFACE)
    for i, d in enumerate(data, 1):
        ax.text(i, d.median() + 1.5, f"{d.median():.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, 6), [f"{s_}★" for s_ in range(1, 6)])
    ax.set_ylabel("Review length (words)")
    ax.set_title("Length by star rating", loc="left", fontsize=10.5, color=INK2)
    rho, p = stats.spearmanr(df["score"], df["review_length"])
    r_up, p_up = stats.spearmanr(df["review_length"], df["thumbs_up"])
    summary["review_length"] = {"spearman_score_vs_length": round(float(rho), 4), "spearman_p": float(p),
                                "median_words_by_score": {s_: float(d.median()) for s_, d in zip(range(1, 6), data)},
                                "spearman_length_vs_thumbs": round(float(r_up), 4)}

    t = np.sort(df["thumbs_up"].to_numpy())
    cum = np.cumsum(t) / t.sum()
    x = np.arange(1, len(t) + 1) / len(t)
    gini = 1 - 2 * float(np.sum((cum[1:] + cum[:-1]) / 2 * np.diff(x)) + cum[0] * x[0] / 2)
    top1 = t[int(len(t) * 0.99):].sum() / t.sum()
    ax = axes[1]
    ax.plot(x, cum, color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.fill_between(x, cum, x, color=BLUE, alpha=0.08)
    ax.set_xlabel("Cumulative share of reviews (least → most upvoted)")
    ax.set_ylabel("Cumulative share of all upvotes")
    ax.set_title(f"Upvote concentration (Gini = {gini:.2f})", loc="left", fontsize=10.5, color=INK2)
    ax.text(0.3, 0.42, f"Top 1% of reviews hold\n{top1 * 100:.0f}% of all upvotes", fontsize=9.5)

    share = df.groupby("score")["thumbs_up"].sum() / df["thumbs_up"].sum() * 100
    rshare = df["score"].value_counts(normalize=True).sort_index() * 100
    ax = axes[2]
    w = 0.38
    xs = np.arange(1, 6)
    ax.bar(xs - w / 2, rshare.values, w, color=MUTED, label="% of reviews")
    ax.bar(xs + w / 2, share.values, w, color=[STAR_COLORS[s_] for s_ in xs], label="% of upvotes")
    for xi, a, b in zip(xs, rshare.values, share.values):
        ax.text(xi - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [f"{s_}★" for s_ in xs])
    ax.set_ylabel("%")
    ax.set_title("Share of reviews vs share of upvotes", loc="left", fontsize=10.5, color=INK2)
    ax.legend(fontsize=8.5)
    summary["thumbs_up"] = {"gini": round(gini, 4), "top1pct_share_of_upvotes": round(float(top1 * 100), 2),
                            "share_of_upvotes_by_score_pct": {int(k): round(float(v), 2) for k, v in share.items()},
                            "mean": round(float(df["thumbs_up"].mean()), 2), "median": float(df["thumbs_up"].median()),
                            "mean_by_app": {a: round(float(v), 1) for a, v in df.groupby("app_name", observed=True)["thumbs_up"].mean().items()}}
    save(fig, "eda_03_length_and_upvotes",
         f"Unhappy users write twice as much ({data[0].median():.0f} vs {data[4].median():.0f} words); upvotes are winner-take-all (top 1% = {top1 * 100:.0f}%)",
         f"1★ is {rshare[1]:.0f}% of reviews but {share[1]:.0f}% of upvotes, so upvotes alone do not explain the negative skew. Use median or log(upvotes), never the mean.")


def chart_01_coverage_timeline(df, window):
    """Monthly sample volume per app: reveals the unequal time windows."""
    monthly = df.groupby(["month", "app_name"], observed=True).size().unstack(fill_value=0)
    idx = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M").astype(str)
    monthly = monthly.reindex(idx, fill_value=0)
    fig, axes = plt.subplots(len(APPS), 1, figsize=(11, 8.6), sharex=True)
    xs = np.arange(len(idx))
    for ax, app in zip(axes, APPS):
        ax.bar(xs, monthly[app].values, color=APP_COLORS[app], width=0.85)
        if window:
            ax.axvspan(idx.get_loc(window[0]) - .5, idx.get_loc(window[-1]) + .5, color="#000", alpha=0.05, lw=0)
        ax.set_ylabel(app, rotation=0, ha="right", va="center", fontweight="bold", color=INK)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=8)
        n = int(monthly[app].sum())
        ax.text(0.005, 0.9, f"{n:,} reviews · {(monthly[app] > 0).sum()} months with data",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    ticks = [i for i, m in enumerate(idx) if m.endswith("-01")]
    axes[-1].set_xticks(ticks, [m[:4] for m in [idx[i] for i in ticks]])
    axes[-1].set_xlabel("Review month (grey band = common comparison window)")
    save(fig, "eda_01_coverage_timeline",
         "Three apps are almost entirely 2026 reviews; only Paytm and PhonePe reach back to 2018",
         "Monthly reviews in the scraped sample (not true review volume). Cross-app trends use the grey window only.")


# ============================================================================
# B. RATINGS, LENGTH & ENGAGEMENT
# ============================================================================
def chart_04_correlation(df, summary):
    cols = {"score": "Star rating", "sentiment_compound": "VADER compound", "sentiment_neg": "VADER neg",
            "sentiment_pos": "VADER pos", "issue_count": "Issue count", "review_length": "Review length",
            "log_thumbs": "log(1+upvotes)"}
    corr = df[list(cols)].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    labels = list(cols.values())
    im = heatmap(ax, corr.values, labels, labels, DIV, fmt="{:.2f}", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Spearman ρ")
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
    summary["strongest_correlations"] = [{"a": a, "b": b, "spearman": round(float(v), 3)} for (a, b), v in pairs.head(6).items()]
    r_len_issue = corr.loc["issue_count", "review_length"]
    save(fig, "eda_04_correlation_matrix",
         f"Issue tags track review length (ρ={r_len_issue:.2f}); upvotes ignore rating and sentiment",
         "Spearman rank correlations. Longer text has more chances to match a keyword rule, so recall of the tagger depends on length. Sentiment↔rating reproduces Person 2's validation.")


# ============================================================================
# C. ISSUE ANALYSIS
# ============================================================================
def chart_06_issue_by_app(df, summary):
    tbl = pd.DataFrame({ISSUE_LABELS[i]: df.groupby("app_name", observed=True)[f"issue_{i}"].mean() * 100 for i in ISSUES}).T[APPS]
    summary["issue_prevalence_by_app_pct"] = json.loads(tbl.round(2).to_json())
    # chi-square + Cramer's V per issue: does the issue depend on the app?
    rows = []
    for i in ISSUES:
        ct = pd.crosstab(df["app_name"], df[f"issue_{i}"])
        chi2, p, dof, v = cramers_v(ct)
        rows.append({"issue": i, "chi2": round(chi2, 1), "p_value": p, "cramers_v": round(v, 3)})
    summary["issue_vs_app_chi2"] = rows
    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = heatmap(ax, tbl.values, APPS, list(tbl.index), SEQ, fmt="{:.0f}%", vmin=0, vmax=tbl.values.max())
    top = max(rows, key=lambda r: r["cramers_v"])
    cs = tbl.loc["Customer Support"]
    maxp = max(r["p_value"] for r in rows)
    save(fig, "eda_06_issue_by_app",
         "Each app has its own failure fingerprint: delivery for food apps, returns for Myntra, crashes for Paytm",
         f"% of each app's reviews tagged with the issue. Strongest app effect: {ISSUE_LABELS[top['issue']]} (Cramér's V={top['cramers_v']:.2f}); "
         f"support {cs.idxmax()} {cs.max():.0f}% vs {cs.idxmin()} {cs.min():.0f}%. All 9 chi-square tests: max p={maxp:.1e}.")


def chart_07_cooccurrence(df, summary):
    X = df[ISSUE_COLS].to_numpy()
    n = len(X)
    both = X.T @ X
    cnt = np.diag(both).astype(float)
    lift = both * n / np.outer(cnt, cnt)
    labels = [ISSUE_LABELS[i] for i in ISSUES]
    mask = np.eye(len(ISSUES), dtype=bool)
    show = np.where(mask, np.nan, np.log2(lift))
    fig, ax = plt.subplots(figsize=(9.2, 7))
    im = heatmap(ax, show, labels, labels, DIV, fmt="{:+.1f}", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="log2(lift): + = co-occur more than chance")
    pairs = []
    for a in range(len(ISSUES)):
        for b in range(a + 1, len(ISSUES)):
            pairs.append({"issue_a": ISSUES[a], "issue_b": ISSUES[b], "co_count": int(both[a, b]), "lift": round(float(lift[a, b]), 2)})
    pairs = sorted(pairs, key=lambda r: -r["lift"])
    summary["issue_cooccurrence_top_pairs"] = [p for p in pairs if p["co_count"] >= 50][:6]
    summary["issue_cooccurrence_bottom_pairs"] = [p for p in pairs if p["co_count"] >= 0][-3:]
    top = summary["issue_cooccurrence_top_pairs"][0]
    save(fig, "eda_07_issue_cooccurrence",
         f"{ISSUE_LABELS[top['issue_a']]} + {ISSUE_LABELS[top['issue_b']]} co-occur {top['lift']:.1f}× more than chance (n={top['co_count']})",
         "Pairwise lift = P(A∧B) / (P(A)·P(B)), shown as log2. Blue = issues travel together, red = tend to be mutually exclusive.")


def chart_08_issue_by_rating(df, summary):
    tbl = pd.DataFrame({ISSUE_LABELS[i]: df.groupby("score")[f"issue_{i}"].mean() * 100 for i in ISSUES}).T
    tbl.loc["Any issue"] = df.groupby("score")["has_issue"].mean() * 100
    tbl.columns = [f"{c}★" for c in tbl.columns]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    heatmap(ax, tbl.values, list(tbl.columns), list(tbl.index), SEQ, fmt="{:.0f}%", vmin=0, vmax=np.nanmax(tbl.values))
    ax.axhline(len(ISSUES) - .5, color=INK, lw=1.2)
    summary["issue_any_by_rating_pct"] = {c: round(float(v), 2) for c, v in tbl.loc["Any issue"].items()}
    fp = df[(df["score"] == 5) & (df["issue_cancellation_return"] == 1)]
    summary["cancellation_tag_on_5star"] = {"count": int(len(fp)), "myntra_share_pct": round(float((fp["app_name"] == "Myntra").mean() * 100), 1)}
    save(fig, "eda_08_issue_by_rating",
         f"Tags track rating ({tbl.loc['Any issue', '1★']:.0f}% of 1★ vs {tbl.loc['Any issue', '5★']:.0f}% of 5★); 5★ 'Cancellation & Return' is a false-positive cluster",
         f"{len(fp)} five-star reviews carry the Cancellation & Return tag, {(fp['app_name'] == 'Myntra').mean() * 100:.0f}% of them Myntra praising easy returns/exchanges: the rule ignores sentiment polarity.")


def chart_05_issue_priority(df, summary):
    rows = []
    base_score = df["score"].mean()
    for i in ISSUES:
        s = df[df[f"issue_{i}"] == 1]
        rows.append({"issue": i, "label": ISSUE_LABELS[i], "n": len(s), "prevalence_pct": len(s) / len(df) * 100,
                     "mean_score": s["score"].mean(), "mean_sentiment": s["sentiment_compound"].mean(),
                     "pct_1_2_star": s["is_low"].mean() * 100, "median_thumbs": s["thumbs_up"].median(),
                     "mean_thumbs": s["thumbs_up"].mean(), "pct_50plus_thumbs": (s["thumbs_up"] >= 50).mean() * 100})
    q = pd.DataFrame(rows).set_index("issue")
    summary["issue_priority"] = json.loads(q.round(3).to_json(orient="index"))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(q["prevalence_pct"], q["mean_score"], s=np.sqrt(q["n"]) * 9, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=2)
    offsets = {"Pricing & Fraud": (-10, 12), "Delivery Delay": (12, -12), "Payment & Refund": (-12, 12),
               "Cancellation & Return": (-10, 14), "Order Quality": (12, -4), "Customer Support": (-12, 14)}
    for _, r in q.iterrows():
        dx, dy = offsets.get(r["label"], (10, 6))
        ax.annotate(r["label"], (r["prevalence_pct"], r["mean_score"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha="left" if dx > 0 else "right", color=INK)
    ax.axhline(base_score, color=MUTED, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], base_score + 0.02, f"dataset mean {base_score:.2f}★", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("Prevalence: % of all reviews (bubble area ∝ review count)")
    ax.set_ylabel("Mean star rating of reviews with the issue")
    worst, big = q["mean_score"].idxmin(), q["prevalence_pct"].idxmax()
    save(fig, "eda_05_issue_priority",
         f"{ISSUE_LABELS[big]} is the largest problem ({q.loc[big, 'prevalence_pct']:.0f}%, {q.loc[big, 'mean_score']:.2f}★); {ISSUE_LABELS[worst]} is the most damaging ({q.loc[worst, 'mean_score']:.2f}★)",
         "Bottom-right = frequent AND low-rated. Only UI/UX & Update reviews are rated above the dataset mean (dashed line).")


# ============================================================================
# D. TEXT SIGNAL
# ============================================================================
BRAND_STOP = {"swiggy", "zomato", "myntra", "paytm", "phonepe", "phone", "pe", "app", "application", "im", "ive", "dont",
              "doesnt", "didnt", "cant", "just", "really", "like", "get", "got", "use", "using", "one"}
STOP = list(ENGLISH_STOP_WORDS | BRAND_STOP)


def distinctive_terms(docs_a, docs_b, apps_a, apps_b, k=14, min_df=25):
    """Log-odds (Laplace smoothed) of document frequency in A vs B."""
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=STOP, min_df=min_df, binary=True,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    cv.fit(pd.concat([docs_a, docs_b]))
    xa, xb = cv.transform(docs_a), cv.transform(docs_b)
    da, db = np.asarray(xa.sum(0)).ravel(), np.asarray(xb.sum(0)).ravel()
    na, nb = xa.shape[0], xb.shape[0]
    lo = np.log((da + .5) / (na - da + .5)) - np.log((db + .5) / (nb - db + .5))
    vocab = np.array(cv.get_feature_names_out())

    def pick(order):
        chosen, used = [], set()
        for j in order:
            toks = set(vocab[j].split())
            if toks & used:
                continue
            chosen.append(j)
            used |= toks
            if len(chosen) == k:
                break
        return chosen

    ja = pick(np.argsort(-lo))
    jb = pick(np.argsort(lo))
    def top_app(x, apps, js):
        out = []
        for j in js:
            vc = apps.iloc[x[:, j].nonzero()[0]].value_counts()
            out.append((str(vc.index[0]), float(vc.iloc[0] / vc.sum() * 100)))
        return out

    def mk(js, d, n, sign, x, apps):
        t = pd.DataFrame({"term": vocab[js], "odds_ratio": np.exp(sign * lo[js]), "df_pct": d[js] / n * 100})
        t["top_app"], t["top_app_share_pct"] = zip(*top_app(x.tocsc(), apps, js))
        return t

    return mk(ja, da, na, 1, xa, apps_a), mk(jb, db, nb, -1, xb, apps_b), (na, nb)


def hbar_terms(ax, t, color, title, na):
    t = t.iloc[::-1]
    ax.barh(t["term"], t["odds_ratio"], color=color, height=0.65)
    for yi, (o, d, a, sh) in enumerate(zip(t["odds_ratio"], t["df_pct"], t["top_app"], t["top_app_share_pct"])):
        ax.text(o * 1.02, yi, f"{o:.1f}×  ({d:.1f}% of reviews)  ·  {a} {sh:.0f}%", va="center", fontsize=8.3)
    ax.set_xlim(0, t["odds_ratio"].max() * 1.9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio vs the other group")


def chart_09_taxonomy_gap(df, summary):
    low = df[df["score"] <= 2]
    un, tg = low.loc[low["has_issue"] == 0, "content"], low.loc[low["has_issue"] == 1, "content"]
    a, _, (na, nb) = distinctive_terms(un, tg, low.loc[un.index, "app_name"], low.loc[tg.index, "app_name"], k=16)
    pct_un = len(un) / len(low) * 100
    summary["taxonomy_gap"] = {"low_star_reviews": int(len(low)), "low_star_untagged": int(len(un)),
                               "low_star_untagged_pct": round(float(pct_un), 2), "terms": a["term"].tolist(),
                               "median_words_untagged": float(low.loc[low["has_issue"] == 0, "review_length"].median()),
                               "median_words_tagged": float(low.loc[low["has_issue"] == 1, "review_length"].median())}
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    hbar_terms(ax, a, "#eb6834", f"Terms over-represented in untagged 1–2★ reviews (n={na:,}) vs tagged (n={nb:,})", na)
    save(fig, "eda_09_taxonomy_gap",
         f"{pct_un:.0f}% of 1–2★ reviews have no issue tag; security-scan alerts, QR/scan and notifications are the gaps",
         f"Terms over-represented in untagged vs tagged low-star reviews. Untagged ones are shorter (median {summary['taxonomy_gap']['median_words_untagged']:.0f} vs {summary['taxonomy_gap']['median_words_tagged']:.0f} words); 'malicious'/'detected' is almost all Paytm.")


# ============================================================================
# E. TRENDS, SPIKES & VERSIONS
# ============================================================================
def monthly_table(df, window):
    w = df[df["month"].isin(window)]
    g = w.groupby(["app_name", "month"], observed=True).agg(
        n=("score", "size"), mean_rating=("score", "mean"), pct_low_star=("is_low", "mean"),
        pct_has_issue=("has_issue", "mean"), mean_sentiment=("sentiment_compound", "mean"),
        mean_issue_count=("issue_count", "mean"), median_length=("review_length", "median")).reset_index()
    for c in ["pct_low_star", "pct_has_issue"]:
        g[c] *= 100
    return g


def line_panel(ax, g, col, ylabel, window, fmt="{:.1f}"):
    xs = {m: i for i, m in enumerate(window)}
    for app in APPS:
        s = g[(g["app_name"] == app)].set_index("month").reindex(window)
        y = s[col].where(s["n"] >= MIN_CELL_N)
        ax.plot(range(len(window)), y.values, color=APP_COLORS[app], lw=2.2, marker="o", ms=5, label=app)
        last = y.dropna()
        if len(last):
            ax.text(xs[last.index[-1]] + 0.12, last.iloc[-1], f" {app}", color=INK, fontsize=8.5, va="center")
    ax.set_xticks(range(len(window)), [month_label(m) + ("*" if m == PARTIAL_MONTH else "") for m in window])
    ax.set_xlim(-0.3, len(window) - 1 + 1.1)
    ax.set_ylabel(ylabel)


def app_legend(fig, h):
    handles = [plt.Line2D([], [], color=APP_COLORS[a], lw=2.2, marker="o", ms=5, label=a) for a in APPS]
    fig.legend(handles=handles, ncol=5, loc="upper right", bbox_to_anchor=(0.99, 1 - 0.80 / h), fontsize=9, frameon=False)


REGIME_START = pd.Timestamp("2026-07-01")   # all five apps change sampling regime here (see chart 11)


LEN_BINS = [0, 20, 40, 70, 10_000]
LEN_LABELS = ["≤20 words", "21–40", "41–70", "71+"]


def regime_adjusted(df, window):
    """Pre/post-July change per app, raw vs standardised to the app's own review-length mix."""
    w = df[df["month"].isin(window)].copy()
    w["post"] = w["review_date"] >= REGIME_START
    w["lb"] = pd.cut(w["review_length"], LEN_BINS, labels=LEN_LABELS)
    rows = []
    for app, g in w.groupby("app_name", observed=True):
        wt = g["lb"].value_counts(normalize=True)
        row = {"app_name": app, "n_pre": int((~g["post"]).sum()), "n_post": int(g["post"].sum()),
               "median_len_pre": float(g.loc[~g["post"], "review_length"].median()),
               "median_len_post": float(g.loc[g["post"], "review_length"].median())}
        for col in ["score", "has_issue"]:
            for post, tag in [(False, "pre"), (True, "post")]:
                gg = g[g["post"] == post]
                m = gg.groupby("lb", observed=True)[col].mean()
                ww = wt.reindex(m.index)
                ww /= ww.sum()
                row[f"{col}_raw_{tag}"] = gg[col].mean()
                row[f"{col}_adj_{tag}"] = (m * ww).sum()
            row[f"{col}_raw_change"] = row[f"{col}_raw_post"] - row[f"{col}_raw_pre"]
            row[f"{col}_adj_change"] = row[f"{col}_adj_post"] - row[f"{col}_adj_pre"]
        rows.append(row)
    return w, pd.DataFrame(rows).set_index("app_name").loc[APPS]


def chart_11_regime_shift(df, window, summary):
    """All five apps change character at the same time in July 2026 -> a collection artefact, not app behaviour."""
    w, adj = regime_adjusted(df, window)
    summary["regime_shift"] = {
        "start": str(REGIME_START.date()),
        "median_length_pre_post": {a: [adj.loc[a, "median_len_pre"], adj.loc[a, "median_len_post"]] for a in APPS},
        "rating_change_raw": {a: round(float(adj.loc[a, "score_raw_change"]), 3) for a in APPS},
        "rating_change_length_adjusted": {a: round(float(adj.loc[a, "score_adj_change"]), 3) for a in APPS},
        "issue_rate_change_raw_pp": {a: round(float(adj.loc[a, "has_issue_raw_change"] * 100), 1) for a in APPS},
        "issue_rate_change_length_adjusted_pp": {a: round(float(adj.loc[a, "has_issue_adj_change"] * 100), 1) for a in APPS},
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    med = w.groupby(["month", "app_name"], observed=True)["review_length"].median().unstack().reindex(window)
    ax = axes[0]
    for a in APPS:
        ax.plot(range(len(window)), med[a], color=APP_COLORS[a], lw=2.2, marker="o", ms=5, label=a)
    cut = window.index(f"{REGIME_START.year}-{REGIME_START.month:02d}") - 0.5
    ax.axvline(cut, color=INK2, ls="--", lw=1)
    ax.text(cut + 0.05, ax.get_ylim()[1] * 0.98, "regime shift", fontsize=8.5, color=INK2, va="top")
    ax.set_xticks(range(len(window)), [month_label(m) for m in window], rotation=30)
    ax.set_ylabel("Median review length (words)")
    ax.set_title("Median review length by month", loc="left", fontsize=10.5, color=INK2)
    for ax, col, ylabel, ttl, fmt in [(axes[1], "has_issue", "% with ≥1 issue tag", "Issue-tag rate by length band", "{:.0f}"),
                                      (axes[2], "score", "Mean star rating", "Mean rating by length band", "{:.2f}")]:
        g = w.groupby(["lb", "post"], observed=True)[col].agg(["mean", "size"]).reset_index()
        scale = 100 if col == "has_issue" else 1
        x = np.arange(len(LEN_LABELS))
        for k, (post, colr, lab) in enumerate([(False, "#9ec5f4", "Apr–Jun '26"), (True, "#1c5cab", "Jul–Sep '26")]):
            gg = g[g["post"] == post].set_index("lb").reindex(LEN_LABELS)
            bars = ax.bar(x + (k - .5) * 0.38, gg["mean"] * scale, 0.36, color=colr, label=lab)
            for xi, v in zip(x + (k - .5) * 0.38, gg["mean"] * scale):
                ax.text(xi, v + (1 if col == "has_issue" else 0.03), fmt.format(v), ha="center", fontsize=8)
        ax.set_xticks(x, LEN_LABELS)
        ax.set_xlabel("Review length band")
        ax.set_ylabel(ylabel)
        ax.set_title(ttl, loc="left", fontsize=10.5, color=INK2)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(fontsize=8.5, loc="upper center", ncol=2)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8.5, loc="upper right", ncol=1)
    tot = w.groupby("post").size()
    days = {False: (REGIME_START - w["review_date"].min().normalize()).days, True: (w["review_date"].max().normalize() - REGIME_START).days + 1}
    vol_ratio = (tot[True] / days[True]) / (tot[False] / days[False])
    shrink = 1 - adj["median_len_post"] / adj["median_len_pre"]
    short = w[w["lb"] == LEN_LABELS[0]].groupby("post")["score"].mean()
    ia, ir = adj["has_issue_adj_change"] * 100, adj["has_issue_raw_change"] * 100
    summary["regime_shift"]["interpretation"] = (
        f"Reviews/day rise {vol_ratio:.1f}x and median review length falls {shrink.min() * 100:.0f}-{shrink.max() * 100:.0f}% in all five apps "
        f"in July 2026. Issue-tag rate changes of {ir.min():.0f} to {ir.max():.0f} pp shrink to {ia.min():.0f} to {ia.max():.0f} pp after "
        "standardising to each app's own review-length mix, so most apparent post-July change is a length-composition effect of the "
        "collection rather than a change in the apps.")
    summary["regime_shift"]["reviews_per_day_ratio_post_vs_pre"] = round(float(vol_ratio), 2)
    summary["regime_shift"]["median_length_shrink_pct"] = {a: round(float(v * 100), 1) for a, v in shrink.items()}
    save(fig, "eda_11_july_regime_shift",
         f"In July 2026 all five apps shift together: {vol_ratio:.1f}× more reviews per day, {shrink.min() * 100:.0f}–{shrink.max() * 100:.0f}% shorter; issue rates barely move within length bands",
         f"{tot[False]:,} reviews before vs {tot[True]:,} from 1 Jul. Exceptions: 21–40 word reviews are tagged more often after July, and ≤20-word reviews are rated {short[True] - short[False]:+.1f}★ higher. Most post-July 'improvement' is a length-mix effect of the collection.")


def chart_10_monthly_trend(df, window, summary):
    g = monthly_table(df, window)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    line_panel(axes[0], g, "mean_rating", "Mean star rating", window)
    axes[0].set_title("Common window, monthly mean rating", loc="left", fontsize=10.5, color=INK2)
    line_panel(axes[1], g, "pct_low_star", "% rated 1–2★", window)
    axes[1].set_title("Common window, share of 1–2★ reviews", loc="left", fontsize=10.5, color=INK2)
    app_legend(fig, fig.get_figheight())
    ch = g.pivot(index="month", columns="app_name", values="mean_rating").reindex(window)
    delta = (ch.iloc[-2] - ch.iloc[0]) if len(window) > 2 else ch.iloc[-1] - ch.iloc[0]
    summary["monthly_rating_change_first_to_last_full_month"] = {a: round(float(v), 3) for a, v in delta.items()}
    # Kruskal-Wallis of rating across apps in the window + pairwise-friendly effect size
    w = df[df["month"].isin(window)]
    groups = [w.loc[w["app_name"] == a, "score"] for a in APPS]
    h, p = stats.kruskal(*groups)
    ct = pd.crosstab(w["app_name"], w["score"])
    chi2, pc, dof, v = cramers_v(ct)
    summary["rating_differs_by_app"] = {"kruskal_H": round(float(h), 1), "kruskal_p": float(p), "chi2": round(chi2, 1),
                                        "chi2_p": pc, "cramers_v": round(v, 3), "window_n": int(len(w))}
    _, adj = regime_adjusted(df, window)
    ceil = ch[["Swiggy", "Zomato"]].max().max()
    save(fig, "eda_10_monthly_trend",
         f"Swiggy and Zomato never exceed {ceil:.1f}★; other apps' gains after July shrink to {adj['score_adj_change'].drop(['Swiggy', 'Zomato']).min():+.2f}…{adj['score_adj_change'].drop(['Swiggy', 'Zomato']).max():+.2f}★ once review length is controlled",
         f"Points need ≥{MIN_CELL_N} reviews. * = partial month. Rating differs by app: Cramér's V = {v:.2f} (Kruskal–Wallis p<0.001). See chart 11 for the length adjustment.")


def chart_12_app_versions(df, summary):
    """Rating by app version: candidate 'bad release' detector (uses top versions per app)."""
    d = df.dropna(subset=["app_version"])
    rows = []
    for app in APPS:
        a = d[d["app_name"] == app]
        mu, sd = a["score"].mean(), a["score"].std()
        for v, grp in a.groupby("app_version"):
            if len(grp) < MIN_CELL_N:
                continue
            se = sd / np.sqrt(len(grp))
            rows.append({"app_name": app, "app_version": v, "n": len(grp), "mean_rating": grp["score"].mean(),
                         "pct_low_star": grp["is_low"].mean() * 100, "pct_has_issue": grp["has_issue"].mean() * 100,
                         "mean_sentiment": grp["sentiment_compound"].mean(), "first_seen": grp["review_date"].min(),
                         "median_date": grp["review_date"].median(), "z_vs_app_mean": (grp["score"].mean() - mu) / se})
    vt = pd.DataFrame(rows)
    vt["flag"] = np.where(vt["z_vs_app_mean"] <= -2, "worse than app average",
                          np.where(vt["z_vs_app_mean"] >= 2, "better than app average", ""))
    vt.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c).to_csv(TABLES_DIR / "version_metrics.csv", index=False)
    summary["version_analysis"] = {
        "versions_with_min_n": int(len(vt)), "min_n": MIN_CELL_N,
        "worse_than_average": vt[vt["flag"] == "worse than app average"][["app_name", "app_version", "n", "mean_rating"]].round(2).to_dict("records"),
        "better_than_average": vt[vt["flag"] == "better than app average"][["app_name", "app_version", "n", "mean_rating"]].round(2).to_dict("records"),
        "caveat": "Reviews are attributed to the reviewer's installed version, not the version at posting time; PhonePe mixes two version schemes.",
    }
    fig, axes = plt.subplots(len(APPS), 1, figsize=(12.5, 11.5))
    for ax, app in zip(axes, APPS):
        v = vt[vt["app_name"] == app].nlargest(10, "n").sort_values("median_date")
        mu = df.loc[df["app_name"] == app, "score"].mean()
        colors = [APP_COLORS[app] if f == "" else ("#c93a39" if f.startswith("worse") else "#1c5cab") for f in v["flag"]]
        ax.bar(range(len(v)), v["mean_rating"], color=colors, width=0.62)
        ax.axhline(mu, color=INK2, ls="--", lw=1)
        for i, (r, n, f) in enumerate(zip(v["mean_rating"], v["n"], v["flag"])):
            ax.text(i, r + 0.04, f"{r:.2f}", ha="center", fontsize=8.5, fontweight="bold")
            ax.text(i, 0.05, f"n={n}", ha="center", fontsize=7.5, color="#ffffff" if r > 0.5 else INK)
            if f:
                ax.text(i, r + 0.42, "▼" if f.startswith("worse") else "▲", ha="center", fontsize=9, color=colors[i])
        ax.set_xticks(range(len(v)), v["app_version"], fontsize=8.5)
        ax.set_ylim(0, v["mean_rating"].max() + 1.0)
        ax.set_ylabel(f"{app}\n(app mean {mu:.2f}, dashed)", rotation=0, ha="right", va="center", fontweight="bold", color=INK)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("Ten most-reviewed versions per app, ordered by median review date (▼/▲ = |z|>2 vs the app's mean)")
    nw = len(summary["version_analysis"]["worse_than_average"])
    exp_fp = 0.0228 * len(vt)
    summary["version_analysis"]["expected_false_flags_per_direction"] = round(float(exp_fp), 1)
    save(fig, "eda_12_app_versions",
         f"{nw} of {len(vt)} versions rate significantly worse than their app's mean (~{exp_fp:.0f} expected by chance); version is confounded with time",
         f"Mean rating per version with ≥{MIN_CELL_N} reviews. Red ▼ z ≤ −2, dark blue ▲ z ≥ 2. Newer PhonePe/Paytm versions rate higher partly because of the July sample shift (chart 11). Full table: data/eda/version_metrics.csv.")
    return vt


def chart_13_version_issue_mix(df, vt, summary):
    """For the versions that rate significantly worse: WHICH failure is elevated vs the app's baseline?"""
    worse = vt[vt["flag"].str.startswith("worse")].copy()
    worse["order"] = worse["app_name"].map({a: i for i, a in enumerate(APPS)})
    worse = worse.sort_values(["order", "median_date"])
    diff = np.full((len(worse), len(ISSUES)), np.nan)
    signals = []
    for r, (_, v) in enumerate(worse.iterrows()):
        app_df = df[df["app_name"] == v["app_name"]]
        ver_df = app_df[app_df["app_version"] == v["app_version"]]
        n = len(ver_df)
        for c, i in enumerate(ISSUES):
            p0, p1 = app_df[f"issue_{i}"].mean(), ver_df[f"issue_{i}"].mean()
            se = np.sqrt(max(p0 * (1 - p0), 1e-9) / n)
            if abs(p1 - p0) > 2 * se and abs(p1 - p0) >= 0.03:      # significant AND at least 3 pp
                diff[r, c] = (p1 - p0) * 100
                signals.append({"app_name": v["app_name"], "app_version": v["app_version"], "issue": i,
                                "version_pct": round(float(p1 * 100), 1), "app_baseline_pct": round(float(p0 * 100), 1)})
    summary["version_issue_signals"] = signals
    rows = [f"{a} {v}  (n={n}, {m:.2f}★)" for a, v, n, m in zip(worse["app_name"], worse["app_version"], worse["n"], worse["mean_rating"])]
    fig, ax = plt.subplots(figsize=(11.5, 0.55 * len(rows) + 2.6))
    im = heatmap(ax, diff, [ISSUE_LABELS[i] for i in ISSUES], rows, DIV, fmt="{:+.0f}", vmin=-25, vmax=25)
    ax.set_xticks(range(len(ISSUES)), [ISSUE_LABELS[i] for i in ISSUES], rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="pp vs the app's overall rate")
    ups = [x for x in signals if x["version_pct"] > x["app_baseline_pct"]]
    ops = [x for x in ups if x["issue"] in ("delivery_delay", "customer_support", "order_quality_fulfillment", "cancellation_return")]
    crash = [x for x in ups if x["issue"] in ("crash_bugs_stability", "ui_ux_update")]
    quiet = len(worse) - len({(x["app_name"], x["app_version"]) for x in signals})
    summary["version_issue_summary"] = {"flagged_versions": int(len(worse)), "elevated_signals": len(ups),
                                        "operational_signals": len(ops), "crash_or_ui_signals": len(crash),
                                        "versions_with_no_specific_signal": int(quiet)}
    save(fig, "eda_13_version_issue_mix",
         f"Worse-rated versions show more fulfilment/support complaints ({len(ops)} of {len(ups)} signals), never crash or UI ({len(crash)})",
         f"Issue rate in each flagged version minus its app's overall rate (pp). Blank = not significant (|diff| < 2 SE or < 3 pp). {quiet} of {len(worse)} flagged versions show no specific issue: a general drop not tied to one issue type.")


def export_monthly_trend(df, window):
    """Monthly per-app metrics over ALL months (with flags) - the trend dataset for Person 5."""
    g = df.groupby(["app_name", "month"], observed=True).agg(
        n=("score", "size"), mean_rating=("score", "mean"), pct_low_star=("is_low", "mean"),
        pct_has_issue=("has_issue", "mean"), mean_sentiment=("sentiment_compound", "mean"),
        mean_issue_count=("issue_count", "mean"), median_length=("review_length", "median"),
        **{f"pct_{i}": (f"issue_{i}", "mean") for i in ISSUES}).reset_index()
    pct_cols = [c for c in g.columns if c.startswith("pct_")]
    g[pct_cols] = g[pct_cols] * 100
    g["in_common_window"] = g["month"].isin(window)
    g["post_regime_shift"] = g["month"] >= f"{REGIME_START.year}-{REGIME_START.month:02d}"
    g["partial_month"] = g["month"] == PARTIAL_MONTH
    g["n_ge_min"] = g["n"] >= MIN_CELL_N
    g = g.sort_values(["app_name", "month"])
    g.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c).to_csv(TABLES_DIR / "monthly_trend.csv", index=False)
    print(f"  saved monthly_trend.csv ({len(g)} app-months)")


# ============================================================================
# Statistical tests
# ============================================================================
def statistical_tests(df, summary):
    print("\n[Stats] hypothesis tests")
    t = {}
    ct = pd.crosstab(df["app_name"], df["score"])
    chi2, p, dof, v = cramers_v(ct)
    t["rating_vs_app_chi2"] = {"chi2": round(chi2, 1), "dof": dof, "p": p, "cramers_v": round(v, 3)}
    groups = [df.loc[df["app_name"] == a, "score"] for a in APPS]
    h, p = stats.kruskal(*groups)
    t["rating_by_app_kruskal"] = {"H": round(float(h), 1), "p": float(p)}
    a, b = df.loc[df["has_issue"] == 1, "thumbs_up"], df.loc[df["has_issue"] == 0, "thumbs_up"]
    u = stats.mannwhitneyu(a, b)
    t["thumbs_up_tagged_vs_untagged"] = {"median_tagged": float(a.median()), "median_untagged": float(b.median()),
                                         "mannwhitney_p": float(u.pvalue),
                                         "rank_biserial": round(float(1 - 2 * u.statistic / (len(a) * len(b))), 4)}
    hi_len, lo_len = df.loc[df["has_issue"] == 1, "review_length"], df.loc[df["has_issue"] == 0, "review_length"]
    t["length_tagged_vs_untagged"] = {"median_tagged": float(hi_len.median()), "median_untagged": float(lo_len.median()),
                                      "mannwhitney_p": float(stats.mannwhitneyu(hi_len, lo_len).pvalue)}
    r = stats.spearmanr(df["review_length"], df["thumbs_up"])
    t["spearman_length_vs_thumbs"] = {"rho": round(float(r.statistic), 4), "p": float(r.pvalue)}
    s = stats.spearmanr(df["sentiment_compound"], df["thumbs_up"])
    t["spearman_sentiment_vs_thumbs"] = {"rho": round(float(s.statistic), 4), "p": float(s.pvalue)}
    # issue effect on rating: Mann-Whitney per issue
    eff = {}
    for i in ISSUES:
        x, y = df.loc[df[f"issue_{i}"] == 1, "score"], df.loc[df[f"issue_{i}"] == 0, "score"]
        m = stats.mannwhitneyu(x, y)
        eff[i] = {"mean_score_with": round(float(x.mean()), 3), "mean_score_without": round(float(y.mean()), 3),
                  "rank_biserial": round(float(1 - 2 * m.statistic / (len(x) * len(y))), 3), "p": float(m.pvalue)}
    t["rating_effect_of_each_issue"] = eff
    summary["statistical_tests"] = t
    print("  ", json.dumps({k: v for k, v in t.items() if k != "rating_effect_of_each_issue"})[:400], "...")


# ============================================================================
def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    meta = pd.read_csv(META_PATH)
    window, _ = common_window(df)
    print(f"Loaded {len(df):,} reviews; common window = {window}")
    summary = {"input_rows": int(len(df)),
               "common_window": {"months": window, "min_reviews_per_app_month": MIN_MONTH_N, "partial_month": PARTIAL_MONTH}}

    a_quality_audit(df, summary)
    print("[A] coverage, ratings, engagement")
    chart_01_coverage_timeline(df, window)
    chart_02_ratings_and_bias(df, meta, summary)
    chart_03_engagement(df, summary)
    chart_04_correlation(df, summary)
    print("[B] issues")
    chart_05_issue_priority(df, summary)
    chart_06_issue_by_app(df, summary)
    chart_07_cooccurrence(df, summary)
    chart_08_issue_by_rating(df, summary)
    chart_09_taxonomy_gap(df, summary)
    print("[C] time and versions")
    chart_10_monthly_trend(df, window, summary)
    chart_11_regime_shift(df, window, summary)
    vt = chart_12_app_versions(df, summary)
    chart_13_version_issue_mix(df, vt, summary)
    export_monthly_trend(df, window)
    statistical_tests(df, summary)

    summary["charts"] = CHART_INDEX
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {len(CHART_INDEX)} charts to {CHARTS_DIR}")
    print(f"Saved version table to {TABLES_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
