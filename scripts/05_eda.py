"""
Stage 4: Exploratory Data Analysis (Persons 3 and 2)

Reads data/tagged/*.csv.gz (11 apps, every English/India review since 1 Apr 2026) and produces
  * 15 charts              -> data/charts/eda/eda_XX_*.png
  * data/eda_summary.json  -> every number quoted in EDA.md, plus the statistical tests
  * data/eda/monthly_trend.csv   -> monthly per-app metrics (full months flagged)
  * data/eda/weekly_trend.csv    -> weekly per-app metrics (complete weeks only)
  * data/eda/version_metrics.csv -> rating per app version with z-scores

Charts
  01 coverage (reviews/day)      08 issues by star rating (tagger false positives)
  02 ratings vs public rating    09 taxonomy coverage gap
  03 length + upvotes            10 weekly trends, one panel per domain
  04 correlation matrix          11 July 2026 check (did the old sample's shift survive?)
  05 issue size vs severity      12 rating by app version
  06 issues by app               13 which issues are elevated in the worse-rated versions
  07 issue co-occurrence         14 domain comparison
                                 15 late-April feed gap (positive reviews missing)

Every app is collected over the same fixed window (Sort.NEWEST back to 1 Apr 2026), so
cross-app comparisons need no common-window restriction. Eleven series never share one
panel: time charts are faceted by domain (Food & Grocery, Shopping, Payments).
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
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, DOMAIN_COLORS, DOMAINS  # noqa: E402
from data_io import read_stage  # noqa: E402

# ----------------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
META_PATH = REPO_ROOT / "data" / "app_metadata.csv"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "eda"
TABLES_DIR = REPO_ROOT / "data" / "eda"
SUMMARY_PATH = REPO_ROOT / "data" / "eda_summary.json"

APPS = APP_NAMES
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

WINDOW_START = pd.Timestamp("2026-04-01")
JULY = pd.Timestamp("2026-07-01")
# 21 Apr - 5 May 2026: short positive reviews are largely missing from the Play Store feed for several apps
# while negative reviews fall far less (chart 15). Share-based metrics are distorted
# in this window, so trend comparisons, the July check and the version analysis exclude it.
GAP_START, GAP_END = pd.Timestamp("2026-04-21"), pd.Timestamp("2026-05-05")
MIN_CELL_N = 30            # min reviews before a rate/mean is plotted or a version is analysed
VERSION_MIN_GAP = 0.25     # a version is flagged only if it is >= 0.25 stars from its app mean ...
VERSION_Z = 3              # ... AND |z| >= 3 (with ~1.5M rows, z >= 2 alone flags trivial gaps)

# ----------------------------------------------------------------------------
# Visual style (palette validated with dataviz validate_palette.js)
# ----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
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
    CHART_INDEX.append({"file": f"data/charts/eda/{name}.png", "title": title, "subtitle": sub})
    print(f"  saved {path.name}")


def text_on(rgb_or_hex):
    r, g, b = to_rgb(rgb_or_hex)
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def heatmap(ax, data, xlabels, ylabels, cmap, fmt="{:.0f}", vmin=None, vmax=None, fontsize=8.5):
    vals = np.asarray(data, dtype=float)
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
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=fontsize,
                    color=text_on(im.cmap(im.norm(vals[i, j]))[:3]))
    return im


def domain_dividers(ax, axis="x"):
    """Thin rules between domain blocks on a heatmap whose apps are in APPS order."""
    pos, k = [], 0
    for d in DOMAINS[:-1]:
        k += sum(APP_DOMAIN[a] == d for a in APPS)
        pos.append(k - 0.5)
    for p in pos:
        (ax.axvline if axis == "x" else ax.axhline)(p, color=INK, lw=1.2)


def cramers_v(table):
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(chi2), float(p), int(dof), float(np.sqrt(chi2 / (n * k)))


def month_label(m):
    return pd.Period(m).strftime("%b '%y")


def domain_apps(domain):
    return [a for a in APPS if APP_DOMAIN[a] == domain]


def label_line_ends(ax, series_by_app, x_of):
    """Direct labels at line ends, nudged apart so they never overlap."""
    ends = sorted(((s.iloc[-1], a, x_of(s.index[-1])) for a, s in series_by_app.items() if len(s)), key=lambda t: t[0])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    placed = []
    for y, a, x in ends:
        y_adj = max(y, placed[-1] + gap) if placed else y
        placed.append(y_adj)
        ax.text(x, y_adj, f"  {a}", color=INK, fontsize=8.5, va="center")


# ----------------------------------------------------------------------------
# Load & prepare
# ----------------------------------------------------------------------------
def load():
    df = read_stage("tagged", parse_dates=["review_date"])
    df.attrs["n_source_columns"] = df.shape[1]
    df["app_name"] = pd.Categorical(df["app_name"], APPS, ordered=True)
    df["domain"] = pd.Categorical(df["domain"], DOMAINS, ordered=True)
    df["week"] = df["review_date"].dt.to_period("W-SUN").dt.start_time
    df["is_low"] = (df["score"] <= 2).astype(np.int8)         # Person 4's `is_problematic`
    df["log_thumbs"] = np.log1p(df["thumbs_up"])
    day = df["review_date"].dt.normalize()
    df["in_gap"] = (day >= GAP_START) & (day <= GAP_END)
    return df


def shade_gap(ax):
    ax.axvspan(GAP_START, GAP_END + pd.Timedelta(days=1), color="#000", alpha=0.06, lw=0)


def periods(df):
    """Full months and complete weeks inside the collection window shared by every app."""
    end = df.groupby("app_name", observed=True)["review_date"].max().min().normalize()   # last day every app has
    months = [str(p) for p in pd.period_range(WINDOW_START, end, freq="M") if p.end_time.normalize() <= end]
    weeks = sorted(w for w in df["week"].unique() if w >= WINDOW_START and w + pd.Timedelta(days=6) <= end)
    return end, months, weeks


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
        "outlier_policy": "No rows removed. thumbs_up outliers are genuine heavily-upvoted reviews (kept, handled with medians / "
                          "log1p / rank statistics); review_length and sentiment_compound outliers are plausible.",
    }
    per_app = df.groupby("app_name", observed=True).agg(
        domain=("domain", "first"), reviews=("review_id", "size"), first_review=("review_date", "min"),
        last_review=("review_date", "max"), distinct_versions=("app_version", "nunique"),
        version_missing_pct=("app_version", lambda s: s.isna().mean() * 100),
        mean_rating=("score", "mean"), pct_low_star=("is_low", "mean"),
        median_thumbs_up=("thumbs_up", "median"), median_length_words=("review_length", "median"),
    )
    per_app["pct_low_star"] *= 100
    per_app = per_app.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c)
    summary["coverage_by_app"] = json.loads(per_app.reset_index().astype(str).to_json(orient="records"))
    summary["reviews_by_domain"] = {d: int(n) for d, n in df["domain"].value_counts().reindex(DOMAINS).items()}


def chart_01_coverage(df, end, weeks, summary):
    """Reviews per day by complete week, one panel per domain."""
    w = df[df["week"].isin(weeks)]
    per_day = w.groupby(["week", "app_name"], observed=True).size().unstack(fill_value=0) / 7
    days = (end - WINDOW_START).days + 1                     # the last common day is complete, so it counts
    avg = df[df["review_date"] < end + pd.Timedelta(days=1)].groupby("app_name", observed=True).size() / days
    summary["reviews_per_day"] = {a: round(float(v), 1) for a, v in avg.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True)
    for ax, d in zip(axes, DOMAINS):
        series = {}
        for a in domain_apps(d):
            s = per_day[a]
            ax.plot(s.index, s.values, color=APP_COLORS[a], lw=2, label=a)
            series[a] = s
        shade_gap(ax)
        ax.set_title(d, loc="left", fontsize=10.5, color=INK2)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
        ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4)
    axes[0].set_ylabel("Reviews per day (weekly average)")
    hi, lo = avg.idxmax(), avg.idxmin()
    n_days = (df["review_date"].max().normalize() - WINDOW_START).days + 1
    days_with = df.assign(d=df["review_date"].dt.normalize()).groupby("app_name", observed=True)["d"].nunique()
    summary["days_with_reviews"] = {"window_days": int(n_days), **{a: int(v) for a, v in days_with.items()}}
    full = int((days_with == n_days).sum())
    gaps = [f"{a} {int(v)}/{n_days}" for a, v in days_with.items() if v < n_days]
    save(fig, "eda_01_coverage",
         f"{full} of {len(APPS)} apps have reviews on all {n_days} days from 1 Apr 2026; volume ranges {avg[lo]:,.0f}/day ({lo}) to {avg[hi]:,.0f}/day ({hi})"
         + (f"; {', '.join(gaps)} days" if gaps else ""),
         f"{len(df):,} reviews in total. Complete weeks only ({pd.Timestamp(weeks[0]):%d %b} – {pd.Timestamp(weeks[-1]) + pd.Timedelta(days=6):%d %b}). "
         f"Grey band = {GAP_START:%d %b}–{GAP_END:%d %b} feed gap (chart 15). The time window is identical for every app, so apps and weeks compare directly.")


def chart_02_ratings_vs_public(df, meta, summary):
    """Rating mix per app (left) and sample mean vs the public Play Store rating (right)."""
    ct = pd.crosstab(df["app_name"], df["score"], normalize="index") * 100
    dom = pd.crosstab(df["domain"], df["score"], normalize="index") * 100
    ct = pd.concat([pd.DataFrame([df["score"].value_counts(normalize=True).sort_index() * 100], index=["All apps"]),
                    dom, ct])
    order = ["All apps"] + DOMAINS + APPS
    ct = ct.loc[order]
    summary["rating_distribution_pct"] = {k: {int(s_): round(float(v), 2) for s_, v in r.items()} for k, r in ct.iterrows()}
    tbl = pd.DataFrame({"public_rating": meta.set_index("app_name")["current_score"],
                        "sample_mean_rating": df.groupby("app_name", observed=True)["score"].mean()}).loc[APPS]
    tbl["gap"] = tbl["public_rating"] - tbl["sample_mean_rating"]
    summary["sample_vs_public_rating"] = json.loads(tbl.round(3).to_json(orient="index"))
    chi2, p, dof, v = cramers_v(pd.crosstab(df["app_name"], df["score"]))
    summary["rating_vs_app_chi2"] = {"chi2": round(chi2, 1), "dof": dof, "p": p, "cramers_v": round(v, 3)}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7.4), gridspec_kw={"width_ratios": [6, 5]})
    y = np.arange(len(order))[::-1].astype(float)
    y[1 + len(DOMAINS):] -= 0.5          # small gap between the domain block and the app block
    y[1:] -= 0.5                          # and between "All apps" and the domains
    left = np.zeros(len(order))
    for s_ in range(1, 6):
        w = ct[s_].values
        ax.barh(y, w, left=left, color=STAR_COLORS[s_], edgecolor=SURFACE, linewidth=2, height=0.72, label=f"{s_}★")
        for yi, li, wi in zip(y, left, w):
            if wi >= 5:
                ax.text(li + wi / 2, yi, f"{wi:.0f}%", ha="center", va="center", fontsize=8.5, color=text_on(STAR_COLORS[s_]))
        left += w
    ax.set_yticks(y, order)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in ["All apps"] + DOMAINS:
            lbl.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of reviews")
    ax.set_title("Rating mix", loc="left", fontsize=10.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8.5)

    yy = np.arange(len(APPS))[::-1]
    for yi, app in zip(yy, APPS):
        r = tbl.loc[app]
        bx.plot([r["sample_mean_rating"], r["public_rating"]], [yi, yi], color=GRID, lw=3, zorder=1)
        bx.scatter(r["public_rating"], yi, s=90, color=MUTED, zorder=3, edgecolor=SURFACE, linewidth=2)
        bx.scatter(r["sample_mean_rating"], yi, s=90, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
        bx.text(r["sample_mean_rating"] - 0.06, yi, f"{r['sample_mean_rating']:.2f}", ha="right", va="center", fontsize=9)
        bx.text(r["public_rating"] + 0.06, yi, f"{r['public_rating']:.2f}", ha="left", va="center", fontsize=9, color=INK2)
    bx.set_yticks(yy, APPS)
    bx.set_xlim(1.5, 5.3)
    bx.set_xlabel("Mean star rating: blue = written reviews since 1 Apr, grey = public Play Store rating")
    bx.set_title("Written reviews vs public rating", loc="left", fontsize=10.5, color=INK2)
    bx.grid(axis="y", visible=False)
    harsh = ct.loc[APPS, [1, 2]].sum(axis=1)
    save(fig, "eda_02_ratings_vs_public",
         f"{ct.loc['All apps', 5]:.0f}% of reviews are 5★ and {ct.loc['All apps', 1]:.0f}% are 1★; "
         f"{harsh.idxmax()} is the harshest app ({harsh.max():.0f}% 1–2★), {harsh.idxmin()} the mildest ({harsh.min():.0f}%)",
         f"Written reviews sit {tbl['gap'].min():.1f}–{tbl['gap'].max():.1f} stars below each app's public rating, which also counts "
         f"ratings without a written review. App effect on rating: Cramér's V = {v:.2f}.")


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
        ax.text(i, d.median() + 0.8, f"{d.median():.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, 6), [f"{s_}★" for s_ in range(1, 6)])
    ax.set_ylabel("Review length (words)")
    ax.set_title("Length by star rating", loc="left", fontsize=10.5, color=INK2)
    rho, p = stats.spearmanr(df["score"], df["review_length"])
    r_up, _ = stats.spearmanr(df["review_length"], df["thumbs_up"])
    summary["review_length"] = {"spearman_score_vs_length": round(float(rho), 4), "spearman_p": float(p),
                                "median_words_by_score": {s_: float(d.median()) for s_, d in zip(range(1, 6), data)},
                                "spearman_length_vs_thumbs": round(float(r_up), 4)}

    t = np.sort(df["thumbs_up"].to_numpy())
    cum = np.cumsum(t) / t.sum()
    x = np.arange(1, len(t) + 1) / len(t)
    gini = 1 - 2 * float(np.sum((cum[1:] + cum[:-1]) / 2 * np.diff(x)) + cum[0] * x[0] / 2)
    top1 = t[int(len(t) * 0.99):].sum() / t.sum()
    zero = (t == 0).mean()
    ax = axes[1]
    ax.plot(x, cum, color=BLUE, lw=2)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.fill_between(x, cum, x, color=BLUE, alpha=0.08)
    ax.set_xlabel("Cumulative share of reviews (least → most upvoted)")
    ax.set_ylabel("Cumulative share of all upvotes")
    ax.set_title(f"Upvote concentration (Gini = {gini:.2f})", loc="left", fontsize=10.5, color=INK2)
    ax.text(0.05, 0.6, f"{zero * 100:.0f}% of reviews have 0 upvotes;\ntop 1% hold {top1 * 100:.0f}% of all upvotes", fontsize=9.5)

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
                            "zero_upvotes_pct": round(float(zero * 100), 2),
                            "share_of_upvotes_by_score_pct": {int(k): round(float(v), 2) for k, v in share.items()},
                            "mean": round(float(df["thumbs_up"].mean()), 2), "median": float(df["thumbs_up"].median())}
    ratio = data[0].median() / max(data[4].median(), 1)
    save(fig, "eda_03_length_and_upvotes",
         f"1★ reviews are {ratio:.0f}× longer than 5★ ({data[0].median():.0f} vs {data[4].median():.0f} words); upvotes are winner-take-all (Gini {gini:.2f})",
         f"1★ is {rshare[1]:.0f}% of reviews but {share[1]:.0f}% of upvotes"
         + (": other users endorse complaints far more than praise. " if share[1] > 1.5 * rshare[1] else ". ") +
         f"{zero * 100:.0f}% of reviews have no upvotes, so use log(1+upvotes) or medians, never the mean.")


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
    r_sent = corr.loc["score", "sentiment_compound"]
    r_issue = corr.loc["score", "issue_count"]
    save(fig, "eda_04_correlation_matrix",
         f"Rating tracks sentiment (ρ={r_sent:.2f}) and issue tags (ρ={r_issue:.2f}); issue tags also track review length (ρ={r_len_issue:.2f})",
         "Spearman rank correlations. Longer text has more chances to match a keyword rule, so the model must not learn length alone.")


# ============================================================================
# C. ISSUE ANALYSIS
# ============================================================================
def chart_05_issue_priority(df, summary):
    rows = []
    base_score = df["score"].mean()
    for i in ISSUES:
        s = df[df[f"issue_{i}"] == 1]
        rows.append({"issue": i, "label": ISSUE_LABELS[i], "n": len(s), "prevalence_pct": len(s) / len(df) * 100,
                     "mean_score": s["score"].mean(), "mean_sentiment": s["sentiment_compound"].mean(),
                     "pct_1_2_star": s["is_low"].mean() * 100, "median_thumbs": s["thumbs_up"].median(),
                     "mean_thumbs": s["thumbs_up"].mean()})
    q = pd.DataFrame(rows).set_index("issue")
    summary["issue_priority"] = json.loads(q.round(3).to_json(orient="index"))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(q["prevalence_pct"], q["mean_score"], s=np.sqrt(q["n"]) * 2.2, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=2)
    placed = []
    for _, r in q.sort_values("mean_score", ascending=False).iterrows():
        near = any(abs(r["prevalence_pct"] - px) < 0.3 and abs(r["mean_score"] - py) < 0.12 for px, py in placed)
        ax.annotate(r["label"], (r["prevalence_pct"], r["mean_score"]), xytext=(10, -14 if near else 6),
                    textcoords="offset points", fontsize=9, ha="left", color=INK)
        placed.append((r["prevalence_pct"], r["mean_score"]))
    ax.axhline(base_score, color=MUTED, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], base_score + 0.03, f"dataset mean {base_score:.2f}★", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("Prevalence: % of all reviews (bubble area ∝ review count)")
    ax.set_ylabel("Mean star rating of reviews with the issue")
    worst, big = q["mean_score"].idxmin(), q["prevalence_pct"].idxmax()
    above = [ISSUE_LABELS[i] for i in q.index if q.loc[i, "mean_score"] > base_score]
    title = (f"{ISSUE_LABELS[big]} is both the most common issue ({q.loc[big, 'prevalence_pct']:.1f}% of reviews) and the most damaging ({q.loc[big, 'mean_score']:.2f}★)"
             if big == worst else
             f"{ISSUE_LABELS[big]} is the most common issue ({q.loc[big, 'prevalence_pct']:.1f}%, {q.loc[big, 'mean_score']:.2f}★); "
             f"{ISSUE_LABELS[worst]} is the most damaging ({q.loc[worst, 'mean_score']:.2f}★)")
    save(fig, "eda_05_issue_priority", title,
         "Bottom-right = frequent AND low-rated. " + (f"Rated above the dataset mean: {', '.join(above)}." if above
                                                    else "Every issue category is rated below the dataset mean (dashed line)."))


def chart_06_issue_by_app(df, summary):
    tbl = pd.DataFrame({ISSUE_LABELS[i]: df.groupby("app_name", observed=True)[f"issue_{i}"].mean() * 100 for i in ISSUES}).T[APPS]
    summary["issue_prevalence_by_app_pct"] = json.loads(tbl.round(2).to_json())
    rows = []
    for i in ISSUES:
        chi2, p, dof, v = cramers_v(pd.crosstab(df["app_name"], df[f"issue_{i}"]))
        rows.append({"issue": i, "chi2": round(chi2, 1), "p_value": p, "cramers_v": round(v, 3)})
    summary["issue_vs_app_chi2"] = rows
    fig, ax = plt.subplots(figsize=(13, 5.6))
    heatmap(ax, tbl.values, APPS, list(tbl.index), SEQ, fmt="{:.1f}", vmin=0, vmax=tbl.values.max())
    ax.set_xticks(range(len(APPS)), APPS, rotation=30, ha="right")
    domain_dividers(ax, "x")
    top = max(rows, key=lambda r: r["cramers_v"])
    pooled = pd.DataFrame({ISSUE_LABELS[i]: df.groupby("domain", observed=True)[f"issue_{i}"].mean() for i in ISSUES})
    leaders = {d: pooled.loc[d].idxmax() for d in DOMAINS}
    summary["top_issue_by_domain"] = leaders
    grouped = {}
    for d in DOMAINS:
        grouped.setdefault(leaders[d], []).append(d.lower())
    load = tbl.sum()
    save(fig, "eda_06_issue_by_app",
         "; ".join(f"{iss} leads {' and '.join(ds)}" for iss, ds in grouped.items())
         + f"; {load.idxmax()} carries the heaviest issue load",
         f"% of each app's reviews tagged with the issue (vertical rules separate domains). Strongest app effect: "
         f"{ISSUE_LABELS[top['issue']]} (Cramér's V={top['cramers_v']:.2f}). All 9 chi-square tests p<0.001.")


def chart_07_cooccurrence(df, summary):
    X = df[ISSUE_COLS].to_numpy(dtype=np.int64)
    n = len(X)
    both = X.T @ X
    cnt = np.diag(both).astype(float)
    lift = both * n / np.outer(cnt, cnt)
    labels = [ISSUE_LABELS[i] for i in ISSUES]
    show = np.where(np.eye(len(ISSUES), dtype=bool), np.nan, np.log2(lift))
    fig, ax = plt.subplots(figsize=(9.2, 7))
    im = heatmap(ax, show, labels, labels, DIV, fmt="{:+.1f}", vmin=-2, vmax=2)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="log2(lift): + = co-occur more than chance")
    pairs = []
    for a in range(len(ISSUES)):
        for b in range(a + 1, len(ISSUES)):
            pairs.append({"issue_a": ISSUES[a], "issue_b": ISSUES[b], "co_count": int(both[a, b]), "lift": round(float(lift[a, b]), 2)})
    pairs = sorted(pairs, key=lambda r: -r["lift"])
    summary["issue_cooccurrence_top_pairs"] = [p for p in pairs if p["co_count"] >= 200][:6]
    summary["issue_cooccurrence_bottom_pairs"] = pairs[-3:]
    top = summary["issue_cooccurrence_top_pairs"][0]
    save(fig, "eda_07_issue_cooccurrence",
         f"{ISSUE_LABELS[top['issue_a']]} + {ISSUE_LABELS[top['issue_b']]} co-occur {top['lift']:.1f}× more than chance (n={top['co_count']:,})",
         "Pairwise lift = P(A∧B) / (P(A)·P(B)), shown as log2. Blue = issues travel together, red = tend to be mutually exclusive.")


def chart_08_issue_by_rating(df, summary):
    tbl = pd.DataFrame({ISSUE_LABELS[i]: df.groupby("score")[f"issue_{i}"].mean() * 100 for i in ISSUES}).T
    tbl.loc["Any issue"] = df.groupby("score")["has_issue"].mean() * 100
    tbl.columns = [f"{c}★" for c in tbl.columns]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    heatmap(ax, tbl.values, list(tbl.columns), list(tbl.index), SEQ, fmt="{:.1f}", vmin=0, vmax=np.nanmax(tbl.values))
    ax.axhline(len(ISSUES) - .5, color=INK, lw=1.2)
    summary["issue_any_by_rating_pct"] = {c: round(float(v), 2) for c, v in tbl.loc["Any issue"].items()}
    five = tbl["5★"].drop("Any issue")
    fp_issue = [i for i in ISSUES if ISSUE_LABELS[i] == five.idxmax()][0]
    fp = df[(df["score"] == 5) & (df[f"issue_{fp_issue}"] == 1)]
    fp_top = fp["app_name"].value_counts()
    summary["five_star_false_positive_cluster"] = {"issue": fp_issue, "count": int(len(fp)),
                                                   "top_app": str(fp_top.index[0]),
                                                   "top_app_share_pct": round(float(fp_top.iloc[0] / len(fp) * 100), 1)}
    save(fig, "eda_08_issue_by_rating",
         f"Tags track rating ({tbl.loc['Any issue', '1★']:.0f}% of 1★ vs {tbl.loc['Any issue', '5★']:.0f}% of 5★ carry an issue); "
         f"5★ '{ISSUE_LABELS[fp_issue]}' is the largest false-positive cluster",
         f"{len(fp):,} five-star reviews carry the {ISSUE_LABELS[fp_issue]} tag ({fp_top.iloc[0] / len(fp) * 100:.0f}% of them {fp_top.index[0]}): "
         "keyword rules ignore polarity, so praise such as 'easy returns' still matches.")


BRAND_STOP = {"swiggy", "zomato", "myntra", "paytm", "phonepe", "phone", "pe", "blinkit", "dominos", "domino", "flipkart",
              "amazon", "meesho", "google", "gpay", "pay", "app", "application", "im", "ive", "dont", "doesnt", "didnt",
              "cant", "just", "really", "like", "get", "got", "use", "using", "one"}
STOP = list(ENGLISH_STOP_WORDS | BRAND_STOP)


def distinctive_terms(docs_a, docs_b, apps_a, k=14, min_df=100):
    """Log-odds (Laplace smoothed) of document frequency in A vs B."""
    cv = CountVectorizer(ngram_range=(1, 2), stop_words=STOP, min_df=min_df, binary=True,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    cv.fit(pd.concat([docs_a, docs_b]))
    xa, xb = cv.transform(docs_a), cv.transform(docs_b)
    da, db = np.asarray(xa.sum(0)).ravel(), np.asarray(xb.sum(0)).ravel()
    na, nb = xa.shape[0], xb.shape[0]
    lo = np.log((da + .5) / (na - da + .5)) - np.log((db + .5) / (nb - db + .5))
    vocab = np.array(cv.get_feature_names_out())
    chosen, used = [], set()
    for j in np.argsort(-lo):
        toks = set(vocab[j].split())
        if toks & used:
            continue
        chosen.append(j)
        used |= toks
        if len(chosen) == k:
            break
    xa = xa.tocsc()
    tops = []
    for j in chosen:
        vc = apps_a.iloc[xa[:, j].nonzero()[0]].value_counts()
        tops.append((str(vc.index[0]), float(vc.iloc[0] / vc.sum() * 100)))
    t = pd.DataFrame({"term": vocab[chosen], "odds_ratio": np.exp(lo[chosen]), "df_pct": da[chosen] / na * 100})
    t["top_app"], t["top_app_share_pct"] = zip(*tops)
    return t, (na, nb)


def chart_09_taxonomy_gap(df, summary):
    low = df[df["score"] <= 2]
    un, tg = low.loc[low["has_issue"] == 0, "content"], low.loc[low["has_issue"] == 1, "content"]
    a, (na, nb) = distinctive_terms(un.fillna(""), tg.fillna(""), low.loc[un.index, "app_name"].astype(str), k=16)
    pct_un = len(un) / len(low) * 100
    summary["taxonomy_gap"] = {"low_star_reviews": int(len(low)), "low_star_untagged": int(len(un)),
                               "low_star_untagged_pct": round(float(pct_un), 2),
                               "terms": a[["term", "odds_ratio", "df_pct", "top_app", "top_app_share_pct"]].round(2).to_dict("records"),
                               "median_words_untagged": float(low.loc[low["has_issue"] == 0, "review_length"].median()),
                               "median_words_tagged": float(low.loc[low["has_issue"] == 1, "review_length"].median())}
    fig, ax = plt.subplots(figsize=(10, 6.6))
    t = a.iloc[::-1]
    ax.barh(t["term"], t["odds_ratio"], color="#eb6834", height=0.65)
    for yi, (o, d, ap, sh) in enumerate(zip(t["odds_ratio"], t["df_pct"], t["top_app"], t["top_app_share_pct"])):
        ax.text(o * 1.02, yi, f"{o:.1f}×  ({d:.1f}% of reviews)  ·  {ap} {sh:.0f}%", va="center", fontsize=8.3)
    ax.set_xlim(0, t["odds_ratio"].max() * 1.9)
    ax.set_title(f"Terms over-represented in untagged 1–2★ reviews (n={na:,}) vs tagged (n={nb:,})", loc="left", fontsize=10.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Odds ratio vs the other group")
    head = ", ".join(f"'{x}'" for x in a["term"].head(4))
    one_word = float((low.loc[low["has_issue"] == 0, "review_length"] <= 2).mean() * 100)
    summary["taxonomy_gap"]["untagged_le2_words_pct"] = round(one_word, 1)
    save(fig, "eda_09_taxonomy_gap",
         f"{pct_un:.0f}% of 1–2★ reviews get no issue tag; the most distinctive untagged terms are {head}",
         f"Untagged low-star reviews are short (median {summary['taxonomy_gap']['median_words_untagged']:.0f} vs "
         f"{summary['taxonomy_gap']['median_words_tagged']:.0f} words; {one_word:.0f}% are 1–2 words), so many carry no issue to tag. "
         "Praise words among them point to users who mis-rate. Right-hand labels give the app that uses each term most.")


# ============================================================================
# E. TRENDS, JULY CHECK, VERSIONS, DOMAINS
# ============================================================================
def weekly_table(df, weeks):
    w = df[df["week"].isin(weeks)]
    g = w.groupby(["app_name", "week"], observed=True).agg(
        n=("score", "size"), mean_rating=("score", "mean"), pct_low_star=("is_low", "mean"),
        pct_has_issue=("has_issue", "mean"), mean_sentiment=("sentiment_compound", "mean"),
        median_length=("review_length", "median"),
        **{f"pct_{i}": (f"issue_{i}", "mean") for i in ISSUES}).reset_index()
    pct_cols = [c for c in g.columns if c.startswith("pct_")]
    g[pct_cols] *= 100
    g["domain"] = g["app_name"].map(APP_DOMAIN)
    g["overlaps_gap"] = (g["week"] <= GAP_END) & (g["week"] + pd.Timedelta(days=6) >= GAP_START)
    return g


def chart_10_weekly_trend(df, weeks, summary):
    g = weekly_table(df, weeks)
    g.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c).to_csv(TABLES_DIR / "weekly_trend.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True, sharey="row")
    for col_i, d in enumerate(DOMAINS):
        for row_i, (col, ylabel) in enumerate([("mean_rating", "Mean star rating"), ("pct_low_star", "% rated 1–2★")]):
            ax = axes[row_i, col_i]
            for a in domain_apps(d):
                s = g[g["app_name"] == a].set_index("week")[col]
                ax.plot(s.index, s.values, color=APP_COLORS[a], lw=2, marker="o", ms=3.5, label=a)
            shade_gap(ax)
            if row_i == 0:
                ax.set_title(d, loc="left", fontsize=10.5, color=INK2, pad=24)
                ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
            if col_i == 0:
                ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    # Compare two clean periods outside the gap: May 6 - Jun 30 vs Aug 1 - end.
    early = g[(g["week"] > GAP_END) & (g["week"] + pd.Timedelta(days=6) < JULY)]
    late = g[g["week"] >= pd.Timestamp("2026-08-01")]
    d_low = (late.groupby("app_name", observed=True)["pct_low_star"].mean()
             - early.groupby("app_name", observed=True)["pct_low_star"].mean()).reindex(APPS)
    d_rating = (late.groupby("app_name", observed=True)["mean_rating"].mean()
                - early.groupby("app_name", observed=True)["mean_rating"].mean()).reindex(APPS)
    summary["change_may_jun_to_aug_sep"] = {"pct_low_star_pp": {a: round(float(v), 2) for a, v in d_low.items()},
                                            "mean_rating": {a: round(float(v), 3) for a, v in d_rating.items()}}
    clean = g[~g["overlaps_gap"]]
    vol = clean.groupby("app_name", observed=True)["mean_rating"].std().reindex(APPS)
    summary["weekly_rating_sd_outside_gap"] = {a: round(float(v), 3) for a, v in vol.items()}
    w = df[df["week"].isin(weeks) & ~df["in_gap"]]
    h, p = stats.kruskal(*[w.loc[w["app_name"] == a, "score"] for a in APPS])
    chi2, pc, dof, v = cramers_v(pd.crosstab(w["app_name"], w["score"]))
    summary["rating_differs_by_app"] = {"kruskal_H": round(float(h), 1), "kruskal_p": float(p), "chi2": round(chi2, 1),
                                        "chi2_p": pc, "cramers_v": round(v, 3), "n": int(len(w))}
    worse, better = d_low.idxmax(), d_low.idxmin()
    save(fig, "eda_10_weekly_trend",
         f"From May–Jun to Aug–Sep, {worse} worsened most ({d_low[worse]:+.1f} pp 1–2★) and {better} improved most ({d_low[better]:+.1f} pp); "
         f"{int((d_low.abs() < 2.5).sum())} of {len(APPS)} apps move by less than 2.5 pp",
         f"Complete weeks; grey band = {GAP_START:%d %b}–{GAP_END:%d %b} feed gap, excluded from the comparison (chart 15). "
         f"Differences between apps (Cramér's V = {v:.2f}) are far larger than any app's movement over time.")


LEN_BINS = [0, 5, 15, 40, 10_000]
LEN_LABELS = ["≤5 words", "6–15", "16–40", "41+"]


def chart_11_july_check(df, months, summary):
    """The old MOST_RELEVANT sample showed a July-2026 'regime shift'. Does it exist in a complete NEWEST census?"""
    w = df[df["month"].isin(months) & ~df["in_gap"]].copy()
    w["post"] = w["review_date"] >= JULY
    w["lb"] = pd.cut(w["review_length"], LEN_BINS, labels=LEN_LABELS)
    m = w.groupby(["app_name", "month"], observed=True).agg(n=("score", "size"), med_len=("review_length", "median"))
    days_present = w.assign(d=w["review_date"].dt.normalize()).groupby("month")["d"].nunique()   # gap days excluded
    m["per_day"] = m["n"] / m.index.get_level_values("month").map(days_present).to_numpy()
    per_day = m["per_day"].unstack()[months]
    med_len = m["med_len"].unstack()[months]
    pre_m = [x for x in months if pd.Period(x).start_time < JULY]
    post_m = [x for x in months if pd.Period(x).start_time >= JULY]
    vol_ratio = per_day[post_m].mean(axis=1) / per_day[pre_m].mean(axis=1)
    len_change = med_len[post_m].median(axis=1) - med_len[pre_m].median(axis=1)
    rows = {}
    for app, g in w.groupby("app_name", observed=True):
        wt = g["lb"].value_counts(normalize=True)
        row = {}
        for col in ["score", "has_issue"]:
            vals = {}
            for post in (False, True):
                gg = g[g["post"] == post]
                mm = gg.groupby("lb", observed=True)[col].mean()
                ww = wt.reindex(mm.index) / wt.reindex(mm.index).sum()
                vals[post] = (gg[col].mean(), (mm * ww).sum())
            row[f"{col}_raw_change"] = vals[True][0] - vals[False][0]
            row[f"{col}_adj_change"] = vals[True][1] - vals[False][1]
        rows[app] = row
    adj = pd.DataFrame(rows).T.loc[APPS]
    summary["july_check"] = {
        "reviews_per_day_ratio_jul_sep_vs_apr_jun": {a: round(float(v), 2) for a, v in vol_ratio.items()},
        "median_length_change_words": {a: float(v) for a, v in len_change.items()},
        "rating_change_raw": {a: round(float(v), 3) for a, v in adj["score_raw_change"].items()},
        "rating_change_length_adjusted": {a: round(float(v), 3) for a, v in adj["score_adj_change"].items()},
        "issue_rate_change_raw_pp": {a: round(float(v * 100), 2) for a, v in adj["has_issue_raw_change"].items()},
        "issue_rate_change_length_adjusted_pp": {a: round(float(v * 100), 2) for a, v in adj["has_issue_adj_change"].items()},
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    idx = per_day.div(per_day[pre_m].mean(axis=1), axis=0) * 100
    heatmap(axes[0], idx.values, [month_label(x) for x in months], APPS, DIV, fmt="{:.0f}", vmin=0, vmax=200, fontsize=8)
    axes[0].axvline(len(pre_m) - 0.5, color=INK, lw=1.2)
    domain_dividers(axes[0], "y")
    axes[0].set_title("Reviews per day, indexed to the app's Apr–Jun average (=100)", loc="left", fontsize=10.5, color=INK2)
    heatmap(axes[1], med_len.values, [month_label(x) for x in months], APPS, SEQ, fmt="{:.0f}",
            vmin=0, vmax=np.nanmax(med_len.values), fontsize=8)
    axes[1].axvline(len(pre_m) - 0.5, color=INK, lw=1.2)
    domain_dividers(axes[1], "y")
    axes[1].set_title("Median review length (words)", loc="left", fontsize=10.5, color=INK2)
    up = int((vol_ratio > 1.5).sum())
    shorter = int((len_change <= -2).sum())
    common_shift = up > len(APPS) / 2 and shorter > len(APPS) / 2
    summary["july_check"]["apps_with_volume_up_50pct"] = up
    summary["july_check"]["apps_with_reviews_2plus_words_shorter"] = shorter
    summary["july_check"]["verdict"] = ("A common July shift exists in the full review stream." if common_shift else
                                        "No common July shift in the full review stream; the July 'regime shift' in the old "
                                        "MOST_RELEVANT sample was produced by that sampling method.")
    title = (f"July 2026 shift confirmed: {up} of {len(APPS)} apps jump ≥1.5× in volume" if common_shift else
             f"No July 2026 shift in the full review stream: {up} of {len(APPS)} apps jump ≥1.5× in volume, {shorter} get shorter reviews")
    save(fig, "eda_11_july_check", title,
         f"The old MOST_RELEVANT sample showed a sudden July change in all apps; this complete NEWEST collection tests it. Full months, "
         f"vertical rule = 1 Jul, gap days excluded. Length-adjusted rating change Jul–Sep vs Apr–Jun: "
         f"{adj['score_adj_change'].min():+.2f} to {adj['score_adj_change'].max():+.2f}★.")


def chart_12_app_versions(df, summary):
    """Rating by app version: candidate 'bad release' detector (gap days excluded: they understate ratings)."""
    d = df[~df["in_gap"]].dropna(subset=["app_version"])
    rows = []
    for app in APPS:
        a = d[d["app_name"] == app]
        mu, sd = a["score"].mean(), a["score"].std()
        for v, grp in a.groupby("app_version"):
            if len(grp) < MIN_CELL_N:
                continue
            se = sd / np.sqrt(len(grp))
            rows.append({"app_name": app, "domain": APP_DOMAIN[app], "app_version": v, "n": len(grp),
                         "mean_rating": grp["score"].mean(), "gap_vs_app_mean": grp["score"].mean() - mu,
                         "pct_low_star": grp["is_low"].mean() * 100, "pct_has_issue": grp["has_issue"].mean() * 100,
                         "mean_sentiment": grp["sentiment_compound"].mean(), "first_seen": grp["review_date"].min(),
                         "median_date": grp["review_date"].median(), "z_vs_app_mean": (grp["score"].mean() - mu) / se})
    vt = pd.DataFrame(rows)
    worse = (vt["z_vs_app_mean"] <= -VERSION_Z) & (vt["gap_vs_app_mean"] <= -VERSION_MIN_GAP)
    better = (vt["z_vs_app_mean"] >= VERSION_Z) & (vt["gap_vs_app_mean"] >= VERSION_MIN_GAP)
    vt["flag"] = np.where(worse, "worse than app average", np.where(better, "better than app average", ""))
    vt.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c).to_csv(TABLES_DIR / "version_metrics.csv", index=False)
    summary["version_analysis"] = {
        "versions_with_min_n": int(len(vt)), "min_n": MIN_CELL_N,
        "flag_rule": f"|z| >= {VERSION_Z} AND |gap| >= {VERSION_MIN_GAP} stars vs the app's mean",
        "worse_than_average": vt[worse][["app_name", "app_version", "n", "mean_rating", "gap_vs_app_mean"]].round(2).to_dict("records"),
        "better_than_average_count": int(better.sum()),
        "caveat": "Reviews are attributed to the reviewer's installed version, not necessarily the version that caused the problem.",
    }
    fig, axes = plt.subplots(4, 3, figsize=(16, 12.5))
    for ax, app in zip(axes.ravel(), APPS):
        v = vt[vt["app_name"] == app].nlargest(8, "n").sort_values("median_date")
        mu = d.loc[d["app_name"] == app, "score"].mean()
        colors = [MUTED if f == "" else ("#c93a39" if f.startswith("worse") else "#1c5cab") for f in v["flag"]]
        ax.bar(range(len(v)), v["mean_rating"], color=colors, width=0.62)
        ax.axhline(mu, color=INK2, ls="--", lw=1)
        for i, r in enumerate(v["mean_rating"]):
            ax.text(i, r + 0.06, f"{r:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(v)), [str(x).split(" (")[0][:14] for x in v["app_version"]], fontsize=6.5, rotation=35, ha="right")
        ax.set_ylim(0, 5.3)
        ax.set_title(f"{app}  (mean {mu:.2f}★, dashed)", loc="left", fontsize=10, color=INK2)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(APPS):]:
        ax.set_visible(False)
    nw = int(worse.sum())
    save(fig, "eda_12_app_versions",
         f"{nw} of {len(vt)} app versions rate clearly worse than their app's average (≥{VERSION_MIN_GAP}★ below, z ≤ −{VERSION_Z})",
         f"Eight most-reviewed versions per app, ordered by median review date. Red = flagged worse, dark blue = flagged better, grey = not flagged. "
         f"Versions need ≥{MIN_CELL_N} reviews; {GAP_START:%d %b}–{GAP_END:%d %b} gap days excluded. Full table: data/eda/version_metrics.csv.")
    return vt


def chart_13_version_issue_mix(df, vt, summary):
    """For the worst flagged versions: WHICH failure is elevated vs the app's baseline?"""
    worse = vt[vt["flag"].str.startswith("worse")].nsmallest(15, "z_vs_app_mean").copy()
    worse["order"] = worse["app_name"].map({a: i for i, a in enumerate(APPS)})
    worse = worse.sort_values(["order", "median_date"])
    diff = np.full((len(worse), len(ISSUES)), np.nan)
    signals = []
    clean = df[~df["in_gap"]]
    for r, (_, v) in enumerate(worse.iterrows()):
        app_df = clean[clean["app_name"] == v["app_name"]]
        ver_df = app_df[app_df["app_version"] == v["app_version"]]
        n = len(ver_df)
        for c, i in enumerate(ISSUES):
            p0, p1 = app_df[f"issue_{i}"].mean(), ver_df[f"issue_{i}"].mean()
            se = np.sqrt(max(p0 * (1 - p0), 1e-9) / n)
            if abs(p1 - p0) > 3 * se and abs(p1 - p0) >= 0.02:
                diff[r, c] = (p1 - p0) * 100
                signals.append({"app_name": v["app_name"], "app_version": v["app_version"], "issue": i,
                                "version_pct": round(float(p1 * 100), 1), "app_baseline_pct": round(float(p0 * 100), 1)})
    summary["version_issue_signals"] = signals
    if worse.empty:
        summary["version_issue_summary"] = {"flagged_versions": 0}
        return
    rows = [f"{a} {str(v).split(' (')[0][:14]}  (n={n:,}, {m:.2f}★)" for a, v, n, m in zip(worse["app_name"], worse["app_version"], worse["n"], worse["mean_rating"])]
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(rows) + 2.8))
    vmax = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 10
    im = heatmap(ax, diff, [ISSUE_LABELS[i] for i in ISSUES], rows, DIV, fmt="{:+.0f}", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ISSUES)), [ISSUE_LABELS[i] for i in ISSUES], rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="pp vs the app's overall rate")
    ups = [x for x in signals if x["version_pct"] > x["app_baseline_pct"]]
    by_issue = pd.Series([x["issue"] for x in ups]).value_counts()
    quiet = len(worse) - len({(x["app_name"], x["app_version"]) for x in signals})
    summary["version_issue_summary"] = {"versions_shown": int(len(worse)), "elevated_signals": len(ups),
                                        "elevated_by_issue": {k: int(v) for k, v in by_issue.items()},
                                        "versions_with_no_specific_signal": int(quiet)}
    top_n = int(by_issue.iloc[0]) if len(by_issue) else 0
    leads = [ISSUE_LABELS[i] for i, n in by_issue.items() if n == top_n]
    title = (f"In the worst-rated versions, {leads[0]} is the issue most often elevated ({top_n} of {len(ups)} signals)" if len(leads) == 1 else
             f"In the worst-rated versions, {' and '.join(leads)} are the issues most often elevated ({top_n} signals each, of {len(ups)})")
    save(fig, "eda_13_version_issue_mix", title,
         f"Issue rate in each of the {len(worse)} most significantly worse versions minus its app's overall rate (pp). Blank = not significant "
         f"(|diff| < 3 SE or < 2 pp). {quiet} of {len(worse)} {'shows' if quiet == 1 else 'show'} no specific issue: a general drop rather than one failure type.")


def chart_14_domains(df, summary):
    """Domain comparison: issue prevalence and rating mix."""
    prev = pd.DataFrame({ISSUE_LABELS[i]: df.groupby("domain", observed=True)[f"issue_{i}"].mean() * 100 for i in ISSUES}).T[DOMAINS]
    low = df.groupby("domain", observed=True)["is_low"].mean().reindex(DOMAINS) * 100
    summary["domain_comparison"] = {"issue_prevalence_pct": json.loads(prev.round(2).to_json()),
                                    "pct_low_star": {d: round(float(v), 2) for d, v in low.items()},
                                    "mean_rating": {d: round(float(v), 3) for d, v in df.groupby("domain", observed=True)["score"].mean().items()}}
    fig, ax = plt.subplots(figsize=(13, 5.4))
    order = prev.max(axis=1).sort_values(ascending=False).index
    x = np.arange(len(order))
    w = 0.26
    for k, d in enumerate(DOMAINS):
        vals = prev.loc[order, d].values
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=DOMAIN_COLORS[d], label=f"{d} ({low[d]:.0f}% rated 1–2★)")
        for xi, v in zip(x + (k - 1) * w, vals):
            if v >= 0.5:
                ax.text(xi, v + 0.12, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("% of the domain's reviews")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9)
    lead = {d: prev[d].idxmax() for d in DOMAINS}
    ratio = (prev.max(axis=1) + 0.05) / (prev.min(axis=1) + 0.05)
    gap_issue = ratio.idxmax()
    summary["domain_comparison"]["most_domain_specific_issue"] = {"issue": gap_issue, "high_domain": prev.loc[gap_issue].idxmax(),
                                                                  "low_domain": prev.loc[gap_issue].idxmin(),
                                                                  "ratio": round(float(ratio[gap_issue]), 1)}
    save(fig, "eda_14_domain_comparison",
         f"{low.idxmax()} reviews are the most negative ({low.max():.0f}% 1–2★ vs {low.min():.0f}% for {low.idxmin()}); "
         f"{gap_issue} is the most domain-specific issue",
         "Top issue: " + "; ".join(f"{d} – {lead[d]}" for d in DOMAINS) + f". {gap_issue}: "
         f"{prev.loc[gap_issue].max():.1f}% in {prev.loc[gap_issue].idxmax()} vs {prev.loc[gap_issue].min():.1f}% in {prev.loc[gap_issue].idxmin()}. "
         "Bars sorted by the highest domain.")


def chart_15_april_gap(df, summary):
    """21 Apr - 5 May: positive reviews largely vanish from the feed while negative reviews fall far less."""
    span = 13
    periods_ = {"Before\n(8–20 Apr)": (GAP_START - pd.Timedelta(days=span), GAP_START - pd.Timedelta(days=1)),
                "Gap\n(21 Apr–5 May)": (GAP_START, GAP_END),
                "After\n(6–18 May)": (GAP_END + pd.Timedelta(days=1), GAP_END + pd.Timedelta(days=span))}
    day = df["review_date"].dt.normalize()
    pos, neg, med = {}, {}, {}
    for label, (a, b) in periods_.items():
        w = df[(day >= a) & (day <= b)]
        n_days = (b - a).days + 1
        pos[label] = w[w["score"] >= 4].groupby("app_name", observed=True).size().reindex(APPS) / n_days
        neg[label] = w[w["score"] <= 2].groupby("app_name", observed=True).size().reindex(APPS) / n_days
        med[label] = w.groupby("app_name", observed=True)["review_length"].median().reindex(APPS)
    pos, neg, med = pd.DataFrame(pos), pd.DataFrame(neg), pd.DataFrame(med)
    base = (pos.iloc[:, 0] + pos.iloc[:, 2]) / 2
    nbase = (neg.iloc[:, 0] + neg.iloc[:, 2]) / 2
    pos_idx = pos.div(base, axis=0) * 100
    neg_idx = neg.div(nbase, axis=0) * 100
    gap_col = list(periods_)[1]
    affected = pos_idx.index[(pos_idx[gap_col] < 60) & (pos_idx[gap_col] < 0.6 * neg_idx[gap_col])].tolist()
    summary["april_gap"] = {
        "window": [str(GAP_START.date()), str(GAP_END.date())],
        "positive_per_day": json.loads(pos.round(1).to_json(orient="index")),
        "negative_per_day": json.loads(neg.round(1).to_json(orient="index")),
        "median_words": json.loads(med.to_json(orient="index")),
        "affected_apps": affected,
        "rule": "affected = positive reviews/day in the gap < 60% of the before/after average AND fallen to less than 0.6x "
                "the relative level of negative reviews/day",
        "reviews_in_gap": int(df["in_gap"].sum()),
    }
    labels = [p.replace("\n", " ") for p in periods_]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 3, 3]})
    for ax, data, ttl, cmap, fmt, vmax in [
        (axes[0], pos_idx, "Positive (4–5★) reviews/day, index (before/after avg = 100)", DIV, "{:.0f}", 200),
        (axes[1], neg_idx, "Negative (1–2★) reviews/day, same index", DIV, "{:.0f}", 200),
        (axes[2], med, "Median review length (words)", SEQ, "{:.0f}", float(np.nanmax(med.values)))]:
        heatmap(ax, data.values, labels if ax is not axes[0] else labels, APPS, cmap, fmt=fmt, vmin=0, vmax=vmax, fontsize=8.5)
        ax.set_xticks(range(3), [p for p in periods_], fontsize=8.5)
        domain_dividers(ax, "y")
        ax.set_title(ttl, loc="left", fontsize=10, color=INK2)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    drop = 100 - pos_idx.loc[affected, gap_col].median() if affected else 0
    save(fig, "eda_15_april_gap",
         f"21 Apr–5 May: in {len(affected)} of {len(APPS)} apps positive reviews fall ~{drop:.0f}% while negative reviews fall far less — a feed gap, not an incident",
         f"Affected: {', '.join(affected) if affected else 'none'}. Short positive reviews ('good', 'nice') are what disappear, so median length jumps. "
         "Shares such as '% rated 1–2★' are inflated in this window; counts of negative reviews per day are far less affected. The gap is kept in the dataset "
         "and flagged; trend, version and July analyses exclude it.")


def export_monthly_trend(df, months):
    g = df.groupby(["app_name", "month"], observed=True).agg(
        n=("score", "size"), mean_rating=("score", "mean"), pct_low_star=("is_low", "mean"),
        pct_has_issue=("has_issue", "mean"), mean_sentiment=("sentiment_compound", "mean"),
        mean_issue_count=("issue_count", "mean"), median_length=("review_length", "median"),
        **{f"pct_{i}": (f"issue_{i}", "mean") for i in ISSUES}).reset_index()
    pct_cols = [c for c in g.columns if c.startswith("pct_")]
    g[pct_cols] = g[pct_cols] * 100
    g["domain"] = g["app_name"].map(APP_DOMAIN)
    g["full_month"] = g["month"].isin(months)
    g["post_july"] = g["month"] >= "2026-07"
    g["contains_gap_days"] = g["month"].isin(["2026-04", "2026-05"])
    g = g.sort_values(["app_name", "month"])
    g.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c).to_csv(TABLES_DIR / "monthly_trend.csv", index=False)
    print(f"  saved monthly_trend.csv ({len(g)} app-months)")


# ============================================================================
# Statistical tests
# ============================================================================
def statistical_tests(df, summary):
    print("\n[Stats] hypothesis tests")
    t = {}
    chi2, p, dof, v = cramers_v(pd.crosstab(df["app_name"], df["score"]))
    t["rating_vs_app_chi2"] = {"chi2": round(chi2, 1), "dof": dof, "p": p, "cramers_v": round(v, 3)}
    chi2, p, dof, v = cramers_v(pd.crosstab(df["domain"], df["score"]))
    t["rating_vs_domain_chi2"] = {"chi2": round(chi2, 1), "dof": dof, "p": p, "cramers_v": round(v, 3)}
    h, p = stats.kruskal(*[df.loc[df["app_name"] == a, "score"] for a in APPS])
    t["rating_by_app_kruskal"] = {"H": round(float(h), 1), "p": float(p)}
    a, b = df.loc[df["has_issue"] == 1, "thumbs_up"], df.loc[df["has_issue"] == 0, "thumbs_up"]
    u = stats.mannwhitneyu(a, b)
    t["thumbs_up_tagged_vs_untagged"] = {"mean_tagged": round(float(a.mean()), 3), "mean_untagged": round(float(b.mean()), 3),
                                         "mannwhitney_p": float(u.pvalue),
                                         "rank_biserial": round(float(1 - 2 * u.statistic / (len(a) * len(b))), 4)}
    hi_len, lo_len = df.loc[df["has_issue"] == 1, "review_length"], df.loc[df["has_issue"] == 0, "review_length"]
    t["length_tagged_vs_untagged"] = {"median_tagged": float(hi_len.median()), "median_untagged": float(lo_len.median()),
                                      "mannwhitney_p": float(stats.mannwhitneyu(hi_len, lo_len).pvalue)}
    r = stats.spearmanr(df["review_length"], df["thumbs_up"])
    t["spearman_length_vs_thumbs"] = {"rho": round(float(r.statistic), 4), "p": float(r.pvalue)}
    eff = {}
    for i in ISSUES:
        x, y = df.loc[df[f"issue_{i}"] == 1, "score"], df.loc[df[f"issue_{i}"] == 0, "score"]
        m = stats.mannwhitneyu(x, y)
        eff[i] = {"mean_score_with": round(float(x.mean()), 3), "mean_score_without": round(float(y.mean()), 3),
                  "rank_biserial": round(float(1 - 2 * m.statistic / (len(x) * len(y))), 3), "p": float(m.pvalue)}
    t["rating_effect_of_each_issue"] = eff
    summary["statistical_tests"] = t


# ============================================================================
def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHARTS_DIR.glob("eda_*.png"):
        old.unlink()
    df = load()
    meta = pd.read_csv(META_PATH)
    end, months, weeks = periods(df)
    print(f"Loaded {len(df):,} reviews; full months = {months}; {len(weeks)} complete weeks; last common day {end:%Y-%m-%d}")
    summary = {"input_rows": int(len(df)),
               "window": {"start": str(WINDOW_START.date()), "last_common_day": str(end.date()),
                          "full_months": months, "complete_weeks": len(weeks)}}

    a_quality_audit(df, summary)
    print("[A] coverage, ratings, engagement")
    chart_01_coverage(df, end, weeks, summary)
    chart_02_ratings_vs_public(df, meta, summary)
    chart_03_engagement(df, summary)
    chart_04_correlation(df, summary)
    print("[B] issues")
    chart_05_issue_priority(df, summary)
    chart_06_issue_by_app(df, summary)
    chart_07_cooccurrence(df, summary)
    chart_08_issue_by_rating(df, summary)
    chart_09_taxonomy_gap(df, summary)
    print("[C] time, versions, domains")
    chart_10_weekly_trend(df, weeks, summary)
    chart_11_july_check(df, months, summary)
    vt = chart_12_app_versions(df, summary)
    chart_13_version_issue_mix(df, vt, summary)
    chart_14_domains(df, summary)
    chart_15_april_gap(df, summary)
    export_monthly_trend(df, months)
    statistical_tests(df, summary)

    summary["charts"] = CHART_INDEX
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {len(CHART_INDEX)} charts to {CHARTS_DIR}")
    print(f"Saved tables to {TABLES_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
