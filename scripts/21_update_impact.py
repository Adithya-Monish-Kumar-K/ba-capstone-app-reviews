"""
Stage 5 of Review 2, Method 2 (Time-Series Analysis, evaluation and interpretation): update impact (#138).

Did a new app version make things worse? For each of the 224 release events in
data/timeseries/release_events.csv (Stage 1, #131) the script looks at the 7 days before and the 7 days
after the release in two ways:

  A. Before vs after (the app as a whole). 1-2 star reviews per day, the share of reviews rated 1-2 stars,
     the mean rating and the issue mix in the 7 days before the release against the 7 days from the release.
     Both windows cover one full Monday-Sunday cycle, so the weekday pattern of the food apps does not bias them.
     This view also picks up anything else that changed that week (a sale, an outage, the previous release).

  B. New version vs older versions, same days. In the 7 days from the release, reviews written on the new
     version against reviews written on older versions of the same app over exactly the same days.
     Same days means the same weekday mix, the same news and the same feed gap, so the difference
     belongs to the version. This is the main test.

Users on the newest version are not a random sample: on some apps they are always a little more negative
than users still on older builds (Flipkart and Amazon: about +6 points in almost every release), whatever the
release. So each release is compared with **its app's usual gap**, the median gap of the app's other releases
(leave-one-out). A release is listed as having **made things worse** when its gap is at least 5 percentage
points above the app's usual gap and that excess is significant after the Benjamini-Hochberg correction across
all releases (q < 0.05). It is **confirmed at app level** when view A also shows a significant rise in 1-2 star
reviews per day (q < 0.05). Releases better by the same rule are listed too. A significant change in the issue
mix (view B, q < 0.05 and at least 2 points) names what got worse.

Release day: `first_seen_10th_review` (the day the version reached its 10th review), which ignores the
handful of early-access reviews that appear before a real rollout. A sensitivity run with `first_seen`
is stored in the summary.

Input:  data/tagged/*.csv.gz                      (review level: app, date, version, score, issue flags)
        data/timeseries/daily_app.csv             (Stage 1: daily counts and the feed-gap flag)
        data/timeseries/release_events.csv        (Stage 1: release events)
        data/timeseries/unusual_days_unique.csv   (Stage 4: days outside the 95% forecast interval)
Output: data/update_impact/
          release_impact.csv        one row per release: both views, tests, verdict, most elevated issue
          release_issue_changes.csv one row per release and issue: share on the new vs older versions, test
          worse_releases.csv        the releases that made things worse, per app, worst first
          impact_by_app.csv         per app: releases tested, worse, better, typical change
        data/charts/update_impact/  01 to 04
        data/update_impact_summary.json

Run from the repository root:  python scripts/21_update_impact.py  
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, DOMAINS  # noqa: E402
from data_io import read_stage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TS_DIR = REPO_ROOT / "data" / "timeseries"
OUT_DIR = REPO_ROOT / "data" / "update_impact"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "update_impact"
SUMMARY_PATH = REPO_ROOT / "data" / "update_impact_summary.json"

WINDOW = 7                     # days before and after a release (one full weekly cycle each)
MIN_REVIEWS = 30               # reviews needed on the new version and on older versions in the after window
MIN_SHARE_DIFF_PP = 5.0        # smallest difference in 1-2 star share (percentage points) that counts as a real change
MIN_ISSUE_DIFF_PP = 2.0        # smallest difference in an issue's share that counts
ALPHA = 0.05                   # false-discovery rate (Benjamini-Hochberg)
DATA_START, DATA_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-20")

ISSUES = {
    "issue_crash_bugs_stability": "Crash & Stability",
    "issue_payment_refund": "Payment & Refund",
    "issue_delivery_delay": "Delivery Delay",
    "issue_order_quality_fulfillment": "Order Quality",
    "issue_cancellation_return": "Cancellation & Return",
    "issue_customer_support": "Customer Support",
    "issue_account_login_otp": "Account / Login / OTP",
    "issue_pricing_charges_fraud": "Pricing & Fraud",
    "issue_ui_ux_update": "UI/UX & Update",
}
VERDICT_ORDER = ["Worse (confirmed at app level)", "Worse (new version only)", "No clear change",
                 "Better", "Not testable"]
VERDICT_COLORS = {"Worse (confirmed at app level)": "#c93a39", "Worse (new version only)": "#ee8f8a",
                  "No clear change": "#d9d8d4", "Better": "#2a78d6", "Not testable": "#f2f1ee"}

# ----------------------------------------------------------------------------
# Visual style (same palette and layout as scripts 15 and 19)
# ----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})
CHART_INDEX = []


def save(fig, path, title, sub=None):
    """Left-aligned title and muted subtitle, then write the PNG (same helper as scripts 15 and 19)."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    CHART_INDEX.append({"file": str(path.relative_to(REPO_ROOT)), "title": title, "subtitle": sub})
    print(f"  saved {path.relative_to(REPO_ROOT)}")


def version_key(v):
    """Same version ordering as scripts/15_time_series_data.py: compare the numbers in the version string."""
    return tuple(int(x) for x in re.findall(r"\d+", str(v)))


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load():
    cols = ["app_name", "review_date", "app_version", "score"] + list(ISSUES)
    reviews = read_stage("tagged", columns=cols)
    reviews["day"] = pd.to_datetime(reviews["review_date"]).dt.normalize()
    reviews["neg"] = (reviews["score"] <= 2).astype(int)
    reviews["vkey"] = reviews["app_version"].map(version_key, na_action="ignore")
    daily = pd.read_csv(TS_DIR / "daily_app.csv", parse_dates=["day"])
    releases = pd.read_csv(TS_DIR / "release_events.csv", parse_dates=["first_seen", "first_seen_10th_review"])
    releases = releases[releases["use_for_update_impact"]].copy()
    alerts = pd.read_csv(TS_DIR / "unusual_days_unique.csv", parse_dates=["day"])
    return reviews, daily, releases, alerts


# ----------------------------------------------------------------------------
# 2. Tests
# ----------------------------------------------------------------------------
def two_prop_test(k1, n1, k2, n2):
    """Two-sided z-test for p1 = p2 (pooled variance). Returns (difference p1 - p2, p-value)."""
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, 1.0
    z = (p1 - p2) / se
    return p1 - p2, 2 * stats.norm.sf(abs(z))


def diff_ci(k1, n1, k2, n2, z=1.96):
    """95% Wald interval for the difference of two proportions."""
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2 - z * se, p1 - p2 + z * se


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def analyse_release(rel, app_reviews, app_daily, alerts_app, day_col):
    """Both views for one release. Returns the release row and its per-issue rows."""
    t0 = rel[day_col] if pd.notna(rel[day_col]) else rel["first_seen"]
    before = (t0 - pd.Timedelta(days=WINDOW), t0 - pd.Timedelta(days=1))
    after = (t0, t0 + pd.Timedelta(days=WINDOW - 1))
    row = {"app_name": rel["app_name"], "domain": rel["domain"], "app_version": rel["app_version"],
           "release_day": t0.date(), "first_seen": rel["first_seen"].date(),
           "version_reviews_total": int(rel["reviews"])}
    complete = before[0] >= DATA_START and after[1] <= DATA_END
    row["windows_complete"] = bool(complete)

    # ---- View A: app before vs after ----------------------------------------------------------
    db = app_daily[(app_daily["day"] >= before[0]) & (app_daily["day"] <= before[1])]
    da = app_daily[(app_daily["day"] >= after[0]) & (app_daily["day"] <= after[1])]
    rb = app_reviews[(app_reviews["day"] >= before[0]) & (app_reviews["day"] <= before[1])]
    ra = app_reviews[(app_reviews["day"] >= after[0]) & (app_reviews["day"] <= after[1])]
    gap = bool(db["feed_gap"].any() or da["feed_gap"].any())
    row["feed_gap_in_window"] = gap
    row["neg_per_day_before"] = db["neg_reviews_filled"].mean()
    row["neg_per_day_after"] = da["neg_reviews_filled"].mean()
    row["neg_per_day_change_pct"] = 100 * (row["neg_per_day_after"] / row["neg_per_day_before"] - 1) if row["neg_per_day_before"] else np.nan
    row["p_neg_per_day"] = welch(da["neg_reviews_filled"], db["neg_reviews_filled"]) if complete else np.nan
    row["neg_share_before"] = rb["neg"].mean()
    row["neg_share_after"] = ra["neg"].mean()
    _, row["p_neg_share_before_after"] = two_prop_test(ra["neg"].sum(), len(ra), rb["neg"].sum(), len(rb))
    row["rating_before"] = rb["score"].mean()
    row["rating_after"] = ra["score"].mean()
    row["p_rating_before_after"] = welch(ra["score"], rb["score"])
    row["alerts_above_in_week_after"] = int(((alerts_app["day"] >= after[0]) & (alerts_app["day"] <= after[1])
                                             & (alerts_app["alert_direction"].str.startswith("Above"))).sum())

    # ---- View B: new version vs older versions, same days ---------------------------------------
    vk = version_key(rel["app_version"])
    new = ra[ra["app_version"] == rel["app_version"]]
    old = ra[ra["vkey"].notna() & ra["vkey"].map(lambda k: k < vk if isinstance(k, tuple) else False)]
    row["n_new_version"], row["n_older_versions"] = len(new), len(old)
    row["neg_share_new"] = new["neg"].mean() if len(new) else np.nan
    row["neg_share_older"] = old["neg"].mean() if len(old) else np.nan
    row["rating_new"] = new["score"].mean() if len(new) else np.nan
    row["rating_older"] = old["score"].mean() if len(old) else np.nan
    testable = complete and len(new) >= MIN_REVIEWS and len(old) >= MIN_REVIEWS
    row["testable"] = bool(testable)
    issue_rows = []
    if testable:
        diff, p = two_prop_test(new["neg"].sum(), len(new), old["neg"].sum(), len(old))
        lo, hi = diff_ci(new["neg"].sum(), len(new), old["neg"].sum(), len(old))
        row.update(neg_share_diff_pp=100 * diff, neg_share_diff_ci_low_pp=100 * lo, neg_share_diff_ci_high_pp=100 * hi,
                   neg_share_diff_se_pp=100 * (hi - lo) / (2 * 1.96),
                   p_neg_share_new_vs_older=p, p_rating_new_vs_older=welch(new["score"], old["score"]),
                   rating_diff=row["rating_new"] - row["rating_older"])
        for col, label in ISSUES.items():
            d_i, p_i = two_prop_test(new[col].sum(), len(new), old[col].sum(), len(old))
            issue_rows.append({"app_name": rel["app_name"], "app_version": rel["app_version"], "release_day": t0.date(),
                               "issue": label, "share_new_pct": 100 * new[col].mean(), "share_older_pct": 100 * old[col].mean(),
                               "diff_pp": 100 * d_i, "p_value": p_i})
    return row, issue_rows


def run(reviews, daily, releases, alerts, day_col):
    rows, issue_rows = [], []
    for app in APP_NAMES:
        ar = reviews[reviews["app_name"] == app]
        ad = daily[daily["app_name"] == app].sort_values("day")
        al = alerts[alerts["app_name"] == app]
        for _, rel in releases[releases["app_name"] == app].sort_values("first_seen").iterrows():
            r, ir = analyse_release(rel, ar, ad, al, day_col)
            rows.append(r)
            issue_rows += ir
    res = pd.DataFrame(rows)
    issues = pd.DataFrame(issue_rows)

    # Each release against its app's usual new-version gap (median of the app's other testable releases)
    t = res["testable"]
    res["app_usual_gap_pp"] = np.nan
    for app, g in res[t].groupby("app_name", sort=False):
        d = g["neg_share_diff_pp"]
        for i in g.index:
            res.loc[i, "app_usual_gap_pp"] = d.drop(i).median() if len(d) > 1 else 0.0
    res["excess_gap_pp"] = res["neg_share_diff_pp"] - res["app_usual_gap_pp"]
    z = res["excess_gap_pp"] / res["neg_share_diff_se_pp"]
    res["p_excess_gap"] = 2 * stats.norm.sf(z.abs())
    res.loc[~t, "p_excess_gap"] = np.nan

    # Benjamini-Hochberg across all testable releases, one family per test
    for p_col, q_col in [("p_neg_share_new_vs_older", "q_neg_share_new_vs_older"),
                         ("p_rating_new_vs_older", "q_rating_new_vs_older"),
                         ("p_excess_gap", "q_excess_gap")]:
        res[q_col] = np.nan
        res.loc[t, q_col] = multipletests(res.loc[t, p_col], method="fdr_bh")[1]
    ok = res["windows_complete"] & res["p_neg_per_day"].notna()
    res["q_neg_per_day"] = np.nan
    res.loc[ok, "q_neg_per_day"] = multipletests(res.loc[ok, "p_neg_per_day"], method="fdr_bh")[1]
    if len(issues):
        issues["q_value"] = multipletests(issues["p_value"], method="fdr_bh")[1]
        issues["significant_increase"] = (issues["q_value"] < ALPHA) & (issues["diff_pp"] >= MIN_ISSUE_DIFF_PP)

    def verdict(r):
        if not r["testable"]:
            return "Not testable"
        sig = r["q_excess_gap"] < ALPHA
        if sig and r["excess_gap_pp"] >= MIN_SHARE_DIFF_PP and r["neg_share_diff_pp"] > 0:
            app_level = (r["q_neg_per_day"] < ALPHA) and (r["neg_per_day_change_pct"] > 0) and not r["feed_gap_in_window"]
            return "Worse (confirmed at app level)" if app_level else "Worse (new version only)"
        if sig and r["excess_gap_pp"] <= -MIN_SHARE_DIFF_PP and r["neg_share_diff_pp"] < 0:
            return "Better"
        return "No clear change"

    res["verdict"] = res.apply(verdict, axis=1)
    res["made_worse"] = res["verdict"].str.startswith("Worse")
    if len(issues):
        top = (issues[issues["significant_increase"]].sort_values("diff_pp", ascending=False)
               .groupby(["app_name", "app_version"], sort=False).head(1)
               .set_index(["app_name", "app_version"]))
        key = list(zip(res["app_name"], res["app_version"]))
        res["most_elevated_issue"] = [top["issue"].get(k, "") for k in key]
        res["most_elevated_issue_pp"] = [top["diff_pp"].get(k, np.nan) for k in key]
    return res, issues


# ----------------------------------------------------------------------------
# 3. Charts
# ----------------------------------------------------------------------------
def domain_apps(domain):
    return [a for a in APP_NAMES if APP_DOMAIN[a] == domain]


def chart_verdicts(res):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    apps = APP_NAMES[::-1]
    counts = res.groupby(["app_name", "verdict"]).size().unstack(fill_value=0).reindex(index=apps, columns=VERDICT_ORDER, fill_value=0)
    left = np.zeros(len(apps))
    for v in VERDICT_ORDER:
        ax.barh(apps, counts[v], left=left, color=VERDICT_COLORS[v], label=v, edgecolor=SURFACE, linewidth=1.2)
        for i, (x, w) in enumerate(zip(left, counts[v])):
            if w > 0 and v != "Not testable":
                ax.text(x + w / 2, i, str(w), ha="center", va="center", fontsize=8.5,
                        color="white" if v in ("Worse (confirmed at app level)", "Better") else INK)
        left += counts[v].to_numpy()
    for i, a in enumerate(apps):
        ax.text(left[i] + 0.4, i, f"{APP_DOMAIN[a]}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Release events")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.1), fontsize=8.5)
    n_worse = int(res["made_worse"].sum())
    n_test = int(res["testable"].sum())
    save(fig, CHARTS_DIR / "01_release_verdicts_by_app.png",
         f"{n_worse} of {n_test} testable releases made things worse; most releases changed nothing measurable",
         f"Verdict for each of the {len(res)} release events. 'Worse' = over the same 7 days, the new version's share of 1–2★ reviews "
         f"exceeds older versions' by at least {MIN_SHARE_DIFF_PP:.0f} points more than the app's usual new-version gap (Benjamini–Hochberg "
         f"q < {ALPHA}). 'Confirmed' = the app's 1–2★ reviews per day also rose significantly against the week before.")


def chart_effects(res):
    fig, axes = plt.subplots(len(DOMAINS), 4, figsize=(16, 10.5), sharey=True)
    t = res[res["testable"]]
    lim = max(abs(t["neg_share_diff_ci_low_pp"].min()), abs(t["neg_share_diff_ci_high_pp"].max())) * 1.05
    for i, dom in enumerate(DOMAINS):
        apps = domain_apps(dom)
        for j in range(4):
            ax = axes[i][j]
            if j >= len(apps):
                ax.axis("off")
                continue
            app = apps[j]
            g = t[t["app_name"] == app].copy()
            g["release_day"] = pd.to_datetime(g["release_day"])
            usual = g["neg_share_diff_pp"].median() if len(g) else 0
            ax.axhspan(usual - MIN_SHARE_DIFF_PP, usual + MIN_SHARE_DIFF_PP, color=GRID, alpha=0.5, lw=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.axhline(usual, color=INK2, lw=0.9, ls="--")
            for v in VERDICT_ORDER[:-1]:
                gv = g[g["verdict"] == v]
                ax.errorbar(gv["release_day"], gv["neg_share_diff_pp"],
                            yerr=[gv["neg_share_diff_pp"] - gv["neg_share_diff_ci_low_pp"],
                                  gv["neg_share_diff_ci_high_pp"] - gv["neg_share_diff_pp"]],
                            fmt="o", ms=5, color=VERDICT_COLORS[v] if v != "No clear change" else MUTED,
                            ecolor=GRID if v == "No clear change" else VERDICT_COLORS[v], elinewidth=1.2, capsize=0)
            ax.set_title(app, loc="left", fontsize=11, color=INK2)
            ax.set_ylim(-lim, lim)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(labelsize=8.5)
        axes[i][0].set_ylabel(f"{dom}\nnew − older versions\n(points of 1–2★ share)", fontsize=9)
    save(fig, CHARTS_DIR / "02_release_effects.png",
         "How each release compares with the versions it replaced",
         "One dot per release: share of 1–2★ reviews on the new version minus the share on older versions, over the same 7 days "
         "from the release, with its 95% interval. Dashed line = the app's usual gap (median over its releases); grey band = within ±5 points of it. "
         "Flipkart and Amazon users on the newest version are always more negative, so only releases well above that line count. "
         "Red = made things worse, blue = better, grey dots = no clear change.")


def chart_examples(res, daily, reviews, n=4):
    worst = res[res["verdict"] == "Worse (confirmed at app level)"].sort_values("excess_gap_pp", ascending=False)
    if len(worst) < n:
        worst = pd.concat([worst, res[res["verdict"] == "Worse (new version only)"].sort_values("excess_gap_pp", ascending=False)])
    worst = worst.drop_duplicates("app_name").head(n)
    if worst.empty:
        return
    fig, axes = plt.subplots(1, len(worst), figsize=(4.2 * len(worst), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, worst.iterrows()):
        t0 = pd.Timestamp(r["release_day"])
        d = daily[(daily["app_name"] == r["app_name"]) & (daily["day"] >= t0 - pd.Timedelta(days=21))
                  & (daily["day"] <= t0 + pd.Timedelta(days=13))].sort_values("day")
        color = APP_COLORS[r["app_name"]]
        ax.axvspan(t0 - pd.Timedelta(days=WINDOW) - pd.Timedelta(hours=12), t0 - pd.Timedelta(hours=12), color=GRID, alpha=0.6, lw=0)
        ax.axvspan(t0 - pd.Timedelta(hours=12), t0 + pd.Timedelta(days=WINDOW - 1, hours=12), color=color, alpha=0.12, lw=0)
        ax.plot(d["day"], d["neg_reviews_filled"], color=color, lw=1.8, marker="o", ms=3)
        ax.axvline(t0, color=INK2, lw=0.9, ls=":")
        ax.set_title(f"{r['app_name']} {r['app_version']}  ({t0:%d %b})", loc="left", fontsize=10.5, color=INK2)
        ax.text(0.02, 0.96, f"new version: {r['neg_share_new']:.0%} rated 1–2★\nolder versions: {r['neg_share_older']:.0%} (usual gap {r['app_usual_gap_pp']:+.0f} pts)\n"
                f"1–2★ per day: {r['neg_per_day_before']:.0f} → {r['neg_per_day_after']:.0f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID, alpha=0.95))
        ax.set_ylim(0, d["neg_reviews_filled"].max() * 1.45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=8, axis="x", rotation=30)
    axes[0].set_ylabel("1–2★ reviews per day")
    save(fig, CHARTS_DIR / "03_worst_release_examples.png",
         "Releases that made things worse: daily 1–2★ reviews around the release",
         "Grey band = the 7 days before the release; coloured band = the 7 days from the release (dotted line = release day). "
         "The worst release of each app among those flagged as worse.")


def chart_issue_mix(res, issues):
    worse = res[res["made_worse"]]
    if worse.empty or issues.empty:
        return
    m = issues.merge(worse[["app_name", "app_version"]], on=["app_name", "app_version"])
    m["label"] = m["app_name"] + " " + m["app_version"].astype(str)
    piv = m.pivot_table(index="label", columns="issue", values="diff_pp", aggfunc="first")
    sig = m.pivot_table(index="label", columns="issue", values="significant_increase", aggfunc="first").reindex_like(piv)
    order = worse.assign(label=worse["app_name"] + " " + worse["app_version"].astype(str)).sort_values(
        ["app_name", "release_day"], key=lambda s: s.map({a: i for i, a in enumerate(APP_NAMES)}) if s.name == "app_name" else s)["label"]
    piv = piv.reindex(order)[list(ISSUES.values())]
    sig = sig.reindex(order)[list(ISSUES.values())].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=(12, 0.34 * len(piv) + 2.2))
    lim = np.nanmax(np.abs(piv.to_numpy())) or 1
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if pd.notna(v) and abs(v) >= 1:
                ax.text(j, i, f"{v:+.0f}{'*' if sig.iat[i, j] else ''}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > lim * 0.6 else INK)
    ax.set_xticks(range(piv.shape[1]), [c.replace(" & ", " &\n").replace(" / ", " /\n") for c in piv.columns], fontsize=8.5)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    ax.xaxis.tick_top()
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("new − older (points)", fontsize=8.5)
    save(fig, CHARTS_DIR / "04_issue_mix_of_worse_releases.png",
         "What got worse: issue mix of the releases that made things worse",
         "Share of reviews mentioning each issue on the new version minus older versions, same 7 days (keyword tags; a review can have "
         f"several). * = significant increase (q < {ALPHA}, at least {MIN_ISSUE_DIFF_PP:.0f} points). Red = more on the new version.")


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    reviews, daily, releases, alerts = load()
    print(f"Loaded {len(reviews):,} reviews, {len(releases)} release events")
    res, issues = run(reviews, daily, releases, alerts, day_col="first_seen_10th_review")
    sens, _ = run(reviews, daily, releases, alerts, day_col="first_seen")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = dict(index=False, float_format="%.4f")
    res.to_csv(OUT_DIR / "release_impact.csv", **fmt)
    issues.to_csv(OUT_DIR / "release_issue_changes.csv", **fmt)
    worse_cols = ["app_name", "domain", "app_version", "release_day", "verdict", "n_new_version", "neg_share_new",
                  "neg_share_older", "neg_share_diff_pp", "neg_share_diff_ci_low_pp", "neg_share_diff_ci_high_pp",
                  "app_usual_gap_pp", "excess_gap_pp", "q_excess_gap", "rating_new", "rating_older", "neg_per_day_before", "neg_per_day_after",
                  "neg_per_day_change_pct", "q_neg_per_day", "alerts_above_in_week_after", "most_elevated_issue",
                  "most_elevated_issue_pp"]
    worse = res[res["made_worse"]].sort_values(["app_name", "excess_gap_pp"], ascending=[True, False],
                                               key=lambda s: s.map({a: i for i, a in enumerate(APP_NAMES)}) if s.name == "app_name" else s)
    worse[worse_cols].to_csv(OUT_DIR / "worse_releases.csv", **fmt)

    by_app = (res.groupby(["app_name", "domain"], sort=False)
              .agg(releases=("app_version", "size"), testable=("testable", "sum"), made_worse=("made_worse", "sum"),
                   confirmed_at_app_level=("verdict", lambda s: (s == "Worse (confirmed at app level)").sum()),
                   better=("verdict", lambda s: (s == "Better").sum()),
                   usual_gap_pp=("neg_share_diff_pp", "median"),
                   worst_diff_pp=("neg_share_diff_pp", "max"))
              .reset_index())
    by_app["share_worse_pct"] = 100 * by_app["made_worse"] / by_app["testable"].where(by_app["testable"] > 0)
    worst_ver = worse.groupby("app_name", sort=False).head(1).set_index("app_name")
    by_app["worst_release"] = by_app["app_name"].map(worst_ver["app_version"]).fillna("")
    by_app["worst_release_day"] = by_app["app_name"].map(worst_ver["release_day"])
    by_app.to_csv(OUT_DIR / "impact_by_app.csv", **fmt)

    print("Charts")
    chart_verdicts(res)
    chart_effects(res)
    chart_examples(res, daily, reviews)
    chart_issue_mix(res, issues)

    t = res[res["testable"]]
    summary = {
        "method": {"window_days": WINDOW, "release_day": "first_seen_10th_review", "min_reviews": MIN_REVIEWS,
                   "min_share_diff_pp": MIN_SHARE_DIFF_PP, "min_issue_diff_pp": MIN_ISSUE_DIFF_PP,
                   "false_discovery_rate": ALPHA,
                   "main_test": "new version vs older versions over the same 7 days: gap in the share of 1-2 star reviews, "
                                "compared with the app's usual gap (leave-one-out median); z-test on the excess, Benjamini-Hochberg across releases",
                   "app_level_test": "Welch t-test on 1-2 star reviews per day, 7 days before vs 7 days from the release"},
        "releases": int(len(res)), "testable": int(len(t)),
        "not_testable_reasons": {"window_outside_data": int((~res["windows_complete"]).sum()),
                                 "too_few_reviews": int((res["windows_complete"] & ~res["testable"]).sum())},
        "verdicts": {v: int((res["verdict"] == v).sum()) for v in VERDICT_ORDER},
        "share_worse_pct_of_testable": round(100 * res["made_worse"].sum() / max(len(t), 1), 1),
        "median_neg_share_diff_pp": round(float(t["neg_share_diff_pp"].median()), 2),
        "usual_gap_by_app_pp": {a: round(float(v), 2) for a, v in t.groupby("app_name")["neg_share_diff_pp"].median().items()},
        "raw_rule_without_app_baseline_would_flag": int(((t["q_neg_share_new_vs_older"] < ALPHA) & (t["neg_share_diff_pp"] >= MIN_SHARE_DIFF_PP)).sum()),
        "significant_before_after_count_rises": int(((res["q_neg_per_day"] < ALPHA) & (res["neg_per_day_change_pct"] > 0)).sum()),
        "significant_before_after_count_falls": int(((res["q_neg_per_day"] < ALPHA) & (res["neg_per_day_change_pct"] < 0)).sum()),
        "worse_releases_with_forecast_alert_in_week_after": int((worse["alerts_above_in_week_after"] > 0).sum()),
        "most_elevated_issue_in_worse_releases": worse["most_elevated_issue"].replace("", np.nan).value_counts().to_dict(),
        "by_app": {r.app_name: {"testable": int(r.testable), "made_worse": int(r.made_worse),
                                "confirmed_at_app_level": int(r.confirmed_at_app_level), "better": int(r.better),
                                "worst_release": r.worst_release} for r in by_app.itertuples()},
        "worst_releases": [{"app": r.app_name, "version": r.app_version, "release_day": str(r.release_day),
                            "neg_share_new": round(r.neg_share_new, 3), "neg_share_older": round(r.neg_share_older, 3),
                            "diff_pp": round(r.neg_share_diff_pp, 1), "excess_over_usual_pp": round(r.excess_gap_pp, 1), "verdict": r.verdict,
                            "issue": r.most_elevated_issue}
                           for r in res[res["made_worse"]].sort_values("excess_gap_pp", ascending=False).head(10).itertuples()],
        "sensitivity_release_day_first_seen": {"testable": int(sens["testable"].sum()),
                                               "made_worse": int(sens["made_worse"].sum()),
                                               "same_releases_flagged": int(len(set(zip(res.loc[res.made_worse, "app_name"], res.loc[res.made_worse, "app_version"]))
                                                                             & set(zip(sens.loc[sens.made_worse, "app_name"], sens.loc[sens.made_worse, "app_version"]))))},
        "charts": CHART_INDEX,
    }
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  saved {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print(json.dumps({k: summary[k] for k in ("releases", "testable", "verdicts", "median_neg_share_diff_pp")}, indent=2))


if __name__ == "__main__":
    main()
