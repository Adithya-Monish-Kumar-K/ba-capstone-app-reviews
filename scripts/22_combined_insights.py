"""
Stage 5 of Review 2: combined business insights and recommendations (#141).

Joins the outputs of every stage into one per-app priority table and writes recommendations per domain and
per app, each backed by a number from that table. Nothing is re-estimated here: the script only reads the
saved results of the earlier stages.

Sources 
  Volume and trend       data/timeseries/daily_app.csv, patterns_by_app.csv          (Stage 1, #131)
  Complaint topics       data/textmining/topic_share_by_app.csv, topic_sentiment_by_app.csv  (Stage 2, #134/#136)
  Forecast               data/timeseries/forecast_weekly.csv, evaluation_metrics_by_app.csv  (Stages 3-4, #135/#137)
  Forecast alerts        data/timeseries/unusual_days_unique.csv                     (Stage 4, #137)
  Releases               data/update_impact/release_impact.csv, impact_by_app.csv    (Stage 5, #138)
  Problem-review model   data/model_predictions.csv.gz, model_results_by_group.csv   (Review 1)

Priority score: one point for each warning sign below, 0 to 6 (rules in PRIORITY_RULES):
  many 1-2 star reviews a day, a high share of 1-2 star reviews, a rising trend, a forecast above the recent
  level, repeated days above the forecast interval, and a release that made things worse.
  3 or more points = High priority, 2 = Medium, 0-1 = Low.

Output: data/combined/priority_table.csv        one row per app with every signal and the priority
        data/combined/recommendations.csv        one row per recommendation: level, domain, app, action, evidence
        data/charts/combined/ci_01_priority_overview.png
        data/combined_insights_summary.json

Run from the repository root:  python scripts/22_combined_insights.py   
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_DOMAIN, APP_NAMES, DOMAINS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
TS, TM, UI = DATA / "timeseries", DATA / "textmining", DATA / "update_impact"
OUT_DIR = DATA / "combined"
CHARTS_DIR = DATA / "charts" / "combined"
SUMMARY_PATH = DATA / "combined_insights_summary.json"

ISSUES = {
    "issue_crash_bugs_stability": "Crash & Stability", "issue_payment_refund": "Payment & Refund",
    "issue_delivery_delay": "Delivery Delay", "issue_order_quality_fulfillment": "Order Quality",
    "issue_cancellation_return": "Cancellation & Return", "issue_customer_support": "Customer Support",
    "issue_account_login_otp": "Account / Login / OTP", "issue_pricing_charges_fraud": "Pricing & Fraud",
    "issue_ui_ux_update": "UI/UX & Update",
}
RISING_PCT = 10            # last 4 weeks at least 10% above the first 4 weeks
FORECAST_UP_PCT = 5        # week-1 forecast at least 5% above the last 4 weeks
UNUSUAL_DAYS_MIN = 5       # at least 5 distinct days above the 95% forecast interval in the backtest
UNRELIABLE_CHANGE_PCT = 30 # week-1 forecast this far from the last 4 weeks: read with care (Stage 3 rule)
MIN_TOPIC_SHARE = 10       # a topic needs at least 10% of the app's complaints to be called its most severe
PRIORITY_RULES = {
    "sig_high_volume": "1–2★ reviews per day above the median of the 11 apps",
    "sig_high_share": "share of 1–2★ reviews (feed-gap days excluded) above the all-app share",
    "sig_rising": f"last 4 weeks at least {RISING_PCT}% above the first 4 weeks (STL patterns, Stage 1)",
    "sig_forecast_up": f"week-1 forecast at least {FORECAST_UP_PCT}% above the last 4 weeks (Stage 3)",
    "sig_unusual_days": f"at least {UNUSUAL_DAYS_MIN} days above the 95% forecast interval (Stage 4)",
    "sig_worse_release": "at least one release made things worse (update impact, #138)",
}

SIGN_LABELS = {
    "sig_high_volume": "many 1–2★ reviews a day", "sig_high_share": "high share of 1–2★ reviews",
    "sig_rising": "rising trend", "sig_forecast_up": "forecast above recent level",
    "sig_unusual_days": "repeated days above the forecast interval", "sig_worse_release": "a release made things worse",
}

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
})
CHART_INDEX = []


def save(fig, path, title, sub=None):
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


# ----------------------------------------------------------------------------
# 1. One row per app from every stage
# ----------------------------------------------------------------------------
def volume_and_trend():
    daily = pd.read_csv(TS / "daily_app.csv", parse_dates=["day"])
    clean = daily[~daily["feed_gap"]]
    g = daily.groupby("app_name").agg(reviews=("reviews", "sum"), neg_reviews=("neg_reviews", "sum"),
                                      days=("day", "nunique"), **{c: (c, "sum") for c in ISSUES})
    gc = clean.groupby("app_name").agg(reviews_ng=("reviews", "sum"), neg_ng=("neg_reviews", "sum"))
    out = pd.DataFrame(index=APP_NAMES)
    out["domain"] = [APP_DOMAIN[a] for a in APP_NAMES]
    out["reviews"] = g["reviews"]
    out["neg_reviews"] = g["neg_reviews"]
    out["neg_per_day"] = g["neg_reviews"] / g["days"]
    out["neg_share_pct"] = 100 * gc["neg_ng"] / gc["reviews_ng"]
    rates = g[list(ISSUES)].div(g["reviews"], axis=0) * 100
    out["top_keyword_issue"] = rates.idxmax(axis=1).map(ISSUES)
    out["top_keyword_issue_pct"] = rates.max(axis=1)
    pat = pd.read_csv(TS / "patterns_by_app.csv").set_index("app_name")
    out["trend_change_pct"] = pat["trend_change_pct"]
    out["peak_weekday"] = pat["peak_weekday"]
    all_share = 100 * gc["neg_ng"].sum() / gc["reviews_ng"].sum()
    return out, all_share


def topics():
    share = pd.read_csv(TM / "topic_share_by_app.csv")
    sent = pd.read_csv(TM / "topic_sentiment_by_app.csv").rename(columns={"dominant_topic_label": "topic_label"})
    m = share.merge(sent[["app_name", "topic_label", "mean_sentiment", "pct_strongly_neg"]], on=["app_name", "topic_label"], how="left")
    rows = {}
    for app, g in m.groupby("app_name"):
        top = g.sort_values("share_pct", ascending=False).iloc[0]
        sev = g[g["share_pct"] >= MIN_TOPIC_SHARE].sort_values("mean_sentiment").iloc[0]
        rows[app] = {"top_topic": top["topic_label"], "top_topic_share_pct": top["share_pct"],
                     "top_topic_sentiment": top["mean_sentiment"], "top_topic_strongly_neg_pct": top["pct_strongly_neg"],
                     "most_severe_topic": sev["topic_label"], "most_severe_topic_share_pct": sev["share_pct"],
                     "most_severe_topic_sentiment": sev["mean_sentiment"],
                     "most_severe_topic_strongly_neg_pct": sev["pct_strongly_neg"],
                     "topic_reviews": int(g["reviews"].sum())}
    return pd.DataFrame.from_dict(rows, orient="index")


def forecast():
    fw = pd.read_csv(TS / "forecast_weekly.csv")
    w1 = fw[fw["week"] == 1].set_index("app_name")
    ev = pd.read_csv(TS / "evaluation_metrics_by_app.csv")
    ar = ev[ev["model"] == "ARIMA"].set_index("app_name")
    sn = ev[ev["model"] == "Seasonal naive"].set_index("app_name")
    al = pd.read_csv(TS / "unusual_days_unique.csv")
    out = pd.DataFrame(index=APP_NAMES)
    out["forecast_week1"] = w1["forecast"]
    out["forecast_week1_low80"] = w1["lower_80"]
    out["forecast_week1_high80"] = w1["upper_80"]
    out["last_4_weeks_avg"] = w1["last_4_weeks_avg"]
    out["forecast_change_pct"] = w1["change_vs_last_4_weeks_pct"]
    out["arima_mase"] = ar["MASE"]
    out["arima_mae"] = ar["MAE"]
    out["naive_mae"] = sn["MAE"]
    out["arima_beats_naive"] = ar["MAE"] < sn["MAE"]
    out["forecast_reliable"] = out["arima_beats_naive"] & (out["forecast_change_pct"].abs() <= UNRELIABLE_CHANGE_PCT)
    out["unusual_days_above"] = al[al["alert_direction"].str.startswith("Above")].groupby("app_name").size()
    out["unusual_days_below"] = al[al["alert_direction"].str.startswith("Below")].groupby("app_name").size()
    out[["unusual_days_above", "unusual_days_below"]] = out[["unusual_days_above", "unusual_days_below"]].fillna(0).astype(int)
    top_day = (al[al["alert_direction"].str.startswith("Above")].sort_values("absolute_error", ascending=False)
               .groupby("app_name").head(1).set_index("app_name"))
    out["largest_unusual_day"] = top_day["day"]
    out["largest_unusual_day_actual"] = top_day["actual"]
    out["largest_unusual_day_forecast"] = top_day["forecast"]
    return out


def releases():
    by = pd.read_csv(UI / "impact_by_app.csv").set_index("app_name")
    res = pd.read_csv(UI / "release_impact.csv")
    worse = res[res["made_worse"]].sort_values("excess_gap_pp", ascending=False).groupby("app_name").head(1).set_index("app_name")
    out = pd.DataFrame(index=APP_NAMES)
    out["releases_tested"] = by["testable"]
    out["releases_worse"] = by["made_worse"]
    out["releases_better"] = by["better"]
    out["worst_release"] = worse["app_version"]
    out["worst_release_day"] = worse["release_day"]
    out["worst_release_neg_share_new_pct"] = 100 * worse["neg_share_new"]
    out["worst_release_neg_share_older_pct"] = 100 * worse["neg_share_older"]
    out["worst_release_excess_pp"] = worse["excess_gap_pp"]
    out["worst_release_confirmed"] = worse["verdict"].eq("Worse (confirmed at app level)")
    out["worst_release_issue"] = worse["most_elevated_issue"]
    out["worst_release_in_feed_gap"] = worse["feed_gap_in_window"]
    return out


def classifier():
    p = pd.read_csv(DATA / "model_predictions.csv.gz")
    g = p.groupby("app_name").agg(test_reviews=("actual", "size"), flag_rate_pct=("random_forest_prediction", "mean"))
    g["flag_rate_pct"] *= 100
    grp = pd.read_csv(DATA / "model_results_by_group.csv")
    rf = grp[(grp["level"] == "app_name") & (grp["model"] == "random_forest")].set_index("group")
    g["classifier_precision"] = rf["precision"]
    g["classifier_recall"] = rf["recall"]
    g["classifier_pr_auc"] = rf["pr_auc"]
    return g.reindex(APP_NAMES)


def build_table():
    vol, all_share = volume_and_trend()
    t = vol.join(topics()).join(forecast()).join(releases()).join(classifier())
    t.index.name = "app_name"
    med = t["neg_per_day"].median()
    t["sig_high_volume"] = t["neg_per_day"] > med
    t["sig_high_share"] = t["neg_share_pct"] > all_share
    t["sig_rising"] = t["trend_change_pct"] >= RISING_PCT
    t["sig_forecast_up"] = t["forecast_reliable"] & (t["forecast_change_pct"] >= FORECAST_UP_PCT)
    t["sig_unusual_days"] = t["unusual_days_above"] >= UNUSUAL_DAYS_MIN
    t["sig_worse_release"] = t["releases_worse"].fillna(0) > 0
    sigs = list(PRIORITY_RULES)
    t["priority_score"] = t[sigs].sum(axis=1).astype(int)
    t["priority"] = np.select([t["priority_score"] >= 3, t["priority_score"] == 2], ["High", "Medium"], "Low")
    t["warning_signs"] = t[sigs].apply(lambda r: "; ".join(SIGN_LABELS[s] for s in sigs if r[s]), axis=1)
    t = t.sort_values(["priority_score", "neg_per_day"], ascending=False)
    return t.reset_index(), all_share, med


# ----------------------------------------------------------------------------
# 2. Recommendations, each with the number behind it
# ----------------------------------------------------------------------------
def app_recommendations(r):
    recs = []

    def add(kind, action, evidence):
        recs.append({"level": "App", "domain": r["domain"], "app_name": r["app_name"], "priority": r["priority"],
                     "type": kind, "action": action, "evidence": evidence})

    add("Fix first", f"Fix '{r['top_topic']}' first: it is the largest complaint topic.",
        f"{r['top_topic_share_pct']:.0f}% of the app's topic-assigned 1–2★ reviews; mean VADER sentiment {r['top_topic_sentiment']:+.2f}")
    if r["most_severe_topic"] != r["top_topic"]:
        add("Most damaging", f"Treat '{r['most_severe_topic']}' as the most damaging complaint: smaller but angriest.",
            f"{r['most_severe_topic_share_pct']:.0f}% of complaints, sentiment {r['most_severe_topic_sentiment']:+.2f}, "
            f"{r['most_severe_topic_strongly_neg_pct']:.0f}% strongly negative")
    if r["sig_worse_release"]:
        conf = "confirmed: 1–2★ reviews per day also rose" if r["worst_release_confirmed"] else "seen on the new version only"
        if r["worst_release_in_feed_gap"] is True:
            conf += "; released during the late-April feed gap, so the shares are inflated, but both groups cover the same days"
        issue = f"; most elevated issue: {r['worst_release_issue']}" if isinstance(r["worst_release_issue"], str) and r["worst_release_issue"] else ""
        add("Release check", f"Review release {r['worst_release']} ({r['worst_release_day']}) and add a pre-release check for what it broke.",
            f"{r['worst_release_neg_share_new_pct']:.0f}% of reviews on the new version rated 1–2★ vs {r['worst_release_neg_share_older_pct']:.0f}% "
            f"on older versions, {r['worst_release_excess_pp']:+.0f} points above the app's usual gap ({conf}){issue}")
    if r["forecast_reliable"]:
        add("Plan capacity", "Staff the support queue for the forecast volume next week.",
            f"{r['forecast_week1']:,.0f} 1–2★ reviews expected 21–27 Sep (80% interval {r['forecast_week1_low80']:,.0f}–"
            f"{r['forecast_week1_high80']:,.0f}), {r['forecast_change_pct']:+.0f}% vs the last 4 weeks")
    else:
        why = (f"ARIMA did not beat the seasonal-naive baseline (backtest MAE {r['arima_mae']:.1f} vs "
               f"{r['naive_mae']:.1f} reviews a day)" if not r["arima_beats_naive"]
               else f"week-1 forecast is {r['forecast_change_pct']:+.0f}% from the last 4 weeks")
        add("Plan capacity", "Do not plan on the ARIMA forecast alone; use last week's level and watch daily counts.",
            f"Forecast flagged as unreliable: {why}")
    if r["sig_unusual_days"]:
        add("Monitor", "Set a daily alert when 1–2★ reviews leave the forecast interval, and investigate each alert.",
            f"{r['unusual_days_above']} days above the 95% interval in the backtest; largest {r['largest_unusual_day']} "
            f"({r['largest_unusual_day_actual']:.0f} vs {r['largest_unusual_day_forecast']:.0f} expected)")
    if r["sig_rising"]:
        add("Trend", "Find what changed: complaints are rising.", f"Last 4 weeks {r['trend_change_pct']:+.0f}% vs the first 4 weeks")
    add("Triage", "Route reviews flagged by the problem-review classifier to support first.",
        f"Flags {r['flag_rate_pct']:.0f}% of reviews; {r['classifier_precision'] * 100:.0f}% of flags are real 1–2★ reviews "
        f"and it finds {r['classifier_recall'] * 100:.0f}% of them (test set)")
    return recs


def domain_recommendations(t):
    sent = pd.read_csv(TM / "topic_sentiment.csv")
    recs = []
    for dom in DOMAINS:
        d = t[t["domain"] == dom]
        s = sent[sent["domain"] == dom]
        big = s.sort_values("reviews", ascending=False).iloc[0]
        worst = s.sort_values("mean_sentiment").iloc[0]
        share = 100 * big["reviews"] / s["reviews"].sum()
        recs.append({"level": "Domain", "domain": dom, "app_name": "", "priority": "", "type": "Fix first",
                     "action": f"Across {dom}, '{big['dominant_topic_label']}' is the most common complaint; fix it in every app.",
                     "evidence": f"{share:.0f}% of the domain's topic-assigned 1–2★ reviews ({int(big['reviews']):,}); sentiment {big['mean_sentiment']:+.2f}"})
        recs.append({"level": "Domain", "domain": dom, "app_name": "", "priority": "", "type": "Most damaging",
                     "action": f"'{worst['dominant_topic_label']}' produces the angriest reviews in {dom}.",
                     "evidence": f"Mean sentiment {worst['mean_sentiment']:+.2f}; {worst['pct_strongly_neg']:.0f}% strongly negative"})
        hi = d[d["priority"] == "High"]["app_name"].tolist()
        lead = d.sort_values("neg_per_day", ascending=False).iloc[0]
        recs.append({"level": "Domain", "domain": dom, "app_name": "", "priority": "", "type": "Focus",
                     "action": (f"Focus on {', '.join(hi)} (High priority)." if hi else "No app in this domain is High priority; monitor.")
                     + f" {lead['app_name']} has the most 1–2★ reviews a day.",
                     "evidence": f"{lead['app_name']}: {lead['neg_per_day']:.0f} a day, {lead['neg_share_pct']:.1f}% of its reviews"})
    return recs


# ----------------------------------------------------------------------------
# 3. Chart
# ----------------------------------------------------------------------------
def chart_priority(t):
    sigs = list(PRIORITY_RULES)
    labels = ["Many 1–2★\nper day", "High 1–2★\nshare", "Rising\ntrend", "Forecast\nabove recent", "Days above\nforecast", "Release made\nthings worse"]
    tt = t.set_index("app_name")
    fig, ax = plt.subplots(figsize=(12, 6.2))
    colors = {"High": "#c93a39", "Medium": "#eda100", "Low": "#8a8985"}
    for i, app in enumerate(tt.index):
        for j, s in enumerate(sigs):
            on = bool(tt.loc[app, s])
            ax.scatter(j, i, s=260, color=colors[tt.loc[app, "priority"]] if on else SURFACE,
                       edgecolor=colors[tt.loc[app, "priority"]] if on else GRID, linewidth=1.5, zorder=3)
        ax.text(len(sigs) - 0.3, i, f"{tt.loc[app, 'priority_score']}  {tt.loc[app, 'priority']}", va="center", fontsize=10,
                color=colors[tt.loc[app, "priority"]], fontweight="bold")
        ax.text(len(sigs) + 0.75, i, f"{tt.loc[app, 'top_topic']} ({tt.loc[app, 'top_topic_share_pct']:.0f}%)", va="center",
                fontsize=9, color=INK2)
    ax.set_xticks(range(len(sigs)), labels, fontsize=9)
    ax.set_yticks(range(len(tt)), [f"{a}  ·  {APP_DOMAIN[a]}" for a in tt.index], fontsize=9.5)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xlim(-0.6, len(sigs) + 4.6)
    ax.text(len(sigs) - 0.3, -0.9, "Score", fontsize=9, color=INK2)
    ax.text(len(sigs) + 0.75, -0.9, "Largest complaint topic", fontsize=9, color=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save(fig, CHARTS_DIR / "ci_01_priority_overview.png",
         "Which apps need attention first: six warning signs from all stages",
         "Filled dot = the sign applies. Score = number of signs (3+ High, 2 Medium, 0–1 Low). Signs: 1–2★ reviews per day above the median app; "
         "share of 1–2★ reviews above all apps; last 4 weeks 10%+ above the first 4; reliable week-1 forecast 5%+ above the last 4 weeks; "
         "5+ days above the 95% forecast interval; a release that made things worse.")


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    t, all_share, med = build_table()
    recs = domain_recommendations(t)
    for _, r in t.iterrows():
        recs += app_recommendations(r)
    recs = pd.DataFrame(recs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_DIR / "priority_table.csv", index=False, float_format="%.3f")
    recs.to_csv(OUT_DIR / "recommendations.csv", index=False)
    print(f"  saved data/combined/priority_table.csv ({len(t)} apps), recommendations.csv ({len(recs)} rows)")
    chart_priority(t)
    summary = {
        "priority_rules": PRIORITY_RULES,
        "thresholds": {"all_app_neg_share_pct_excl_feed_gap": round(all_share, 2), "median_neg_per_day": round(med, 1),
                       "rising_pct": RISING_PCT, "forecast_up_pct": FORECAST_UP_PCT, "unusual_days_min": UNUSUAL_DAYS_MIN,
                       "unreliable_forecast_change_pct": UNRELIABLE_CHANGE_PCT},
        "priority": {r.app_name: {"score": int(r.priority_score), "priority": r.priority, "signs": r.warning_signs}
                     for r in t.itertuples()},
        "high_priority_apps": t.loc[t["priority"] == "High", "app_name"].tolist(),
        "unreliable_forecasts": t.loc[~t["forecast_reliable"], "app_name"].tolist(),
        "recommendations": int(len(recs)),
        "charts": CHART_INDEX,
    }
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  saved {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print(t[["app_name", "priority_score", "priority", "warning_signs"]].to_string(index=False))


if __name__ == "__main__":
    main()
