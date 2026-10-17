"""Slide-sized charts for the Review 2 presentation (docs/Review_2_Presentation.pptx).

Every chart is drawn at the size it has on the slide (half slide: 5.8 x 3.45 in, full width: 11.9 x 3.45 in), so its
labels stay readable when projected. All numbers come from the outputs of the earlier scripts; nothing is recomputed
except simple aggregations of those outputs.

Input:  data/eda_summary.json, data/tagged/, data/timeseries/, data/textmining/, data/update_impact/
Output: data/charts/slides/
          eda_share_by_app.png, eda_service_vs_app.png          slide 7  (Unit 1)
          eda_feed_gap.png, eda_review_length.png               slide 8  (Unit 1)
          tm_topics.png                                         slide 12 (Method 1 results)
          ts_forecast_swiggy.png, ts_paytm_release.png          slides 16 and 18 (Method 2)
        The other charts on those slides are the existing ones (ts_03, tm_08, update impact 01, ci_01).
Run:    python scripts/26_slide_charts.py   (about 1 minute; reads the 11 tagged files once)
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_DOMAIN, DOMAIN_COLORS, DOMAINS  # noqa: E402
from data_io import read_stage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "charts" / "slides"
HALF, WIDE = (5.8, 3.45), (11.9, 3.45)
INK2, GRID = "#52514e", "#c9c8c3"
ORANGE, BLUE, GREEN, YELLOW, GREY = "#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#b9b8b3"
GAP_APPS = ["Swiggy", "Blinkit", "Domino's", "Flipkart", "Amazon"]
GENERAL_TOPICS = {"General Bad Service & Quality", "General Bad Experience", "Worst-Ever Experience"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GRID, "text.color": "#1f2933"})


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print("saved", (OUT / name).relative_to(ROOT))


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold")


# ----------------------------------------------------------------------------------------------------------- Unit 1
def eda_charts(tagged):
    eda = json.loads((DATA / "eda_summary.json").read_text())
    cov = pd.DataFrame(eda["coverage_by_app"]).astype({"reviews": int, "pct_low_star": float})

    fig, ax = plt.subplots(figsize=HALF)
    rows, ys, y = [], [], 0
    for dom in reversed(DOMAINS):
        for _, r in cov[cov.domain == dom].sort_values("pct_low_star").iterrows():
            rows.append(r); ys.append(y); y += 1
        y += 0.6
    for r, yy in zip(rows, ys):
        ax.barh(yy, r.pct_low_star, color=DOMAIN_COLORS[r.domain], height=0.75)
        ax.text(r.pct_low_star + 0.8, yy, f"{r.pct_low_star:.0f}%", va="center", fontsize=9)
    ax.set_yticks(ys, [r.app_name for r in rows], fontsize=9)
    ax.axvline(19.5, color=INK2, ls="--", lw=1)
    ax.text(20, ys[-1] + 0.9, "all apps 19.5%", fontsize=8, color=INK2)
    for dom in DOMAINS:
        yy = [y_ for r, y_ in zip(rows, ys) if r.domain == dom]
        ax.text(62, np.mean(yy), dom, color=DOMAIN_COLORS[dom], fontsize=9.5, fontweight="bold", ha="right", va="center")
    ax.set_xlim(0, 62); ax.set_ylim(-0.7, ys[-1] + 1.4); ax.set_xlabel("% of the app's reviews rated 1–2★", fontsize=9)
    title(ax, "Share of 1–2★ reviews by app")
    save(fig, "eda_share_by_app.png")

    ca = eda["complaint_attribution"]
    cats = ["Service issue only", "Service and app issue", "App issue only", "Payment & Refund only"]
    low = cov.set_index("app_name").reviews * cov.set_index("app_name").pct_low_star / 100
    cnt = pd.DataFrame(ca["by_app_pct"]).T[cats].mul(low, axis=0) / 100
    groups = {"All apps": cnt.sum()} | {d: cnt.loc[[a for a in cnt.index if APP_DOMAIN[a] == d]].sum() for d in DOMAINS}
    share = pd.DataFrame(groups).T
    share = share.div(share.sum(axis=1), axis=0) * 100
    colors = dict(zip(cats, [ORANGE, GREEN, BLUE, YELLOW]))
    labels = dict(zip(cats, ["Service only", "Service + app", "App only", "Payment & refund"]))
    fig, ax = plt.subplots(figsize=HALF)
    order = list(share.index)[::-1]
    for i, g in enumerate(order):
        left = 0
        for c in cats:
            v = share.loc[g, c]
            ax.barh(i, v, left=left, color=colors[c], height=0.6, label=labels[c] if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
            left += v
    ax.set_yticks(range(len(order)), order, fontsize=9.5); ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 100); ax.set_xlabel("% of 1–2★ reviews that name an issue (keyword tags)", fontsize=9)
    ax.text(103, len(order) - 0.45, "service · app", fontsize=8, color=INK2, clip_on=False)
    for i, g in enumerate(order):
        svc = share.loc[g, ["Service issue only", "Service and app issue"]].sum()
        app = share.loc[g, ["Service and app issue", "App issue only"]].sum()
        ax.text(103, i, f"{svc:.0f}% · {app:.0f}%", va="center", fontsize=9, clip_on=False,
                fontweight="bold" if g == "All apps" else "normal")
    title(ax, "What 1–2★ reviews name")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=1.2, columnspacing=1)
    save(fig, "eda_service_vs_app.png", rect=(0, 0, 0.9, 1))

    d = pd.read_csv(DATA / "timeseries" / "daily_app.csv", parse_dates=["day"])
    g = d[d.app_name.isin(GAP_APPS)].groupby("day")[["pos_reviews", "neg_reviews"]].sum().loc["2026-04-01":"2026-06-15"]
    inside, around = g.loc["2026-04-21":"2026-05-05"].mean(), pd.concat([g.loc["2026-04-06":"2026-04-20"], g.loc["2026-05-06":"2026-05-20"]]).mean()
    drop = (inside / around - 1) * 100
    fig, ax = plt.subplots(figsize=HALF)
    ax.axvspan(pd.Timestamp("2026-04-21"), pd.Timestamp("2026-05-06"), color="#000", alpha=0.07, lw=0)
    ax.plot(g.index, g.pos_reviews, color=BLUE, lw=1.8, label="4–5★ reviews")
    ax.plot(g.index, g.neg_reviews, color=ORANGE, lw=1.8, label="1–2★ reviews")
    mid = pd.Timestamp("2026-04-28")
    ax.text(mid, g.pos_reviews.max() * 0.97, "feed gap\n21 Apr – 5 May", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.text(mid, 1900, f"4–5★ {drop.pos_reviews:+.0f}%".replace("-", "−"), ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.text(mid, 280, f"1–2★ {drop.neg_reviews:+.0f}%".replace("-", "−"), ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    ax.set_ylim(0, g.pos_reviews.max() * 1.05); ax.set_ylabel("reviews per day", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    title(ax, "Feed gap: positive reviews vanish, complaints do not")
    ax.text(0, -0.2, "Swiggy, Blinkit, Domino's, Flipkart, Amazon combined; change vs the 15 days either side",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    save(fig, "eda_feed_gap.png")

    bins, lab = [0, 1, 3, 10, 30, 10**6], ["1", "2–3", "4–10", "11–30", "31+"]
    rows = {}
    for name, sub in (("neg", tagged[tagged.score <= 2]), ("pos", tagged[tagged.score >= 4])):
        c = pd.cut(sub.review_length, bins, labels=lab)
        rows[name] = sub.groupby(c, observed=True).agg(n=("has_issue", "size"), tagged=("has_issue", "mean"))
        rows[name]["share"] = rows[name].n / rows[name].n.sum() * 100
    med = {k: int(tagged[sel].review_length.median()) for k, sel in (("neg", tagged.score <= 2), ("pos", tagged.score >= 4))}
    fig, ax = plt.subplots(figsize=HALF)
    x, w = np.arange(len(lab)), 0.38
    ax.bar(x - w / 2, rows["neg"].share, w, color=ORANGE, label=f"1–2★ reviews (median {med['neg']} words)")
    ax.bar(x + w / 2, rows["pos"].share, w, color=BLUE, label=f"4–5★ reviews (median {med['pos']} words)")
    for i, (s, tg) in enumerate(zip(rows["neg"].share, rows["neg"].tagged)):
        ax.text(i - w / 2, s + 1, f"{tg * 100:.0f}%", ha="center", fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.text(4.45, 39.5, "orange labels = share of 1–2★ reviews\nin which a keyword tag finds an issue", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xticks(x, lab, fontsize=9.5); ax.set_xlabel("review length (words)", fontsize=9)
    ax.set_ylabel("% of reviews", fontsize=9); ax.set_ylim(0, 50)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    title(ax, "Praise is short; keyword tags miss most complaints")
    save(fig, "eda_review_length.png")


# ------------------------------------------------------------------------------------------------- Method 1 (text)
def text_mining_charts():
    ts = pd.read_csv(DATA / "textmining" / "topic_sentiment.csv")
    ts["share"] = 100 * ts.reviews / ts.groupby("domain").reviews.transform("sum")
    fig, axes = plt.subplots(1, 3, figsize=WIDE)
    for ax, dom in zip(axes, DOMAINS):
        s = ts[ts.domain == dom].sort_values("share")
        colors = [GREY if lbl in GENERAL_TOPICS else DOMAIN_COLORS[dom] for lbl in s.dominant_topic_label]
        ax.barh(range(len(s)), s.share, color=colors, height=0.7)
        for i, (sh, se) in enumerate(zip(s.share, s.mean_sentiment)):
            ax.text(sh + 1, i, f"{sh:.0f}%  ({se:+.2f})".replace("-", "−"), va="center", fontsize=8)
        ax.set_yticks(range(len(s)), [textwrap.fill(lbl, 20) for lbl in s.dominant_topic_label], fontsize=8.3)
        ax.set_xlim(0, 75); ax.set_xticks([0, 25, 50]); ax.tick_params(axis="x", labelsize=8)
        ax.set_title(dom, loc="left", fontsize=10, fontweight="bold", color=DOMAIN_COLORS[dom])
    axes[1].set_xlabel("% of the domain's complaints with a topic   (mean VADER sentiment in brackets; grey = general dissatisfaction, no specific fault)", fontsize=8.5)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.2, wspace=0.95)
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "tm_topics.png", dpi=300); plt.close(fig)
    print("saved", (OUT / "tm_topics.png").relative_to(ROOT))

# ------------------------------------------------------------------------------------------- Method 2 (time series)
def time_series_charts():
    daily = pd.read_csv(DATA / "timeseries" / "daily_app.csv", parse_dates=["day"])
    fc = pd.read_csv(DATA / "timeseries" / "forecast_daily.csv", parse_dates=["day"])
    hist = daily[daily.app_name == "Swiggy"].set_index("day")["neg_reviews"].iloc[-56:]
    f = fc[(fc.app_name == "Swiggy") & (fc.model == "ARIMA")].set_index("day")
    fig, ax = plt.subplots(figsize=HALF)
    ax.fill_between(f.index, f.lower_95, f.upper_95, color=BLUE, alpha=0.15, lw=0, label="95% interval")
    ax.fill_between(f.index, f.lower_80, f.upper_80, color=BLUE, alpha=0.3, lw=0, label="80% interval")
    ax.plot(hist.index, hist.values, color="#1f2933", lw=1.4, label="actual")
    ax.plot(f.index, f.forecast, color=BLUE, lw=2, label="ARIMA forecast")
    ax.axvline(hist.index[-1], color=INK2, ls=":", lw=1)
    sundays = f.index[f.index.dayofweek == 6]
    ax.plot(sundays, f.loc[sundays, "forecast"], "o", color=BLUE, ms=4)
    ax.text(hist.index[-1], ax.get_ylim()[1] * 0.97, "  forecast from 21 Sep", fontsize=8, color=INK2, va="top")
    ax.set_ylabel("1–2★ reviews per day", fontsize=9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=8); ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, max(hist.max(), f.upper_95.max()) * 1.15)
    title(ax, "Swiggy: 4-week forecast keeps the Sunday peak (dots)")
    save(fig, "ts_forecast_swiggy.png")

    paytm = daily[daily.app_name == "Paytm"].set_index("day")["neg_reviews"].loc["2026-08-01":]
    alerts = pd.read_csv(DATA / "timeseries" / "unusual_days_unique.csv", parse_dates=["day"])
    alerts = alerts[(alerts.app_name == "Paytm") & alerts.alert_direction.str.startswith("Above")]
    rel = pd.read_csv(DATA / "update_impact" / "release_impact.csv", parse_dates=["release_day"])
    rel = rel[(rel.app_name == "Paytm") & (rel.app_version == "10.84.5")].iloc[0]
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(paytm.index, paytm.values, color="#1f2933", lw=1.6, label="1–2★ reviews per day")
    ax.plot(alerts.day, alerts.actual, "^", color="#c93a39", ms=7, label=f"unusual day (above 95% interval): {len(alerts)}")
    ax.axvline(rel.release_day, color="#c93a39", lw=1.5)
    ax.text(rel.release_day, paytm.max() * 1.12, "  release 10.84.5 (3 Sep)", color="#c93a39", fontsize=8.5, va="top")
    ax.text(pd.Timestamp("2026-08-02"), paytm.max() * 0.98,
            f"new version: {rel.neg_share_new * 100:.0f}% 1–2★ vs {rel.neg_share_older * 100:.0f}% on older versions\n"
            f"1–2★ per day: {rel.neg_per_day_before:.0f} → {rel.neg_per_day_after:.0f} (week before vs after)",
            fontsize=8.5, va="top", color="#1f2933")
    ax.set_ylim(0, paytm.max() * 1.15); ax.set_ylabel("1–2★ reviews per day", fontsize=9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=8); ax.legend(fontsize=8, frameon=False, loc="lower left")
    title(ax, "Paytm: one bad release, caught two ways")
    save(fig, "ts_paytm_release.png")


def main():
    tagged = read_stage("tagged", columns=["score", "review_length", "has_issue"])
    eda_charts(tagged)
    text_mining_charts()
    time_series_charts()


if __name__ == "__main__":
    main()
