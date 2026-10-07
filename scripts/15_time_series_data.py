"""
Stage 1 of Review 2, Method 2 (Time-Series Analysis): data preparation.

Builds the time series the forecasting models use, the release-event table for the update-impact
analysis, and the time-based patterns (trend, weekly pattern, stationarity, autocorrelation).

Input:  data/tagged/<app>.csv.gz
Output: data/timeseries/
          daily_app.csv        one row per app per day (1 Apr - 20 Sep 2026): review counts, 1-2 star counts,
                               rating, sentiment, issue counts, data-quality flags
          daily_domain.csv     the same, summed per domain
          weekly_app.csv       Monday-Sunday weeks per app (complete_week marks full weeks)
          weekly_domain.csv    the same, per domain
          release_events.csv   one row per app version: first date seen, reviews, rating
          stl_components.csv   STL decomposition of daily 1-2 star counts per app
          patterns_by_app.csv  trend and seasonal strength, weekday effect, trend change per app
          stationarity.csv     ADF tests on the level, first difference and weekly difference
          acf_pacf.csv         ACF and PACF values (lags 0-28) per app and series
        data/charts/timeseries/  overview charts, plus one ACF/PACF chart per app in acf_pacf/
        data/timeseries_summary.json

The modelled series is the daily count of 1-2 star reviews per app. Counts are used rather than
shares because the 21 Apr - 5 May feed gap removes mostly positive reviews, which distorts shares
far more than counts (EDA chart 13).
"""
import json
import re
import sys
from datetime import timedelta
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, pacf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, APPS, DOMAINS  # noqa: E402
from data_io import read_stage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "timeseries"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "timeseries"
SUMMARY_PATH = REPO_ROOT / "data" / "timeseries_summary.json"
SLUG = {a["name"]: slug for slug, a in APPS.items()}

WINDOW_START, WINDOW_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-20")
DAYS = pd.date_range(WINDOW_START, WINDOW_END, freq="D")

# Play Store feed gap: short positive reviews are largely missing for these apps (EDA chart 13)
GAP_START, GAP_END = pd.Timestamp("2026-04-21"), pd.Timestamp("2026-05-05")
GAP_APPS = ["Swiggy", "Blinkit", "Domino's", "Flipkart", "Amazon"]
# Zomato has no reviews from 23 Jul 22:00 to 25 Jul 12:30 (a gap in the source, not the scraper)
SOURCE_GAP = {"Zomato": (pd.Timestamp("2026-07-23 22:00"), pd.Timestamp("2026-07-25 12:30"))}

ISSUES = ["crash_bugs_stability", "payment_refund", "delivery_delay", "order_quality_fulfillment",
          "cancellation_return", "customer_support", "account_login_otp", "pricing_charges_fraud", "ui_ux_update"]
ISSUE_COLS = [f"issue_{i}" for i in ISSUES]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_LAG = 28
SEASONAL_DIFF_STRENGTH = 0.64   # Hyndman & Athanasopoulos: seasonal strength above ~0.64 suggests one seasonal difference

# ----------------------------------------------------------------------------
# Visual style (same palette and layout as the EDA charts)
# ----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
DIV = LinearSegmentedColormap.from_list("div_rb", ["#e34948", "#f0efec", "#2a78d6"])
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
    """Left-aligned title and muted subtitle, then write the PNG."""
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


def text_on(rgb):
    r, g, b = to_rgb(rgb)
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def shade_gap(ax):
    ax.axvspan(GAP_START.to_pydatetime(), (GAP_END + pd.Timedelta(days=1)).to_pydatetime(), color="#000", alpha=0.06, lw=0)


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def domain_apps(domain):
    return [a for a in APP_NAMES if APP_DOMAIN[a] == domain]


# ----------------------------------------------------------------------------
# 1. Daily and weekly series
# ----------------------------------------------------------------------------
def load():
    cols = ["app_name", "domain", "score", "app_version", "review_date", "sentiment_compound"] + ISSUE_COLS
    df = read_stage("tagged", columns=cols, parse_dates=["review_date"])
    df["day"] = df["review_date"].dt.normalize()
    df = df[(df["day"] >= WINDOW_START) & (df["day"] <= WINDOW_END)].copy()
    df["neg"] = (df["score"] <= 2).astype(int)
    df["pos"] = (df["score"] >= 4).astype(int)
    return df


def coverage_of_day(app, day):
    """Share of the day's 24 hours that the source covers (1 except inside a source gap)."""
    if app not in SOURCE_GAP:
        return 1.0
    a, b = (t.to_pydatetime() for t in SOURCE_GAP[app])
    start = day.to_pydatetime()
    end = start + timedelta(days=1)
    missing = max(timedelta(0), min(end, b) - max(start, a))
    return round(1 - missing / timedelta(days=1), 3)


def build_daily(df):
    agg = {"reviews": ("score", "size"), "neg_reviews": ("neg", "sum"), "pos_reviews": ("pos", "sum"),
           "sum_score": ("score", "sum"), "sum_sentiment": ("sentiment_compound", "sum")}
    agg.update({c: (c, "sum") for c in ISSUE_COLS})
    daily = df.groupby(["app_name", "day"]).agg(**agg)
    grid = pd.MultiIndex.from_product([APP_NAMES, DAYS], names=["app_name", "day"])
    daily = daily.reindex(grid, fill_value=0).reset_index()
    daily.insert(1, "domain", daily["app_name"].map(APP_DOMAIN))
    daily["mean_rating"] = (daily["sum_score"] / daily["reviews"]).round(4)
    daily["mean_sentiment"] = (daily["sum_sentiment"] / daily["reviews"]).round(4)
    daily["neg_share"] = (daily["neg_reviews"] / daily["reviews"]).round(4)
    daily["weekday"] = daily["day"].dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    in_gap = (daily["day"] >= GAP_START) & (daily["day"] <= GAP_END)
    daily["feed_gap"] = in_gap & daily["app_name"].isin(GAP_APPS)
    daily["source_coverage"] = [coverage_of_day(a, d) for a, d in zip(daily["app_name"], daily["day"])]
    daily["source_gap"] = daily["source_coverage"] < 1
    daily["neg_reviews_filled"] = fill_source_gaps(daily)
    return daily.drop(columns=["sum_score", "sum_sentiment"])


def fill_source_gaps(daily):
    """A day the source misses completely takes the mean of the same weekday one and two weeks either side.
    Partly covered days keep their real count (reviews can arrive in a burst after an outage).
    Used only for STL, ADF and ACF/PACF; the raw count is kept in neg_reviews."""
    out = daily["neg_reviews"].astype(float).copy()
    for app in SOURCE_GAP:
        a = daily[daily["app_name"] == app].set_index("day")
        for day, row in a[a["source_gap"]].iterrows():
            idx = daily.index[(daily["app_name"] == app) & (daily["day"] == day)][0]
            if row["source_coverage"] < 0.25:
                same = [day + pd.DateOffset(days=k) for k in (-14, -7, 7, 14)]
                out[idx] = a.loc[a.index.isin(same), "neg_reviews"].mean()
    return out.round(2)


def build_domain(daily):
    num = ["reviews", "neg_reviews", "pos_reviews", "neg_reviews_filled"] + ISSUE_COLS
    d = daily.assign(w_rating=daily["mean_rating"] * daily["reviews"], w_sent=daily["mean_sentiment"] * daily["reviews"])
    g = d.groupby(["domain", "day"]).agg(**{c: (c, "sum") for c in num + ["w_rating", "w_sent"]},
                                         feed_gap=("feed_gap", "any"), source_gap=("source_gap", "any")).reset_index()
    g["mean_rating"] = (g.pop("w_rating") / g["reviews"]).round(4)
    g["mean_sentiment"] = (g.pop("w_sent") / g["reviews"]).round(4)
    g["neg_share"] = (g["neg_reviews"] / g["reviews"]).round(4)
    g["domain"] = pd.Categorical(g["domain"], DOMAINS, ordered=True)
    return g.sort_values(["domain", "day"]).reset_index(drop=True)


def build_weekly(daily, key):
    d = daily.assign(week=daily["day"] - pd.to_timedelta(daily["day"].dt.dayofweek, unit="D"),
                     w_rating=daily["mean_rating"].fillna(0) * daily["reviews"])
    g = d.groupby([key, "week"], observed=True).agg(
        days_in_window=("day", "size"), reviews=("reviews", "sum"), neg_reviews=("neg_reviews", "sum"),
        pos_reviews=("pos_reviews", "sum"), w_rating=("w_rating", "sum"),
        feed_gap_days=("feed_gap", "sum"), source_gap_days=("source_gap", "sum")).reset_index()
    g["mean_rating"] = (g.pop("w_rating") / g["reviews"]).round(4)
    g["neg_share"] = (g["neg_reviews"] / g["reviews"]).round(4)
    g["complete_week"] = g["days_in_window"] == 7
    return g


# ----------------------------------------------------------------------------
# 2. Release events (first date each app version appears)
# ----------------------------------------------------------------------------
def build_releases(df):
    d = df.dropna(subset=["app_version"]).sort_values("review_date")
    rows = []
    for (app, ver), g in d.groupby(["app_name", "app_version"], sort=False):
        rows.append({
            "app_name": app, "domain": APP_DOMAIN[app], "app_version": ver,
            "first_seen": g["review_date"].iloc[0].normalize(),
            "first_seen_10th_review": g["review_date"].iloc[9].normalize() if len(g) >= 10 else pd.NaT,
            "last_seen": g["review_date"].iloc[-1].normalize(),
            "reviews": len(g), "days_present": g["day"].nunique(),
            "mean_rating": round(g["score"].mean(), 3), "neg_share": round(g["neg"].mean(), 4),
        })
    r = pd.DataFrame(rows)
    # A version first seen on the first day of the window was already released before it.
    r["first_seen_censored"] = r["first_seen"] <= WINDOW_START
    r["min_reviews_ok"] = r["reviews"] >= 30
    r["app_name"] = pd.Categorical(r["app_name"], APP_NAMES, ordered=True)
    r = r.sort_values(["app_name", "first_seen", "app_version"]).reset_index(drop=True)
    r["release_order"] = r.groupby("app_name", observed=True).cumcount() + 1
    # Reviewers on old builds make old versions show up late. A version is a new release only if its number is
    # higher than every version (with 30+ reviews) seen before it.
    r["is_new_release"] = False
    for app, g in r[r["min_reviews_ok"]].groupby("app_name", observed=True):
        highest = ()
        for i, row in g.iterrows():
            v = version_key(row["app_version"])
            if v > highest:
                r.loc[i, "is_new_release"] = not row["first_seen_censored"]
                highest = v
    r["use_for_update_impact"] = r["is_new_release"]
    return r


def version_key(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v)))


# ----------------------------------------------------------------------------
# 3. Time-based patterns: STL, weekday effect, stationarity, ACF/PACF
# ----------------------------------------------------------------------------
def strength(component, resid):
    """Hyndman & Athanasopoulos strength measure: 1 - Var(R) / Var(component + R), floored at 0."""
    return float(max(0.0, 1 - np.var(resid) / np.var(component + resid)))


def analyse_patterns(daily):
    stl_rows, pat_rows, adf_rows, corr_rows = [], [], [], []
    for app in APP_NAMES:
        a = daily[daily["app_name"] == app].set_index("day")
        y = a["neg_reviews_filled"].asfreq("D")
        res = STL(y, period=7, robust=True).fit()
        stl_rows.append(pd.DataFrame({"app_name": app, "day": y.index, "observed": y.values,
                                      "trend": res.trend.values.round(3), "seasonal": res.seasonal.values.round(3),
                                      "resid": res.resid.values.round(3)}))
        f_trend, f_season = strength(res.trend, res.resid), strength(res.seasonal, res.resid)

        # Weekday effect: each day against its centred 7-day mean (removes the trend), averaged by weekday.
        # Feed-gap and source-gap days are left out.
        ratio = (y / y.rolling(7, center=True).mean())
        keep = ~(a["feed_gap"] | a["source_gap"])
        wd = ratio[keep].groupby(ratio[keep].index.dayofweek).mean()
        effect = {WEEKDAYS[k]: round(float((v - 1) * 100), 2) for k, v in wd.items()}

        clean = ~(a["feed_gap"] | a["source_gap"])
        first4 = y[clean].iloc[:28].mean()
        last4 = y[clean].iloc[-28:].mean()
        pat_rows.append({"app_name": app, "domain": APP_DOMAIN[app],
                         "mean_neg_per_day": round(float(y.mean()), 2),
                         "trend_strength": round(f_trend, 3), "seasonal_strength": round(f_season, 3),
                         "first_4_weeks_mean": round(float(first4), 2), "last_4_weeks_mean": round(float(last4), 2),
                         "trend_change_pct": round(float((last4 / first4 - 1) * 100), 1),
                         "peak_weekday": max(effect, key=effect.get), "low_weekday": min(effect, key=effect.get),
                         **{f"weekday_{k}_pct": v for k, v in effect.items()}})

        series = {"level": y, "diff1": y.diff().dropna(), "diff7": y.diff(7).dropna(),
                  "diff1_diff7": y.diff(7).diff().dropna()}
        for name, s in series.items():
            stat, p, lags, nobs, crit, _ = adfuller(s.values, autolag="AIC", result_object=False)
            adf_rows.append({"app_name": app, "series": name, "adf_statistic": round(float(stat), 3),
                             "p_value": round(float(p), 4), "lags_used": int(lags), "n_obs": int(nobs),
                             "critical_5pct": round(float(crit["5%"]), 3), "stationary_at_5pct": bool(p < 0.05)})
            if name in ("level", "diff7"):
                ac, pc = acf(s.values, nlags=MAX_LAG, fft=True), pacf(s.values, nlags=MAX_LAG, method="ywm")
                band = 1.96 / np.sqrt(len(s))
                corr_rows.append(pd.DataFrame({"app_name": app, "series": name, "lag": range(MAX_LAG + 1),
                                               "acf": ac.round(4), "pacf": pc.round(4), "ci_95": round(band, 4)}))

    stl = pd.concat(stl_rows, ignore_index=True)
    patterns = pd.DataFrame(pat_rows)
    adf = pd.DataFrame(adf_rows)
    corr = pd.concat(corr_rows, ignore_index=True)

    # Suggested differencing for ARIMA: d from the ADF test on the level, D from the seasonal strength.
    level_p = adf[adf["series"] == "level"].set_index("app_name")["p_value"]
    patterns["adf_p_level"] = patterns["app_name"].map(level_p)
    patterns["suggested_d"] = (patterns["adf_p_level"] >= 0.05).astype(int)
    patterns["suggested_D"] = (patterns["seasonal_strength"] >= SEASONAL_DIFF_STRENGTH).astype(int)
    return stl, patterns, adf, corr


# ----------------------------------------------------------------------------
# 4. Charts
# ----------------------------------------------------------------------------
def chart_daily(daily):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True)
    for ax, dom in zip(axes, DOMAINS):
        shade_gap(ax)
        for app in domain_apps(dom):
            s = daily[daily["app_name"] == app].set_index("day")["neg_reviews"]
            ax.plot(s.index, s.values, color=APP_COLORS[app], lw=0.6, alpha=0.25)
            r = s.rolling(7, center=True).mean()
            ax.plot(r.index, r.values, color=APP_COLORS[app], lw=2, label=app)
        ax.set_title(dom, loc="left", fontsize=11, color=INK2)
        ax.set_ylim(bottom=0)
        date_axis(ax)
        ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    axes[0].set_ylabel("1–2★ reviews per day")
    save(fig, CHARTS_DIR / "ts_01_daily_negative_reviews.png",
         "Daily 1–2★ reviews per app: weekly ripples, short spikes and level shifts",
         "Thin line = daily count; thick line = 7-day centred mean. Grey band = 21 Apr – 5 May feed gap "
         "(positive reviews missing for Swiggy, Blinkit, Domino's, Flipkart and Amazon; negative counts are far less affected).")


def chart_trend(stl):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for ax, dom in zip(axes, DOMAINS):
        shade_gap(ax)
        for app in domain_apps(dom):
            s = stl[stl["app_name"] == app].set_index("day")["trend"]
            idx = s / s.mean() * 100
            ax.plot(idx.index, idx.values, color=APP_COLORS[app], lw=2, label=app)
        ax.axhline(100, color=MUTED, lw=1, ls="--")
        ax.set_title(dom, loc="left", fontsize=11, color=INK2)
        date_axis(ax)
        ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    axes[0].set_ylabel("STL trend (app's average = 100)")
    save(fig, CHARTS_DIR / "ts_02_stl_trend.png",
         "Trend of daily 1–2★ reviews (STL decomposition, weekly period)",
         "Each app's trend component indexed to its own average, so apps of different sizes compare directly. "
         "Robust STL with a 7-day period; the dashed line is the app's average level.")


def chart_weekday(patterns):
    vals = patterns.set_index("app_name")[[f"weekday_{d}_pct" for d in WEEKDAYS]].loc[APP_NAMES]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    lim = float(np.nanmax(np.abs(vals.values)))
    im = ax.imshow(vals.values, cmap=DIV.reversed(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(7), WEEKDAYS)
    ax.set_yticks(range(len(APP_NAMES)), APP_NAMES)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(vals.shape[0]):
        for j in range(7):
            v = vals.values[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=8.5, color=text_on(im.cmap(im.norm(v))[:3]))
    k = 0
    for dom in DOMAINS[:-1]:
        k += len(domain_apps(dom))
        ax.axhline(k - 0.5, color=INK, lw=1.2)
    save(fig, CHARTS_DIR / "ts_03_weekday_effect.png",
         "Weekday effect on 1–2★ reviews per app",
         "Each day's count against its centred 7-day mean, averaged by weekday (feed-gap and source-gap days left out). "
         "Red = more complaints than usual on that weekday, blue = fewer.")


def chart_releases(releases):
    r = releases[releases["use_for_update_impact"]]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    shade_gap(ax)
    for i, app in enumerate(APP_NAMES):
        s = r[r["app_name"] == app]
        ax.scatter(s["first_seen"], [i] * len(s), s=26, color=APP_COLORS[app], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.text((WINDOW_END + pd.Timedelta(days=2)).to_pydatetime(), i, f"{len(s)}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(APP_NAMES)), APP_NAMES)
    ax.invert_yaxis()
    ax.set_xlim((WINDOW_START - pd.Timedelta(days=2)).to_pydatetime(), (WINDOW_END + pd.Timedelta(days=8)).to_pydatetime())
    date_axis(ax)
    ax.grid(axis="y", visible=False)
    save(fig, CHARTS_DIR / "ts_04_release_events.png",
         "Release events: the first day each app version appears in the reviews",
         "One dot per new version: at least 30 reviews, first seen after 1 Apr, and numbered higher than every earlier version "
         "(old builds that some users still run are left out). "
         "The number at the right is the count of release events per app.")


def chart_acf_pacf(corr, adf, app):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    titles = {"level": "daily 1–2★ count", "diff7": "after a weekly difference (y − y₇)"}
    for row, series in enumerate(["level", "diff7"]):
        c = corr[(corr["app_name"] == app) & (corr["series"] == series)]
        band = c["ci_95"].iloc[0]
        for col, kind in enumerate(["acf", "pacf"]):
            ax = axes[row, col]
            ax.axhspan(-band, band, color=APP_COLORS[app], alpha=0.12, lw=0)
            ax.vlines(c["lag"][1:], 0, c[kind][1:], color=APP_COLORS[app], lw=2)
            ax.scatter(c["lag"][1:], c[kind][1:], s=12, color=APP_COLORS[app], zorder=3)
            ax.axhline(0, color=INK2, lw=0.8)
            for lag in (7, 14, 21, 28):
                ax.axvline(lag, color=GRID, lw=1, zorder=0)
            ax.set_ylim(-1, 1)
            ax.set_title(f"{kind.upper()}: {titles[series]}", loc="left", fontsize=10, color=INK2)
    for ax in axes[1]:
        ax.set_xlabel("lag (days)")
    p_level = adf[(adf["app_name"] == app) & (adf["series"] == "level")]["p_value"].iloc[0]
    p_d7 = adf[(adf["app_name"] == app) & (adf["series"] == "diff7")]["p_value"].iloc[0]
    save(fig, CHARTS_DIR / "acf_pacf" / f"{SLUG[app]}.png",
         f"{app}: autocorrelation of daily 1–2★ reviews",
         f"Shaded band = 95% limits (±1.96/√n); vertical lines at weekly lags. "
         f"ADF p-value: level {p_level:.3f}, after weekly difference {p_d7:.3f}.")


# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading tagged reviews...")
    df = load()
    print(f"  {len(df):,} reviews, {df['day'].nunique()} days")

    daily = build_daily(df)
    domain = build_domain(daily)
    weekly_app = build_weekly(daily, "app_name")
    weekly_dom = build_weekly(daily, "domain")
    releases = build_releases(df)
    stl, patterns, adf, corr = analyse_patterns(daily)

    fmt = {"date_format": "%Y-%m-%d", "index": False}
    daily.to_csv(OUT_DIR / "daily_app.csv", **fmt)
    domain.to_csv(OUT_DIR / "daily_domain.csv", **fmt)
    weekly_app.to_csv(OUT_DIR / "weekly_app.csv", **fmt)
    weekly_dom.to_csv(OUT_DIR / "weekly_domain.csv", **fmt)
    releases.to_csv(OUT_DIR / "release_events.csv", **fmt)
    stl.to_csv(OUT_DIR / "stl_components.csv", **fmt)
    patterns.to_csv(OUT_DIR / "patterns_by_app.csv", **fmt)
    adf.to_csv(OUT_DIR / "stationarity.csv", **fmt)
    corr.to_csv(OUT_DIR / "acf_pacf.csv", **fmt)
    print(f"Saved tables to {OUT_DIR.relative_to(REPO_ROOT)}/")

    print("Charts...")
    chart_daily(daily)
    chart_trend(stl)
    chart_weekday(patterns)
    chart_releases(releases)
    for app in APP_NAMES:
        chart_acf_pacf(corr, adf, app)

    usable = releases[releases["use_for_update_impact"]]
    level = adf[adf["series"] == "level"].set_index("app_name")
    summary = {
        "window": [str(WINDOW_START.date()), str(WINDOW_END.date())], "days": len(DAYS),
        "reviews": int(len(df)), "neg_reviews": int(df["neg"].sum()),
        "series_modelled": "daily count of 1-2 star reviews per app",
        "feed_gap": {"window": [str(GAP_START.date()), str(GAP_END.date())], "apps": GAP_APPS,
                     "app_days_flagged": int(daily["feed_gap"].sum())},
        "source_gap": {app: {"from": str(a), "to": str(b),
                             "days": daily[(daily["app_name"] == app) & daily["source_gap"]]["day"].dt.strftime("%Y-%m-%d").tolist()}
                       for app, (a, b) in SOURCE_GAP.items()},
        "release_events": {"versions": int(len(releases)), "with_30_plus_reviews": int(releases["min_reviews_ok"].sum()),
                           "new_releases_for_update_impact": int(len(usable)),
                           "per_app": usable.groupby("app_name", observed=True).size().astype(int).to_dict()},
        "patterns": patterns.set_index("app_name")[["mean_neg_per_day", "trend_strength", "seasonal_strength",
                                                   "trend_change_pct", "peak_weekday", "low_weekday",
                                                   "suggested_d", "suggested_D"]].to_dict(orient="index"),
        "stationarity": {
            "level_stationary_apps": level.index[level["stationary_at_5pct"]].tolist(),
            "level_nonstationary_apps": level.index[~level["stationary_at_5pct"]].tolist(),
            "rule": "ADF with AIC lag selection; stationary if p < 0.05. suggested_d = 1 if the level is not stationary; "
                    f"suggested_D = 1 if STL seasonal strength >= {SEASONAL_DIFF_STRENGTH}.",
        },
        "charts": CHART_INDEX,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print("\nPatterns by app:")
    print(patterns[["app_name", "mean_neg_per_day", "trend_strength", "seasonal_strength", "trend_change_pct",
                    "peak_weekday", "low_weekday", "adf_p_level", "suggested_d", "suggested_D"]].to_string(index=False))


if __name__ == "__main__":
    main()
