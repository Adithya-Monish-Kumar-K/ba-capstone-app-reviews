"""
Data access and aggregation helpers for the interactive dashboard (Person 5, issue #112).

The dashboard only READS outputs that the pipeline scripts already produced:
  * data/app_reviews_tagged.csv              (Person 2: issue flags + VADER sentiment per review)
  * data/timeseries/*.csv, data/timeseries_summary.json   (scripts/13_time_series_forecast.py)
  * data/eda_summary.json, data/sentiment_summary.json, data/app_metadata.csv
  * data/model_results.csv, data/prediction_error_summary.csv

Nothing here re-tags, re-scores or re-forecasts. Filtered views are plain counts, shares and means
of columns that already exist. A missing file returns None so the app can show a notice instead of crashing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Paths & constants (same names and palette as scripts/05_eda.py and scripts/13_time_series_forecast.py)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

FILES = {
    "reviews": "app_reviews_tagged.csv",
    "weekly_app": "timeseries/weekly_app_metrics.csv",
    "weekly_overall": "timeseries/weekly_overall_metrics.csv",
    "forecast": "timeseries/weekly_forecast.csv",
    "accuracy": "timeseries/forecast_accuracy.csv",
    "backtest": "timeseries/forecast_backtest.csv",
    "ts_summary": "timeseries_summary.json",
    "eda_summary": "eda_summary.json",
    "sentiment_summary": "sentiment_summary.json",
    "app_metadata": "app_metadata.csv",
    "model_results": "model_results.csv",
    "prediction_errors": "prediction_error_summary.csv",
}

OVERALL = "All apps"
APPS = ["Swiggy", "Zomato", "Myntra", "Paytm", "PhonePe"]
APP_COLORS = dict(zip(APPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SERIES_COLORS = {**APP_COLORS, OVERALL: "#52514e"}
PALETTE = list(APP_COLORS.values())
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLORS = {"Negative": "#c93a39", "Neutral": "#c9c8c3", "Positive": "#2a78d6"}
STAR_COLORS = {1: "#c93a39", 2: "#ee8f8a", 3: "#c9c8c3", 4: "#86b6ef", 5: "#2a78d6"}
RATING_BANDS = {"pct_low_star": ("1–2★", "#c93a39"), "pct_mid_star": ("3★", "#c9c8c3"), "pct_high_star": ("4–5★", "#2a78d6")}
SEQ_SCALE = ["#eef5fd", "#9ec5f4", "#2a78d6", "#0d366b"]

ISSUES = [
    "crash_bugs_stability", "payment_refund", "delivery_delay", "order_quality_fulfillment", "cancellation_return",
    "customer_support", "account_login_otp", "pricing_charges_fraud", "ui_ux_update",
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

MIN_WEEK_N = 30   # same floor as the EDA and time-series stages: rates on fewer reviews are too noisy to plot

# Metrics the user can pick. "fmt" is a d3 format for Plotly, "suffix" is appended to displayed values.
METRICS = {
    "review_count": {"label": "Weekly review volume", "axis": "Reviews per week", "fmt": ",.0f", "suffix": "", "is_rate": False},
    "mean_rating": {"label": "Average rating", "axis": "Mean star rating", "fmt": ".2f", "suffix": "★", "is_rate": True},
    "pct_low_star": {"label": "% rated 1–2★ (negative reviews)", "axis": "% rated 1–2★", "fmt": ".1f", "suffix": "%", "is_rate": True},
    "pct_high_star": {"label": "% rated 4–5★", "axis": "% rated 4–5★", "fmt": ".1f", "suffix": "%", "is_rate": True},
    "pct_negative_sentiment": {"label": "% negative sentiment (VADER)", "axis": "% negative sentiment", "fmt": ".1f", "suffix": "%", "is_rate": True},
    "pct_has_issue": {"label": "% with at least one issue tag", "axis": "% with ≥1 issue tag", "fmt": ".1f", "suffix": "%", "is_rate": True},
    "median_length": {"label": "Median review length (words)", "axis": "Median words", "fmt": ".0f", "suffix": "", "is_rate": True},
}
FORECAST_METRICS = ["pct_low_star", "mean_rating", "pct_negative_sentiment", "pct_has_issue"]   # the four the pipeline forecast
MODEL_LABELS = {
    "naive": "Naive (last week)",
    "hist_mean": "Mean of all history",
    "shift_mean": "Level-shift mean",
    "moving_avg": "4-week moving average",
    "ses": "Exponential smoothing (SES)",
}
ERROR_MEASURES = {
    "mae_h1_4": "Mean absolute error, 1–4 weeks ahead",
    "mae_h1": "Mean absolute error, 1 week ahead",
    "rmse_h1_4": "Root mean squared error, 1–4 weeks ahead",
}


def fmt_value(value, metric):
    """Human-readable value for KPI cards and captions."""
    if value is None or pd.isna(value):
        return "–"
    spec = METRICS[metric]
    return f"{value:{spec['fmt']}}{spec['suffix']}"


# ----------------------------------------------------------------------------
# Loading (cached: files are read once per session, not on every interaction)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(key, parse_dates=()):
    path = DATA_DIR / FILES[key]
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=list(parse_dates))


@st.cache_data(show_spinner=False)
def load_json(key):
    path = DATA_DIR / FILES[key]
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading reviews…")
def load_reviews():
    """Review-level tagged dataset without the free text (not needed for any chart)."""
    path = DATA_DIR / FILES["reviews"]
    if not path.exists():
        return None
    cols = ["app_name", "score", "review_date", "review_length", "has_issue", "issue_count", "primary_issue",
            "sentiment_compound", "sentiment_label"] + [f"issue_{i}" for i in ISSUES]
    df = pd.read_csv(path, usecols=cols, parse_dates=["review_date"])
    df["date"] = df["review_date"].dt.normalize()
    df["week_start"] = df["review_date"].dt.to_period("W-SUN").dt.start_time   # Monday-Sunday weeks, as in the pipeline
    df["is_low"] = (df["score"] <= 2).astype(int)
    df["is_mid"] = (df["score"] == 3).astype(int)
    df["is_high"] = (df["score"] >= 4).astype(int)
    df["is_negative"] = (df["sentiment_label"] == "Negative").astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_weekly():
    """Per-app and pooled weekly metrics in one long table (column `series`), plus 3★ / 4–5★ shares.

    The 4–5★ and 3★ shares are not in the pipeline tables; they are counted here from the same
    reviews and the same Monday-Sunday weeks, so 1–2★ + 3★ + 4–5★ = 100% in every week.
    """
    app_tbl = load_csv("weekly_app", parse_dates=("week_start", "week_end"))
    all_tbl = load_csv("weekly_overall", parse_dates=("week_start", "week_end"))
    if app_tbl is None and all_tbl is None:
        return None
    parts = []
    if app_tbl is not None:
        parts.append(app_tbl.rename(columns={"app_name": "series"}))
    if all_tbl is not None:
        parts.append(all_tbl.assign(series=OVERALL))
    weekly = pd.concat(parts, ignore_index=True)

    reviews = load_reviews()
    if reviews is not None:
        by_app = reviews.groupby(["app_name", "week_start"])[["is_mid", "is_high"]].sum().reset_index().rename(columns={"app_name": "series"})
        pooled = reviews.groupby("week_start")[["is_mid", "is_high"]].sum().reset_index().assign(series=OVERALL)
        stars = pd.concat([by_app, pooled], ignore_index=True).rename(columns={"is_mid": "n_mid_star", "is_high": "n_high_star"})
        weekly = weekly.merge(stars, on=["series", "week_start"], how="left")
        weekly[["n_mid_star", "n_high_star"]] = weekly[["n_mid_star", "n_high_star"]].fillna(0).astype(int)
        denom = weekly["review_count"].where(weekly["review_count"] > 0)
        weekly["pct_mid_star"] = weekly["n_mid_star"] / denom * 100
        weekly["pct_high_star"] = weekly["n_high_star"] / denom * 100
    else:
        weekly["pct_mid_star"] = np.nan
        weekly["pct_high_star"] = np.nan
    return weekly.sort_values(["series", "week_start"]).reset_index(drop=True)


def data_status():
    """One row per input file: used for the 'Data behind this dashboard' table and missing-file notices."""
    rows = []
    for key, rel in FILES.items():
        path = DATA_DIR / rel
        rows.append({"Dataset": f"data/{rel}", "Available": path.exists(),
                     "Size (KB)": round(path.stat().st_size / 1024, 1) if path.exists() else None})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------
def filter_reviews(reviews, app, start, end):
    """Reviews dated within [start, end] for one app, or for all apps when app == OVERALL."""
    mask = (reviews["date"] >= pd.Timestamp(start)) & (reviews["date"] <= pd.Timestamp(end))
    if app != OVERALL:
        mask &= reviews["app_name"] == app
    return reviews[mask]


def filter_weekly(weekly, series, start, end, hide_small=True, metrics=()):
    """Weekly rows whose week starts within [start, end]; rate metrics on small weeks are blanked, not dropped."""
    names = [series] if isinstance(series, str) else list(series)
    rows = weekly[weekly["series"].isin(names) & (weekly["week_start"] >= pd.Timestamp(start))
                  & (weekly["week_start"] <= pd.Timestamp(end))].copy()
    if hide_small:
        small = rows["review_count"] < MIN_WEEK_N
        for m in metrics:
            if m in rows.columns and METRICS.get(m, {"is_rate": True})["is_rate"]:
                rows.loc[small, m] = np.nan
    return rows


def preset_ranges(reviews, ts_summary):
    """Date presets offered in the sidebar. The common window comes from the time-series summary."""
    first, last = reviews["date"].min().date(), reviews["date"].max().date()
    presets = {}
    window = ts_summary.get("weekly_aggregation", {}).get("common_window")
    if window:
        start = pd.Timestamp(window[0]).date()
        end = (pd.Timestamp(window[1]) + pd.Timedelta(days=6)).date()
        presets[f"Common window ({start:%d %b} – {end:%d %b %Y})"] = (start, end)
    year_start = pd.Timestamp(year=last.year, month=1, day=1).date()
    presets[f"{last.year} to date"] = (max(first, year_start), last)
    presets["Full history"] = (first, last)
    return presets, first, last


# ----------------------------------------------------------------------------
# Aggregations on filtered reviews (counts, shares and means of existing columns only)
# ----------------------------------------------------------------------------
def kpis(df):
    if df is None or len(df) == 0:
        return None
    return {
        "review_count": int(len(df)),
        "n_apps": int(df["app_name"].nunique()),
        "first": df["date"].min(),
        "last": df["date"].max(),
        "mean_rating": float(df["score"].mean()),
        "pct_low_star": float(df["is_low"].mean() * 100),
        "pct_high_star": float(df["is_high"].mean() * 100),
        "pct_negative_sentiment": float(df["is_negative"].mean() * 100),
        "pct_has_issue": float(df["has_issue"].mean() * 100),
        "median_length": float(df["review_length"].median()),
        "mean_sentiment": float(df["sentiment_compound"].mean()),
    }


def metric_by_app(df, metric):
    """Value of one metric per app over the whole selected period (review_count = reviews in the period)."""
    g = df.groupby("app_name")
    values = {
        "review_count": g.size,
        "mean_rating": lambda: g["score"].mean(),
        "pct_low_star": lambda: g["is_low"].mean() * 100,
        "pct_high_star": lambda: g["is_high"].mean() * 100,
        "pct_negative_sentiment": lambda: g["is_negative"].mean() * 100,
        "pct_has_issue": lambda: g["has_issue"].mean() * 100,
        "median_length": lambda: g["review_length"].median(),
    }[metric]()
    out = pd.DataFrame({"app_name": APPS}).merge(
        pd.DataFrame({"app_name": values.index, "value": values.to_numpy(dtype=float), "n": g.size().to_numpy()}), how="left")
    return out


def app_summary_table(df):
    """One row per app for the selected period."""
    if len(df) == 0:
        return pd.DataFrame()
    g = df.groupby("app_name")
    out = pd.DataFrame({
        "Reviews": g.size(),
        "Average rating": g["score"].mean(),
        "% rated 1–2★": g["is_low"].mean() * 100,
        "% rated 4–5★": g["is_high"].mean() * 100,
        "% negative sentiment": g["is_negative"].mean() * 100,
        "% with issue tag": g["has_issue"].mean() * 100,
        "Median words": g["review_length"].median(),
    }).reindex([a for a in APPS if a in g.groups])
    top = {}
    for app, sub in g:
        rates = sub[[f"issue_{i}" for i in ISSUES]].mean()
        top[app] = ISSUE_LABELS[rates.idxmax()[len("issue_"):]] if rates.max() > 0 else "–"
    out["Top issue"] = pd.Series(top)
    return out.rename_axis("App").reset_index()


def sentiment_distribution(df, groups):
    """Share of Negative / Neutral / Positive reviews for each (label, dataframe) group."""
    rows = []
    for label, sub in groups:
        if len(sub) == 0:
            continue
        share = sub["sentiment_label"].value_counts(normalize=True) * 100
        for s in SENTIMENT_ORDER:
            rows.append({"group": label, "sentiment": s, "pct": float(share.get(s, 0.0)),
                         "n": int((sub["sentiment_label"] == s).sum()), "total": int(len(sub))})
    return pd.DataFrame(rows)


def rating_distribution(df):
    counts = df["score"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    return pd.DataFrame({"score": counts.index, "n": counts.to_numpy(), "pct": counts.to_numpy() / max(len(df), 1) * 100})


def issue_table(df):
    """Per issue category: reviews tagged, share of the selection, mean rating and mean sentiment of tagged reviews."""
    rows = []
    for i in ISSUES:
        sub = df[df[f"issue_{i}"] == 1]
        rows.append({"issue": i, "label": ISSUE_LABELS[i], "n": int(len(sub)),
                     "pct": len(sub) / len(df) * 100 if len(df) else np.nan,
                     "mean_rating": float(sub["score"].mean()) if len(sub) else np.nan,
                     "mean_sentiment": float(sub["sentiment_compound"].mean()) if len(sub) else np.nan,
                     "pct_low_star": float(sub["is_low"].mean() * 100) if len(sub) else np.nan})
    return pd.DataFrame(rows)


def issue_by_app(df):
    """Rows = apps, columns = issue labels, values = % of that app's reviews carrying the tag."""
    cols = [f"issue_{i}" for i in ISSUES]
    m = df.groupby("app_name")[cols].mean() * 100
    m = m.reindex([a for a in APPS if a in m.index])
    m.columns = [ISSUE_LABELS[i] for i in ISSUES]
    return m, df.groupby("app_name").size()


def sentiment_by_star(df):
    g = df.groupby("score")["sentiment_compound"].agg(["mean", "size"]).reindex([1, 2, 3, 4, 5])
    return g.reset_index().rename(columns={"mean": "mean_compound", "size": "n"})


# ----------------------------------------------------------------------------
# Forecast helpers (read-only views of the pipeline outputs)
# ----------------------------------------------------------------------------
def forecast_view(weekly, forecast, backtest, series, metric, model, horizon):
    """History (common window), forward forecast, and the chosen hold-out forecasts for one series and metric."""
    hist = weekly[(weekly["series"] == series) & weekly["in_common_window"]][["week_start", "week_end", "review_count", metric]]
    fc = forecast[(forecast["series"] == series) & (forecast["metric"] == metric)].sort_values("horizon")
    bt = pd.DataFrame()
    if backtest is not None:
        bt = backtest[(backtest["series"] == series) & (backtest["metric"] == metric) & (backtest["model"] == model)
                      & (backtest["horizon"] == horizon)].sort_values("target_week")
    return hist, fc, bt


def accuracy_view(accuracy, series, metric):
    a = accuracy[(accuracy["series"] == series) & (accuracy["metric"] == metric)].copy()
    a["Model"] = a["model"].map(MODEL_LABELS).fillna(a["model"])
    return a.sort_values("mae_h1_4")
