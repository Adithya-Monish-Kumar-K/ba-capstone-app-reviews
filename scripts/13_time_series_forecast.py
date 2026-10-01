import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "app_reviews_tagged.csv"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "timeseries"
TABLES_DIR = REPO_ROOT / "data" / "timeseries"
SUMMARY_PATH = REPO_ROOT / "data" / "timeseries_summary.json"

DATE_COL = "review_date"
OVERALL = "All apps"
APPS = ["Swiggy", "Zomato", "Myntra", "Paytm", "PhonePe"]
SERIES = APPS + [OVERALL]
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

WEEK_RULE = "W-SUN"                         
MIN_WEEK_N = 30                              
REGIME_START = pd.Timestamp("2026-07-01")   

# Forecast design
FORECAST_METRICS = {                         
    "pct_low_star": ("% rated 1–2★", "pp", 0.0, 100.0),
    "mean_rating": ("Mean star rating", "★", 1.0, 5.0),
    "pct_negative_sentiment": ("% negative sentiment (VADER)", "pp", 0.0, 100.0),
    "pct_has_issue": ("% with ≥1 issue tag", "pp", 0.0, 100.0),
}
PRIMARY_METRIC = "pct_low_star"              
TEST_WEEKS = 6
HORIZONS = [1, 2, 3, 4]                      
FORECAST_HORIZON = 4                        
MA_WINDOW = 4
MIN_TRAIN_WEEKS = 8                          
SES_ALPHAS = np.round(np.arange(0.05, 1.0, 0.05), 2)
SES_INIT_WEEKS = 4                           
SPIKE_Z = 3.0
SPIKE_MIN_EXPECTED = 5                       
FINAL_MODEL = "ses"                         
PI_LEVEL = 0.80
PI_Z = float(stats.norm.ppf(0.5 + PI_LEVEL / 2))

MODEL_LABELS = {
    "naive": "Naive (last week)",
    "hist_mean": "Mean of all history",
    "shift_mean": "Level-shift mean",
    "moving_avg": f"{MA_WINDOW}-week moving avg",
    "ses": "Exp. smoothing (SES)",
}


SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
APP_COLORS = dict(zip(APPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SERIES_COLORS = {**APP_COLORS, OVERALL: INK2}
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fd", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

CHART_INDEX = []   # (filename, title) for the summary json / TIME_SERIES.md


def save(fig, name, title, sub=None, top_pad=0.0):
    """Left-aligned action title + muted subtitle (fixed inch offsets), then write the PNG."""
    h = fig.get_figheight()
    lines = textwrap.wrap(sub, int(fig.get_figwidth() * 13.5)) if sub else []
    head = ((0.95 + 0.17 * (len(lines) - 1)) if sub else 0.6) + top_pad
    fig.tight_layout(rect=[0, 0, 1, 1 - head / h])
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
    CHART_INDEX.append({"file": f"data/charts/timeseries/{name}.png", "title": title})
    print(f"  saved {path.name}")


def text_on(rgb_or_hex):
    r, g, b = to_rgb(rgb_or_hex)
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def week_axis(ax):
    """Month ticks on a weekly date axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="x", visible=False)


def mark_regime(ax, cut, label=True):
    """Dashed line between the last pre-shift and first post-shift week."""
    ax.axvline(cut, color=INK2, ls="--", lw=1)
    if label:
        ax.annotate("sampling shift", (cut, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                    textcoords="offset points", fontsize=8, color=INK2, va="top")


# Load & weekly aggregation

def load():
    df = pd.read_csv(INPUT_PATH, parse_dates=[DATE_COL])
    df["app_name"] = pd.Categorical(df["app_name"], categories=APPS, ordered=True)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    # Monday of the review's calendar week (review_date has no stored timezone; used as scraped)
    df["week_start"] = df[DATE_COL].dt.to_period(WEEK_RULE).dt.start_time
    df["is_low"] = (df["score"] <= 2).astype(int)
    df["is_negative"] = (df["sentiment_label"] == "Negative").astype(int)
    return df


def aggregate_weekly(d, last_week):
    """Weekly metrics for one group of reviews, on a gap-free weekly index up to last_week."""
    g = d.groupby("week_start").agg(
        review_count=("score", "size"), mean_rating=("score", "mean"), sd_rating=("score", "std"),
        mean_sentiment=("sentiment_compound", "mean"), mean_issue_count=("issue_count", "mean"),
        median_length=("review_length", "median"),
        n_low_star=("is_low", "sum"), n_negative_sentiment=("is_negative", "sum"), n_has_issue=("has_issue", "sum"),
        **{f"n_{i}": (f"issue_{i}", "sum") for i in ISSUES})
    # Missing weeks become explicit rows: counts are a true 0, means/rates stay NaN (nothing to average)
    g = g.reindex(pd.date_range(g.index.min(), last_week, freq="7D", name="week_start"))
    count_cols = ["review_count"] + [c for c in g.columns if c.startswith("n_")]
    g[count_cols] = g[count_cols].fillna(0).astype(int)
    denom = g["review_count"].where(g["review_count"] > 0)
    for c in count_cols[1:]:
        g["pct_" + c[2:]] = g[c] / denom * 100
    return g.reset_index()


def add_flags(g, window, regime_week, partial_week):
    g.insert(g.columns.get_loc("week_start") + 1, "week_end", g["week_start"] + pd.Timedelta(days=6))
    g["in_common_window"] = g["week_start"].isin(window)
    g["post_regime_shift"] = g["week_start"] >= regime_week
    g["partial_week"] = g["week_start"] == partial_week
    g["n_ge_min"] = g["review_count"] >= MIN_WEEK_N
    return g


def column_order(g, lead):
    core = ["review_count", "mean_rating", "sd_rating", "pct_low_star", "pct_negative_sentiment", "pct_has_issue",
            "mean_sentiment", "mean_issue_count", "median_length", "n_low_star", "n_negative_sentiment", "n_has_issue"]
    flags = ["in_common_window", "post_regime_shift", "partial_week", "n_ge_min"]
    return g[lead + ["week_start", "week_end"] + core + [f"pct_{i}" for i in ISSUES] + [f"n_{i}" for i in ISSUES] + flags]


def find_partial_week(df):
    """The last calendar week is partial when the newest review is dated before that week's Sunday."""
    last_week = df["week_start"].max()
    is_partial = df[DATE_COL].max().normalize() < last_week + pd.Timedelta(days=6)
    return last_week, (last_week if is_partial else pd.NaT)


def common_window(df, partial_week):
    """Latest contiguous run of complete weeks in which EVERY app has >= MIN_WEEK_N reviews."""
    counts = df.groupby(["week_start", "app_name"], observed=False).size().unstack(fill_value=0)
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="7D"), fill_value=0)
    if pd.notna(partial_week):
        counts = counts[counts.index < partial_week]
    ok = (counts.min(axis=1) >= MIN_WEEK_N).to_numpy()
    start = len(ok)
    while start > 0 and ok[start - 1]:
        start -= 1
    return list(counts.index[start:])


def build_weekly_tables(df):
    last_week, partial_week = find_partial_week(df)
    window = common_window(df, partial_week)
    # A week counts as post-shift when most of its days fall on or after REGIME_START
    regime_week = (REGIME_START + pd.Timedelta(days=3)).to_period(WEEK_RULE).start_time
    per_app = []
    for app in APPS:
        g = aggregate_weekly(df[df["app_name"] == app], last_week)
        g.insert(0, "app_name", app)
        per_app.append(g)
    app_tbl = column_order(add_flags(pd.concat(per_app, ignore_index=True), window, regime_week, partial_week), ["app_name"])
    all_tbl = column_order(add_flags(aggregate_weekly(df, last_week), window, regime_week, partial_week), [])
    return app_tbl, all_tbl, window, regime_week, partial_week


def write_table(g, name):
    g.apply(lambda c: c.round(3) if c.dtype.kind == "f" else c).to_csv(TABLES_DIR / name, index=False)
    print(f"  saved {name} ({len(g):,} rows)")


def window_series(app_tbl, all_tbl, series):
    """Chronologically ordered common-window rows for one app or the pooled series."""
    g = all_tbl if series == OVERALL else app_tbl[app_tbl["app_name"] == series]
    return g[g["in_common_window"]].sort_values("week_start").reset_index(drop=True)



# Forecast models 
def fit_ses_alpha(y):
    """Smoothing weight that minimises one-step-ahead squared error inside the training slice."""
    if len(y) < 3:
        return 0.5
    best, best_sse = SES_ALPHAS[0], np.inf
    for a in SES_ALPHAS:
        level, sse = np.mean(y[:SES_INIT_WEEKS]), 0.0
        for v in y:
            sse += (v - level) ** 2
            level += a * (v - level)
        if sse < best_sse - 1e-12:
            best, best_sse = a, sse
    return float(best)


def ses_level(y, alpha):
    level = np.mean(y[:SES_INIT_WEEKS])
    for v in y:
        level += alpha * (v - level)
    return float(level)


def forecast_naive(y, post):
    return float(y[-1])


def forecast_hist_mean(y, post):
    return float(np.mean(y))


def forecast_shift_mean(y, post):
    """Mean of the weeks since the July-2026 sampling shift (step level-shift model); all history before it."""
    return float(np.mean(y[post])) if post.any() else float(np.mean(y))


def forecast_moving_avg(y, post):
    return float(np.mean(y[-MA_WINDOW:]))


def forecast_ses(y, post):
    return ses_level(y, fit_ses_alpha(y))


MODELS = {"naive": forecast_naive, "hist_mean": forecast_hist_mean, "shift_mean": forecast_shift_mean,
          "moving_avg": forecast_moving_avg, "ses": forecast_ses}


def run_backtest(app_tbl, all_tbl):
    """Rolling-origin hold-out: each test week is forecast 1-4 weeks ahead from data up to the origin only."""
    rows = []
    for series in SERIES:
        g = window_series(app_tbl, all_tbl, series)
        weeks, post = g["week_start"].to_numpy(), g["post_regime_shift"].to_numpy()
        for metric in FORECAST_METRICS:
            y = g[metric].to_numpy(dtype=float)
            for t in range(len(y) - TEST_WEEKS, len(y)):
                for h in HORIZONS:
                    origin = t - h
                    y_train, post_train = y[:origin + 1], post[:origin + 1]
                    for model, fn in MODELS.items():
                        f = fn(y_train, post_train)
                        rows.append({"series": series, "metric": metric, "model": model, "horizon": h,
                                     "origin_week": weeks[origin], "target_week": weeks[t], "n_train_weeks": origin + 1,
                                     "forecast": f, "actual": y[t], "error": f - y[t]})
    return pd.DataFrame(rows)


def sampling_noise_floor(g, metric):
    """Expected absolute error if a week's value were predicted perfectly up to sampling noise (0.798 x SE)."""
    t = g.tail(TEST_WEEKS)
    if metric == "mean_rating":
        se = t["sd_rating"] / np.sqrt(t["review_count"])
    else:
        p = t[metric] / 100
        se = np.sqrt(p * (1 - p) / t["review_count"]) * 100
    return float(np.sqrt(2 / np.pi) * se.mean())


def score_backtest(bt, app_tbl, all_tbl):
    rows = []
    for (series, metric, model), g in bt.groupby(["series", "metric", "model"], sort=False):
        h1 = g[g["horizon"] == 1]
        rows.append({"series": series, "metric": metric, "model": model,
                     "mae_h1": h1["error"].abs().mean(), "mae_h1_4": g["error"].abs().mean(),
                     "rmse_h1_4": np.sqrt((g["error"] ** 2).mean()), "bias_h1_4": g["error"].mean(),
                     "n_forecasts": len(g)})
    acc = pd.DataFrame(rows)
    naive = acc[acc["model"] == "naive"].set_index(["series", "metric"])["mae_h1_4"]
    acc["skill_vs_naive"] = 1 - acc["mae_h1_4"] / acc.set_index(["series", "metric"]).index.map(naive).to_numpy()
    floor = {(s, m): sampling_noise_floor(window_series(app_tbl, all_tbl, s), m) for s in SERIES for m in FORECAST_METRICS}
    acc["sampling_noise_floor"] = [floor[(s, m)] for s, m in zip(acc["series"], acc["metric"])]
    acc["rank_in_series"] = acc.groupby(["series", "metric"])["mae_h1_4"].rank(method="min").astype(int)
    return acc


def horizon_rmse(y, post, fn, h):
    """Spread of past h-step-ahead errors of one model, each forecast made from the weeks before it."""
    errs = [fn(y[:o + 1], post[:o + 1]) - y[o + h] for o in range(MIN_TRAIN_WEEKS - 1, len(y) - h)]
    return float(np.sqrt(np.mean(np.square(errs)))), len(errs)


def make_forecasts(app_tbl, all_tbl):
    """Forecast FORECAST_HORIZON weeks past the last complete week with FINAL_MODEL, fitted on the whole window."""
    fn, rows = MODELS[FINAL_MODEL], []
    for series in SERIES:
        g = window_series(app_tbl, all_tbl, series)
        post, last_week = g["post_regime_shift"].to_numpy(), g["week_start"].iloc[-1]
        for metric, (_, _, lo, hi) in FORECAST_METRICS.items():
            y = g[metric].to_numpy(dtype=float)
            point, widest = fn(y, post), 0.0
            for h in range(1, FORECAST_HORIZON + 1):
                rmse, n_err = horizon_rmse(y, post, fn, h)
                rmse = widest = max(rmse, widest)   # few errors per horizon: never let the interval narrow with h
                rows.append({"series": series, "metric": metric, "model": FINAL_MODEL,
                             "ses_alpha": fit_ses_alpha(y) if FINAL_MODEL == "ses" else np.nan,
                             "last_observed_week": last_week, "week_start": last_week + pd.Timedelta(days=7 * h),
                             "horizon": h, "forecast": point, "last_observed": y[-1],
                             "pi80_lower": float(np.clip(point - PI_Z * rmse, lo, hi)),
                             "pi80_upper": float(np.clip(point + PI_Z * rmse, lo, hi)),
                             "error_rmse_at_horizon": rmse, "n_errors_for_interval": n_err})
    return pd.DataFrame(rows)


def trend_statistics(app_tbl, all_tbl):
    """Pre/post-shift weekly means and Kendall tau trend tests (whole window, and post-shift weeks only)."""
    out = {}
    for series in SERIES:
        g = window_series(app_tbl, all_tbl, series)
        post = g["post_regime_shift"].to_numpy()
        out[series] = {}
        for metric in list(FORECAST_METRICS) + ["median_length", "review_count"]:
            y = g[metric].to_numpy(dtype=float)
            tau_all, p_all = stats.kendalltau(np.arange(len(y)), y)
            tau_post, p_post = stats.kendalltau(np.arange(post.sum()), y[post])
            out[series][metric] = {
                "mean_pre_shift": round(float(y[~post].mean()), 3), "mean_post_shift": round(float(y[post].mean()), 3),
                "change_post_minus_pre": round(float(y[post].mean() - y[~post].mean()), 3),
                "min": round(float(y.min()), 3), "max": round(float(y.max()), 3),
                "weekly_sd_post_shift": round(float(y[post].std(ddof=1)), 3),
                "kendall_tau_window": round(float(tau_all), 3), "kendall_p_window": float(p_all),
                "kendall_tau_post_shift": round(float(tau_post), 3), "kendall_p_post_shift": float(p_post),
            }
    return out


def signal_agreement(app_tbl, all_tbl):
    """Do the text-mining signals move with the star-rating signal week to week? (levels and weekly changes)"""
    out = {}
    for series in SERIES:
        g = window_series(app_tbl, all_tbl, series)
        out[series] = {}
        for col in ["pct_negative_sentiment", "pct_has_issue"]:
            r_level = stats.pearsonr(g[PRIMARY_METRIC], g[col])[0]
            r_diff = stats.pearsonr(g[PRIMARY_METRIC].diff().dropna(), g[col].diff().dropna())[0]
            out[series][col] = {"r_levels": round(float(r_level), 3), "r_weekly_changes": round(float(r_diff), 3)}
    return out


def issue_statistics(app_tbl):
    """Per app and issue: pre/post-shift weekly mean rate and the post-shift trend."""
    out = {}
    for app in APPS:
        g = window_series(app_tbl, None, app)
        post = g["post_regime_shift"].to_numpy()
        out[app] = {}
        for i in ISSUES:
            y = g[f"pct_{i}"].to_numpy(dtype=float)
            tau, p = stats.kendalltau(np.arange(post.sum()), y[post])
            out[app][i] = {"mean_pre_shift": round(float(y[~post].mean()), 2), "mean_post_shift": round(float(y[post].mean()), 2),
                           "kendall_tau_post_shift": round(float(tau), 3), "kendall_p_post_shift": float(p)}
    return out


def issue_spike_scan(app_tbl):
    """Weeks where an app's issue rate is far above that app's own rate in the same sampling regime."""
    w = app_tbl[app_tbl["in_common_window"]]
    rows, cells = [], 0
    for (app, post), g in w.groupby(["app_name", "post_regime_shift"], sort=False):
        for i in ISSUES:
            p = g[f"n_{i}"].sum() / g["review_count"].sum()
            testable = g[g["review_count"] * p >= SPIKE_MIN_EXPECTED]
            cells += len(testable)
            z = (testable[f"n_{i}"] / testable["review_count"] - p) / np.sqrt(p * (1 - p) / testable["review_count"])
            for idx in z[z > SPIKE_Z].index:
                r = testable.loc[idx]
                rows.append({"app_name": app, "issue": i, "week_start": str(r["week_start"].date()),
                             "review_count": int(r["review_count"]), "n_tagged": int(r[f"n_{i}"]),
                             "pct_week": round(float(r[f"pct_{i}"]), 1), "pct_regime_baseline": round(float(p * 100), 1),
                             "z": round(float(z[idx]), 2)})
    expected = cells * float(stats.norm.sf(SPIKE_Z))
    return {"z_threshold": SPIKE_Z, "app_issue_weeks_tested": int(cells), "expected_by_chance": round(expected, 1),
            "n_flagged": len(rows), "flagged": sorted(rows, key=lambda r: -r["z"])}


LEN_BINS = [0, 20, 40, 70, 10_000]           # review-length bands used for standardisation in 05_eda.py


def length_band_sparsity(df, window):
    """Why the EDA's length standardisation is not repeated weekly: many app-week-band cells are empty."""
    w = df[df["week_start"].isin(window)]
    band = pd.cut(w["review_length"], LEN_BINS, labels=["<=20", "21-40", "41-70", "71+"])
    cells = w.groupby(["app_name", "week_start", band], observed=False).size()
    return {"cells": int(len(cells)), "empty_cells": int((cells == 0).sum()), "cells_under_5_reviews": int((cells < 5).sum())}


def coverage_statistics(df, app_tbl, window, partial_week):
    out = {}
    for app in APPS:
        g = app_tbl[app_tbl["app_name"] == app]
        hist = g[~g["in_common_window"] & ~g["partial_week"]]
        w = g[g["in_common_window"]]
        out[app] = {"first_week": str(g["week_start"].min().date()), "weeks_spanned": int(len(g)),
                    "weeks_with_no_reviews": int((g["review_count"] == 0).sum()),
                    "weeks_below_min_n_outside_window": int((hist["review_count"] < MIN_WEEK_N).sum()),
                    "weeks_outside_window": int(len(hist)),
                    "window_reviews": int(w["review_count"].sum()),
                    "window_weekly_n_min": int(w["review_count"].min()), "window_weekly_n_max": int(w["review_count"].max())}
    in_window = df["week_start"].isin(window)
    out["_total"] = {"reviews": int(len(df)), "reviews_in_window": int(in_window.sum()),
                     "pct_reviews_in_window": round(float(in_window.mean() * 100), 1),
                     "reviews_in_partial_week": int((df["week_start"] == partial_week).sum())}
    return out


# Charts

def regime_cut(regime_week):
    return regime_week - pd.Timedelta(days=3.5)


def app_legend(fig, top_inches, series=APPS):
    handles = [plt.Line2D([], [], color=SERIES_COLORS[a], lw=2.2, label=a) for a in series]
    fig.legend(handles=handles, ncol=len(series), loc="upper right",
               bbox_to_anchor=(0.99, 1 - top_inches / fig.get_figheight()), fontsize=9, frameon=False)


def chart_01_volume(app_tbl, all_tbl, window, regime_week, partial_week, summary):
    start = window[0] - pd.Timedelta(days=28)
    a = all_tbl[all_tbl["week_start"] >= start]
    full, part = a[~a["partial_week"]], a[a["partial_week"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    ax.plot(full["week_start"], full["review_count"], color=INK2, lw=2.2, marker="o", ms=4)
    ax.plot(part["week_start"], part["review_count"], ls="none", marker="o", ms=6, mfc=SURFACE, mec=INK2)
    if len(part):
        ax.annotate("partial week\n(excluded)", (part["week_start"].iloc[0], part["review_count"].iloc[0]), xytext=(-6, 8),
                    textcoords="offset points", fontsize=8, color=INK2, ha="right")
    ax.set_title("All five apps, reviews per week", loc="left", fontsize=10.5, color=INK2)
    ax = axes[1]
    for app in APPS:
        g = app_tbl[(app_tbl["app_name"] == app) & (app_tbl["week_start"] >= start) & ~app_tbl["partial_week"]]
        ax.plot(g["week_start"], g["review_count"], color=APP_COLORS[app], lw=2, label=app)
    ax.axhline(MIN_WEEK_N, color=MUTED, lw=1, ls=":")
    ax.annotate(f"n = {MIN_WEEK_N} floor", (1, MIN_WEEK_N), xycoords=("axes fraction", "data"), xytext=(-2, -10),
                textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    ax.set_title("By app (complete weeks)", loc="left", fontsize=10.5, color=INK2)
    ax.legend(ncol=1, fontsize=8.5, loc="upper left")
    for ax in axes:
        ax.axvspan(window[0] - pd.Timedelta(days=3.5), window[-1] + pd.Timedelta(days=3.5), color=GRID, alpha=0.45, lw=0)
        mark_regime(ax, regime_cut(regime_week))
        ax.set_ylabel("Reviews per week")
        ax.set_ylim(0, None)
        week_axis(ax)
    w = all_tbl[all_tbl["in_common_window"]]
    pre, post = w[~w["post_regime_shift"]]["review_count"].mean(), w[w["post_regime_shift"]]["review_count"].mean()
    summary["volume"] = {"mean_weekly_reviews_pre_shift": round(float(pre), 1), "mean_weekly_reviews_post_shift": round(float(post), 1),
                         "ratio_post_vs_pre": round(float(post / pre), 2)}
    save(fig, "ts_01_weekly_volume",
         f"Weekly review volume steps up {post / pre:.1f}× at the July sampling shift: a property of the feed, not of demand",
         f"Shaded = common window ({len(window)} complete weeks with ≥{MIN_WEEK_N} reviews for every app). Each app was scraped to a fixed 3,000-review "
         f"quota, so weekly counts show where the MOST_RELEVANT feed drew its reviews. Volume is described here but not forecast.")


def chart_02_failure_trend(all_tbl, window, regime_week, summary):
    w = all_tbl[all_tbl["in_common_window"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    for col, colr, lab in [("pct_low_star", BLUE, "Rated 1–2★"), ("pct_has_issue", ORANGE, "≥1 issue tag"),
                           ("pct_negative_sentiment", GREEN, "Negative sentiment")]:
        ax.plot(w["week_start"], w[col], color=colr, lw=2.2, marker="o", ms=4, label=lab)
        ax.annotate(f" {w[col].iloc[-1]:.0f}%", (w["week_start"].iloc[-1], w[col].iloc[-1]), fontsize=8.5, color=INK, va="center")
    ax.set_ylabel("% of that week's reviews")
    ax.set_title("Failure signals, all five apps pooled", loc="left", fontsize=10.5, color=INK2)
    ax.legend(ncol=3, fontsize=8.5, loc="lower left")
    ax = axes[1]
    ax.plot(w["week_start"], w["median_length"], color=INK2, lw=2.2, marker="o", ms=4)
    ax.set_ylabel("Median review length (words)")
    ax.set_ylim(0, None)
    ax.set_title("Median review length", loc="left", fontsize=10.5, color=INK2)
    for ax in axes:
        mark_regime(ax, regime_cut(regime_week))
        ax.set_xlim(window[0] - pd.Timedelta(days=5), window[-1] + pd.Timedelta(days=12))
        week_axis(ax)
    t = summary["trend_statistics"][OVERALL]
    save(fig, "ts_02_weekly_failure_trend",
         f"All three failure signals step down at the July shift, together with review length ({t['median_length']['mean_pre_shift']:.0f} → "
         f"{t['median_length']['mean_post_shift']:.0f} words)",
         f"Weekly means before → after: 1–2★ {t['pct_low_star']['mean_pre_shift']:.0f}% → {t['pct_low_star']['mean_post_shift']:.0f}%, issue-tagged "
         f"{t['pct_has_issue']['mean_pre_shift']:.0f}% → {t['pct_has_issue']['mean_post_shift']:.0f}%, negative sentiment "
         f"{t['pct_negative_sentiment']['mean_pre_shift']:.0f}% → {t['pct_negative_sentiment']['mean_post_shift']:.0f}%. EDA.md §6.1 attributes most of the "
         f"step to shorter sampled reviews, so it should not be read as the apps improving. Review length keeps drifting down after July.")


def chart_03_trend_by_app(app_tbl, window, regime_week, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    for ax, col, ylabel in [(axes[0], "pct_low_star", "% rated 1–2★"), (axes[1], "mean_rating", "Mean star rating"),
                            (axes[2], "pct_has_issue", "% with ≥1 issue tag")]:
        for app in APPS:
            g = window_series(app_tbl, None, app)
            ax.plot(g["week_start"], g[col], color=APP_COLORS[app], lw=2, label=app)
            ax.annotate(f" {app}", (g["week_start"].iloc[-1], g[col].iloc[-1]), fontsize=8.5, color=INK, va="center")
        mark_regime(ax, regime_cut(regime_week), label=ax is axes[0])
        ax.set_xlim(window[0] - pd.Timedelta(days=5), window[-1] + pd.Timedelta(days=30))
        ax.set_title(ylabel + ", weekly", loc="left", fontsize=10.5, color=INK2)
        ax.set_ylabel(ylabel)
        week_axis(ax)
    app_legend(fig, 0.80)
    t = summary["trend_statistics"]
    drop = {a: t[a]["pct_low_star"]["change_post_minus_pre"] for a in APPS}
    big = sorted(drop, key=drop.get)[:2]
    small = sorted(drop, key=lambda a: abs(drop[a]))[:2]
    save(fig, "ts_03_weekly_trend_by_app",
         f"{big[0]} and {big[1]} carry the July step ({drop[big[0]]:+.1f} and {drop[big[1]]:+.1f} pp in 1–2★ share); "
         f"{small[0]} and {small[1]} move least ({drop[small[0]]:+.1f} and {drop[small[1]]:+.1f} pp)",
         f"Common window, {len(window)} complete weeks, every point has ≥{MIN_WEEK_N} reviews. Dashed line = July 2026 sampling shift. "
         f"Levels reflect the MOST_RELEVANT sample, not the apps' public ratings.", top_pad=0.3)


def chart_04_issue_categories(app_tbl, window, regime_week, summary):
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 10), sharex=True)
    for ax, issue in zip(axes.ravel(), ISSUES):
        for app in APPS:
            g = window_series(app_tbl, None, app)
            ax.plot(g["week_start"], g[f"pct_{issue}"], color=APP_COLORS[app], lw=1.6)
        mark_regime(ax, regime_cut(regime_week), label=False)
        ax.set_title(ISSUE_LABELS[issue], loc="left", fontsize=10.5, color=INK2)
        ax.set_ylim(0, None)
        week_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of app's weekly reviews")
    app_legend(fig, 0.80)
    n_trend = sum(v["kendall_p_post_shift"] < 0.05 for a in APPS for v in summary["issue_statistics"][a].values())
    sp = summary["issue_spike_scan"]
    save(fig, "ts_04_weekly_issue_categories",
         f"Each app keeps its own issue mix week after week: {sp['n_flagged']} of {sp['app_issue_weeks_tested']} app-issue-weeks spike above {SPIKE_Z:.0f} SE, "
         f"{n_trend} of {len(APPS) * len(ISSUES)} series trend after July",
         f"Share of each app's weekly reviews carrying the tag (multi-label), common window; each panel has its own y-scale. Dashed line = July 2026 "
         f"sampling shift. By chance alone about {sp['expected_by_chance']:.0f} spike and {len(APPS) * len(ISSUES) * 0.05:.0f} trends would be expected. "
         f"Rare tags (<5%) rest on a handful of reviews per week.", top_pad=0.3)


def chart_05_accuracy(acc, summary):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5))
    cols = list(MODELS) + ["floor"]
    xlabels = [textwrap.fill(MODEL_LABELS[m], 12) for m in MODELS] + ["Sampling\nnoise floor"]
    for ax, metric in zip(axes, [PRIMARY_METRIC, "mean_rating"]):
        a = acc[acc["metric"] == metric]
        m = a.pivot(index="series", columns="model", values="mae_h1_4").loc[SERIES, list(MODELS)]
        m["floor"] = a.drop_duplicates("series").set_index("series")["sampling_noise_floor"].loc[SERIES]
        vals = m[cols].to_numpy()
        im = ax.imshow(vals, cmap=SEQ, vmin=0, vmax=np.nanmax(vals[:, :-1]), aspect="auto")
        ax.set_xticks(range(len(cols)), xlabels, fontsize=8.5)
        ax.set_yticks(range(len(SERIES)), SERIES)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fmt = "{:.1f}" if metric != "mean_rating" else "{:.2f}"
        for i in range(vals.shape[0]):
            best = int(np.argmin(vals[i, :-1]))
            for j in range(vals.shape[1]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=SURFACE, lw=2))
                ax.text(j, i, fmt.format(vals[i, j]) + (" ◂" if j == best else ""), ha="center", va="center", fontsize=9,
                        fontweight="bold" if j == best else "normal", color=text_on(im.cmap(im.norm(vals[i, j]))[:3]))
        ax.axvline(len(MODELS) - 0.5, color=SURFACE, lw=6)
        label, unit = FORECAST_METRICS[metric][0], FORECAST_METRICS[metric][1]
        ax.set_title(f"{label}: mean absolute error ({unit}), 1–4 weeks ahead", loc="left", fontsize=10.5, color=INK2)
    s = summary["forecast_accuracy"]
    save(fig, "ts_05_forecast_backtest_accuracy",
         f"SES forecasts the 1–2★ share with {s['primary_metric_mae_final_model_min']:.1f}–{s['primary_metric_mae_final_model_max']:.1f} pp error; "
         f"the all-history mean, blind to the July step, is worst in {s['hist_mean_worst_count']} of {s['n_series_metric']} series",
         f"Hold-out = last {TEST_WEEKS} complete weeks; each cell averages {TEST_WEEKS * len(HORIZONS)} forecasts made from earlier weeks only. "
         f"◂ = lowest error in the row. The noise floor is the error a perfect forecast of the true rate would still make from weekly sample size alone.")


def forecast_panel(ax, g, bt, fc, series, metric, color, show_labels=True):
    """History, one-step hold-out forecasts and the forward forecast with its 80% interval, for one series."""
    test_start = g["week_start"].iloc[-TEST_WEEKS]
    ax.axvspan(test_start - pd.Timedelta(days=3.5), g["week_start"].iloc[-1] + pd.Timedelta(days=3.5), color=GRID, alpha=0.5, lw=0)
    ax.plot(g["week_start"], g[metric], color=color, lw=2, marker="o", ms=3.5)
    b = bt[(bt["series"] == series) & (bt["metric"] == metric) & (bt["model"] == FINAL_MODEL) & (bt["horizon"] == 1)]
    ax.plot(b["target_week"], b["forecast"], color=INK, lw=1.2, ls=":", marker="o", ms=5, mfc=SURFACE, mec=INK)
    f = fc[(fc["series"] == series) & (fc["metric"] == metric)]
    xs = [g["week_start"].iloc[-1]] + list(f["week_start"])
    ax.plot(xs, [g[metric].iloc[-1]] + list(f["forecast"]), color=color, lw=2, ls="--")
    ax.fill_between(f["week_start"], f["pi80_lower"], f["pi80_upper"], color=color, alpha=0.18, lw=0)
    fmt = "{:.2f}" if metric == "mean_rating" else "{:.0f}%"
    ax.annotate(" " + fmt.format(f["forecast"].iloc[-1]), (f["week_start"].iloc[-1], f["forecast"].iloc[-1]),
                fontsize=8.5, color=INK, va="center")
    if show_labels:
        ax.annotate("hold-out", (test_start, 1), xycoords=("data", "axes fraction"), xytext=(0, -2),
                    textcoords="offset points", fontsize=8, color=INK2, va="top")
    ax.set_xlim(g["week_start"].iloc[0] - pd.Timedelta(days=5), f["week_start"].iloc[-1] + pd.Timedelta(days=16))
    week_axis(ax)


def forecast_legend(fig, top_inches):
    handles = [plt.Line2D([], [], color=INK2, lw=2, marker="o", ms=3.5, label="Observed week"),
               plt.Line2D([], [], color=INK, lw=1.2, ls=":", marker="o", ms=5, mfc=SURFACE, mec=INK, label="1-week-ahead hold-out forecast"),
               plt.Line2D([], [], color=INK2, lw=2, ls="--", label=f"{FORECAST_HORIZON}-week forecast"),
               plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.18, lw=0, label=f"{PI_LEVEL:.0%} prediction interval")]
    fig.legend(handles=handles, ncol=4, loc="upper right", bbox_to_anchor=(0.99, 1 - top_inches / fig.get_figheight()),
               fontsize=9, frameon=False)


def chart_06_forecast_overall(all_tbl, bt, fc, acc, summary):
    g = window_series(None, all_tbl, OVERALL)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, (metric, (label, unit, _, _)) in zip(axes.ravel(), FORECAST_METRICS.items()):
        forecast_panel(ax, g, bt, fc, OVERALL, metric, SERIES_COLORS[OVERALL], show_labels=ax is axes[0, 0])
        a = acc[(acc["series"] == OVERALL) & (acc["metric"] == metric) & (acc["model"] == FINAL_MODEL)].iloc[0]
        ax.set_title(f"{label}  ·  hold-out MAE {a['mae_h1_4']:.2f} {unit}", loc="left", fontsize=10.5, color=INK2)
        ax.set_ylabel(label)
    forecast_legend(fig, 0.80)
    f = summary["forecasts"][OVERALL][PRIMARY_METRIC]
    save(fig, "ts_06_forecast_overall",
         f"Pooled forecast: about {f['forecast']:.0f}% of reviews rated 1–2★ over the next {FORECAST_HORIZON} weeks "
         f"(80% interval {f['pi80_lower_h4']:.0f}–{f['pi80_upper_h4']:.0f}% by week {FORECAST_HORIZON})",
         f"{MODEL_LABELS[FINAL_MODEL]} fitted on {len(g)} complete weeks; flat forecast because the data supports a level, not a trend or season. "
         f"Shaded = {TEST_WEEKS}-week hold-out. Valid only if the sampling regime stays as it has been since July.", top_pad=0.3)


def chart_forecast_by_app(app_tbl, all_tbl, bt, fc, acc, summary, metric, name):
    label, unit = FORECAST_METRICS[metric][0], FORECAST_METRICS[metric][1]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    for ax, series in zip(axes.ravel(), SERIES):
        g = window_series(app_tbl, all_tbl, series)
        forecast_panel(ax, g, bt, fc, series, metric, SERIES_COLORS[series], show_labels=series == SERIES[0])
        a = acc[(acc["series"] == series) & (acc["metric"] == metric) & (acc["model"] == FINAL_MODEL)].iloc[0]
        ax.set_title(f"{series}  ·  hold-out MAE {a['mae_h1_4']:.2f} {unit}", loc="left", fontsize=10.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(label)
    forecast_legend(fig, 0.80)
    f = {a: summary["forecasts"][a][metric]["forecast"] for a in APPS}
    hi, lo = max(f, key=f.get), min(f, key=f.get)
    fmt = "{:.2f}★" if metric == "mean_rating" else "{:.0f}%"
    save(fig, name,
         f"{label}, {FORECAST_HORIZON}-week forecast by app: from {fmt.format(f[lo])} ({lo}) to {fmt.format(f[hi])} ({hi})",
         f"{MODEL_LABELS[FINAL_MODEL]} per app, each panel on its own y-scale. Shaded = {TEST_WEEKS}-week hold-out; band = {PI_LEVEL:.0%} prediction "
         f"interval from past forecast errors at the same horizon. Levels describe the scraped sample, not each app's user base.", top_pad=0.3)


def summarise_accuracy(acc):
    final = acc[acc["model"] == FINAL_MODEL]
    prim = final[final["metric"] == PRIMARY_METRIC]
    best = acc[acc["rank_in_series"] == 1]
    worst = acc.loc[acc.groupby(["series", "metric"])["mae_h1_4"].idxmax()]
    table = {}
    for (series, metric), g in acc.groupby(["series", "metric"], sort=False):
        table.setdefault(series, {})[metric] = {
            "mae_h1_4": {m: round(float(v), 3) for m, v in zip(g["model"], g["mae_h1_4"])},
            "mae_h1": {m: round(float(v), 3) for m, v in zip(g["model"], g["mae_h1"])},
            "bias_h1_4": {m: round(float(v), 3) for m, v in zip(g["model"], g["bias_h1_4"])},
            "best_model": g.loc[g["mae_h1_4"].idxmin(), "model"],
            "sampling_noise_floor": round(float(g["sampling_noise_floor"].iloc[0]), 3),
        }
    return {
        "by_series": table,
        "mean_rank_by_model": {m: round(float(v), 2) for m, v in acc.groupby("model")["rank_in_series"].mean().items()},
        "times_best_by_model": {m: int(v) for m, v in best["model"].value_counts().items()},
        "n_series_metric": int(acc.groupby(["series", "metric"]).ngroups),
        "hist_mean_worst_count": int((worst["model"] == "hist_mean").sum()),
        "final_model_beats_naive_count": int((final["skill_vs_naive"] > 0).sum()),
        "final_model_mean_skill_vs_naive": round(float(final["skill_vs_naive"].mean()), 3),
        "primary_metric_mae_final_model_min": round(float(prim["mae_h1_4"].min()), 2),
        "primary_metric_mae_final_model_max": round(float(prim["mae_h1_4"].max()), 2),
    }


def summarise_forecasts(fc):
    out = {}
    for (series, metric), g in fc.groupby(["series", "metric"], sort=False):
        g = g.sort_values("horizon")
        out.setdefault(series, {})[metric] = {
            "forecast": round(float(g["forecast"].iloc[0]), 3), "last_observed": round(float(g["last_observed"].iloc[0]), 3),
            "ses_alpha": None if pd.isna(g["ses_alpha"].iloc[0]) else round(float(g["ses_alpha"].iloc[0]), 2),
            "pi80_lower_h1": round(float(g["pi80_lower"].iloc[0]), 3), "pi80_upper_h1": round(float(g["pi80_upper"].iloc[0]), 3),
            "pi80_lower_h4": round(float(g["pi80_lower"].iloc[-1]), 3), "pi80_upper_h4": round(float(g["pi80_upper"].iloc[-1]), 3),
        }
    return out


def interval_coverage(bt, app_tbl, all_tbl):
    """Share of 1-week-ahead hold-out actuals inside the 80% interval built only from errors before each origin."""
    fn, hits, total = MODELS[FINAL_MODEL], 0, 0
    for series in SERIES:
        g = window_series(app_tbl, all_tbl, series)
        post = g["post_regime_shift"].to_numpy()
        for metric in FORECAST_METRICS:
            y = g[metric].to_numpy(dtype=float)
            for t in range(len(y) - TEST_WEEKS, len(y)):
                y_tr, p_tr = y[:t], post[:t]
                rmse, _ = horizon_rmse(y_tr, p_tr, fn, 1)
                hits += abs(fn(y_tr, p_tr) - y[t]) <= PI_Z * rmse
                total += 1
    return {"nominal": PI_LEVEL, "empirical_h1": round(hits / total, 3), "n": total}


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"Loaded {len(df):,} reviews, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

    print("Weekly aggregation")
    app_tbl, all_tbl, window, regime_week, partial_week = build_weekly_tables(df)
    write_table(app_tbl, "weekly_app_metrics.csv")
    write_table(all_tbl, "weekly_overall_metrics.csv")
    n_pre = sum(w < regime_week for w in window)
    print(f"  common window: {window[0].date()} to {window[-1].date()} ({len(window)} weeks; {n_pre} before / {len(window) - n_pre} after the shift)")
    if len(window) < MIN_TRAIN_WEEKS + TEST_WEEKS:
        raise SystemExit(f"Common window has {len(window)} weeks; need at least {MIN_TRAIN_WEEKS + TEST_WEEKS} to forecast.")

    print("Forecast backtest")
    bt = run_backtest(app_tbl, all_tbl)
    acc = score_backtest(bt, app_tbl, all_tbl)
    fc = make_forecasts(app_tbl, all_tbl)
    write_table(bt, "forecast_backtest.csv")
    write_table(acc, "forecast_accuracy.csv")
    write_table(fc, "weekly_forecast.csv")

    summary = {
        "input": {"file": "data/app_reviews_tagged.csv", "date_column": DATE_COL, "rows": int(len(df)),
                  "first_review": str(df[DATE_COL].min()), "last_review": str(df[DATE_COL].max())},
        "weekly_aggregation": {"week_definition": "Monday-Sunday (pandas W-SUN), labelled by week_start",
                               "min_reviews_per_app_week": MIN_WEEK_N,
                               "common_window": [str(window[0].date()), str(window[-1].date())],
                               "common_window_weeks": len(window), "weeks_pre_shift": n_pre, "weeks_post_shift": len(window) - n_pre,
                               "regime_shift_date": str(REGIME_START.date()), "first_post_shift_week": str(regime_week.date()),
                               "partial_week_excluded": None if pd.isna(partial_week) else str(partial_week.date())},
        "coverage": coverage_statistics(df, app_tbl, window, partial_week),
        "length_band_sparsity": length_band_sparsity(df, window),
        "trend_statistics": trend_statistics(app_tbl, all_tbl),
        "signal_agreement_with_low_star": signal_agreement(app_tbl, all_tbl),
        "issue_statistics": issue_statistics(app_tbl),
        "issue_spike_scan": issue_spike_scan(app_tbl),
        "forecast_setup": {"metrics": list(FORECAST_METRICS), "primary_metric": PRIMARY_METRIC, "models": MODEL_LABELS,
                           "final_model": FINAL_MODEL, "test_weeks": TEST_WEEKS, "horizons": HORIZONS,
                           "train_weeks_at_first_origin": len(window) - TEST_WEEKS - max(HORIZONS) + 1,
                           "test_window": [str(window[-TEST_WEEKS].date()), str(window[-1].date())],
                           "forecast_weeks": [str((window[-1] + pd.Timedelta(days=7 * h)).date()) for h in range(1, FORECAST_HORIZON + 1)],
                           "prediction_interval": PI_LEVEL},
        "forecast_accuracy": summarise_accuracy(acc),
        "interval_coverage": interval_coverage(bt, app_tbl, all_tbl),
        "forecasts": summarise_forecasts(fc),
    }

    print("Charts")
    chart_01_volume(app_tbl, all_tbl, window, regime_week, partial_week, summary)
    chart_02_failure_trend(all_tbl, window, regime_week, summary)
    chart_03_trend_by_app(app_tbl, window, regime_week, summary)
    chart_04_issue_categories(app_tbl, window, regime_week, summary)
    chart_05_accuracy(acc, summary)
    chart_06_forecast_overall(all_tbl, bt, fc, acc, summary)
    chart_forecast_by_app(app_tbl, all_tbl, bt, fc, acc, summary, PRIMARY_METRIC, "ts_07_forecast_low_star_by_app")
    chart_forecast_by_app(app_tbl, all_tbl, bt, fc, acc, summary, "mean_rating", "ts_08_forecast_rating_by_app")

    summary["charts"] = CHART_INDEX
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
