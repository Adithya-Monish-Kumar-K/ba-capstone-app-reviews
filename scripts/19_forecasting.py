"""
Stage 3 of Review 2, Method 2 (Time-Series Analysis): ARIMA forecasting with a rolling backtest.

Forecasts the daily count of 1-2 star reviews per app. The main model is ARIMA, with seasonal terms
for a weekly cycle (SARIMA, period 7) where the data support them; for the five apps hit by the 21 Apr - 5 May feed gap the gap is an
extra input (SARIMAX with a 0/1 `feed_gap` dummy), so the model does not read the dip as a real change.
Two comparison models are fitted at every step: a seasonal-naive baseline ("same weekday last week")
and Holt-Winters exponential smoothing (damped additive trend, additive weekly season).

Method
  1. Orders are chosen per app from the ACF/PACF of the training series (which lags are significant
     decide how many AR, MA and seasonal terms are tried) and then by the corrected AIC (AICc); among orders within 2 AICc points of the best, the one with the fewest terms is taken.
     The differencing order d comes from the ADF test in data/timeseries/patterns_by_app.csv.
  2. Rolling-origin backtest: the models are fitted on the first 12 weeks (84 days), forecast the next
     28 days, then the origin moves forward one week and the models are refitted, until the data end.
     The orders are chosen on the first 84 days only, so the backtest never uses future data.
  3. Final forecast: the orders are chosen again on all 173 days and the next 28 days (21 Sep - 18 Oct
     2026, four Monday-Sunday weeks) are forecast with 80% and 95% prediction intervals. Weekly totals
     and their intervals come from 2,000 simulated future paths.

Input:  data/timeseries/daily_app.csv, data/timeseries/patterns_by_app.csv  (from 15_time_series_data.py)
Output: data/timeseries/
          arima_orders.csv        the chosen orders per app, for the backtest and the final forecast
          arima_candidates.csv    every order tried with its AICc, so the choice can be checked
          backtest_forecasts.csv  every backtest forecast: app, model, origin, horizon, actual, forecast, intervals
          backtest_summary.csv    MAE and RMSE per app, model and horizon band
          forecast_daily.csv      next 28 days per app and model (ARIMA with 80% and 95% intervals)
          forecast_weekly.csv     next 4 weeks per app: total and interval for 1-2 star reviews
        data/charts/timeseries/   ts_05 to ts_07
        data/forecasting_summary.json

Evaluating the forecasts further (MASE, residual checks, interval coverage, unusual days) is Stage 4 (#137).
"""
import itertools
import json
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, DOMAINS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TS_DIR = REPO_ROOT / "data" / "timeseries"
CHARTS_DIR = REPO_ROOT / "data" / "charts" / "timeseries"
SUMMARY_PATH = REPO_ROOT / "data" / "forecasting_summary.json"

SEASON = 7                      # weekly cycle
STRONG_WEEKDAY_PCT = 15         # a weekday this far (%) from the weekly mean counts as a strong weekly pattern (Stage 1)
INITIAL_TRAIN = 84              # first backtest origin: 12 weeks of data
HORIZON = 28                    # forecast four weeks ahead
STEP = 7                        # the origin moves one week at a time
MAX_P, MAX_Q = 3, 3             # largest non-seasonal orders tried
AICC_TOLERANCE = 2              # orders this close to the best AICc count as equally good; the simplest is taken
N_SIM = 2000                    # simulated paths for the weekly totals
GAP_APPS = ["Swiggy", "Blinkit", "Domino's", "Flipkart", "Amazon"]
BANDS = {"1-7 days": (1, 7), "8-14 days": (8, 14), "15-28 days": (15, 28), "all (1-28 days)": (1, 28)}
MODELS = ["ARIMA", "Seasonal naive", "Holt-Winters"]
MODEL_COLORS = {"ARIMA": "#2a78d6", "Seasonal naive": "#8a8985", "Holt-Winters": "#eb6834"}

# ----------------------------------------------------------------------------
# Visual style (same palette and layout as 15_time_series_data.py)
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


def domain_apps(domain):
    return [a for a in APP_NAMES if APP_DOMAIN[a] == domain]


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load():
    """Daily series per app: the count to forecast and the feed-gap dummy (None for apps without a gap)."""
    daily = pd.read_csv(TS_DIR / "daily_app.csv", parse_dates=["day"])
    patterns = pd.read_csv(TS_DIR / "patterns_by_app.csv").set_index("app_name")
    series = {}
    for app in APP_NAMES:
        d = daily[daily["app_name"] == app].sort_values("day").set_index("day").asfreq("D")
        y = d["neg_reviews_filled"].astype(float)
        exog = d[["feed_gap"]].astype(float) if app in GAP_APPS else None
        weekday = patterns.loc[app, [f"weekday_{d}_pct" for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")]]
        series[app] = {"y": y, "exog": exog, "d": int(patterns.loc[app, "suggested_d"]),
                       "weekly_pattern": bool(weekday.abs().max() >= STRONG_WEEKDAY_PCT)}
    return series


# ----------------------------------------------------------------------------
# 2. Order selection (ACF/PACF to propose, AICc to choose)
# ----------------------------------------------------------------------------
def propose_orders(y, d, weekly_pattern):
    """Candidate (p, d, q)(P, 0, Q, 7) orders from the ACF/PACF of the (differenced) training series.

    p goes up to the last significant PACF lag (1-3), q up to the last significant ACF lag (1-3),
    and a seasonal AR (or MA) term is tried if the PACF (or ACF) is significant at lag 7, or if Stage 1 found
    a strong weekday effect for the app (weekly_pattern). Significant means outside +-1.96 / sqrt(n).
    """
    z = y.diff(d).dropna() if d else y
    bound = 1.96 / np.sqrt(len(z))
    r, pr = acf(z, nlags=SEASON, fft=False), pacf(z, nlags=SEASON, method="ywm")
    last_sig = lambda v, top: max([k for k in range(1, top + 1) if abs(v[k]) > bound], default=1)  # noqa: E731
    p_max, q_max = last_sig(pr, MAX_P), last_sig(r, MAX_Q)
    seasonal_p = [0, 1] if weekly_pattern or abs(pr[SEASON]) > bound else [0]
    seasonal_q = [0, 1] if weekly_pattern or abs(r[SEASON]) > bound else [0]
    grid = itertools.product(range(p_max + 1), range(q_max + 1), seasonal_p, seasonal_q)
    return [((p, d, q), (P, 0, Q, SEASON)) for p, q, P, Q in grid], {"p_max": p_max, "q_max": q_max,
                                                                       "seasonal_ar": 1 in seasonal_p, "seasonal_ma": 1 in seasonal_q}


def fit_arima(y, exog, order, seasonal_order):
    trend = "c" if order[1] == 0 else "n"          # a constant only makes sense without differencing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend).fit(disp=False, maxiter=300)


def select_orders(app, s, n_train, stage):
    """Fit every proposed order on the first n_train days and keep the one with the lowest AICc."""
    y = s["y"].iloc[:n_train]
    exog = None if s["exog"] is None else s["exog"].iloc[:n_train]
    cands, info = propose_orders(y, s["d"], s["weekly_pattern"])
    rows = []
    for order, seasonal in cands:
        try:
            res = fit_arima(y, exog, order, seasonal)
            rows.append({"app_name": app, "stage": stage, "order": order, "seasonal_order": seasonal, "aicc": res.aicc,
                         "aic": res.aic, "converged": bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else True})
        except Exception as exc:                       # a candidate that cannot be fitted is skipped, and recorded
            rows.append({"app_name": app, "stage": stage, "order": order, "seasonal_order": seasonal, "aicc": np.nan,
                         "aic": np.nan, "converged": False, "error": str(exc)[:80]})
    table = pd.DataFrame(rows)
    ok = table[table["converged"] & table["aicc"].notna()].copy()
    # Parsimony: any order within 2 AICc points of the minimum fits about as well, so take the one with the fewest terms
    ok["terms"] = ok.apply(lambda r: r["order"][0] + r["order"][2] + r["seasonal_order"][0] + r["seasonal_order"][2], axis=1)
    best = ok[ok["aicc"] <= ok["aicc"].min() + AICC_TOLERANCE].sort_values(["terms", "aicc"]).iloc[0]
    table["chosen"] = (table["order"] == best["order"]) & (table["seasonal_order"] == best["seasonal_order"])
    chosen = {"app_name": app, "stage": stage, "train_days": n_train, "p": best["order"][0], "d": best["order"][1],
              "q": best["order"][2], "P": best["seasonal_order"][0], "D": 0, "Q": best["seasonal_order"][2], "season": SEASON,
              "feed_gap_input": s["exog"] is not None, "aicc": round(best["aicc"], 2), "candidates_tried": len(table), **info}
    return best["order"], best["seasonal_order"], chosen, table


# ----------------------------------------------------------------------------
# 3. Models: ARIMA and the two comparison models
# ----------------------------------------------------------------------------
def forecast_arima(res, steps, exog):
    f = res.get_forecast(steps=steps, exog=exog)
    out = {"forecast": f.predicted_mean.to_numpy()}
    for level, alpha in ((80, 0.20), (95, 0.05)):
        ci = f.conf_int(alpha=alpha).to_numpy()
        out[f"lower_{level}"], out[f"upper_{level}"] = ci[:, 0], ci[:, 1]
    return {k: np.clip(v, 0, None) for k, v in out.items()}


def forecast_seasonal_naive(y, steps):
    last = y.to_numpy()[-SEASON:]
    return {"forecast": np.array([last[i % SEASON] for i in range(steps)])}


def forecast_holt_winters(y, steps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=SEASON,
                                   initialization_method="estimated").fit()
    return {"forecast": np.clip(res.forecast(steps).to_numpy(), 0, None)}


def all_forecasts(s, n_train, order, seasonal, steps):
    """Fit the three models on the first n_train days and forecast the next `steps` days."""
    y = s["y"].iloc[:n_train]
    days = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    exog_train = None if s["exog"] is None else s["exog"].iloc[:n_train]
    res = fit_arima(y, exog_train, order, seasonal)
    return days, res, {"ARIMA": forecast_arima(res, steps, future_exog_for(s, days)),
                       "Seasonal naive": forecast_seasonal_naive(y, steps),
                       "Holt-Winters": forecast_holt_winters(y, steps)}


def future_exog_for(s, days):
    """The feed gap is in the past, so the dummy is 0 for every forecast day."""
    return None if s["exog"] is None else pd.DataFrame({"feed_gap": 0.0}, index=days)


# ----------------------------------------------------------------------------
# 4. Rolling-origin backtest
# ----------------------------------------------------------------------------
def backtest(app, s, order, seasonal):
    n = len(s["y"])
    rows = []
    for origin in range(INITIAL_TRAIN, n - HORIZON + 1, STEP):
        days, _, fc = all_forecasts(s, origin, order, seasonal, HORIZON)
        actual = s["y"].iloc[origin:origin + HORIZON].to_numpy()
        for model, f in fc.items():
            for h in range(HORIZON):
                rows.append({"app_name": app, "model": model, "origin": s["y"].index[origin - 1], "horizon": h + 1,
                             "day": days[h], "actual": actual[h], "forecast": f["forecast"][h],
                             **{k: f[k][h] for k in ("lower_80", "upper_80", "lower_95", "upper_95") if k in f}})
    return rows


def cover(bt, level):
    """Share of ARIMA backtest days whose actual count falls inside the `level`% prediction interval."""
    a = bt[bt["model"] == "ARIMA"]
    return ((a["actual"] >= a[f"lower_{level}"]) & (a["actual"] <= a[f"upper_{level}"])).mean()


def summarise_backtest(bt):
    rows = []
    for (app, model), g in bt.groupby(["app_name", "model"], observed=True):
        err = g["forecast"] - g["actual"]
        for band, (lo, hi) in BANDS.items():
            m = g["horizon"].between(lo, hi)
            rows.append({"app_name": app, "domain": APP_DOMAIN[app], "model": model, "horizon_band": band,
                         "mae": err[m].abs().mean(), "rmse": np.sqrt((err[m] ** 2).mean()),
                         "mean_actual": g.loc[m, "actual"].mean(), "forecasts": int(m.sum()),
                         "origins": g["origin"].nunique()})
    out = pd.DataFrame(rows)
    out["mae_pct_of_mean"] = out["mae"] / out["mean_actual"] * 100
    return out.round(3)


# ----------------------------------------------------------------------------
# 5. Final forecast
# ----------------------------------------------------------------------------
def final_forecast(app, s, order, seasonal):
    n = len(s["y"])
    days, res, fc = all_forecasts(s, n, order, seasonal, HORIZON)
    daily = []
    for model, f in fc.items():
        daily.append(pd.DataFrame({"app_name": app, "model": model, "day": days, "horizon": np.arange(1, HORIZON + 1),
                                   **{k: np.round(v, 2) for k, v in f.items()}}))
    # weekly totals: simulate future paths from the fitted ARIMA and sum each Monday-Sunday week
    rng = np.random.default_rng(sum(map(ord, app)))      # fixed seed per app so re-runs are identical
    sims = np.asarray(res.simulate(HORIZON, repetitions=N_SIM, anchor="end", exog=future_exog_for(s, days),
                                   rng=rng)).reshape(HORIZON, N_SIM)
    sims = np.clip(sims, 0, None)
    weekly = []
    for w in range(HORIZON // 7):
        sl = slice(w * 7, (w + 1) * 7)
        tot = sims[sl].sum(axis=0)
        point = fc["ARIMA"]["forecast"][sl].sum()
        weekly.append({"app_name": app, "week": w + 1, "week_start": days[sl.start], "week_end": days[sl.stop - 1],
                       "forecast": round(point, 1), "lower_80": round(np.percentile(tot, 10), 1),
                       "upper_80": round(np.percentile(tot, 90), 1), "lower_95": round(np.percentile(tot, 2.5), 1),
                       "upper_95": round(np.percentile(tot, 97.5), 1),
                       "last_4_weeks_avg": round(s["y"].iloc[-28:].sum() / 4, 1)})
        weekly[-1]["change_vs_last_4_weeks_pct"] = round((point / weekly[-1]["last_4_weeks_avg"] - 1) * 100, 1)
    return pd.concat(daily, ignore_index=True), weekly, res


# ----------------------------------------------------------------------------
# 6. Charts
# ----------------------------------------------------------------------------
def panel_grid(figsize):
    fig, axes = plt.subplots(len(DOMAINS), 4, figsize=figsize)
    cells = {}
    for i, dom in enumerate(DOMAINS):
        apps = domain_apps(dom)
        for j in range(4):
            ax = axes[i][j]
            if j < len(apps):
                cells[apps[j]] = ax
            else:
                ax.axis("off")
    return fig, cells


def chart_forecast(series, fdaily):
    fig, cells = panel_grid((16, 10.5))
    cutoff = series["Swiggy"]["y"].index[-1]
    for app, ax in cells.items():
        y = series[app]["y"].iloc[-70:]
        f = fdaily[(fdaily["app_name"] == app) & (fdaily["model"] == "ARIMA")]
        color = APP_COLORS[app]
        ax.plot(y.index, y.values, color=color, lw=1.1, alpha=0.9)
        ax.fill_between(f["day"], f["lower_95"], f["upper_95"], color=color, alpha=0.13, lw=0)
        ax.fill_between(f["day"], f["lower_80"], f["upper_80"], color=color, alpha=0.25, lw=0)
        ax.plot(f["day"], f["forecast"], color=color, lw=2)
        ax.axvline(cutoff, color=MUTED, lw=0.9, ls=":")
        ax.set_title(app, loc="left", fontsize=11, color=INK2)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=8.5)
    for i, dom in enumerate(DOMAINS):
        fig.axes[i * 4].set_ylabel(f"{dom}\n1–2★ reviews per day", fontsize=9)
    save(fig, CHARTS_DIR / "ts_05_forecast_next_4_weeks.png",
         "ARIMA forecast of daily 1–2★ reviews, 21 Sep – 18 Oct 2026",
         "Line = last 10 weeks of actual counts, then the ARIMA forecast (weekly seasonal terms for the four Food & Grocery apps only). Dark band = 80% and light band = 95% prediction interval. "
         "Dotted line = last day of data (20 Sep). The interval widens with the horizon; food apps keep their Sunday peak.")


def chart_backtest(series, bt):
    fig, cells = panel_grid((16, 10.5))
    arima = bt[bt["model"] == "ARIMA"]
    for app, ax in cells.items():
        y = series[app]["y"].iloc[INITIAL_TRAIN - 7:]
        color = APP_COLORS[app]
        for _, g in arima[arima["app_name"] == app].groupby("origin"):
            ax.plot(g["day"], g["forecast"], color=color, lw=1.1, alpha=0.5)
        ax.plot(y.index, y.values, color=INK, lw=1.0)
        ax.set_title(app, loc="left", fontsize=11, color=INK2)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=8.5)
    for i, dom in enumerate(DOMAINS):
        fig.axes[i * 4].set_ylabel(f"{dom}\n1–2★ reviews per day", fontsize=9)
    n_orig = arima["origin"].nunique()
    save(fig, CHARTS_DIR / "ts_06_rolling_backtest.png",
         "Rolling-origin backtest: each coloured line is one 28-day forecast made from a different week",
         f"Black = actual daily 1–2★ reviews. {n_orig} origins one week apart, the first after 84 days (23 Jun). "
         "Each forecast uses only data up to its origin; the orders are fixed from the first 84 days.")


def chart_model_comparison(summary):
    s = summary[summary["horizon_band"] == "all (1-28 days)"]
    order = (s[s["model"] == "ARIMA"].sort_values("mae_pct_of_mean")["app_name"]).tolist()[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    h = 0.26
    for k, model in enumerate(MODELS):
        v = s[s["model"] == model].set_index("app_name").reindex(order)["mae_pct_of_mean"]
        pos = np.arange(len(order)) + (1 - k) * h
        ax.barh(pos, v.values, height=h * 0.86, color=MODEL_COLORS[model], label=model)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Mean absolute error as % of the app's average daily count (lower is better)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    save(fig, CHARTS_DIR / "ts_07_backtest_mae_vs_baseline.png",
         "Backtest error of ARIMA against the seasonal-naive baseline and Holt-Winters",
         "Average over all 28 forecast days and all backtest origins. Error is relative to each app's level so apps of different size can be compared. "
         "Full evaluation (MASE, residual checks, interval coverage) is in the forecast-evaluation stage.")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    series = load()
    n = len(next(iter(series.values()))["y"])
    print(f"{len(series)} apps, {n} days; backtest origins {INITIAL_TRAIN}..{n - HORIZON} every {STEP} days, horizon {HORIZON}")

    orders, candidates, bt_rows, fdaily, fweekly = [], [], [], [], []
    for app in APP_NAMES:
        s = series[app]
        order, seasonal, chosen, table = select_orders(app, s, INITIAL_TRAIN, "backtest")
        orders.append(chosen)
        candidates.append(table)
        bt_rows += backtest(app, s, order, seasonal)

        f_order, f_seasonal, f_chosen, f_table = select_orders(app, s, n, "final")
        orders.append(f_chosen)
        candidates.append(f_table)
        daily, weekly, res = final_forecast(app, s, f_order, f_seasonal)
        fdaily.append(daily)
        fweekly += weekly
        print(f"  {app:<11} backtest {order}{seasonal[:3]}  final {f_order}{f_seasonal[:3]}  ({len(f_table)} candidates)")

    orders = pd.DataFrame(orders)
    candidates = pd.concat(candidates, ignore_index=True)
    candidates["order"] = candidates["order"].astype(str)
    candidates["seasonal_order"] = candidates["seasonal_order"].astype(str)
    bt = pd.DataFrame(bt_rows)
    fdaily = pd.concat(fdaily, ignore_index=True)
    fweekly = pd.DataFrame(fweekly)
    summary = summarise_backtest(bt)

    fmt = {"date_format": "%Y-%m-%d", "index": False}
    orders.to_csv(TS_DIR / "arima_orders.csv", **fmt)
    candidates.round(3).to_csv(TS_DIR / "arima_candidates.csv", **fmt)
    bt.round({c: 2 for c in bt.select_dtypes("number")}).to_csv(TS_DIR / "backtest_forecasts.csv", **fmt)
    summary.to_csv(TS_DIR / "backtest_summary.csv", **fmt)
    fdaily.to_csv(TS_DIR / "forecast_daily.csv", **fmt)
    fweekly.to_csv(TS_DIR / "forecast_weekly.csv", **fmt)
    print(f"Saved tables to {TS_DIR.relative_to(REPO_ROOT)}/")

    print("Charts...")
    chart_forecast(series, fdaily)
    chart_backtest(series, bt)
    chart_model_comparison(summary)

    allband = summary[summary["horizon_band"] == "all (1-28 days)"].pivot(index="app_name", columns="model", values="mae")
    allband = allband.reindex(APP_NAMES)
    beats = (allband["ARIMA"] < allband["Seasonal naive"]).sum()
    out = {
        "series_modelled": "daily count of 1-2 star reviews per app (neg_reviews_filled)",
        "train_window": ["2026-04-01", "2026-09-20"], "days": n,
        "backtest": {"initial_train_days": INITIAL_TRAIN, "horizon_days": HORIZON, "step_days": STEP,
                     "origins": int(bt["origin"].nunique()), "forecasts_per_app_and_model": int(len(bt) / len(APP_NAMES) / len(MODELS)),
                     "orders_chosen_on": "first 84 days only (no future data)"},
        "final_forecast": {"from": str(fdaily["day"].min())[:10], "to": str(fdaily["day"].max())[:10], "days": HORIZON,
                           "intervals": "80% and 95%; weekly totals from 2,000 simulated paths"},
        "order_selection": "ACF/PACF of the training series propose the orders (p, q up to 3; seasonal AR/MA if significant at lag 7 or the app has a strong weekday effect); lowest AICc wins (within 2 points the fewest terms); d from the ADF test",
        "models": MODELS,
        "arima_beats_seasonal_naive_on_mae": {"apps": int(beats), "of": len(APP_NAMES),
                                              "does_not_beat": allband.index[allband["ARIMA"] >= allband["Seasonal naive"]].tolist()},
        "interval_coverage_check": {"nominal_80": round(float(cover(bt, 80)), 3), "nominal_95": round(float(cover(bt, 95)), 3),
                                    "note": "share of backtest days inside the interval, pooled over apps; the full check is in #137"},
        "large_week1_changes": {"rule": "week-1 forecast differs from the average of the last 4 weeks by more than 30%; read these forecasts with care",
                                "apps": fweekly[(fweekly["week"] == 1) & (fweekly["change_vs_last_4_weeks_pct"].abs() > 30)]
                                .set_index("app_name")["change_vs_last_4_weeks_pct"].to_dict()},
        "mae_all_horizons": allband.round(2).to_dict(orient="index"),
        "orders": orders[orders["stage"] == "final"].set_index("app_name")[["p", "d", "q", "P", "D", "Q", "season", "feed_gap_input"]]
                  .to_dict(orient="index"),
        "charts": CHART_INDEX,
    }
    SUMMARY_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print("\nBacktest MAE over all 28 days (reviews per day):")
    print(allband.round(1).to_string())
    print(f"\nARIMA beats the seasonal-naive baseline on MAE for {beats} of {len(APP_NAMES)} apps")


if __name__ == "__main__":
    main()
