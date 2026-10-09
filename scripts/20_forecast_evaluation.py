"""Evaluate rolling-origin forecasts and flag unusual negative-review days."""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "data" / "timeseries"
CHARTS = ROOT / "data" / "charts" / "timeseries"
GAP_APPS = {"Amazon", "Blinkit", "Domino's", "Flipkart", "Swiggy"}
SEASON = 7


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def main():
    forecasts = pd.read_csv(TS / "backtest_forecasts.csv", parse_dates=["day", "origin"])
    daily = pd.read_csv(TS / "daily_app.csv", parse_dates=["day"])
    orders = pd.read_csv(TS / "arima_orders.csv")
    forecasts["error"] = forecasts["actual"] - forecasts["forecast"]

    # MASE scaling uses the first 84 days, matching the initial backtest training window.
    scale = {}
    for app, group in daily.groupby("app_name"):
        y = (group.sort_values("day")["neg_reviews_filled"]
             .astype(float).iloc[:84].to_numpy())
        scale[app] = float(np.mean(np.abs(y[SEASON:] - y[:-SEASON])))

    metric_rows = []
    for (app, model), g in forecasts.groupby(["app_name", "model"]):
        denom = scale[app]
        metric_rows.append({
            "app_name": app,
            "model": model,
            "forecast_rows": len(g),
            "origins": g["origin"].nunique(),
            "MAE": mae(g["actual"], g["forecast"]),
            "RMSE": rmse(g["actual"], g["forecast"]),
            "MASE": mae(g["actual"], g["forecast"]) / denom if denom else np.nan,
            "seasonal_naive_scale": denom,
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(TS / "evaluation_metrics_by_app.csv", index=False)

    # Horizon-band metrics support checking whether performance worsens further out.
    bands = [(1, 7, "1-7 days"), (8, 14, "8-14 days"), (15, 28, "15-28 days")]
    band_rows = []
    for (app, model), g in forecasts.groupby(["app_name", "model"]):
        for lo, hi, label in bands:
            h = g[g["horizon"].between(lo, hi)]
            if h.empty:
                continue
            band_rows.append({
                "app_name": app, "model": model, "horizon_band": label,
                "forecast_rows": len(h), "MAE": mae(h.actual, h.forecast),
                "RMSE": rmse(h.actual, h.forecast),
                "MASE": mae(h.actual, h.forecast) / scale[app] if scale[app] else np.nan,
            })
    pd.DataFrame(band_rows).to_csv(TS / "evaluation_metrics_by_horizon.csv", index=False)

    # Prediction interval coverage: only ARIMA rows have intervals.
    arima = forecasts[forecasts["model"].eq("ARIMA")].copy()
    coverage_rows = []
    for app, g in list(arima.groupby("app_name")) + [("ALL_APPS", arima)]:
        row = {"app_name": app, "forecast_rows": len(g)}
        for level in (80, 95):
            lower, upper = g[f"lower_{level}"], g[f"upper_{level}"]
            valid = lower.notna() & upper.notna() & g["actual"].notna()
            row[f"coverage_{level}_pct"] = (
                100 * ((g.loc[valid, "actual"] >= lower[valid]) &
                       (g.loc[valid, "actual"] <= upper[valid])).mean()
                if valid.any() else np.nan
            )
            row[f"interval_rows_{level}"] = int(valid.sum())
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(TS / "prediction_interval_coverage.csv", index=False)

    # Alert at forecast-row level, preserving origin and horizon because windows overlap.
    arima["alert_direction"] = np.select(
        [arima["actual"] < arima["lower_95"], arima["actual"] > arima["upper_95"]],
        ["Below 95% interval", "Above 95% interval"],
        default="",
    )
    alerts = arima[arima["alert_direction"].ne("")].copy()
    alerts["absolute_error"] = (alerts["actual"] - alerts["forecast"]).abs()
    alerts = alerts[[
        "app_name", "day", "origin", "horizon", "actual", "forecast",
        "lower_95", "upper_95", "alert_direction", "absolute_error"
    ]].sort_values(["day", "app_name", "origin"])
    alerts.to_csv(TS / "unusual_day_alerts.csv", index=False)

    # Deduplicate alerts by app and calendar day. Keep the shortest-horizon
    # flagged forecast as the representative alert, and retain alert frequency.
    if not alerts.empty:
        alert_counts = (
            alerts.groupby(["app_name", "day"])
            .size().rename("flagged_forecast_rows").reset_index()
        )
        unique_alerts = (
            alerts.sort_values(["app_name", "day", "horizon"])
            .drop_duplicates(["app_name", "day"])
            .merge(alert_counts, on=["app_name", "day"], how="left")
            .sort_values(["day", "app_name"])
        )
    else:
        unique_alerts = alerts.copy()
        unique_alerts["flagged_forecast_rows"] = pd.Series(dtype=int)
    unique_alerts.to_csv(TS / "unusual_days_unique.csv", index=False)

    # Residual diagnostics from each app's saved FINAL ARIMA/SARIMAX order.
    # These are in-sample fitted-model residuals, not overlapping backtest errors.
    residual_rows = []
    residual_acf_rows = []
    final_orders = orders[orders["stage"].eq("final")]
    for _, order_row in final_orders.iterrows():
        app = order_row["app_name"]
        d = daily[daily["app_name"].eq(app)].sort_values("day").set_index("day").asfreq("D")
        y = d["neg_reviews_filled"].astype(float)
        exog = d[["feed_gap"]].astype(float) if app in GAP_APPS else None
        order = (int(order_row.p), int(order_row.d), int(order_row.q))
        seasonal = (int(order_row.P), int(order_row.D), int(order_row.Q), int(order_row.season))
        trend = "c" if order[1] == 0 else "n"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    y, exog=exog, order=order, seasonal_order=seasonal, trend=trend
                ).fit(disp=False, maxiter=300)
            resid = pd.Series(fit.resid).replace([np.inf, -np.inf], np.nan).dropna()
            lag = min(14, max(1, len(resid) // 5))
            lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
            residual_rows.append({
                "app_name": app, "order": str(order), "seasonal_order": str(seasonal),
                "residual_count": len(resid), "ljung_box_lag": lag,
                "ljung_box_statistic": float(lb["lb_stat"].iloc[-1]),
                "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[-1]),
                "autocorrelation_flag_p_lt_0_05": bool(lb["lb_pvalue"].iloc[-1] < 0.05),
            })
            centered = resid - resid.mean()
            acf_vals = [1.0] + [
                float(centered.autocorr(lag=k)) for k in range(1, min(28, len(centered) - 1) + 1)
            ]
            for lag_no, value in enumerate(acf_vals):
                residual_acf_rows.append({"app_name": app, "lag": lag_no, "acf": value})
        except Exception as exc:
            residual_rows.append({
                "app_name": app, "order": str(order), "seasonal_order": str(seasonal),
                "residual_count": 0, "ljung_box_lag": np.nan,
                "ljung_box_statistic": np.nan, "ljung_box_pvalue": np.nan,
                "autocorrelation_flag_p_lt_0_05": np.nan, "error": str(exc),
            })

    residuals = pd.DataFrame(residual_rows)
    residuals.to_csv(TS / "residual_diagnostics.csv", index=False)
    acf_df = pd.DataFrame(residual_acf_rows)
    acf_df.to_csv(TS / "residual_acf.csv", index=False)

    CHARTS.mkdir(parents=True, exist_ok=True)
    overall = metrics.groupby("model")[["MAE", "RMSE", "MASE"]].mean().sort_values("MAE")
    fig, ax = plt.subplots(figsize=(9, 5))
    overall["MAE"].plot(kind="bar", ax=ax)
    ax.set_title("Mean absolute error by forecasting model")
    ax.set_ylabel("MAE (negative reviews)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(CHARTS / "ts_08_evaluation_mae.png", dpi=150)
    plt.close(fig)

    all_coverage = coverage[coverage["app_name"].eq("ALL_APPS")].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["80% interval", "95% interval"]
    values = [all_coverage["coverage_80_pct"], all_coverage["coverage_95_pct"]]
    ax.bar(labels, values)
    ax.axhline(80, linestyle="--", linewidth=1, label="Nominal 80%")
    ax.axhline(95, linestyle=":", linewidth=1, label="Nominal 95%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Observed coverage (%)")
    ax.set_title("ARIMA prediction interval coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "ts_09_interval_coverage.png", dpi=150)
    plt.close(fig)

    if not acf_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for app, g in acf_df[acf_df["lag"].between(1, 28)].groupby("app_name"):
            ax.plot(g["lag"], g["acf"], marker=".", linewidth=1, label=app)
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Residual autocorrelation")
        ax.set_title("Final ARIMA/SARIMAX residual ACF by app")
        ax.legend(ncol=3, fontsize=7)
        fig.tight_layout()
        fig.savefig(CHARTS / "ts_10_residual_acf.png", dpi=150)
        plt.close(fig)

    print("Evaluation complete.")
    print("Metrics:", TS / "evaluation_metrics_by_app.csv")
    print("Horizon metrics:", TS / "evaluation_metrics_by_horizon.csv")
    print("Interval coverage:", TS / "prediction_interval_coverage.csv")
    print("Unusual-day alerts:", TS / "unusual_day_alerts.csv", f"({len(alerts)} forecast rows)")
    print("Residual diagnostics:", TS / "residual_diagnostics.csv")
    print("Charts:", CHARTS / "ts_08_evaluation_mae.png, ts_09_interval_coverage.png, ts_10_residual_acf.png")
    print("\nOverall interval coverage:")
    print(coverage[coverage["app_name"].eq("ALL_APPS")].to_string(index=False))
    print("\nResidual diagnostics:")
    print(residuals.to_string(index=False))


if __name__ == "__main__":
    main()
