"""Forecasting (#140): ARIMA forecast of daily 1-2 star reviews with prediction intervals, unusual days and release markers.

Reads the tables written by scripts/19_forecasting.py (forecast_daily.csv, forecast_weekly.csv), scripts/20_forecast_evaluation.py
(evaluation_metrics_by_app.csv, prediction_interval_coverage.csv, unusual_days_unique.csv) and scripts/21_update_impact.py
(release_impact.csv). Nothing is refitted here.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (APP_COLORS, APP_DOMAIN, DATA_DIR, INK2, apply_filters, fmt_int, get_filters, load_daily_app,
                    require_selection, shade_feed_gap, style)

TS_DIR = DATA_DIR / "timeseries"
MODEL_STYLE = {"Seasonal naive": dict(color="#8a8985", dash="dot"), "Holt-Winters": dict(color="#eb6834", dash="dash")}


@st.cache_data
def load_tables():
    t = {
        "daily": pd.read_csv(TS_DIR / "forecast_daily.csv", parse_dates=["day"]),
        "weekly": pd.read_csv(TS_DIR / "forecast_weekly.csv", parse_dates=["week_start", "week_end"]),
        "metrics": pd.read_csv(TS_DIR / "evaluation_metrics_by_app.csv"),
        "coverage": pd.read_csv(TS_DIR / "prediction_interval_coverage.csv"),
        "alerts": pd.read_csv(TS_DIR / "unusual_days_unique.csv", parse_dates=["day", "origin"]),
    }
    path = DATA_DIR / "update_impact" / "release_impact.csv"
    t["releases"] = pd.read_csv(path, parse_dates=["release_day"]) if path.exists() else pd.DataFrame()
    return t


def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


f = get_filters()
require_selection(f)
T = load_tables()

st.title("Forecasting")
st.caption("Daily number of 1–2★ reviews per app: the last weeks of actual counts, then the ARIMA forecast for 21 Sep – 18 Oct 2026 "
           "with 80% and 95% prediction intervals. Forecasts were made by the Stage 3 model; nothing is refitted here.")

app = st.selectbox("App", f.apps, index=0, key="fc_app",
                   format_func=lambda a: f"{a}  ·  {APP_DOMAIN[a]}")
color = APP_COLORS[app]

# ----------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------
w = T["weekly"][T["weekly"]["app_name"] == app].sort_values("week")
m = T["metrics"][T["metrics"]["app_name"] == app].set_index("model")
cov = T["coverage"][T["coverage"]["app_name"] == app]
al = T["alerts"][T["alerts"]["app_name"] == app]
w1 = w.iloc[0]
c = st.columns(5)
c[0].metric("Forecast, 21–27 Sep", fmt_int(w1["forecast"]), f"{w1['change_vs_last_4_weeks_pct']:+.0f}% vs last 4 weeks",
            delta_color="inverse", help="Expected number of 1–2★ reviews in the first forecast week.")
c[1].metric("80% interval (week 1)", f"{fmt_int(w1['lower_80'])} – {fmt_int(w1['upper_80'])}")
c[2].metric("Backtest error (MAE)", f"{m.loc['ARIMA', 'MAE']:.1f} a day",
            f"{m.loc['ARIMA', 'MAE'] - m.loc['Seasonal naive', 'MAE']:+.1f} vs seasonal naive", delta_color="inverse",
            help="Mean absolute error of ARIMA over the 9-origin rolling backtest, against the 'same weekday last week' baseline.")
c[3].metric("95% interval coverage", f"{cov['coverage_95_pct'].iloc[0]:.0f}%" if len(cov) else "–",
            help="Share of backtest days whose actual count fell inside the 95% interval.")
c[4].metric("Unusual days (above / below)", f"{(al['alert_direction'].str.startswith('Above')).sum()} / "
            f"{(al['alert_direction'].str.startswith('Below')).sum()}",
            help="Days whose count fell above / below the 95% interval in the rolling backtest (Stage 4).")

unreliable = []
if m.loc["ARIMA", "MAE"] >= m.loc["Seasonal naive", "MAE"]:
    unreliable.append("ARIMA did not beat the seasonal-naive baseline in the backtest")
if abs(w1["change_vs_last_4_weeks_pct"]) > 30:
    unreliable.append(f"the week-1 forecast is {w1['change_vs_last_4_weeks_pct']:+.0f}% away from the last 4 weeks")
if unreliable:
    st.warning(f"Read this forecast with care: {' and '.join(unreliable)}.", icon=":material/warning:")

# ----------------------------------------------------------------------------
# Forecast chart
# ----------------------------------------------------------------------------
left, mid, right = st.columns([2, 2, 2])
show_models = left.multiselect("Compare with", ["Seasonal naive", "Holt-Winters"], default=[], key="fc_models")
show_alerts = mid.toggle("Show unusual days", value=True, key="fc_alerts")
show_releases = right.toggle("Show releases", value=True, key="fc_releases")

hist = apply_filters(load_daily_app(), f)
hist = hist[hist["app_name"] == app].sort_values("day")
fc = T["daily"][T["daily"]["app_name"] == app]
ar = fc[fc["model"] == "ARIMA"].sort_values("day")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ar["day"], y=ar["upper_95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ar["day"], y=ar["lower_95"], fill="tonexty", fillcolor=hex_rgba(color, 0.12), line=dict(width=0),
                         name="95% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ar["day"], y=ar["upper_80"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ar["day"], y=ar["lower_80"], fill="tonexty", fillcolor=hex_rgba(color, 0.25), line=dict(width=0),
                         name="80% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist["day"], y=hist["neg_reviews_filled"], name="Actual", line=dict(color=color, width=1.6),
                         hovertemplate="%{x|%a %d %b}: %{y:,.0f} 1–2★ reviews<extra>Actual</extra>"))
fig.add_trace(go.Scatter(x=ar["day"], y=ar["forecast"], name="ARIMA forecast", line=dict(color=color, width=2.6),
                         customdata=ar[["lower_80", "upper_80", "lower_95", "upper_95"]],
                         hovertemplate="%{x|%a %d %b}: %{y:,.0f}<br>80%: %{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
                                       "<br>95%: %{customdata[2]:,.0f}–%{customdata[3]:,.0f}<extra>ARIMA</extra>"))
for model in show_models:
    mm = fc[fc["model"] == model].sort_values("day")
    fig.add_trace(go.Scatter(x=mm["day"], y=mm["forecast"], name=model, line=dict(width=1.6, **MODEL_STYLE[model]),
                             hovertemplate="%{x|%a %d %b}: %{y:,.0f}<extra>" + model + "</extra>"))

if show_alerts and len(al):
    a = al[(al["day"].dt.date >= f.start) & (al["day"].dt.date <= f.end)]
    for direction, symbol, col in [("Above", "triangle-up", "#c93a39"), ("Below", "triangle-down", "#2a78d6")]:
        s = a[a["alert_direction"].str.startswith(direction)]
        if len(s):
            fig.add_trace(go.Scatter(x=s["day"], y=s["actual"], mode="markers", name=f"Unusual day ({direction.lower()} 95%)",
                                     marker=dict(symbol=symbol, size=11, color=col, line=dict(color="white", width=1)),
                                     customdata=s[["forecast", "lower_95", "upper_95"]],
                                     hovertemplate="%{x|%a %d %b}: %{y:,.0f} actual<br>expected %{customdata[0]:,.0f} "
                                                   "(95%: %{customdata[1]:,.0f}–%{customdata[2]:,.0f})<extra></extra>"))

rel = T["releases"]
if show_releases and len(rel):
    r = rel[(rel["app_name"] == app) & (rel["release_day"].dt.date >= f.start) & (rel["release_day"].dt.date <= f.end)]
    ymax = max(hist["neg_reviews_filled"].max() if len(hist) else 0, ar["upper_95"].max())
    for _, x in r.iterrows():
        worse = str(x["verdict"]).startswith("Worse")
        fig.add_vline(x=x["release_day"], line=dict(color="#c93a39" if worse else "#b9b8b4", width=1.6 if worse else 0.8,
                                                   dash="solid" if worse else "dot"))
    fig.add_trace(go.Scatter(x=r["release_day"], y=[ymax * 1.04] * len(r), mode="markers", name="Release",
                             marker=dict(symbol="diamond", size=8,
                                         color=["#c93a39" if str(v).startswith("Worse") else "#8a8985" for v in r["verdict"]]),
                             customdata=r[["app_version", "verdict"]],
                             hovertemplate="%{x|%d %b}: version %{customdata[0]}<br>%{customdata[1]}<extra>Release</extra>"))

fig.add_vline(x=pd.Timestamp("2026-09-20"), line=dict(color=INK2, width=1, dash="dot"))
shade_feed_gap(fig, f)
fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0), hovermode="closest",
                  yaxis_title="1–2★ reviews per day", xaxis_title="")
fig.update_yaxes(rangemode="tozero")
st.plotly_chart(style(fig, 470), width="stretch", config={"displayModeBar": False})
st.caption("History follows the sidebar dates; the forecast always covers 21 Sep – 18 Oct. Dotted line = last day of data (20 Sep). "
           "Triangles = days whose count fell outside the 95% interval in the rolling backtest (an alert, not a cause). "
           "Diamonds at the top = new releases; red = a release the update-impact analysis found made things worse.")

# ----------------------------------------------------------------------------
# Unusual days next to releases
# ----------------------------------------------------------------------------
if len(al):
    with st.expander(f"Unusual days for {app} ({len(al)})"):
        tab = al.sort_values("day").copy()
        if len(rel):
            ra = rel[rel["app_name"] == app][["release_day", "app_version", "verdict"]].sort_values("release_day")
            tab = pd.merge_asof(tab.sort_values("day"), ra, left_on="day", right_on="release_day", direction="backward",
                                tolerance=pd.Timedelta(days=6))
            tab["release_in_previous_7_days"] = tab["app_version"].fillna("").astype(str) + \
                tab["verdict"].map(lambda v: f" ({v})" if isinstance(v, str) else "")
        show = tab[["day", "alert_direction", "actual", "forecast", "lower_95", "upper_95"] +
                   (["release_in_previous_7_days"] if "release_in_previous_7_days" in tab else [])]
        st.dataframe(show.rename(columns={"day": "Day", "alert_direction": "Alert", "actual": "Actual", "forecast": "Expected",
                                          "lower_95": "95% low", "upper_95": "95% high",
                                          "release_in_previous_7_days": "Release in the 7 days before"}),
                     hide_index=True, width="stretch",
                     column_config={"Day": st.column_config.DateColumn(format="ddd D MMM"),
                                    "Expected": st.column_config.NumberColumn(format="%.0f"),
                                    "95% low": st.column_config.NumberColumn(format="%.0f"),
                                    "95% high": st.column_config.NumberColumn(format="%.0f")})

# ----------------------------------------------------------------------------
# All selected apps: weekly forecast and model comparison
# ----------------------------------------------------------------------------
st.subheader("Weekly forecast for the selected apps")
wk = T["weekly"][T["weekly"]["app_name"].isin(f.apps)]
piv = wk.pivot(index="app_name", columns="week", values="forecast").reindex([a for a in f.apps if a in set(wk["app_name"])])
piv.columns = [f"Week {c} ({wk[wk['week'] == c]['week_start'].iloc[0]:%d %b})" for c in piv.columns]
last4 = wk.drop_duplicates("app_name").set_index("app_name")["last_4_weeks_avg"]
chg = wk[wk["week"] == 1].set_index("app_name")["change_vs_last_4_weeks_pct"]
piv.insert(0, "Last 4 weeks (avg)", last4)
piv["Week 1 vs last 4 weeks"] = chg.map("{:+.0f}%".format)
st.dataframe(piv.reset_index().rename(columns={"app_name": "App"}), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.0f") for c in piv.columns if c != "Week 1 vs last 4 weeks"})

mt = T["metrics"][T["metrics"]["app_name"].isin(f.apps)]
fig = go.Figure()
for model, col in [("ARIMA", "#2a78d6"), ("Holt-Winters", "#eb6834"), ("Seasonal naive", "#8a8985")]:
    s = mt[mt["model"] == model].set_index("app_name").reindex([a for a in f.apps if a in set(mt["app_name"])])
    fig.add_trace(go.Bar(x=s.index, y=s["MAE_vs_seasonal_naive"], name=model, marker_color=col,
                         customdata=s[["MAE", "MASE"]],
                         hovertemplate="%{x}: %{y:.2f} × the baseline error (MAE %{customdata[0]:.1f}, MASE %{customdata[1]:.2f})"
                                       "<extra>" + model + "</extra>"))
fig.add_hline(y=1, line=dict(color=INK2, width=1, dash="dot"))
fig.update_layout(barmode="group", title="Backtest error relative to 'same weekday last week' (below 1 = better than the baseline)",
                  legend=dict(orientation="h", y=-0.2, x=0), yaxis_title="MAE ÷ seasonal-naive MAE")
st.plotly_chart(style(fig, 360), width="stretch", config={"displayModeBar": False})
