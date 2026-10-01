"""
Plotly figure builders for the dashboard (Person 5, issue #112).

Every function takes already-filtered data and returns a go.Figure. All charts share one layout,
the project's app palette, and hover tooltips that show exact values and the number of reviews behind them.
"""
import pandas as pd
import plotly.graph_objects as go

from utils import (APP_COLORS, APPS, GRID, INK, INK2, METRICS, MODEL_LABELS, MUTED, OVERALL, PALETTE,
                   RATING_BANDS, SENTIMENT_COLORS, SENTIMENT_ORDER, SEQ_SCALE, SERIES_COLORS, STAR_COLORS)

FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=INK)
HALF_WEEK = pd.Timedelta(days=3.5)


def base_layout(fig, height=380, y_title=None, x_title=None, legend=True, hovermode="x unified"):
    fig.update_layout(
        template="plotly_white", height=height, font=FONT, hovermode=hovermode,
        margin=dict(l=10, r=10, t=40 if legend else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font=dict(family=FONT["family"], size=12)),
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False)
    return fig


def pad_for_labels(fig, values, axis="x", share=0.16):
    """Extend a bar chart's value axis so labels drawn outside the bars are not clipped."""
    lo, hi = min(0.0, float(min(values))), max(0.0, float(max(values)))
    span = (hi - lo) or 1.0
    rng = [lo - (span * share if lo < 0 else 0), hi + (span * share if hi > 0 else 0)]
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(range=rng)


def empty_figure(message, height=300):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=MUTED), xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base_layout(fig, height=height, legend=False)


def mark_date(fig, date, label):
    """Dashed vertical reference line with a label (used for the July-2026 sampling shift)."""
    if date is None:
        return
    x = pd.Timestamp(date) - HALF_WEEK
    xr = fig.layout.xaxis.range
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=INK2, width=1, dash="dash"))
    fig.add_annotation(x=x, y=1, yref="paper", text=label, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color=INK2), xshift=4)
    if xr is not None:
        fig.update_xaxes(range=xr)


def _in_range(rows, date):
    return date is not None and len(rows) and rows["week_start"].min() <= pd.Timestamp(date) <= rows["week_start"].max() + pd.Timedelta(days=7)


# ----------------------------------------------------------------------------
# Weekly trends
# ----------------------------------------------------------------------------
def trend_figure(rows, metric, regime_week=None, height=420):
    """One line per series for the chosen weekly metric. Partial weeks are drawn as hollow markers."""
    spec = METRICS[metric]
    fig = go.Figure()
    for name in [s for s in APPS + [OVERALL] if s in set(rows["series"])]:
        g = rows[rows["series"] == name].sort_values("week_start")
        partial = g["partial_week"].to_numpy()
        fig.add_trace(go.Scatter(
            x=g["week_start"], y=g[metric], name=name, mode="lines+markers", connectgaps=False,
            line=dict(color=SERIES_COLORS[name], width=2.2, dash="dot" if name == OVERALL and len(set(rows["series"])) > 1 else "solid"),
            marker=dict(size=6, color=[("#ffffff" if p else SERIES_COLORS[name]) for p in partial],
                        line=dict(color=SERIES_COLORS[name], width=1.5)),
            customdata=g[["review_count", "week_end"]].to_numpy(),
            hovertemplate=f"<b>{name}</b>: %{{y:{spec['fmt']}}}{spec['suffix']} · %{{customdata[0]:,}} reviews<extra></extra>",
        ))
    base_layout(fig, height=height, y_title=spec["axis"])
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    if metric == "review_count":
        fig.update_yaxes(rangemode="tozero")
    if _in_range(rows, regime_week):
        mark_date(fig, regime_week, "sampling shift (Jul 2026)")
    return fig


def rating_mix_figure(rows, height=340):
    """Stacked weekly shares of 1–2★ / 3★ / 4–5★ reviews for a single series."""
    g = rows.sort_values("week_start")
    fig = go.Figure()
    for col, (label, color) in RATING_BANDS.items():
        fig.add_trace(go.Bar(x=g["week_start"], y=g[col], name=label, marker=dict(color=color, line=dict(color="#ffffff", width=1)),
                             customdata=g[["review_count"]].to_numpy(),
                             hovertemplate=f"<b>{label}</b>: %{{y:.1f}}% of %{{customdata[0]:,}} reviews<extra></extra>"))
    base_layout(fig, height=height, y_title="% of the week's reviews")
    fig.update_layout(barmode="stack", bargap=0.15, legend_traceorder="normal")
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    if 0 < len(g) <= 30:   # label bars with their own week-start dates
        fig.update_xaxes(tick0=g["week_start"].iloc[0], dtick=14 * 86_400_000, tickformat="%d %b")
    fig.update_yaxes(range=[0, 100])
    return fig


def mini_trend(rows, metric, color, height=230):
    """Compact single-series line for the app profile."""
    spec = METRICS[metric]
    g = rows.sort_values("week_start")
    fig = go.Figure(go.Scatter(
        x=g["week_start"], y=g[metric], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5),
        customdata=g[["review_count"]].to_numpy(), connectgaps=False,
        hovertemplate=f"%{{y:{spec['fmt']}}}{spec['suffix']} · %{{customdata[0]:,}} reviews<extra></extra>"))
    base_layout(fig, height=height, y_title=spec["axis"], legend=False)
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    if metric == "review_count":
        fig.update_yaxes(rangemode="tozero")
    return fig


def app_comparison_bar(values, metric, selected, height=320):
    """Selected-period value of one metric for each app; the selected app is drawn solid, the rest faded."""
    spec = METRICS[metric]
    v = values.dropna(subset=["value"])
    opacity = [1.0 if selected in (OVERALL, a) else 0.35 for a in v["app_name"]]
    fig = go.Figure(go.Bar(
        x=v["app_name"], y=v["value"], marker=dict(color=[APP_COLORS[a] for a in v["app_name"]], opacity=opacity),
        text=[f"{x:{spec['fmt']}}{spec['suffix']}" for x in v["value"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK), customdata=v[["n"]].to_numpy(),
        hovertemplate=f"<b>%{{x}}</b><br>%{{y:{spec['fmt']}}}{spec['suffix']}<br>%{{customdata[0]:,}} reviews in period<extra></extra>"))
    axis = "Reviews in period" if metric == "review_count" else spec["axis"]
    base_layout(fig, height=height, y_title=axis, legend=False, hovermode="closest")
    if len(v):
        pad_for_labels(fig, v["value"], axis="y")
    return fig


# ----------------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------------
def forecast_figure(hist, fc, bt, metric, series, model, horizon, height=460, legend=True):
    """History, hold-out forecasts of the chosen model/horizon, and the forward forecast with its 80% interval."""
    spec, color = METRICS[metric], SERIES_COLORS[series]
    tip = f"%{{y:{spec['fmt']}}}{spec['suffix']}"
    fig = go.Figure()
    if len(fc):
        fig.add_trace(go.Scatter(x=fc["week_start"], y=fc["pi80_upper"], mode="lines", line=dict(width=0), showlegend=False,
                                 hovertemplate=f"80% interval upper: {tip}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc["week_start"], y=fc["pi80_lower"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(color, 0.18), name="80% prediction interval",
                                 hovertemplate=f"80% interval lower: {tip}<extra></extra>"))
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist[metric], mode="lines+markers", name="Observed week",
                             line=dict(color=color, width=2.2), marker=dict(size=6), customdata=hist[["review_count"]].to_numpy(),
                             hovertemplate=f"Observed: {tip} · %{{customdata[0]:,}} reviews<extra></extra>"))
    if len(bt):
        fig.add_trace(go.Scatter(x=bt["target_week"], y=bt["forecast"], mode="lines+markers",
                                 name=f"Hold-out forecast ({MODEL_LABELS.get(model, model)}, {horizon} wk ahead)",
                                 line=dict(color=INK, width=1.2, dash="dot"), marker=dict(size=8, color="#ffffff", line=dict(color=INK, width=1.5)),
                                 customdata=bt[["error"]].to_numpy(),
                                 hovertemplate=f"Hold-out forecast: {tip} (error %{{customdata[0]:+.2f}})<extra></extra>"))
        fig.add_vrect(x0=bt["target_week"].min() - HALF_WEEK, x1=bt["target_week"].max() + HALF_WEEK, fillcolor=GRID, opacity=0.45,
                      line_width=0, layer="below")
        fig.add_annotation(x=bt["target_week"].min() - HALF_WEEK, y=0, yref="paper", text="hold-out weeks", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=INK2), xshift=4)
    if len(fc) and len(hist):
        xs = [hist["week_start"].iloc[-1]] + list(fc["week_start"])
        ys = [hist[metric].iloc[-1]] + list(fc["forecast"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Forecast", line=dict(color=color, width=2.2, dash="dash"),
                                 marker=dict(size=[0] + [6] * len(fc), color=color), hovertemplate=f"Forecast: {tip}<extra></extra>"))
    base_layout(fig, height=height, y_title=spec["axis"], legend=legend)
    if legend:   # extra headroom: this legend has long entries
        fig.update_layout(margin_t=70, legend_y=1.06)
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    return fig


def accuracy_bar(acc, measure, metric, final_model="ses", height=320):
    """Hold-out error of each model for one series and metric, with the sampling-noise floor as a reference line."""
    spec = METRICS[metric]
    a = acc.sort_values(measure)
    unit = "★" if metric == "mean_rating" else " pp"
    colors = [PALETTE[0] if m == final_model else "#b9c4d2" for m in a["model"]]
    fig = go.Figure(go.Bar(
        y=a["Model"], x=a[measure], orientation="h", marker=dict(color=colors),
        text=[f"{v:.2f}{unit}" for v in a[measure]], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        customdata=a[["bias_h1_4", "skill_vs_naive", "n_forecasts"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>Error: %{x:.2f}" + unit + "<br>Bias: %{customdata[0]:+.2f}" + unit
                      + "<br>Skill vs naive: %{customdata[1]:+.2f}<br>Forecasts scored: %{customdata[2]}<extra></extra>"))
    floor = a["sampling_noise_floor"].iloc[0] if len(a) else None
    if floor is not None and pd.notna(floor):
        fig.add_shape(type="line", x0=floor, x1=floor, y0=0, y1=1, yref="paper", line=dict(color=INK2, width=1, dash="dash"))
        fig.add_annotation(x=floor, y=1, yref="paper", text=f"sampling-noise floor {floor:.2f}", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=INK2), xshift=4)
    base_layout(fig, height=height, x_title=f"Error ({spec['axis']}, {unit.strip()})", legend=False, hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    if len(a):
        pad_for_labels(fig, a[measure])
    fig.update_layout(margin_t=30)
    return fig


def accuracy_heatmap(accuracy, metric, measure, height=330):
    """All series x all models for one metric: which model forecasts which series best."""
    a = accuracy[accuracy["metric"] == metric]
    order = [s for s in APPS + [OVERALL] if s in set(a["series"])]
    m = a.pivot(index="series", columns="model", values=measure).reindex(order)[[c for c in MODEL_LABELS if c in set(a["model"])]]
    unit = "★" if metric == "mean_rating" else " pp"
    fig = go.Figure(go.Heatmap(
        z=m.to_numpy(), x=[MODEL_LABELS[c] for c in m.columns], y=m.index, colorscale=SEQ_SCALE, xgap=3, ygap=3,
        text=m.round(2).astype(str).to_numpy(), texttemplate="%{text}", colorbar=dict(title=unit.strip(), thickness=12),
        hovertemplate="<b>%{y}</b> · %{x}<br>Error: %{z:.2f}" + unit + "<extra></extra>"))
    base_layout(fig, height=height, legend=False, hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------------------------------------------------------
# Sentiment & issues
# ----------------------------------------------------------------------------
def sentiment_figure(dist, height=320):
    """Stacked 100% bars: Negative / Neutral / Positive share per group."""
    fig = go.Figure()
    groups = list(dict.fromkeys(dist["group"]))
    for s in SENTIMENT_ORDER:
        d = dist[dist["sentiment"] == s].set_index("group").reindex(groups)
        fig.add_trace(go.Bar(y=groups, x=d["pct"], name=s, orientation="h",
                             marker=dict(color=SENTIMENT_COLORS[s], line=dict(color="#ffffff", width=2)),
                             text=[f"{v:.0f}%" if v >= 8 else "" for v in d["pct"]], textposition="inside", insidetextanchor="middle",
                             customdata=d[["n", "total"]].to_numpy(),
                             hovertemplate=f"<b>%{{y}}</b> · {s}<br>%{{x:.1f}}% (%{{customdata[0]:,}} of %{{customdata[1]:,}} reviews)<extra></extra>"))
    base_layout(fig, height=height, x_title="% of reviews", hovermode="closest")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(range=[0, 100], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def rating_figure(dist, height=320):
    fig = go.Figure(go.Bar(
        x=[f"{s}★" for s in dist["score"]], y=dist["pct"], marker=dict(color=[STAR_COLORS[s] for s in dist["score"]]),
        text=[f"{v:.1f}%" for v in dist["pct"]], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        customdata=dist[["n"]].to_numpy(), hovertemplate="<b>%{x}</b><br>%{y:.1f}% (%{customdata[0]:,} reviews)<extra></extra>"))
    base_layout(fig, height=height, y_title="% of reviews", legend=False, hovermode="closest")
    pad_for_labels(fig, dist["pct"], axis="y")
    return fig


ISSUE_MEASURES = {
    "pct": ("% of reviews carrying the tag", ".1f", "%"),
    "mean_rating": ("Average rating of tagged reviews", ".2f", "★"),
    "mean_sentiment": ("Mean VADER sentiment of tagged reviews", "+.3f", ""),
}


def issue_bar(table, measure, name, reference=None, height=400):
    """Issue categories ranked by the chosen measure. `reference` adds all-app markers when one app is selected."""
    label, fmt, suffix = ISSUE_MEASURES[measure]
    t = table.dropna(subset=[measure]).sort_values(measure, ascending=measure != "pct")
    fig = go.Figure(go.Bar(
        y=t["label"], x=t[measure], orientation="h", name=name, marker=dict(color=SERIES_COLORS.get(name, PALETTE[0])),
        text=[f"{v:{fmt}}{suffix}" for v in t[measure]], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        customdata=t[["n", "pct", "mean_rating", "mean_sentiment"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} reviews (%{customdata[1]:.1f}%)<br>Average rating %{customdata[2]:.2f}★"
                      "<br>Mean sentiment %{customdata[3]:+.3f}<extra>" + name + "</extra>"))
    if reference is not None:
        r = reference.set_index("label").reindex(t["label"])
        fig.add_trace(go.Scatter(y=r.index, x=r[measure], mode="markers", name=OVERALL,
                                 marker=dict(symbol="line-ns", size=16, line=dict(color=INK, width=2.5)),
                                 hovertemplate=f"<b>%{{y}}</b><br>{OVERALL}: %{{x:{fmt}}}{suffix}<extra></extra>"))
    base_layout(fig, height=height, x_title=label, legend=reference is not None, hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    if len(t):
        values = list(t[measure]) + (list(reference[measure].dropna()) if reference is not None else [])
        pad_for_labels(fig, values)
    return fig


def issue_heatmap(matrix, counts, height=330):
    """App x issue: % of each app's reviews carrying the tag."""
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=[f"{a} (n={counts[a]:,})" for a in matrix.index], colorscale=SEQ_SCALE,
        xgap=3, ygap=3, zmin=0, text=matrix.round(0).astype(int).astype(str).to_numpy(), texttemplate="%{text}",
        colorbar=dict(title="%", thickness=12), hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}% of reviews<extra></extra>"))
    base_layout(fig, height=height, legend=False, hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(margin_b=110)
    return fig


def issue_trend_figure(rows, issues, labels, regime_week=None, height=360):
    """Weekly share of reviews carrying each selected issue tag, for one series."""
    g = rows.sort_values("week_start")
    fig = go.Figure()
    for k, issue in enumerate(issues):
        fig.add_trace(go.Scatter(
            x=g["week_start"], y=g[f"pct_{issue}"], name=labels[issue], mode="lines+markers", connectgaps=False,
            line=dict(color=PALETTE[k % len(PALETTE)], width=2), marker=dict(size=5),
            customdata=g[[f"n_{issue}", "review_count"]].to_numpy(),
            hovertemplate=f"<b>{labels[issue]}</b>: %{{y:.1f}}% (%{{customdata[0]}} of %{{customdata[1]:,}})<extra></extra>"))
    base_layout(fig, height=height, y_title="% of the week's reviews")
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    fig.update_yaxes(rangemode="tozero")
    if _in_range(g, regime_week):
        mark_date(fig, regime_week, "sampling shift (Jul 2026)")
    return fig


def sentiment_by_star_figure(table, height=320):
    t = table.dropna(subset=["mean_compound"])
    fig = go.Figure(go.Bar(
        x=[f"{int(s)}★" for s in t["score"]], y=t["mean_compound"], marker=dict(color=[STAR_COLORS[int(s)] for s in t["score"]]),
        text=[f"{v:+.2f}" for v in t["mean_compound"]], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        customdata=t[["n"]].to_numpy(), hovertemplate="<b>%{x}</b><br>Mean compound %{y:+.3f}<br>%{customdata[0]:,} reviews<extra></extra>"))
    base_layout(fig, height=height, y_title="Mean VADER compound score", legend=False, hovermode="closest")
    if len(t):
        pad_for_labels(fig, t["mean_compound"], axis="y")
    fig.update_yaxes(zeroline=True, zerolinecolor=INK2, zerolinewidth=1)
    return fig


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

