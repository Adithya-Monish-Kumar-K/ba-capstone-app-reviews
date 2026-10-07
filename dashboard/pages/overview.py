"""Overview: review volume, share of 1-2 star reviews and the top issues per app, for the sidebar selection."""
import pandas as pd
import plotly.express as px
import streamlit as st

from common import (APP_COLORS, DOMAIN_COLORS, ISSUE_COLS, ISSUE_LABELS, apply_filters, fmt_int, get_filters, load_daily_app,
                    require_selection, shade_feed_gap, style)

f = get_filters()
require_selection(f)
daily = apply_filters(load_daily_app(), f)
if daily.empty:
    st.warning("No days left for this selection.")
    st.stop()

st.title("Overview")
st.caption(f"{len(f.apps)} app(s) · {f.start:%d %b} – {f.end:%d %b %Y}"
           + (" · feed-gap days excluded" if f.exclude_feed_gap else ""))

# ----------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------
days = daily["day"].nunique()
reviews, neg = daily["reviews"].sum(), daily["neg_reviews"].sum()
rating = (daily["mean_rating"] * daily["reviews"]).sum() / reviews
c = st.columns(5)
c[0].metric("Reviews", fmt_int(reviews))
c[1].metric("1–2★ reviews", fmt_int(neg))
c[2].metric("Share rated 1–2★", f"{neg / reviews:.1%}")
c[3].metric("Average rating", f"{rating:.2f}★")
c[4].metric("1–2★ reviews per day", fmt_int(neg / days))

# ----------------------------------------------------------------------------
# Over time
# ----------------------------------------------------------------------------
st.subheader("Over time")
METRICS = {
    "1–2★ reviews per day": ("neg_reviews", "count"),
    "Share rated 1–2★": ("neg_share", "share"),
    "Average rating": ("mean_rating", "mean"),
    "All reviews per day": ("reviews", "count"),
}
left, right = st.columns([3, 1])
metric = left.segmented_control("Measure", list(METRICS), default="1–2★ reviews per day", key="ov_metric")
metric = metric or "1–2★ reviews per day"
smooth = right.toggle("7-day average", value=True, key="ov_smooth")

col, kind = METRICS[metric]
d = daily.sort_values("day").copy()
if smooth:
    if kind == "count":
        d["value"] = d.groupby("app_name", observed=True)[col].transform(lambda s: s.rolling(7, center=True, min_periods=4).mean())
    else:   # shares and ratings: ratio of 7-day sums, not a mean of daily ratios
        num = d["neg_reviews"] if kind == "share" else d["mean_rating"] * d["reviews"]
        g = d.assign(num=num).groupby("app_name", observed=True)
        roll = lambda s: s.rolling(7, center=True, min_periods=4).sum()  # noqa: E731
        d["value"] = g["num"].transform(roll) / g["reviews"].transform(roll)
else:
    d["value"] = d[col]

# One chart per domain: each domain reuses the same four app colours, so a shared legend would be ambiguous.
doms = [x for x in d["domain"].cat.categories if x in set(d["domain"])]
cols = st.columns(len(doms))
ymax = d["value"].max() * 1.05
for col_, dom in zip(cols, doms):
    dd = d[d["domain"] == dom]
    fig = px.line(dd, x="day", y="value", color="app_name", color_discrete_map=APP_COLORS,
                  labels={"value": metric, "day": "", "app_name": ""}, title=dom)
    fig.update_traces(line_width=2, hovertemplate="%{x|%d %b}: %{y:,.2f}<extra>%{fullData.name}</extra>")
    fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0))
    fig.update_yaxes(range=[0 if kind != "mean" else None, ymax] if kind != "mean" else None,
                     tickformat=".0%" if kind == "share" else None, title_text=metric if dom == doms[0] else "")
    shade_feed_gap(fig, f)
    col_.plotly_chart(style(fig, 380), width="stretch", config={"displayModeBar": False})
if kind == "share" and not f.exclude_feed_gap:
    st.caption("Shares jump during the feed gap because positive reviews are missing, not because complaints rose. "
               "Turn on *Exclude the feed gap* in the sidebar, or compare counts instead.")

# ----------------------------------------------------------------------------
# By app
# ----------------------------------------------------------------------------
per_app = (daily.assign(w_rating=daily["mean_rating"] * daily["reviews"])
           .groupby(["app_name", "domain"], observed=True)
           .agg(reviews=("reviews", "sum"), neg_reviews=("neg_reviews", "sum"), w_rating=("w_rating", "sum"),
                **{c: (c, "sum") for c in ISSUE_COLS})
           .reset_index())
per_app["share_neg"] = per_app["neg_reviews"] / per_app["reviews"]
per_app["mean_rating"] = per_app.pop("w_rating") / per_app["reviews"]

st.subheader("By app")
a, b = st.columns(2)
s = per_app.sort_values("share_neg")
fig = px.bar(s, x="share_neg", y="app_name", color="domain", orientation="h", color_discrete_map=DOMAIN_COLORS,
             text=s["share_neg"].map("{:.0%}".format), labels={"share_neg": "Share rated 1–2★", "app_name": ""})
fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>")
fig.update_layout(title="Share of reviews rated 1–2★", legend=dict(orientation="h", y=-0.2, x=0))
fig.update_yaxes(categoryorder="array", categoryarray=s["app_name"].tolist())
fig.update_xaxes(tickformat=".0%", range=[0, s["share_neg"].max() * 1.18])
a.plotly_chart(style(fig, 400), width="stretch", config={"displayModeBar": False})

s = per_app.sort_values("neg_reviews")
fig = px.bar(s, x="neg_reviews", y="app_name", color="domain", orientation="h", color_discrete_map=DOMAIN_COLORS,
             labels={"neg_reviews": "1–2★ reviews", "app_name": ""})
fig.update_traces(hovertemplate="%{y}: %{x:,}<extra></extra>")
fig.update_layout(title="Number of 1–2★ reviews", legend=dict(orientation="h", y=-0.2, x=0))
fig.update_yaxes(categoryorder="array", categoryarray=s["app_name"].tolist())
b.plotly_chart(style(fig, 400), width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# Top issues
# ----------------------------------------------------------------------------
st.subheader("Top issues per app")
rates = per_app.set_index("app_name")[ISSUE_COLS].div(per_app.set_index("app_name")["reviews"], axis=0) * 100
rates.columns = [ISSUE_LABELS[c].replace(" & ", " &<br>").replace(" / ", " /<br>").replace("Customer ", "Customer<br>").replace("Delivery ", "Delivery<br>").replace("Order ", "Order<br>") for c in ISSUE_COLS]
rates = rates[rates.mean().sort_values(ascending=False).index]
fig = px.imshow(rates.round(2), text_auto=".1f", aspect="auto", color_continuous_scale=["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"],
                labels={"color": "% of reviews", "x": "", "y": ""})
fig.update_traces(hovertemplate="%{y} · %{x}: %{z:.2f}% of reviews<extra></extra>")
fig.update_xaxes(side="top", tickangle=0)
st.plotly_chart(style(fig, 110 + 34 * len(rates)), width="stretch", config={"displayModeBar": False})
st.caption("% of each app's reviews that mention the issue (keyword tags; a review can have several). "
           "Most reviews are too short to name an issue, so these are lower bounds.")

# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------
table = per_app[["app_name", "domain", "reviews", "neg_reviews", "share_neg", "mean_rating"]].copy()
table["top_issue"] = rates.idxmax(axis=1).str.replace("<br>", " ").reindex(table["app_name"]).values
table = table.rename(columns={"app_name": "App", "domain": "Domain", "reviews": "Reviews", "neg_reviews": "1–2★ reviews",
                              "share_neg": "Share 1–2★", "mean_rating": "Average rating", "top_issue": "Most mentioned issue"})
with st.expander("Table by app"):
    st.dataframe(table.style.format({"Reviews": "{:,}", "1–2★ reviews": "{:,}", "Share 1–2★": "{:.1%}",
                                     "Average rating": "{:.2f}"}), hide_index=True, width="stretch")
    st.download_button("Download CSV", table.to_csv(index=False), "overview_by_app.csv", "text/csv")
