"""Text mining: complaint topics per domain and app, sentiment by topic, topic share over time and distinctive terms per app.

Reads the small tables written by scripts/17_topic_modelling.py, 18_topic_evaluation.py and 20_topic_examples.py
(data/textmining/*.csv and data/topic_modelling_summary.json). The 1-2 star complaint reviews of each domain were
split into topics with NMF; sentiment is the VADER compound score (-1 to +1).
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (APP_COLORS, DATA_DIR, DOMAIN_COLORS, DOMAINS, INK2, WINDOW_END, WINDOW_START, empty_figure, fmt_int,
                    get_filters, require_selection, style)

TM_DIR = DATA_DIR / "textmining"
APP_DOMAIN = {}   # filled from topic_share_by_app below
BLUES = ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"]
STRONGLY_NEGATIVE = -0.5     # VADER compound at or below this counts as strongly negative (same cut as the evaluation tables)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_tables():
    t = {name: pd.read_csv(TM_DIR / f"{name}.csv") for name in
         ("topics", "topic_sentiment_by_app", "topic_share_by_app", "monthly_topic_trends", "distinctive_terms",
          "topic_keyword_overlap", "topic_examples", "coherence_scores")}
    t["praise"] = json.loads((DATA_DIR / "topic_modelling_summary.json").read_text())["praise_topics"]
    return t


def short(label, n=34):
    return label if len(label) <= n else label[: n - 1].rstrip() + "…"


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
f = get_filters()
require_selection(f)
T = load_tables()
for _, r in T["topic_share_by_app"][["app_name", "domain"]].drop_duplicates().iterrows():
    APP_DOMAIN[r["app_name"]] = r["domain"]

st.title("Text mining")
st.caption("What do unhappy users write about? The 1–2★ reviews of each domain were split into topics (NMF on TF-IDF, scikit-learn), "
           "and every review was given the topic it fits best. Sentiment is the VADER score from –1 (very negative) to +1.")

domains = [d for d in DOMAINS if any(APP_DOMAIN.get(a) == d for a in f.apps)]
if not domains:
    st.warning("The selected apps have no text-mining results.")
    st.stop()
domain = st.segmented_control("Domain", domains, default=domains[0], key="tm_domain") or domains[0]
if domain not in domains:
    domain = domains[0]
apps = [a for a in f.apps if APP_DOMAIN.get(a) == domain]

topics = T["topics"][T["topics"]["domain"] == domain].sort_values("topic_id")
share = T["topic_share_by_app"]
share = share[(share["domain"] == domain) & share["app_name"].isin(apps)]
sent = T["topic_sentiment_by_app"]
sent = sent[(sent["domain"] == domain) & sent["app_name"].isin(apps)].rename(columns={"dominant_topic_label": "topic_label"})

# Topic table for the selected apps: complaints, share, sentiment (review-weighted over the apps)
by_topic = share.groupby("topic_label", as_index=False)["reviews"].sum()
by_topic["share"] = by_topic["reviews"] / by_topic["reviews"].sum()
s = sent.assign(w=sent["mean_sentiment"] * sent["reviews"], strong=sent["pct_strongly_neg"] / 100 * sent["reviews"]) \
    .groupby("topic_label", as_index=False)[["w", "strong", "reviews"]].sum()
by_topic = by_topic.merge(s.assign(mean_sentiment=s["w"] / s["reviews"], strongly_negative=s["strong"] / s["reviews"])
                          [["topic_label", "mean_sentiment", "strongly_negative"]], on="topic_label")
by_topic = by_topic.merge(topics[["topic_id", "topic_label", "top_terms", "top_weights"]], on="topic_label").sort_values("reviews", ascending=False)

total = int(by_topic["reviews"].sum())
overall_sent = (by_topic["mean_sentiment"] * by_topic["reviews"]).sum() / total
overall_strong = (by_topic["strongly_negative"] * by_topic["reviews"]).sum() / total
c = st.columns(4)
c[0].metric("Complaint reviews with a topic", fmt_int(total))
c[1].metric("Topics in this domain", len(topics))
c[2].metric("Mean sentiment", f"{overall_sent:.2f}")
c[3].metric("Strongly negative", f"{overall_strong:.0%}")
top, worst = by_topic.iloc[0], by_topic.sort_values("mean_sentiment").iloc[0]
st.markdown(f"Largest topic: **{top['topic_label']}** ({top['share']:.0%} of complaints). "
            f"Most negative: **{worst['topic_label']}** (mean {worst['mean_sentiment']:.2f}, {worst['strongly_negative']:.0%} strongly negative).")

# ----------------------------------------------------------------------------
# Topic explorer
# ----------------------------------------------------------------------------
st.subheader("Topic explorer")
left, right = st.columns([1, 1.15])

bars = by_topic.sort_values("reviews")
fig = px.bar(bars, x="share", y="topic_label", orientation="h", color_discrete_sequence=[DOMAIN_COLORS[domain]],
             text=bars["share"].map("{:.0%}".format), custom_data=["reviews", "mean_sentiment", "strongly_negative"])
fig.update_traces(textposition="outside", hovertemplate="<b>%{y}</b><br>%{x:.1%} of complaints (%{customdata[0]:,} reviews)"
                  "<br>mean sentiment %{customdata[1]:.2f}<br>%{customdata[2]:.0%} strongly negative<extra></extra>")
fig.update_layout(title="Share of complaints by topic", yaxis_title="", xaxis_title="")
fig.update_xaxes(tickformat=".0%", range=[0, bars["share"].max() * 1.3])
fig.update_yaxes(tickmode="array", tickvals=bars["topic_label"], ticktext=[short(x) for x in bars["topic_label"]])
left.plotly_chart(style(fig, 360), width="stretch", config={"displayModeBar": False})

labels = by_topic["topic_label"].tolist()
chosen = right.selectbox("Pick a topic", labels, key="tm_topic", format_func=lambda x: f"{x}  ({by_topic.set_index('topic_label').loc[x, 'share']:.0%})")
row = by_topic.set_index("topic_label").loc[chosen]
terms = [t.strip() for t in row["top_terms"].split(";")][:12]
weights = [float(w) for w in str(row["top_weights"]).split(";")][:12]
tw = pd.DataFrame({"term": terms, "weight": weights}).iloc[::-1]
fig = px.bar(tw, x="weight", y="term", orientation="h", color_discrete_sequence=[DOMAIN_COLORS[domain]])
fig.update_traces(hovertemplate="%{y}: %{x:.2f}<extra></extra>")
fig.update_layout(title="Top terms of the topic (weight in the NMF topic)", yaxis_title="", xaxis_title="")
right.plotly_chart(style(fig, 300), width="stretch", config={"displayModeBar": False})
m = right.columns(3)
m[0].metric("Reviews", fmt_int(row["reviews"]))
m[1].metric("Mean sentiment", f"{row['mean_sentiment']:.2f}")
m[2].metric("Strongly negative", f"{row['strongly_negative']:.0%}")

# Example reviews for the chosen topic
ex = T["topic_examples"]
ex = ex[(ex["domain"] == domain) & (ex["topic_label"] == chosen) & ex["app_name"].isin(apps)]
st.markdown(f"**Example reviews — {chosen}**")
if ex.empty:
    st.info("No example reviews for this topic and app selection.")
else:
    per_app = ex.sort_values(["rank", "app_name"]).groupby("app_name").head(2).sort_values(["app_name", "rank"])
    cols = st.columns(min(len(per_app), 3))
    for i, (_, r) in enumerate(per_app.iterrows()):
        with cols[i % len(cols)].container(border=True):
            st.caption(f"**{r['app_name']}** · {int(r['score'])}★ · sentiment {r['sentiment_compound']:+.2f}")
            st.write(r["content"])
    st.caption("The best-fitting reviews (highest topic confidence) of reasonable length, shown as written by the user.")

# What the topic overlaps with in the keyword issue tags
ov = T["topic_keyword_overlap"]
ov = ov[(ov["domain"] == domain) & (ov["topic_label"] == chosen)]
if not ov.empty:
    tags = ov.iloc[0].drop(["domain", "topic_id", "topic_label", "reviews", "Untagged_Pct"]).astype(float).sort_values(ascending=False)
    with st.expander("How this topic compares with the 9 keyword issue tags"):
        st.write(f"**{ov.iloc[0]['Untagged_Pct']:.0f}%** of this topic's reviews carry none of the nine keyword tags, "
                 "so the topic model finds themes the keyword rules miss. Of the tagged reviews, the share per tag:")
        fig = px.bar(pd.DataFrame({"tag": tags.index, "pct": tags.values}).iloc[::-1],
                     x="pct", y="tag", orientation="h", color_discrete_sequence=[DOMAIN_COLORS[domain]])
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}% of the topic's reviews<extra></extra>")
        fig.update_layout(yaxis_title="", xaxis_title="% of the topic's reviews with this tag")
        st.plotly_chart(style(fig, 320), width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# Topics by app
# ----------------------------------------------------------------------------
st.subheader("Topics by app")
metric = st.segmented_control("Show", ["Share of the app's complaints", "Strongly negative reviews"], default="Share of the app's complaints",
                              key="tm_heat") or "Share of the app's complaints"
order = [x for x in by_topic.sort_values("reviews", ascending=False)["topic_label"]]
if metric.startswith("Share"):
    piv = share.pivot(index="topic_label", columns="app_name", values="share_pct").reindex(index=order, columns=apps)
    fmt, suffix, title = ".0f", "%", "Share of each app's complaints, by topic"
else:
    piv = sent.pivot(index="topic_label", columns="app_name", values="pct_strongly_neg").reindex(index=order, columns=apps)
    fmt, suffix, title = ".0f", "%", "Reviews with VADER ≤ –0.5, by topic and app"
piv.index = [short(x, 40) for x in piv.index]
fig = px.imshow(piv, text_auto=fmt, aspect="auto", color_continuous_scale=BLUES, labels={"color": suffix, "x": "", "y": ""})
fig.update_traces(hovertemplate="%{y} · %{x}: %{z:.1f}%<extra></extra>")
fig.update_xaxes(side="top")
fig.update_layout(coloraxis_showscale=False)
st.markdown(f"**{title}**")
st.plotly_chart(style(fig, 90 + 46 * len(piv)), width="stretch", config={"displayModeBar": False})
st.caption("Each column adds up to 100% in the first view. In the second view a dark cell means most of that app's reviews on the "
           "topic are strongly negative (hostile wording), so the topic hurts more than its size suggests.")

# ----------------------------------------------------------------------------
# Topic share over time
# ----------------------------------------------------------------------------
st.subheader("Topic share over time")
tr = T["monthly_topic_trends"]
tr = tr[tr["domain"] == domain].copy()
tr["month_start"] = pd.to_datetime(tr["month"] + "-01")
tr = tr[(tr["month_start"] >= pd.Timestamp(f.start.replace(day=1))) & (tr["month_start"] <= pd.Timestamp(f.end))]
if tr["month"].nunique() < 2:
    st.info("Widen the date range in the sidebar to see how topic shares change month by month.")
else:
    fig = go.Figure()
    for label, g in tr.groupby("topic_label"):
        if label == chosen:
            continue
        fig.add_trace(go.Scatter(x=g["month_start"], y=g["share_pct"], mode="lines", name=short(label), line=dict(color="#c9c8c3", width=1.5),
                                 hovertemplate="%{y:.1f}%<extra>" + short(label) + "</extra>"))
    g = tr[tr["topic_label"] == chosen]
    fig.add_trace(go.Scatter(x=g["month_start"], y=g["share_pct"], mode="lines+markers", name=short(chosen),
                             line=dict(color=DOMAIN_COLORS[domain], width=3), marker=dict(size=8),
                             hovertemplate="%{y:.1f}%<extra>" + short(chosen) + "</extra>"))
    fig.update_layout(title=f"Share of {domain} complaints: {short(chosen, 50)} against the other topics (grey)",
                      yaxis_title="% of the domain's complaints", xaxis_title="", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(style(fig, 360), width="stretch", config={"displayModeBar": False})
    st.caption("Monthly shares for the whole domain. September holds only 1–20 Sep. The topic chosen above is highlighted.")

# ----------------------------------------------------------------------------
# Distinctive terms per app
# ----------------------------------------------------------------------------
st.subheader("Distinctive terms per app")
app = st.segmented_control("App", apps, default=apps[0], key="tm_app") or apps[0]
if app not in apps:
    app = apps[0]
dt = T["distinctive_terms"]
dt = dt[(dt["app_name"] == app)].sort_values("z_score")
if dt.empty:
    st.plotly_chart(empty_figure("No distinctive terms for this app"), width="stretch")
else:
    dt = dt.assign(kind=dt["is_bigram"].map({False: "word", True: "two-word phrase"}))
    fig = px.bar(dt, x="z_score", y="term", orientation="h", color="kind", pattern_shape="kind",
                 color_discrete_map={"word": APP_COLORS[app], "two-word phrase": APP_COLORS[app]},
                 pattern_shape_map={"word": "", "two-word phrase": "/"}, custom_data=["app_count", "domain_count"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>z = %{x:.1f}<br>%{customdata[0]:,} reviews in this app, "
                                    "%{customdata[1]:,} in the domain<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=dt["term"].tolist())
    fig.update_layout(title=f"{app}: words and phrases used far more than in the other {domain} apps", yaxis_title="",
                      xaxis_title="z-score (log-odds against the other apps of the domain)", legend=dict(orientation="h", y=-0.15, x=0))
    st.plotly_chart(style(fig, 440), width="stretch", config={"displayModeBar": False})
    st.caption("Higher means more typical of this app's complaints compared with its domain peers (a z-score above about 2 is unlikely by chance). "
               "Hatched bars are two-word phrases.")

# ----------------------------------------------------------------------------
# Praise contrast and tables
# ----------------------------------------------------------------------------
with st.expander("What satisfied users write (topics of 4–5★ reviews)"):
    praise = T["praise"].get(domain, [])
    st.dataframe(pd.DataFrame([{"Topic": p["label"], "Top terms": ", ".join(p["top_terms"])} for p in praise]),
                 hide_index=True, width="stretch")
    st.caption("Four topics per domain, fitted on 4–5★ reviews of at least three content words. They show what to protect, "
               "the counterpart of the complaint topics above.")

with st.expander("How many topics, and why"):
    cs = T["coherence_scores"]
    cs = cs[cs["domain"] == domain]
    st.write(f"The number of topics for **{domain}** was chosen by comparing coherence (NPMI: do a topic's top words appear together "
             "in the same reviews?) for 3 to 8 topics, then checking that the topics could be named and were not duplicates.")
    st.dataframe(cs.rename(columns={"k": "Topics", "mean_umass": "UMass coherence", "mean_npmi": "NPMI coherence", "is_chosen": "Chosen"}).drop(columns="domain"),
                 hide_index=True, width="stretch")

with st.expander("Table by topic"):
    out = by_topic[["topic_id", "topic_label", "reviews", "share", "mean_sentiment", "strongly_negative", "top_terms"]].rename(columns={
        "topic_id": "Topic", "topic_label": "Name", "reviews": "Reviews", "share": "Share of complaints", "mean_sentiment": "Mean sentiment",
        "strongly_negative": "Strongly negative", "top_terms": "Top terms"})
    st.dataframe(out.style.format({"Reviews": "{:,}", "Share of complaints": "{:.1%}", "Mean sentiment": "{:.2f}", "Strongly negative": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.download_button("Download CSV", out.to_csv(index=False), f"topics_{domain.split()[0].lower()}.csv", "text/csv")
