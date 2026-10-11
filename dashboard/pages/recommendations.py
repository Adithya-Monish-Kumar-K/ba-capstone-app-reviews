"""Recommendations (#140): the per-app priority table and the recommendations from scripts/22_combined_insights.py (#141).

Every recommendation carries the number it is based on. Filtered by the sidebar's domain and app selection.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import DATA_DIR, INK2, fmt_int, get_filters, require_selection, style

CI_DIR = DATA_DIR / "combined"
PRIORITY_COLORS = {"High": "#c93a39", "Medium": "#eda100", "Low": "#8a8985"}
SIGNS = {"sig_high_volume": "Many 1–2★ per day", "sig_high_share": "High 1–2★ share", "sig_rising": "Rising trend",
         "sig_forecast_up": "Forecast above recent", "sig_unusual_days": "Days above forecast",
         "sig_worse_release": "Release made things worse"}


@st.cache_data
def load_tables():
    table = pd.read_csv(CI_DIR / "priority_table.csv")
    recs = pd.read_csv(CI_DIR / "recommendations.csv").fillna({"app_name": "", "priority": ""})
    summary = json.loads((DATA_DIR / "combined_insights_summary.json").read_text())
    return table, recs, summary


f = get_filters()
require_selection(f)
if not (CI_DIR / "priority_table.csv").exists():
    st.error("Run `python scripts/22_combined_insights.py` first.")
    st.stop()
TABLE, RECS, SUMMARY = load_tables()

st.title("Recommendations")
st.caption("All stages joined into one priority table: volume and trend (time series), complaint topics and their sentiment "
           "(text mining), the forecast and its alerts, releases that made things worse (update impact) and the problem-review "
           "classifier. Each app scores one point per warning sign; 3 or more = High priority.")

t = TABLE[TABLE["app_name"].isin(f.apps)].copy()
if t.empty:
    st.warning("No apps in this selection.")
    st.stop()

c = st.columns(4)
for col, p in zip(c[:3], ["High", "Medium", "Low"]):
    names = t.loc[t["priority"] == p, "app_name"].tolist()
    col.metric(f"{p} priority", len(names))
    col.caption(", ".join(names) or "–")
c[3].metric("1–2★ reviews per day (selection)", fmt_int(t["neg_per_day"].sum()))

# ----------------------------------------------------------------------------
# Priority grid
# ----------------------------------------------------------------------------
st.subheader("Priority by app")
t = t.sort_values(["priority_score", "neg_per_day"], ascending=[False, False])
z = t[list(SIGNS)].astype(int).to_numpy()
fig = go.Figure(go.Heatmap(
    z=z, x=list(SIGNS.values()), y=[f"{a} · {p} ({s})" for a, p, s in zip(t["app_name"], t["priority"], t["priority_score"])],
    colorscale=[[0, "#f4f3f0"], [1, "#c93a39"]], showscale=False, xgap=3, ygap=3,
    hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(side="top")
st.plotly_chart(style(fig, 90 + 36 * len(t)), width="stretch", config={"displayModeBar": False})
with st.expander("What each warning sign means"):
    for k, v in SUMMARY["priority_rules"].items():
        st.markdown(f"- **{SIGNS[k]}**: {v}")

cols = {"app_name": "App", "domain": "Domain", "priority": "Priority", "priority_score": "Score",
        "neg_per_day": "1–2★ per day", "neg_share_pct": "1–2★ share %", "trend_change_pct": "Trend %",
        "top_topic": "Largest complaint topic", "top_topic_share_pct": "Topic share %", "top_topic_sentiment": "Topic sentiment",
        "most_severe_topic": "Most damaging topic", "forecast_week1": "Forecast 21–27 Sep", "forecast_change_pct": "Forecast vs last 4 wks %",
        "forecast_reliable": "Forecast reliable", "unusual_days_above": "Days above forecast", "releases_worse": "Releases worse",
        "worst_release": "Worst release", "flag_rate_pct": "Classifier flag rate %", "classifier_precision": "Classifier precision"}
show = t[list(cols)].rename(columns=cols)
st.dataframe(show, hide_index=True, width="stretch",
             column_config={**{k: st.column_config.NumberColumn(format="%.1f") for k in
                               ("1–2★ per day", "1–2★ share %", "Trend %", "Topic share %", "Forecast vs last 4 wks %",
                                "Classifier flag rate %")},
                            "Topic sentiment": st.column_config.NumberColumn(format="%+.2f"),
                            "Forecast 21–27 Sep": st.column_config.NumberColumn(format="%.0f"),
                            "Classifier precision": st.column_config.NumberColumn(format="percent")})
st.download_button("Download priority table", show.to_csv(index=False), "priority_table.csv", "text/csv")

# ----------------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------------
st.subheader("Recommendations")
types = sorted(RECS["type"].unique(), key=lambda x: ["Fix first", "Most damaging", "Focus", "Release check", "Plan capacity",
                                                      "Monitor", "Trend", "Triage"].index(x) if x in
               ["Fix first", "Most damaging", "Focus", "Release check", "Plan capacity", "Monitor", "Trend", "Triage"] else 99)
pick = st.multiselect("Type", types, default=types, key="rec_types")

domains = [d for d in f.domains if d in set(t["domain"])]
for dom in domains:
    st.markdown(f"#### {dom}")
    d = RECS[(RECS["level"] == "Domain") & (RECS["domain"] == dom) & RECS["type"].isin(pick)]
    for _, r in d.iterrows():
        st.markdown(f"- **{r['action']}**  \n  <span style='color:{INK2}'>{r['evidence']}</span>", unsafe_allow_html=True)
    for app in t[t["domain"] == dom]["app_name"]:
        a = RECS[(RECS["level"] == "App") & (RECS["app_name"] == app) & RECS["type"].isin(pick)]
        if a.empty:
            continue
        p = a["priority"].iloc[0]
        with st.expander(f"{app} · {p} priority", expanded=(p == "High")):
            for _, r in a.iterrows():
                st.markdown(f"- **{r['type']}:** {r['action']}  \n  <span style='color:{INK2}'>{r['evidence']}</span>",
                            unsafe_allow_html=True)

sel = RECS[((RECS["level"] == "Domain") & RECS["domain"].isin(domains)) | RECS["app_name"].isin(t["app_name"])]
st.download_button("Download recommendations", sel[sel["type"].isin(pick)].to_csv(index=False), "recommendations.csv", "text/csv")
st.caption("Reviews come from the Play Store's 'newest' feed for 1 Apr – 20 Sep 2026. A 1–2★ review is the user's own judgement, "
           "not a confirmed failure; topics come from a topic model on 1–2★ reviews long enough to carry one.")
