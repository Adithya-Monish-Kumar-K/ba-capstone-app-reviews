"""Update impact (#140): did a new app version make things worse?

Reads the tables written by scripts/21_update_impact.py (data/update_impact/*.csv). For each release the analysis compares
(A) the app's 1-2 star reviews in the 7 days before and the 7 days from the release, and (B) reviews on the new version
against reviews on older versions over the same 7 days, relative to the app's usual new-version gap.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (APP_COLORS, APP_DOMAIN, DATA_DIR, INK2, empty_figure, fmt_int, get_filters, load_daily_app,
                    require_selection, shade_feed_gap, style)

UI_DIR = DATA_DIR / "update_impact"
VERDICTS = ["Worse (confirmed at app level)", "Worse (new version only)", "No clear change", "Better", "Not testable"]
VERDICT_COLORS = {"Worse (confirmed at app level)": "#c93a39", "Worse (new version only)": "#ee8f8a",
                  "No clear change": "#b9b8b4", "Better": "#2a78d6", "Not testable": "#e6e5e1"}
WINDOW = 7


@st.cache_data
def load_tables():
    res = pd.read_csv(UI_DIR / "release_impact.csv", parse_dates=["release_day"])
    issues = pd.read_csv(UI_DIR / "release_issue_changes.csv", parse_dates=["release_day"])
    return res, issues


def fmt_q(q):
    return "q < 0.001" if q < 0.001 else f"q = {q:.3f}"


f = get_filters()
require_selection(f)
if not (UI_DIR / "release_impact.csv").exists():
    st.error("Run `python scripts/21_update_impact.py` first.")
    st.stop()
RES, ISSUES = load_tables()

st.title("Update impact")
st.caption("Did a new app version make things worse? Each release is checked two ways: the app's 1–2★ reviews in the week before "
           "and the week from the release, and reviews on the new version against reviews on older versions over the same days. "
           "A release counts as **worse** when its new-version gap is at least 5 points above the app's usual gap and significant "
           "after correcting for testing 205 releases (Benjamini–Hochberg, q < 0.05).")

res = RES[RES["app_name"].isin(f.apps) & (RES["release_day"].dt.date >= f.start) & (RES["release_day"].dt.date <= f.end)]
if res.empty:
    st.warning("No releases for this selection.")
    st.stop()

c = st.columns(5)
c[0].metric("Releases", fmt_int(len(res)))
c[1].metric("Tested", fmt_int(res["testable"].sum()), help="Releases with full 7-day windows and at least 30 reviews on the new and older versions.")
c[2].metric("Made things worse", fmt_int(res["made_worse"].sum()))
c[3].metric("Confirmed at app level", fmt_int((res["verdict"] == VERDICTS[0]).sum()),
            help="The app's 1–2★ reviews per day also rose significantly against the week before.")
c[4].metric("Better", fmt_int((res["verdict"] == "Better").sum()))

# ----------------------------------------------------------------------------
# Verdicts by app
# ----------------------------------------------------------------------------
counts = res.groupby(["app_name", "verdict"]).size().reset_index(name="releases")
order = [a for a in f.apps if a in set(res["app_name"])]
fig = px.bar(counts, y="app_name", x="releases", color="verdict", orientation="h", category_orders={"verdict": VERDICTS, "app_name": order},
             color_discrete_map=VERDICT_COLORS, labels={"app_name": "", "releases": "Release events", "verdict": ""})
fig.update_traces(hovertemplate="%{y}: %{x} release(s)<extra>%{fullData.name}</extra>")
fig.update_layout(title="Verdict for every release", legend=dict(orientation="h", y=-0.18, x=0))
st.plotly_chart(style(fig, 120 + 32 * len(order)), width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# One app: release effects and the daily series with releases
# ----------------------------------------------------------------------------
st.subheader("Releases of one app")
app = st.selectbox("App", order, key="ui_app", format_func=lambda a: f"{a}  ·  {APP_DOMAIN[a]}")
r = res[res["app_name"] == app].sort_values("release_day")
t = r[r["testable"]]
usual = t["app_usual_gap_pp"].median() if len(t) else 0

left, right = st.columns(2)
fig = go.Figure()
if len(t):
    fig.add_hrect(y0=usual - 5, y1=usual + 5, fillcolor="#e6e5e1", opacity=0.5, line_width=0)
    fig.add_hline(y=0, line=dict(color="#8a8985", width=1))
    fig.add_hline(y=usual, line=dict(color=INK2, width=1, dash="dash"), annotation_text="usual gap",
                  annotation_position="top left", annotation_font_color=INK2)
    for v in VERDICTS[:-1]:
        s = t[t["verdict"] == v]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["release_day"], y=s["neg_share_diff_pp"], mode="markers", name=v,
            marker=dict(size=10, color=VERDICT_COLORS[v], line=dict(color="white", width=1)),
            error_y=dict(type="data", symmetric=False, array=s["neg_share_diff_ci_high_pp"] - s["neg_share_diff_pp"],
                         arrayminus=s["neg_share_diff_pp"] - s["neg_share_diff_ci_low_pp"], color=VERDICT_COLORS[v], thickness=1.2, width=0),
            customdata=s[["app_version", "neg_share_new", "neg_share_older", "n_new_version", "excess_gap_pp"]],
            hovertemplate="%{x|%d %b}: version %{customdata[0]}<br>new %{customdata[1]:.0%} vs older %{customdata[2]:.0%} rated 1–2★"
                          "<br>gap %{y:+.1f} pts (%{customdata[4]:+.1f} vs usual) · %{customdata[3]:,} reviews on new<extra></extra>"))
    fig.update_layout(title="New version vs older versions, same 7 days", yaxis_title="gap in 1–2★ share (points)",
                      legend=dict(orientation="h", y=-0.2, x=0))
    left.plotly_chart(style(fig, 400), width="stretch", config={"displayModeBar": False})
else:
    left.plotly_chart(empty_figure("No testable releases for this app in the selected dates."), width="stretch")

daily = load_daily_app()
d = daily[(daily["app_name"] == app) & (daily["day"].dt.date >= f.start) & (daily["day"].dt.date <= f.end)].sort_values("day")
fig = go.Figure(go.Scatter(x=d["day"], y=d["neg_reviews_filled"], line=dict(color=APP_COLORS[app], width=1.6), name="1–2★ reviews",
                           hovertemplate="%{x|%a %d %b}: %{y:,.0f}<extra></extra>"))
for _, x in r.iterrows():
    fig.add_vline(x=x["release_day"], line=dict(color=VERDICT_COLORS[x["verdict"]], width=2 if x["made_worse"] else 0.8,
                                                dash="solid" if x["made_worse"] else "dot"))
fig.add_trace(go.Scatter(x=r["release_day"], y=[d["neg_reviews_filled"].max() * 1.05] * len(r), mode="markers", name="Release",
                         marker=dict(symbol="diamond", size=8, color=[VERDICT_COLORS[v] for v in r["verdict"]]),
                         customdata=r[["app_version", "verdict"]],
                         hovertemplate="%{x|%d %b}: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
shade_feed_gap(fig, f)
fig.update_layout(title="Daily 1–2★ reviews and release days", showlegend=False, yaxis_title="1–2★ reviews per day")
fig.update_yaxes(rangemode="tozero")
right.plotly_chart(style(fig, 400), width="stretch", config={"displayModeBar": False})
st.caption(f"Left: one dot per release with its 95% interval. Dashed line = {app}'s usual gap ({usual:+.1f} points); the grey band "
           "(±5 points around it) is not counted as a change. Right: release days coloured by verdict.")

# ----------------------------------------------------------------------------
# One release in detail
# ----------------------------------------------------------------------------
st.subheader("One release in detail")
labels = {f"{x['app_version']}  ·  {x['release_day']:%d %b}  ·  {x['verdict']}": i for i, x in r.iterrows()}
default = next((k for k, i in labels.items() if r.loc[i, "made_worse"]), next(iter(labels)))
pick = st.selectbox("Release", list(labels), index=list(labels).index(default), key="ui_release")
x = r.loc[labels[pick]]
t0 = x["release_day"]

c = st.columns(4)
if x["testable"]:
    c[0].metric("1–2★ share, new version", f"{x['neg_share_new']:.1%}", f"{x['neg_share_diff_pp']:+.1f} pts vs older versions",
                delta_color="inverse")
    c[1].metric("Gap vs app's usual gap", f"{x['excess_gap_pp']:+.1f} pts", fmt_q(x["q_excess_gap"]), delta_color="off")
    c[2].metric("Rating, new vs older", f"{x['rating_new']:.2f}★", f"{x['rating_new'] - x['rating_older']:+.2f}★")
else:
    c[0].metric("1–2★ share, new version", "–")
    c[1].info("Not testable: window outside the data or too few reviews.")
c[3].metric("1–2★ reviews per day", f"{x['neg_per_day_before']:.0f} → {x['neg_per_day_after']:.0f}",
            f"{x['neg_per_day_change_pct']:+.0f}% ({fmt_q(x['q_neg_per_day'])})" if pd.notna(x["q_neg_per_day"]) else None,
            delta_color="inverse")
if x["feed_gap_in_window"]:
    st.info("This release falls in the late-April feed gap: shares are inflated because positive reviews are missing, "
            "but the new and older versions cover the same days, so their comparison still holds.", icon=":material/info:")

left, right = st.columns(2)
dd = daily[(daily["app_name"] == app) & (daily["day"] >= t0 - pd.Timedelta(days=21)) & (daily["day"] <= t0 + pd.Timedelta(days=13))].sort_values("day")
fig = go.Figure(go.Scatter(x=dd["day"], y=dd["neg_reviews_filled"], line=dict(color=APP_COLORS[app], width=2), mode="lines+markers",
                           hovertemplate="%{x|%a %d %b}: %{y:,.0f}<extra></extra>"))
fig.add_vrect(x0=t0 - pd.Timedelta(days=WINDOW, hours=12), x1=t0 - pd.Timedelta(hours=12), fillcolor="#8a8985", opacity=0.12,
              line_width=0, annotation_text="7 days before", annotation_position="top left", annotation_font_color=INK2)
fig.add_vrect(x0=t0 - pd.Timedelta(hours=12), x1=t0 + pd.Timedelta(days=WINDOW - 1, hours=12), fillcolor=APP_COLORS[app], opacity=0.12,
              line_width=0, annotation_text="7 days from release", annotation_position="top left", annotation_font_color=INK2)
fig.update_layout(title=f"{app} {x['app_version']}: daily 1–2★ reviews around the release", showlegend=False, yaxis_title="1–2★ reviews per day")
fig.update_yaxes(rangemode="tozero")
left.plotly_chart(style(fig, 360), width="stretch", config={"displayModeBar": False})

iss = ISSUES[(ISSUES["app_name"] == app) & (ISSUES["app_version"].astype(str) == str(x["app_version"]))]
if len(iss):
    iss = iss.sort_values("diff_pp")
    fig = go.Figure(go.Bar(y=iss["issue"], x=iss["diff_pp"], orientation="h",
                           marker_color=["#c93a39" if (s and v > 0) else ("#ee8f8a" if v > 0 else "#9ec5f4")
                                         for s, v in zip(iss["significant_increase"], iss["diff_pp"])],
                           customdata=iss[["share_new_pct", "share_older_pct", "q_value"]],
                           hovertemplate="%{y}: new %{customdata[0]:.1f}% vs older %{customdata[1]:.1f}% of reviews"
                                         "<br>%{x:+.1f} points (q = %{customdata[2]:.3f})<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#8a8985", width=1))
    fig.update_layout(title="Issue mix: new version minus older versions", xaxis_title="points (share of reviews mentioning the issue)")
    right.plotly_chart(style(fig, 360), width="stretch", config={"displayModeBar": False})
    right.caption("Dark red = significant increase (q < 0.05 and at least 2 points). Keyword issue tags; a review can have several.")
else:
    right.plotly_chart(empty_figure("No issue comparison for an untestable release."), width="stretch")

# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------
st.subheader("All releases")
only_worse = st.toggle("Only releases that made things worse", value=False, key="ui_only_worse")
tab = res[res["made_worse"]] if only_worse else res
cols = {"app_name": "App", "app_version": "Version", "release_day": "Release day", "verdict": "Verdict",
        "n_new_version": "Reviews on new", "neg_share_new": "1–2★ new", "neg_share_older": "1–2★ older",
        "neg_share_diff_pp": "Gap (pts)", "app_usual_gap_pp": "Usual gap (pts)", "excess_gap_pp": "Excess (pts)",
        "q_excess_gap": "q", "neg_per_day_before": "1–2★/day before", "neg_per_day_after": "1–2★/day after",
        "most_elevated_issue": "Most elevated issue"}
show = tab.sort_values(["app_name", "release_day"])[list(cols)].rename(columns=cols)
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"Release day": st.column_config.DateColumn(format="D MMM"),
                            "1–2★ new": st.column_config.NumberColumn(format="percent"),
                            "1–2★ older": st.column_config.NumberColumn(format="percent"),
                            **{k: st.column_config.NumberColumn(format="%.1f") for k in
                               ("Gap (pts)", "Usual gap (pts)", "Excess (pts)", "1–2★/day before", "1–2★/day after")},
                            "q": st.column_config.NumberColumn(format="%.3f")})
st.download_button("Download CSV", show.to_csv(index=False), "update_impact.csv", "text/csv")

with st.expander("How the analysis works and its limits"):
    st.markdown(
        "- **Release day** = the day the version reached its 10th review (ignores a few early-access reviews). Using the very first review "
        "gives the same four 'worse' releases.\n"
        "- **Same-days comparison.** Users on the newest version are not a random sample: on Flipkart and Amazon they are about 6 points "
        "more negative in almost every release. So each release is judged against its app's usual gap, not against zero.\n"
        "- **Before vs after** uses one full week on each side, so the weekday pattern cancels out; it also picks up anything else that "
        "happened that week, which is why it only confirms a verdict.\n"
        "- **Limits.** The version is the one on the reviewer's phone, not a confirmed install date; releases come every 4–13 days, so "
        "windows overlap the previous release; and a low rating is not proof that the release caused the problem.")
