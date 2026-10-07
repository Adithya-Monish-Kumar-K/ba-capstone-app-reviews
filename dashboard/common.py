"""Shared data loading, sidebar filters and chart helpers for every dashboard page.

Pages read the filters chosen in the sidebar with `get_filters()` and narrow a table with `apply_filters()`.
All data comes from small precomputed tables in data/ (no raw reviews are loaded), so the app stays fast.
"""
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from apps import APP_COLORS, APP_DOMAIN, APP_NAMES, DOMAIN_COLORS, DOMAINS  # noqa: E402

WINDOW_START, WINDOW_END = date(2026, 4, 1), date(2026, 9, 20)
GAP_START, GAP_END = date(2026, 4, 21), date(2026, 5, 5)

ISSUE_LABELS = {
    "issue_crash_bugs_stability": "Crash & Stability",
    "issue_payment_refund": "Payment & Refund",
    "issue_delivery_delay": "Delivery Delay",
    "issue_order_quality_fulfillment": "Order Quality",
    "issue_cancellation_return": "Cancellation & Return",
    "issue_customer_support": "Customer Support",
    "issue_account_login_otp": "Account / Login / OTP",
    "issue_pricing_charges_fraud": "Pricing & Fraud",
    "issue_ui_ux_update": "UI/UX & Update",
}
ISSUE_COLS = list(ISSUE_LABELS)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_daily_app():
    """One row per app per day: reviews, 1-2 star counts, rating, sentiment, issue counts, quality flags."""
    df = pd.read_csv(DATA_DIR / "timeseries" / "daily_app.csv", parse_dates=["day"])
    df["app_name"] = pd.Categorical(df["app_name"], APP_NAMES, ordered=True)
    df["domain"] = pd.Categorical(df["domain"], DOMAINS, ordered=True)
    return df


@st.cache_data
def load_releases():
    """New-release events per app (first day each new version appears)."""
    df = pd.read_csv(DATA_DIR / "timeseries" / "release_events.csv", parse_dates=["first_seen", "first_seen_10th_review"])
    return df[df["use_for_update_impact"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Filters (drawn once in the sidebar by app.py, read by every page)
# ----------------------------------------------------------------------------
@dataclass
class Filters:
    domains: list
    apps: list
    start: date
    end: date
    exclude_feed_gap: bool


def sidebar_filters():
    with st.sidebar:
        st.header("Filters")
        domains = st.multiselect("Domain", DOMAINS, default=DOMAINS, key="f_domains")
        options = [a for a in APP_NAMES if APP_DOMAIN[a] in domains]
        prev = st.session_state.get("f_apps")
        if prev is not None:
            st.session_state["f_apps"] = [a for a in prev if a in options]
        apps = st.multiselect("App", options, default=options, key="f_apps")
        start, end = st.slider("Date range", min_value=WINDOW_START, max_value=WINDOW_END,
                               value=(WINDOW_START, WINDOW_END), format="D MMM", key="f_dates")
        gap = st.toggle("Exclude the 21 Apr – 5 May feed gap", value=False, key="f_gap",
                        help="For Swiggy, Blinkit, Domino's, Flipkart and Amazon, positive reviews are largely "
                             "missing from the Play Store feed in this window, which inflates the share of 1–2★ reviews.")
        st.caption("Data: every English/India Play Store review of 11 apps, 1 Apr – 20 Sep 2026.")
    st.session_state["filters"] = Filters(domains, apps, start, end, gap)


def get_filters():
    return st.session_state.get("filters") or Filters(DOMAINS, APP_NAMES, WINDOW_START, WINDOW_END, False)


def apply_filters(df, f=None, day_col="day"):
    """Keep the selected apps and dates; drop feed-gap rows if asked (needs a `feed_gap` column)."""
    f = f or get_filters()
    out = df[df["app_name"].isin(f.apps)]
    if day_col in out:
        d = out[day_col].dt.date
        out = out[(d >= f.start) & (d <= f.end)]
    if f.exclude_feed_gap and "feed_gap" in out:
        out = out[~out["feed_gap"]]
    return out


def require_selection(f=None):
    f = f or get_filters()
    if not f.apps:
        st.warning("Select at least one app in the sidebar.")
        st.stop()


# ----------------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------------
def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2), legend_title_text="",
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def shade_feed_gap(fig, f=None):
    """Grey band over the late-April feed gap, if it is inside the selected dates."""
    f = f or get_filters()
    if f.start <= GAP_END and f.end >= GAP_START and not f.exclude_feed_gap:
        fig.add_vrect(x0=str(GAP_START), x1=str(GAP_END), fillcolor="#000", opacity=0.06, line_width=0,
                      annotation_text="feed gap", annotation_position="top left",
                      annotation_font_color=INK2, annotation_font_size=11)
    return fig


def fmt_int(x):
    return f"{int(round(x)):,}"


def empty_figure(text):
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(color=INK2))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, 200)
